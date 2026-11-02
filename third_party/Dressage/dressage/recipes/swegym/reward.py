"""Reward handling for SWE-Gym blackbox rollouts."""

from __future__ import annotations

import json
import logging
from typing import Any

from dressage.recipes.swegym.integrity import zero_reward_for_integrity_violation
from dressage.recipes.swegym.diagnostics import normalize_policy_verifier_result, policy_evaluation_failure
from dressage.reward.registry import register_reward

logger = logging.getLogger(__name__)

REWARD_MARKER = "DRESSAGE_SWEGYM_REWARD_JSON="


def _marker_payload(metadata: dict[str, Any]) -> dict[str, Any] | None:
    for record in metadata.get("execute_cmds", []):
        if (
            record.get("stage") != "after_agent"
            or record.get("name") != "swegym_harness"
        ):
            continue
        result = record.get("cmd_result") or {}
        text = "\n".join(
            str(result.get(key) or "") for key in ("stdout", "stderr", "raw")
        )
        for line in text.splitlines():
            if not line.startswith(REWARD_MARKER):
                continue
            try:
                payload = json.loads(line[len(REWARD_MARKER) :])
            except json.JSONDecodeError:
                return {
                    "parse_failure": True,
                    "failure_reason": "invalid_reward_marker",
                }
            if isinstance(payload, dict):
                return payload
    return None


@register_reward("swegym_harness_marker")
def swegym_harness_marker_reward(
    sample: Any, *, args: Any = None, **kwargs: Any
) -> float:
    """Return the official-harness binary outcome emitted by the sandbox hook."""

    del args, kwargs
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        sample.metadata = metadata

    try:
        diag = metadata.get("swegym_runtime_failure") or _marker_payload(metadata)
    except Exception as exc:
        diag = policy_evaluation_failure(error=exc)
    if diag is None:
        diag = {"parse_failure": True, "failure_reason": "missing_reward_marker"}

    # Retain diagnostics even when infrastructure validation raises and the
    # sampler retries the group; otherwise the useful test traceback is lost.
    metadata["swegym_reward"] = diag
    diag = normalize_policy_verifier_result(diag, candidate_patch_paths=
        (metadata.get("swegym_eval") or {}).get("graded_patch_paths", ()))

    raw_reward = 1.0 if diag.get("resolved") is True else 0.0
    reward = zero_reward_for_integrity_violation(metadata, raw_reward)
    diag["raw_reward"] = raw_reward
    diag["reward"] = reward
    diag["integrity_violation"] = reward != raw_reward
    metadata["swegym_reward"] = diag
    logger.info(
        "SWE-Gym reward=%.1f raw=%.1f integrity=%s instance=%s harness=%s session=%s "
        "f2p=%s/%s p2p=%s/%s reason=%s",
        reward,
        raw_reward,
        diag["integrity_violation"],
        metadata.get("instance_id", "?"),
        metadata.get("harness_id", "?"),
        metadata.get("session_id", getattr(sample, "session_id", "?")),
        diag.get("fail_to_pass_success", 0),
        diag.get("fail_to_pass_total", 0),
        diag.get("pass_to_pass_success", 0),
        diag.get("pass_to_pass_total", 0),
        diag.get("failure_reason", ""),
    )
    return reward
