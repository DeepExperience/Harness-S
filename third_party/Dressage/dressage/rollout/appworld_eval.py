"""Task-level AppWorld validation/test logging for multi-segment rollouts."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _reward(sample: Any) -> float:
    value = getattr(sample, "reward", 0.0)
    if isinstance(value, dict):
        value = value.get("reward", value.get("score", 0.0))
    return float(value or 0.0)


def _status(sample: Any) -> str:
    value = getattr(sample, "status", "unknown")
    if hasattr(value, "value"):
        value = value.value
    return str(value)


def _tool_execution_errors(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    count = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        serialized = json.dumps(message.get("content"), ensure_ascii=False).lower()
        count += int(
            bool(message.get("error"))
            or "execution failed" in serialized
            or '"iserror":true' in serialized.replace(" ", "")
            or "<tool_use_error>" in serialized
        )
    return count


def _failure_type(success: bool, metadata: dict[str, Any]) -> str:
    if success:
        return "success"
    diagnostics = metadata.get("appworld_blackbox_reward") or {}
    if diagnostics.get("failure_reason") == "no_environment_actions":
        return "no_environment_action"
    steps = int(diagnostics.get("steps") or diagnostics.get("action_count") or 0)
    if steps >= 30:
        return "environment_step_budget"
    early_stop = str(metadata.get("blackbox_agent_early_stop_kind") or "").lower()
    if "max" in early_stop and "token" in early_stop:
        return "model_token_budget"
    if "max" in early_stop and ("step" in early_stop or "turn" in early_stop):
        return "model_step_budget"
    if _tool_execution_errors(metadata.get("messages")):
        return "tool_execution_error"
    if diagnostics.get("failure_reason") == "verifier_error":
        return "verifier_error"
    return "official_evaluator_failure"


def _terminal_samples(samples: list[Any]) -> list[Any]:
    grouped: dict[tuple[str, int], list[Any]] = defaultdict(list)
    trajectory_ids: dict[tuple[str, int], set[str]] = defaultdict(set)
    for sample in samples:
        metadata = getattr(sample, "metadata", None) or {}
        instance_id = str(metadata.get("instance_id") or "")
        index = int(getattr(sample, "index", -1))
        if not instance_id or index < 0:
            raise ValueError("AppWorld eval sample is missing instance_id/index")
        key = (instance_id, index)
        grouped[key].append(sample)
        trajectory_id = metadata.get("parent_traj_id") or metadata.get("session_id")
        if trajectory_id:
            trajectory_ids[key].add(str(trajectory_id))
    duplicate_trajectories = {
        key: values for key, values in trajectory_ids.items() if len(values) != 1
    }
    if duplicate_trajectories:
        raise ValueError(
            f"AppWorld eval expected one trajectory per task: {duplicate_trajectories}"
        )
    return [
        max(group, key=lambda sample: int((getattr(sample, "metadata", None) or {}).get("segment_index", 0)))
        for _key, group in sorted(grouped.items(), key=lambda item: item[0][1])
    ]


def _task_result(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None) or {}
    diagnostics = metadata.get("appworld_blackbox_reward") or {}
    reward = _reward(sample)
    success = diagnostics.get("success") is True or reward >= 1.0
    return {
        "task_id": str(metadata.get("task_id") or ""),
        "instance_id": str(metadata.get("instance_id") or ""),
        "task_family": str(metadata.get("task_family") or ""),
        "task_variant": int(metadata.get("task_variant") or 0),
        "original_task_split": str(metadata.get("original_task_split") or ""),
        "success": success,
        "reward": reward,
        "failure_type": _failure_type(success, metadata),
        "environment_actions": int(diagnostics.get("action_count") or 0),
        "response_tokens": int(getattr(sample, "response_length", 0) or 0),
        "segment_count": int(metadata.get("segment_count") or 1),
        "tool_execution_errors": _tool_execution_errors(metadata.get("messages")),
        "status": _status(sample),
        "session_id": str(metadata.get("parent_traj_id") or metadata.get("session_id") or ""),
        "sampling_seed_base": int(
            metadata.get("backend_options", {}).get("proxy", {}).get("sampling_seed_base", 0)
        ),
        "diagnostics": diagnostics,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("AppWorld evaluation produced no task results")
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_family[result["task_family"]].append(result)
    incomplete = {
        family: len(items) for family, items in by_family.items() if len(items) != 3
    }
    if incomplete:
        raise ValueError(f"AppWorld eval families are incomplete: {incomplete}")

    def rate(items: list[dict[str, Any]]) -> float:
        return sum(bool(item["success"]) for item in items) / len(items) if items else 0.0

    normal = [item for item in results if item["original_task_split"] == "test_normal"]
    challenge = [item for item in results if item["original_task_split"] == "test_challenge"]
    robust_successes = sum(all(item["success"] for item in items) for items in by_family.values())
    failures = Counter(item["failure_type"] for item in results)
    return {
        "task_count": len(results),
        "successes": sum(bool(item["success"]) for item in results),
        "overall_success_rate": rate(results),
        "normal_task_count": len(normal),
        "normal_success_rate": rate(normal),
        "challenge_task_count": len(challenge),
        "challenge_success_rate": rate(challenge),
        "family_count": len(by_family),
        "family_robust_successes": robust_successes,
        "family_robust_success_rate": robust_successes / len(by_family),
        "mean_environment_actions": sum(item["environment_actions"] for item in results) / len(results),
        "mean_response_tokens": sum(item["response_tokens"] for item in results) / len(results),
        "tool_execution_errors": sum(item["tool_execution_errors"] for item in results),
        "failure_types": dict(sorted(failures.items())),
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"refusing to overwrite immutable eval result: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def log_eval_rollout_data(
    rollout_id: int,
    args: Any,
    data: dict[str, dict[str, list[Any]]],
    extra_metrics: dict[str, Any] | None,
) -> bool:
    """Collapse segments, validate exact task coverage, and persist JSON."""

    expected = int(os.environ.get("DRESSAGE_APPWORLD_EVAL_EXPECTED_TASKS", "0"))
    stage = os.environ.get("DRESSAGE_APPWORLD_EVAL_STAGE", "validation").strip()
    policy_label = os.environ.get("DRESSAGE_APPWORLD_EVAL_POLICY_LABEL", "unknown").strip()
    output_dir = Path(
        os.environ.get(
            "DRESSAGE_APPWORLD_EVAL_OUTPUT_DIR",
            str(Path(getattr(args, "save", ".")) / "eval-results"),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logical_step = 0 if rollout_id == 0 else int(rollout_id) + 1
    if stage == "validation" and policy_label == "auto":
        policy_label = "base-h0" if logical_step == 0 else f"step-{logical_step}-h0"

    for dataset_name, dataset in data.items():
        terminals = _terminal_samples(list(dataset.get("samples") or []))
        results = [_task_result(sample) for sample in terminals]
        if expected and len(results) != expected:
            raise RuntimeError(
                f"{stage} produced {len(results)} tasks; expected exactly {expected}"
            )
        task_ids = [item["task_id"] for item in results]
        if not all(task_ids) or len(task_ids) != len(set(task_ids)):
            raise RuntimeError(f"{stage} contains missing or duplicate task ids")
        summary = summarize(results)
        payload = {
            "schema_version": "hapt.appworld-task-eval.v1",
            "stage": stage,
            "policy_label": policy_label,
            "dataset": dataset_name,
            "rollout_id": int(rollout_id),
            "logical_step": logical_step,
            "summary": summary,
            "results": results,
        }
        path = output_dir / f"{stage}-step-{logical_step:03d}.json"
        _atomic_write(path, payload)

        dataset["samples"] = terminals
        dataset["rewards"] = [float(item["reward"]) for item in results]
        dataset["truncated"] = [item["status"].lower() == "truncated" for item in results]
        if isinstance(extra_metrics, dict):
            prefix = f"eval/{dataset_name}"
            extra_metrics[f"{prefix}/overall_success_rate"] = summary["overall_success_rate"]
            extra_metrics[f"{prefix}/normal_success_rate"] = summary["normal_success_rate"]
            extra_metrics[f"{prefix}/challenge_success_rate"] = summary["challenge_success_rate"]
            extra_metrics[f"{prefix}/family_robust_success_rate"] = summary[
                "family_robust_success_rate"
            ]

    return False


def generate_rollout(args: Any, rollout_id: int, data_source: Any, evaluation: bool = False):
    """Run Slime eval while respecting the colocated proxy drain boundary."""

    if not evaluation:
        raise ValueError("AppWorld eval rollout function is evaluation-only")

    from dressage.rollout.sync_rollout import (
        _pause_proxy_for_natural_drain,
        _resume_proxy_after_sync_boundary,
        _wait_for_rollout_servers_to_drain,
    )
    from slime.rollout.sglang_rollout import eval_rollout
    from slime.utils.async_utils import run

    async def _run():
        await _resume_proxy_after_sync_boundary()
        try:
            output, _aborted = await eval_rollout(args, rollout_id)
            return output
        finally:
            await _pause_proxy_for_natural_drain()
            await _wait_for_rollout_servers_to_drain(args)

    return run(_run())
