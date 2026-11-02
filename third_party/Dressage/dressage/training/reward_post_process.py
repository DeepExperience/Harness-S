"""Slime reward post-processing hook for GRPO advantage normalization.

Registered via ``--custom-reward-post-process-path``.  Handles:
  - Per-group GRPO mean-subtraction (with optional std normalization).
  - Multi-segment trajectory broadcast: the anchor segment's normalized
    reward is copied to every sibling segment of the trajectory.
"""

from __future__ import annotations

import logging
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CROSS_HARNESS_MODE = "mixed_baseline"
_CROSS_HARNESS_GROUP_LAYOUTS = ("distinct", "shared")


def cross_harness_config() -> tuple[float, tuple[str, ...], str] | None:
    """Return the fail-closed paired mixed-baseline configuration.

    The hook remains byte-for-byte compatible with legacy Dressage behavior
    unless ``DRESSAGE_CROSS_HARNESS_MODE=mixed_baseline`` is explicitly set.
    Lambda zero is the B4 paired-independent condition; positive lambda is M1.
    ``distinct`` preserves the historical two-group layout, while ``shared``
    requires both Harness cells to belong to one atomic rollout group.
    """

    mode = os.environ.get("DRESSAGE_CROSS_HARNESS_MODE", "").strip()
    if not mode:
        return None
    if mode != _CROSS_HARNESS_MODE:
        raise ValueError(
            "DRESSAGE_CROSS_HARNESS_MODE must be empty or "
            f"{_CROSS_HARNESS_MODE!r}; got {mode!r}"
        )
    try:
        coupling = float(os.environ.get("DRESSAGE_CROSS_HARNESS_LAMBDA", ""))
    except ValueError as exc:
        raise ValueError("DRESSAGE_CROSS_HARNESS_LAMBDA must be a float") from exc
    if not 0.0 <= coupling <= 1.0:
        raise ValueError(
            f"DRESSAGE_CROSS_HARNESS_LAMBDA must be in [0, 1]; got {coupling}"
        )
    harnesses = tuple(
        value.strip()
        for value in os.environ.get(
            "DRESSAGE_CROSS_HARNESS_EXPECTED_HARNESSES",
            "minimal,standard-core-native",
        ).split(",")
        if value.strip()
    )
    if len(harnesses) < 2 or len(set(harnesses)) != len(harnesses):
        raise ValueError(
            "DRESSAGE_CROSS_HARNESS_EXPECTED_HARNESSES must contain at least "
            "two unique harness ids"
        )
    group_layout = os.environ.get(
        "DRESSAGE_CROSS_HARNESS_GROUP_LAYOUT", "distinct"
    ).strip()
    if group_layout not in _CROSS_HARNESS_GROUP_LAYOUTS:
        raise ValueError(
            "DRESSAGE_CROSS_HARNESS_GROUP_LAYOUT must be one of "
            f"{_CROSS_HARNESS_GROUP_LAYOUTS}; got {group_layout!r}"
        )
    return coupling, harnesses, group_layout


def _policy_version(sample: Any) -> str:
    metadata = getattr(sample, "metadata", None) or {}
    start = str(metadata.get("dressage_start_token_version", ""))
    end = str(metadata.get("dressage_end_token_version", ""))
    if not start or start == "-1" or start != end:
        raise ValueError(
            "paired cross-Harness trajectory must have one stable real policy "
            f"version; session={metadata.get('session_id')} start={start!r} end={end!r}"
        )
    return start


def _write_cross_harness_audit(payload: dict[str, Any]) -> None:
    path_value = os.environ.get("DRESSAGE_CROSS_HARNESS_AUDIT_PATH", "").strip()
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o640)
    try:
        os.write(descriptor, line.encode("utf-8"))
    finally:
        os.close(descriptor)


def _mixed_baseline_rewards(
    samples: list,
    raw_rewards: list[float],
    *,
    coupling: float,
    expected_harnesses: tuple[str, ...],
    group_layout: str,
    parent_groups: dict[str, list[int]],
    parent_anchor: dict[str, int],
) -> list[float]:
    """Compute W/C/A_CH once per trajectory and broadcast to its segments."""

    representative_indices = set(parent_anchor.values())
    representatives: list[int] = []
    for index, sample in enumerate(samples):
        if getattr(sample, "remove_sample", False):
            continue
        metadata = getattr(sample, "metadata", None) or {}
        if metadata.get("parent_traj_id"):
            if index in representative_indices:
                representatives.append(index)
        else:
            representatives.append(index)

    packet: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    versions: dict[str, set[str]] = defaultdict(set)
    group_indices: dict[tuple[str, str], set[int]] = defaultdict(set)
    for index in representatives:
        sample = samples[index]
        metadata = getattr(sample, "metadata", None) or {}
        task_id = str(metadata.get("task_id") or metadata.get("instance_id") or "")
        harness_id = str(metadata.get("harness_id") or "")
        if not task_id or not harness_id:
            raise ValueError(
                "paired cross-Harness samples require metadata.task_id and metadata.harness_id"
            )
        if sample.group_index is None:
            raise ValueError(
                f"paired cross-Harness sample has no group_index: task={task_id} harness={harness_id}"
            )
        packet[task_id][harness_id].append(index)
        versions[task_id].add(_policy_version(sample))
        group_indices[(task_id, harness_id)].add(int(sample.group_index))

    if not packet:
        raise ValueError("paired cross-Harness packet has no live trajectories")

    expected_set = set(expected_harnesses)
    all_k: set[int] = set()
    for task_id, by_harness in packet.items():
        actual_set = set(by_harness)
        if actual_set != expected_set:
            raise ValueError(
                f"incomplete paired task {task_id}: expected harnesses={sorted(expected_set)} "
                f"actual={sorted(actual_set)}"
            )
        counts = {harness: len(by_harness[harness]) for harness in expected_harnesses}
        if len(set(counts.values())) != 1 or next(iter(counts.values())) < 2:
            raise ValueError(
                f"paired task {task_id} requires equal K>=2 per harness; got {counts}"
            )
        all_k.add(next(iter(counts.values())))
        if len(versions[task_id]) != 1:
            raise ValueError(
                f"paired task {task_id} used multiple policy versions: {sorted(versions[task_id])}"
            )
        for harness_id in expected_harnesses:
            key = (task_id, harness_id)
            if len(group_indices[key]) != 1:
                raise ValueError(
                    f"paired semantics require one group_index for {key}; "
                    f"got {sorted(group_indices[key])}"
                )
        task_group_indices = {
            next(iter(group_indices[(task_id, harness_id)]))
            for harness_id in expected_harnesses
        }
        if group_layout == "distinct" and len(task_group_indices) != len(expected_harnesses):
            raise ValueError(
                "distinct cross-Harness layout requires one different group_index "
                f"per Harness for task {task_id}; got {sorted(task_group_indices)}"
            )
        if group_layout == "shared" and len(task_group_indices) != 1:
            raise ValueError(
                "shared cross-Harness layout requires one common group_index "
                f"for task {task_id}; got {sorted(task_group_indices)}"
            )
    if len(all_k) != 1:
        raise ValueError(f"paired packet must use one fixed K across tasks; got {sorted(all_k)}")

    k = next(iter(all_k))
    rho = (k - 1) / k
    rewards = list(raw_rewards)
    audit_rows: list[dict[str, Any]] = []
    for task_id, by_harness in packet.items():
        means = {
            harness_id: sum(raw_rewards[index] for index in indices) / len(indices)
            for harness_id, indices in by_harness.items()
        }
        for harness_id in expected_harnesses:
            other_indices = [
                index
                for other_harness, indices in by_harness.items()
                if other_harness != harness_id
                for index in indices
            ]
            other_mean = sum(raw_rewards[index] for index in other_indices) / len(other_indices)
            for index in by_harness[harness_id]:
                reward = raw_rewards[index]
                within = reward - means[harness_id]
                cross = rho * (reward - other_mean)
                advantage = (1.0 - coupling) * within + coupling * cross
                rewards[index] = advantage
                metadata = samples[index].metadata
                diagnostic = {
                    "task_id": task_id,
                    "harness_id": harness_id,
                    "group_index": int(samples[index].group_index),
                    "reward": reward,
                    "within_advantage": within,
                    "cross_advantage": cross,
                    "advantage": advantage,
                    "own_mean": means[harness_id],
                    "other_mean": other_mean,
                    "lambda": coupling,
                    "rho": rho,
                    "k": k,
                    "policy_version": next(iter(versions[task_id])),
                }
                metadata["cross_harness_advantage"] = diagnostic
                audit_rows.append(
                    {
                        **diagnostic,
                        "session_id": metadata.get("session_id"),
                        "parent_traj_id": metadata.get("parent_traj_id"),
                    }
                )

    _broadcast_to_segments(rewards, parent_groups, parent_anchor)
    for _parent_id, indices in parent_groups.items():
        anchor = parent_anchor[_parent_id]
        diagnostic = (samples[anchor].metadata or {}).get("cross_harness_advantage")
        for index in indices:
            samples[index].metadata["cross_harness_advantage"] = dict(diagnostic)

    _write_cross_harness_audit(
        {
            "schema_version": "hapt.cross-harness-advantage.v2",
            "lambda": coupling,
            "rho": rho,
            "k": k,
            "group_layout": group_layout,
            "expected_harnesses": list(expected_harnesses),
            "task_count": len(packet),
            "rows": audit_rows,
        }
    )
    return rewards


def _compute_parent_groups(
    samples: list,
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Group sample indices by parent_traj_id and pick each trajectory's anchor.

    Returns:
      parent_groups: ptid -> list of sample indices belonging to that
        trajectory (in input order).
      parent_anchor: ptid -> the single anchor index of that trajectory.
        The anchor is the segment with the highest segment_index — by plan
        §1.3 it's the only segment that ran reward_fn, so it carries the
        trajectory's terminal reward (every other segment was pre-set to
        reward=0.0 by multi_segment.expand_segments_to_samples).
    """
    from dressage.rollout.case_failure import is_case_failure_no_grad
    parent_groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        if getattr(s, "remove_sample", False) and not is_case_failure_no_grad(s):
            continue
        ptid = s.metadata.get("parent_traj_id") if s.metadata else None
        if ptid:
            parent_groups[ptid].append(i)

    def _segment_index_of(idx: int) -> int:
        meta = samples[idx].metadata or {}
        return int(meta.get("segment_index", 0))

    parent_anchor: dict[str, int] = {}
    for ptid, seg_indices in parent_groups.items():
        parent_anchor[ptid] = max(seg_indices, key=_segment_index_of)

    return parent_groups, parent_anchor


def _broadcast_to_segments(
    values: list[float],
    parent_groups: dict[str, list[int]],
    parent_anchor: dict[str, int],
) -> None:
    """In-place broadcast values[anchor_idx] to every segment of each parent.

    Used for ``rewards`` (= GRPO advantage) so every segment of a trajectory
    sees the same advantage. NOT used for ``raw_rewards`` — that value stays
    sparse (anchor carries the trajectory's terminal reward, every other
    segment is 0) so downstream consumers can sum within a trajectory to
    recover the terminal reward (relied on by
    ``log_rollout_data.compute_trajectory_mean_raw_reward`` for the wandb
    trajectory-level mean metric, and by ``slime`` for correct-length stats).
    """
    for ptid, seg_indices in parent_groups.items():
        anchor_idx = parent_anchor[ptid]
        if anchor_idx >= len(values):
            logger.warning(
                "parent_traj_id=%s anchor index %d out of range",
                ptid, anchor_idx,
            )
            continue
        representative = values[anchor_idx]
        for idx in seg_indices:
            values[idx] = representative


def _normalize_group(values, *, standardize):
    mean = sum(values) / len(values) if values else 0.
    centered = [v - mean for v in values]
    if standardize:
        std = (sum(v*v for v in centered) / len(centered)) ** .5 if centered else 0.
        if std > 1e-6:
            centered = [v/std for v in centered]
    return centered


def _dynamic_mixed_rewards(args, samples, raw_rewards):
    """Same GRPO math, grouped by the eight immutable clone slots, not role/hash.

    A real zero-gradient trajectory still counts toward the G8 reward baseline.
    Its zero loss mask prevents a gradient; synthetic padding does not count.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for index, sample in enumerate(samples):
        if sample.group_index is None:
            if not getattr(sample, "remove_sample", False):
                raise ValueError("live mixed sample lacks logical group identity")
            continue
        metadata = sample.metadata or {}
        clone = metadata.get("sampling_clone_index")
        if (type(clone) is not int or not 0 <= clone < 8
                or metadata.get("harness_mix") != getattr(args, "harness_mix", "h0_active_equal")):
            raise ValueError("mixed GRPO sample lacks immutable clone identity")
        groups[sample.group_index][clone].append(index)
    rewards = [0.] * len(samples)
    for clones in groups.values():
        if set(clones) != set(range(8)):
            raise ValueError("mixed GRPO requires eight complete clone outcomes")
        anchors = [max(clones[c], key=lambda i: int((samples[i].metadata or {}).get("segment_index", 0))) for c in range(8)]
        values = [raw_rewards[i] for i in anchors]
        if getattr(args, "rewards_normalization", True):
            values = _normalize_group(values, standardize=getattr(args, "grpo_std_normalization", False))
        for clone, value in enumerate(values):
            for index in clones[clone]:
                rewards[index] = value
    return rewards


def reward_post_process(
    args: Any, samples: list
) -> tuple[list[float], list[float]]:
    """Compute raw and normalized rewards for GRPO advantage estimation.

    For standard trajectories:
      - raw_rewards = [s.reward for s in samples]
      - rewards = normalize per group_index group (GRPO advantage).
      - Samples marked remove_sample do not participate in normalization.

    For rewrite-aware segmented trajectories (samples carry
    ``metadata['parent_traj_id']``):
      - Group segments by parent_traj_id.
      - Normalize at the parent level (anchor = highest segment_index).
      - Broadcast the same advantage to every segment of the trajectory.

    ``raw_rewards`` is intentionally NOT broadcast: it stays sparse, with
    the anchor segment carrying the trajectory's terminal reward and every
    other segment carrying the placeholder 0.0 set by
    ``multi_segment.expand_segments_to_samples``. Broadcasting raw would
    make a long trajectory contribute N times more weight to wandb's
    ``rollout/raw_reward`` than a single-segment trajectory with the same
    terminal reward. The trajectory-level mean is emitted as a separate
    scalar (``raw_reward_trajectory_mean``) by
    ``dressage.rollout.log_rollout`` instead — see that module for
    the sum-per-trajectory invariant relied on here.

    Returns: (raw_rewards, processed_rewards)
    """
    raw_rewards = []
    for s in samples:
        r = s.reward
        if isinstance(r, dict):
            reward_key = getattr(args, "reward_key", None)
            r = r[reward_key] if reward_key else 0.0
        raw_rewards.append(float(r) if r is not None else 0.0)

    parent_groups, parent_anchor = _compute_parent_groups(samples)

    from dressage.training.grpo_algorithm import enabled, group_rewards, legacy_enabled
    if enabled():
        return raw_rewards, group_rewards(args, samples, raw_rewards)

    paired_config = cross_harness_config()
    if (not legacy_enabled() and getattr(args, "harness_mix", "active_only")
            in {"h0_active_equal", "pool_equal", "router_topk", "pool_weighted", "fixed_backend_equal"}):
        if paired_config is not None or getattr(args, "advantage_estimator", "grpo") != "grpo":
            raise ValueError("dynamic mixed rollout requires ordinary pooled GRPO")
        return raw_rewards, _dynamic_mixed_rewards(args, samples, raw_rewards)
    if paired_config is not None:
        coupling, expected_harnesses, group_layout = paired_config
        return raw_rewards, _mixed_baseline_rewards(
            samples,
            raw_rewards,
            coupling=coupling,
            expected_harnesses=expected_harnesses,
            group_layout=group_layout,
            parent_groups=parent_groups,
            parent_anchor=parent_anchor,
        )

    advantage_estimator = getattr(args, "advantage_estimator", "grpo")
    do_normalization = getattr(args, "rewards_normalization", True)

    if advantage_estimator not in ("grpo", "gspo", "reinforce_plus_plus_baseline") or not do_normalization:
        rewards = list(raw_rewards)
        if parent_groups:
            _broadcast_to_segments(rewards, parent_groups, parent_anchor)
        return raw_rewards, rewards

    grpo_std = getattr(args, "grpo_std_normalization", False)

    _NONE_GROUP = -1

    from dressage.rollout.case_failure import is_case_failure_no_grad
    parent_representative_indices = set(parent_anchor.values())
    groups: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for i, s in enumerate(samples):
        if getattr(s, "remove_sample", False) and not is_case_failure_no_grad(s):
            continue
        gi = s.group_index if s.group_index is not None else _NONE_GROUP
        if i in parent_representative_indices:
            groups[gi].append((i, raw_rewards[i]))
        elif not (s.metadata and s.metadata.get("parent_traj_id")):
            groups[gi].append((i, raw_rewards[i]))

    rewards = list(raw_rewards)

    for gi, members in groups.items():
        indices = [m[0] for m in members]
        values = [m[1] for m in members]

        normalized = _normalize_group(values, standardize=advantage_estimator in ("grpo", "gspo") and grpo_std)

        for idx, norm_val in zip(indices, normalized):
            rewards[idx] = norm_val

    _broadcast_to_segments(rewards, parent_groups, parent_anchor)
    for i, sample in enumerate(samples):
        if is_case_failure_no_grad(sample):
            rewards[i] = 0.0

    return raw_rewards, rewards
