"""Pure-Python helpers for trajectory-mean wandb metrics.

Lives in a separate module from ``log_rollout_data.py`` (the slime-facing
wrap factory) so the trajectory-mean math stays testable without going
through the factory + slime mock dance. ``log_rollout_data`` delegates here
for the trajectory-equal raw_reward anchor mean; splitting keeps that file
focused on the slime patch glue.

No external dependencies — these functions take plain ``Iterable`` inputs
and run unchanged in CPU-only unit-test environments.
"""

from __future__ import annotations

from collections.abc import Iterable


def compute_trajectory_mean_raw_reward(
    parent_traj_ids: Iterable[str],
    raw_rewards: Iterable[float],
    segment_indices: Iterable[int],
) -> float | None:
    """Average terminal raw_reward across unique trajectories (one vote per trajectory).

    Slime's default ``rollout/raw_reward`` is ``sum(raw_reward) / N_samples``
    over the flat per-sample list. In multi-segment mode that weights a
    K-segment trajectory K times more than a 1-segment trajectory with the
    same terminal reward. This function computes the trajectory-equal-weighted
    mean instead, so wandb's ``rollout/raw_reward_trajectory_mean`` is a clean
    "fraction of trajectories that got reward 1.0" (for binary rewards).

    Anchor lookup: per trajectory we pick the segment with the maximum
    ``segment_index`` and use its ``raw_reward``. ``expand_segments_to_samples``
    puts ``reward=None`` only on the highest-``segment_index`` segment (so
    slime runs reward_fn there once); every other segment has the placeholder
    ``0.0``. Looking the anchor up explicitly (rather than summing within
    trajectory) is robust to changes in that invariant.

    Returns the mean as a float, or ``None`` when the input has no
    trajectories (caller should fall back to a sentinel like 0.0 so all DP
    ranks ship the same log_dict key set).
    """
    parent_traj_ids = list(parent_traj_ids)
    raw_rewards = list(raw_rewards)
    segment_indices = list(segment_indices)
    if not (len(parent_traj_ids) == len(raw_rewards) == len(segment_indices)):
        raise ValueError(
            f"length mismatch: parent_traj_ids={len(parent_traj_ids)}, "
            f"raw_rewards={len(raw_rewards)}, segment_indices={len(segment_indices)}"
        )

    per_traj_anchor: dict[str, tuple[int, float]] = {}
    for ptid, r, idx in zip(parent_traj_ids, raw_rewards, segment_indices):
        idx_int = int(idx)
        prev = per_traj_anchor.get(ptid)
        if prev is None or idx_int > prev[0]:
            per_traj_anchor[ptid] = (idx_int, float(r))
    if not per_traj_anchor:
        return None
    return sum(r for _, r in per_traj_anchor.values()) / len(per_traj_anchor)
