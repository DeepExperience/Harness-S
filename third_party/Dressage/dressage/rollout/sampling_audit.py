"""Environment-neutral diagnostics of the samples actually retained for GRPO.

Repeated text is a diagnostic, never a reason to resample, deduplicate, change
rewards, or abort training. Missing token arrays are unknown, not identical.
"""

from dressage.rollout.case_failure import is_case_failure_no_grad
from collections import defaultdict
import hashlib
import json


def group_sampling_audit(samples):
    trajectories = defaultdict(list)
    for sample in samples:
        metadata = getattr(sample, "metadata", None) or {}
        identity = metadata.get("parent_traj_id") or getattr(sample, "rollout_id", None)
        if identity is None:
            identity = getattr(sample, "index", None)
        # A dynamic mixed group is one task x G8, including when both roles
        # currently have exactly the same overlay hash. Legacy grouping stays intact.
        group = (getattr(sample, "group_index", None), metadata.get("harness_mix")
                 if metadata.get("harness_mix") in {"h0_active_equal", "pool_equal", "router_topk", "pool_weighted", "fixed_backend_equal"} else
                 str(metadata.get("harness_sha256") or metadata.get("harness_id") or ""))
        trajectories[(group, str(identity))].append(sample)
    groups = defaultdict(list)
    for (group, identity), segments in trajectories.items():
        segments.sort(key=lambda s: int((getattr(s, "metadata", None) or {}).get(
            "selected_segment_index", (getattr(s, "metadata", None) or {}).get("segment_index", 0)) or 0))
        token_arrays = [getattr(s, "tokens", None) for s in segments]
        no_grad_failure = all(is_case_failure_no_grad(s) for s in segments)
        available = not no_grad_failure and all(isinstance(tokens, list) and tokens for tokens in token_arrays)
        digest = (hashlib.sha256(json.dumps(token_arrays, separators=(",", ":")).encode()).hexdigest()
                  if available else None)
        metadata = getattr(segments[0], "metadata", None) or {}
        proxy = (metadata.get("backend_options") or {}).get("proxy") or {}
        seed_records = {}
        for segment in segments:
            extra = (getattr(segment, "metadata", None) or {}).get("proxy_extra_info") or {}
            for record in extra.get("sampling_audit", []):
                seed_records[record["step_id"]] = record.get("sampling_seed")
        groups[group].append({
            "trajectory_id": identity, "token_sha256": digest,
            "case_failure_no_grad": no_grad_failure,
            "harness_role": metadata.get("harness_role", "active"),
            "harness_sha256": metadata.get("harness_sha256"),
            "sampling_clone_index": metadata.get("sampling_clone_index"),
            "configured_seed_base": proxy.get("sampling_seed_base"),
            "engine_sampling_seeds": seed_records,
            "reward_executions": [
                {"segment_index": (getattr(segment, "metadata", None) or {}).get(
                    "selected_segment_index", 0),
                 **dict(segment.metadata["reward_execution"])}
                for segment in segments
                if isinstance((getattr(segment, "metadata", None) or {}).get("reward_execution"), dict)
            ],
        })
    rows = []
    for (index, harness), attempts in groups.items():
        hashes = [a["token_sha256"] for a in attempts if a["token_sha256"] is not None]
        seeds = [a["configured_seed_base"] for a in attempts if a["configured_seed_base"] is not None]
        first_engine_seeds = [next(iter(a["engine_sampling_seeds"].values()), None) for a in attempts]
        first_engine_seeds = [seed for seed in first_engine_seeds if seed is not None]
        rows.append({
            "group_index": index, "harness": harness, "attempts": len(attempts),
            "harness_roles": {role: sum(a["harness_role"] == role for a in attempts)
                              for role in sorted({a["harness_role"] for a in attempts})},
            "token_observed_attempts": len(hashes), "unique_token_sequences": len(set(hashes)),
            "all_tokens_identical": (len(set(hashes)) == 1 if len(hashes) == len(attempts) else None),
            "configured_seed_count": len(seeds), "unique_configured_seeds": len(set(seeds)),
            "initial_engine_seed_count": len(first_engine_seeds),
            "unique_initial_engine_seeds": len(set(first_engine_seeds)),
            "case_failure_no_grad_attempts": sum(a["case_failure_no_grad"] for a in attempts),
            "engine_seed_observed_attempts": sum(bool(a["engine_sampling_seeds"]) and
                all(seed is not None for seed in a["engine_sampling_seeds"].values()) for a in attempts),
            "trajectories": attempts,
        })
    return {"schema_version": 1, "groups": rows,
            "fully_identical_groups": sum(r["all_tokens_identical"] is True and r["attempts"] > 1 for r in rows),
            "seed_collision_groups": sum(r["unique_configured_seeds"] < r["configured_seed_count"] for r in rows),
            "engine_seed_collision_groups": sum(
                r["unique_initial_engine_seeds"] < r["initial_engine_seed_count"] for r in rows),
            "engine_seed_unobserved_groups": sum(
                r["engine_seed_observed_attempts"] + r["case_failure_no_grad_attempts"] < r["attempts"] for r in rows),
            "note": "Diagnostics only; repeated outputs are valid independent draws when seeds are correct."}
