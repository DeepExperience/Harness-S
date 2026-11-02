"""Versioned GRPO recipes and audits shared by every Harness."""
from collections import defaultdict
import json
import math
import os
from pathlib import Path

VERSION = 'group-only-trajectory-mean-v1'
LEGACY_VERSION = 'legacy-grpo-v1'


def algorithm_version():
    value = os.environ.get('HAPT_GRPO_ALGORITHM_VERSION', '')
    if value not in ('', VERSION, LEGACY_VERSION):
        raise ValueError('unknown GRPO algorithm version')
    return value


def enabled():
    return algorithm_version() == VERSION


def legacy_enabled():
    return algorithm_version() == LEGACY_VERSION


def validate_args(args):
    if legacy_enabled():
        expected = dict(advantage_estimator='grpo', normalize_advantages=True,
                        calculate_per_token_loss=False, rewards_normalization=True,
                        grpo_std_normalization=True, use_tis=True,
                        use_rollout_logprobs=False, tis_clip_low=0., tis_clip=2.,
                        eps_clip=.2, eps_clip_high=.28, entropy_coef=0., kl_coef=0.,
                        use_kl_loss=False, num_steps_per_rollout=1)
        if any(getattr(args, key, None) != value for key, value in expected.items()):
            raise ValueError('legacy GRPO runtime differs from the successful frozen recipe')
        if (any(getattr(args, key, None) for key in (
                'custom_advantage_function_path', 'custom_pg_loss_reducer_function_path',
                'custom_loss_function_path', 'custom_tis_function_path', 'mopd_teacher_config',
                'use_opd', 'use_critic')) or os.environ.get('DRESSAGE_CROSS_HARNESS_MODE')):
            raise ValueError('legacy GRPO forbids auxiliary or cross-Harness objectives')
        if (int(args.n_samples_per_prompt) != 8 or
                int(args.global_batch_size) != int(args.rollout_batch_size) * 8):
            raise ValueError('legacy GRPO requires complete task G8 groups')
        return
    if (getattr(args, 'advantage_estimator', None) != 'grpo'
            or getattr(args, 'normalize_advantages', False)
            or getattr(args, 'calculate_per_token_loss', False)
            or not getattr(args, 'rewards_normalization', True)
            or not getattr(args, 'grpo_std_normalization', True)
            or getattr(args, 'custom_advantage_function_path', None)
            or getattr(args, 'custom_pg_loss_reducer_function_path', None)
            or getattr(args, 'custom_loss_function_path', None)
            or getattr(args, 'use_opd', False) or getattr(args, 'use_critic', False)
            or getattr(args, 'mopd_teacher_config', None)
            or getattr(args, 'use_tis', False)
            or getattr(args, 'entropy_coef', 0) != 0
            or getattr(args, 'kl_coef', 0) != 0
            or getattr(args, 'use_kl_loss', False)
            or os.environ.get('DRESSAGE_CROSS_HARNESS_MODE')):
        raise ValueError('group-only GRPO forbids whitening, token-sum loss and auxiliary objectives')
    if int(args.n_samples_per_prompt) != 8:
        raise ValueError('group-only GRPO recipe requires G8')
    if int(args.global_batch_size) != int(args.rollout_batch_size) * 8:
        raise ValueError('GRPO loss denominator must preserve all planned B*8 trajectories')


def apply_update(decision, update):
    """The global controller makes this decision before any rank trains."""
    if decision['update_applied']:
        return update()
    return None


def trajectory_groups(samples):
    groups = defaultdict(lambda: defaultdict(list))
    for index, sample in enumerate(samples):
        if sample.group_index is None:
            if not sample.remove_sample:
                raise ValueError('live GRPO trajectory lacks group identity')
            continue  # Synthetic DP padding is not a real trajectory.
        metadata = sample.metadata or {}
        tid = metadata.get('parent_traj_id')
        if tid is None:
            tid = sample.rollout_id
        if tid is None:
            raise ValueError('GRPO trajectory lacks parent/rollout identity')
        groups[sample.group_index][str(tid)].append(index)
    return groups


def group_rewards(args, samples, raw_rewards):
    validate_args(args)
    from dressage.rollout.case_failure import is_case_failure_no_grad
    from dressage.training.reward_post_process import _normalize_group
    rewards = [0.] * len(samples)
    for trajectories in trajectory_groups(samples).values():
        if len(trajectories) != 8:
            raise ValueError('GRPO group must contain eight complete trajectories')
        members = list(trajectories.values())
        anchors = [max(indices, key=lambda i: int((samples[i].metadata or {}).get('segment_index', 0)))
                   for indices in members]
        advantages = _normalize_group([raw_rewards[i] for i in anchors], standardize=True)
        for indices, advantage in zip(members, advantages, strict=True):
            for i in indices:
                rewards[i] = 0. if is_case_failure_no_grad(samples[i]) else advantage
    return rewards


def math_receipt(args, samples, data):
    """Audit the actual converter output before DP/CP or microbatch partitioning."""
    if legacy_enabled():
        return legacy_math_receipt(args, samples, data)
    validate_args(args)
    groups = trajectory_groups(samples)
    if len(groups) != int(args.rollout_batch_size):
        raise ValueError('GRPO batch task count differs from planned B')
    rows = []
    effective = zero_groups = failed_positive = 0
    for group, trajectories in groups.items():
        if len(trajectories) != 8:
            raise ValueError('GRPO audit requires G8')
        group_signal = False
        for tid, indices in trajectories.items():
            anchor = max(indices, key=lambda i: int((samples[i].metadata or {}).get('segment_index', 0)))
            token_count = sum(sum(data['loss_masks'][i]) for i in indices)
            live_advantages = {data['rewards'][i] for i in indices if any(data['loss_masks'][i])}
            if len(live_advantages) > 1:
                raise ValueError('trajectory segments have different final advantages')
            advantage = next(iter(live_advantages), 0.)
            if not math.isfinite(advantage):
                raise ValueError('non-finite GRPO advantage')
            if any(data['rollout_mask_sums'][i] != token_count for i in indices):
                raise ValueError('whole-trajectory token denominator differs')
            signal = token_count > 0 and advantage != 0.
            group_signal |= signal
            raw = data['raw_reward'][anchor]
            failed_positive += int(raw == 0 and advantage > 0)
            rows.append(dict(group=group, trajectory=tid, reward=raw, advantage=advantage,
                             trainable_tokens=token_count, segments=len(indices),
                             nominal_weight=1./int(args.global_batch_size), has_signal=signal))
        effective += int(group_signal)
        zero_groups += int(not group_signal)
    if failed_positive:
        raise ValueError('failed trajectory has positive GRPO advantage')
    return dict(algorithm_version=VERSION, normalize_advantages=False,
                update_applied=effective > 0,
                skip_reason=None if effective else 'zero_trainable_group_advantages',
                effective_groups=effective, zero_signal_groups=zero_groups,
                total_groups=len(groups), failed_positive_advantages=failed_positive,
                trajectories=rows)


def legacy_math_receipt(args, samples, data):
    """Audit task-token denominators without changing old whitening or updates."""
    validate_args(args)
    from dressage.rollout.case_failure import is_case_failure_no_grad
    groups = trajectory_groups(samples)
    if len(groups) != int(args.rollout_batch_size):
        raise ValueError('legacy GRPO batch task count differs from planned B')
    rows = []
    total_tokens = 0
    weighted_sum = weighted_square = 0.
    all_no_grad = True
    for group, trajectories in groups.items():
        if len(trajectories) != 8:
            raise ValueError('legacy GRPO audit requires G8')
        members = [i for indices in trajectories.values() for i in indices]
        task_tokens = sum(sum(data['loss_masks'][i]) for i in members)
        denom = task_tokens * len(groups) / int(args.global_batch_size)
        for tid, indices in trajectories.items():
            anchor = max(indices, key=lambda i: int((samples[i].metadata or {}).get('segment_index', 0)))
            all_no_grad &= is_case_failure_no_grad(samples[anchor])
            token_count = sum(sum(data['loss_masks'][i]) for i in indices)
            if any(data['rollout_mask_sums'][i] != denom for i in indices):
                raise ValueError('legacy GRPO task-token denominator differs')
            live = {data['rewards'][i] for i in indices if any(data['loss_masks'][i])}
            if len(live) > 1:
                raise ValueError('legacy trajectory segments have different group advantages')
            advantage = next(iter(live), 0.)
            if not math.isfinite(advantage):
                raise ValueError('non-finite legacy GRPO advantage')
            total_tokens += token_count
            weighted_sum += token_count * advantage
            weighted_square += token_count * advantage * advantage
            rows.append(dict(group=group, trajectory=tid, reward=data['raw_reward'][anchor],
                             group_advantage=advantage, trainable_tokens=token_count,
                             task_trainable_tokens=task_tokens, loss_denominator=denom,
                             segments=len(indices), nominal_weight=(
                                 token_count / task_tokens / len(groups) if task_tokens else 0.)))
    mean = weighted_sum / total_tokens if total_tokens else 0.
    variance = max(0., weighted_square / total_tokens - mean * mean) if total_tokens else 0.
    if total_tokens >= 2:
        variance *= total_tokens / (total_tokens - 1)
    for row in rows:
        row['expected_token_whitened_advantage'] = (
            (row['group_advantage'] - mean) / math.sqrt(variance + 1e-8)
            if row['trainable_tokens'] else 0.)
    return dict(algorithm_version=LEGACY_VERSION, normalize_advantages=True, use_tis=True,
                update_applied=not all_no_grad,
                skip_reason='no_trainable_tokens' if all_no_grad else None,
                total_groups=len(groups), token_whitening_mean=mean,
                token_whitening_variance=variance, trajectories=rows)


def write_receipt(args, samples, data, rollout_id):
    receipt = math_receipt(args, samples, data)
    receipt['consumed_batches'] = rollout_id + 1
    path = Path(os.environ['HAPT_JOINT_RUN_ROOT']) / f'training-audit/grpo-step-{rollout_id+1:04d}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    temporary.replace(path)
