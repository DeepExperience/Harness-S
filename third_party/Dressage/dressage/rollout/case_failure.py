"""Audited case failures with no generated tokens, shared by all environments.

The single token is transport padding, never a generated response.  Preserve
the scheduled trajectory and zero reward without inventing logprobs or seeds.
"""
from typing import Any


def is_case_failure_no_grad(sample: Any) -> bool:
    meta = getattr(sample, "metadata", None) or {}
    diag = meta.get("case_failure_diagnostics") or meta.get("swegym_reward") or {}
    return bool(meta.get("case_failure_no_grad") is True
                and diag.get("environment_failure") is True
                and diag.get("scoring_policy") == "environment_failure_zero"
                and diag.get("resolved") is False
                and getattr(sample, "remove_sample", False)
                and getattr(sample, "reward", None) == 0.0
                and getattr(sample, "response_length", None) == 0
                and getattr(sample, "loss_mask", None) == []
                and getattr(sample, "rollout_log_probs", None) == []
                and getattr(sample, "rollout_id", None) is not None)


def mark_case_failure_no_grad(sample: Any, *, session_id: str, diagnostics: dict) -> Any:
    from dressage.rollout.artifacts.samples import set_status
    if diagnostics.get("environment_failure") is not True or diagnostics.get("resolved") is not False:
        raise ValueError("no-gradient case requires explicit failed environment diagnostics")
    # An aborted backend clears sample.session_id so a retry starts fresh.
    # Its physical identity survives in metadata; never overwrite it with None.
    metadata = getattr(sample, 'metadata', None) or {}
    sample.metadata = metadata
    session_id = (session_id or metadata.get('parent_traj_id')
        or metadata.get('last_failed_session_id') or metadata.get('session_id')
        or getattr(sample, 'session_id', None) or diagnostics.get('session_id')
        or f'case-{getattr(sample, "group_index", "unknown")}-{sample.index}')
    sample.metadata.update(case_failure_no_grad=True, case_failure_diagnostics=diagnostics,
                           parent_traj_id=session_id, session_id=session_id,
                           segment_index=0, transport_padding_only=True)
    sample.session_id = session_id
    sample.rollout_id = sample.index
    sample.remove_sample = True
    sample.reward = 0.0
    sample.tokens = [0]
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.rollout_log_probs = []
    set_status(sample, "FAILED")
    return sample


def zero_case_failures(manifest=None):
    import os
    return (manifest.get('runtime', {}).get('case_failure_policy') == 'zero' if manifest is not None
            else os.environ.get('HAPT_CASE_FAILURE_POLICY', 'strict') == 'zero')


def is_global_failure(error, *, contract_errors=True):
    """Contract, numerical and storage corruption cannot be scored as a case."""
    import errno
    if isinstance(error, (MemoryError, ImportError)) or contract_errors and isinstance(error, (ValueError, KeyError, TypeError, AssertionError)):
        return True
    if isinstance(error, OSError) and error.errno in {errno.ENOSPC, errno.EIO, errno.EROFS, errno.EDQUOT}:
        return True
    message = str(error).lower()
    return any(s in message for s in ('out of memory', 'non-finite', 'nan loss', 'checkpoint corrupt',
                                      'identity mismatch', 'weight version', 'source changed', 'swegym_service_rebuild_required',
                                      'policy version mismatch', 'reward function',
                                      'evidence changed', 'artifact changed', 'frozen execution contract'))


def failure_diagnostics(error, *, phase, session_id=None):
    """Explicit case zero, never an invented official verifier result."""
    return dict(case_failure=True, environment_failure=True, resolved=False, success=False,
                reward=0.0, scoring_policy='environment_failure_zero', official_evaluation_ran=False,
                failure_phase=phase, exception_type=type(error).__name__, error=str(error),
                session_id=session_id)


def validate_case_failure_record(record):
    diag = record.get('reward_diagnostics') or {}
    if diag.get('case_failure') is not True:
        return False
    if (record.get('reward') != 0.0 or diag.get('reward') != 0.0
            or diag.get('environment_failure') is not True or diag.get('resolved') is not False
            or diag.get('success') is not False or diag.get('scoring_policy') != 'environment_failure_zero'
            or not diag.get('exception_type') or not diag.get('failure_phase')):
        raise ValueError('invalid case failure reward audit')
    return True


def record_case_failure(sample, diagnostics):
    """Keep evidence outside official grade receipts; storage errors propagate."""
    import os
    from pathlib import Path
    sample.metadata = getattr(sample, 'metadata', None) or {}
    sample.metadata['case_failure_diagnostics'] = diagnostics
    root = os.environ.get('HAPT_JOINT_RUN_ROOT')
    if root:
        from dressage.rollout.artifacts.writer import _write_json_file_atomic as atomic_json
        import hashlib
        identity = diagnostics.get('session_id') or str(getattr(sample, 'index', 'unknown'))
        path = Path(root) / 'case-failures' / (hashlib.sha256(identity.encode()).hexdigest() + '.json')
        atomic_json(path, {**diagnostics, 'task_id': sample.metadata.get('task_id'),
                          'sample_index': getattr(sample, 'index', None)})
    return diagnostics


def zero_failed_samples(samples, error, *, phase):
    """Retain attested generated tokens; mask a case with no usable tokens."""
    import math
    from dressage.rollout.artifacts.samples import set_status
    items = samples if isinstance(samples, list) else [samples]
    if not items:
        raise ValueError('empty trajectory has no scheduled identity')
    session = ((items[0].metadata or {}).get('parent_traj_id')
               or getattr(items[0], 'session_id', None) or f'case-{items[0].index}')
    diag = failure_diagnostics(error, phase=phase, session_id=session)
    usable = True
    for sample in items:
        meta = getattr(sample, 'metadata', None) or {}
        n = getattr(sample, 'response_length', 0) or 0
        tokens = getattr(sample, 'tokens', None) or []
        mask = getattr(sample, 'loss_mask', None) or []
        probs = getattr(sample, 'rollout_log_probs', None) or []
        usable &= bool(n > 0 and len(tokens) > n and len(mask) == len(probs) == n
                       and any(mask) and all(math.isfinite(p) for p in probs)
                       and meta.get('dressage_start_token_version') is not None
                       and meta.get('dressage_end_token_version') is not None)
    for sample in items:
        record_case_failure(sample, diag)
    if not usable:
        return mark_case_failure_no_grad(items[0], session_id=session, diagnostics=diag)
    for sample in items:
        sample.reward = 0.0
        set_status(sample, 'FAILED')
    return samples


async def run_case(operation, sample, *, phase, result_holder=None):
    """One shared boundary around generation AND scoring, below group gather."""
    try:
        return await operation()
    except Exception as exc:
        if not zero_case_failures() or is_global_failure(exc, contract_errors=False):
            raise
        from dressage.paddock.blackbox.failures import expected_abort_from_call_agent_exception
        if expected_abort_from_call_agent_exception(exc) is not None:
            raise
        from dressage.rollout.generate.runtime import discard_proxy_session_best_effort
        await discard_proxy_session_best_effort(getattr(sample, 'session_id', None))
        retained = result_holder.get('samples', sample) if result_holder is not None else sample
        return zero_failed_samples(retained, exc, phase=phase)


def batch_update_decision(records):
    """Only fully attested zero-token case failures can skip the Adam call."""
    if not records:
        raise ValueError('empty batch audit')
    skipped = all(row.get('case_failure_no_grad') is True for row in records)
    if skipped:
        for row in records:
            diag = row.get('reward_diagnostics') or {}
            if (row.get('reward') != 0.0 or row.get('response_length') != 0 or not row.get('remove_sample')
                    or diag.get('resolved') is not False or not diag.get('environment_failure')):
                raise ValueError('invalid zero-token batch audit')
    return {'update_applied': not skipped, 'skip_reason': 'no_trainable_tokens' if skipped else None}
