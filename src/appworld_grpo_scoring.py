"""Scoring fault policy shared by fixed-H0 training and evaluation."""
import time


def policy_error_kind(exc):
    """Only proven malformed-policy signatures, not arbitrary verifier errors."""
    message = str(exc)
    if not ("replay exited" in message or "verifier exited" in message):
        return None
    if all(marker in message for marker in
           ("appworld/requester.py", "jsonable_encoder", "'ellipsis' object is not iterable")):
        return 'invalid_non_json_tool_argument'
    # LibCST rejects a bare except followed by another handler while AppWorld
    # scans the sampled code's imports. This deterministic syntax failure is
    # not repaired by replaying the same actions three times. Do not catch all
    # parser/evaluator exceptions: unrelated SDK faults remain infrastructure.
    if all(marker in message for marker in
           ('appworld/common/safety_guard.py', 'appworld/common/code_parsing.py',
            'parse_imports', 'cst.parse_module', 'CSTValidationError',
            'The bare except: handler must be the last one.')):
        return 'invalid_python_tool_syntax'
    return None


def policy_argument_error(exc):
    return policy_error_kind(exc) is not None


def replay_with_retries(call, *, retries=2, sleep=time.sleep):
    """Retry the same recorded actions; never sample a replacement trajectory."""
    errors = []
    for attempt in range(retries + 1):
        try:
            return call(), attempt + 1, errors
        except Exception as exc:
            if policy_argument_error(exc):
                raise
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt == retries:
                raise
            sleep(attempt + 1)


def policy_failure(exc, action_count):
    kind = policy_error_kind(exc)
    if kind is None:
        raise ValueError('Unclassified verifier error is not a conservative policy failure')
    return {"success": False, "score": 0., "reward": 0.,
            "failure_reason": kind,
            "source": "conservative_policy_failure", "action_count": action_count,
            "policy_replay_error": f"{type(exc).__name__}: {exc}"}
