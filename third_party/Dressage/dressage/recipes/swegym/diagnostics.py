"""Separate runtime environment failures from invalid grading contracts."""

import re

ENVIRONMENT_FAILURE_REASONS = frozenset({
    "patch_extract_failed", "fresh_eval_timeout", "fresh_eval_request_failed",
    "fresh_eval_exception", "agent_environment_exception",
    "fresh_eval_start_failed", "fresh_eval_poll_failed",
    "evaluation_phase_failure",
    "case_evaluation_failure",
})


def normalize_policy_verifier_result(diag, *, candidate_patch_paths=()):
    """Every per-case policy grading failure is an audited zero reward.

    CPU admission still calls validate_verifier_result directly and stays
    strict. Cancellation/SystemExit are not caught here.
    """
    try:
        return _normalize_policy_verifier_result(diag, candidate_patch_paths=candidate_patch_paths)
    except Exception as exc:
        return policy_evaluation_failure(diag, error=exc)


def policy_evaluation_failure(diag=None, *, error=None):
    return {
        "resolved": False, "harness": "swegym",
        "failure_reason": "case_evaluation_failure",
        "original_failure_reason": diag.get("failure_reason") if isinstance(diag, dict) else None,
        "environment_failure": True, "scoring_policy": "environment_failure_zero",
        "raw_verifier_result": diag,
        "error": {"type": type(error).__name__, "message": str(error)} if error is not None else None,
    }


def _normalize_policy_verifier_result(diag, *, candidate_patch_paths=()):
    """Classify returned policy grading failures without relaxing CPU admission.

    A source build runs in the official script's prepare phase. Failure there
    does not by itself imply a broken environment. Preserve the original
    evidence, and never turn malformed markers/parser failures into rewards.
    """
    if not isinstance(diag, dict) or type(diag.get("resolved")) is not bool:
        validate_verifier_result(diag)
    result = dict(diag)
    reason = result.get("failure_reason")
    if (result["resolved"] is False and not result.get("parse_failure")
            and not result.get("environment_failure")
            and result.get("verifier_protocol") == "phased-pytest-v2"
            and reason in {"evaluation_phase_failure", "fresh_eval_timeout"}):
        phases = result.get("phase_returncodes")
        if not isinstance(phases, dict) or any(
                key not in {"prepare", "test", "cleanup"} or type(rc) is not int
                for key, rc in phases.items()):
            raise SWEGymVerifierError("SWE-Gym verifier phase results malformed")
        failed_phase = next((key for key in ("prepare", "cleanup")
                             if key in phases and phases[key] != 0), None)
        if reason == 'fresh_eval_timeout':
            failed_phase = next((key for key in ('prepare', 'test', 'cleanup')
                                 if phases.get(key) in (124, 137)), failed_phase)
        if failed_phase is None and reason == "evaluation_phase_failure":
            # A killed phase may never write its completion record. A valid
            # nonzero process exit remains a runtime failure, not a fake test.
            if type(result.get("eval_returncode")) is int and result["eval_returncode"] != 0:
                failed_phase = next((key for key in ("prepare", "test", "cleanup")
                                     if key not in phases), None)
        if reason == "evaluation_phase_failure" and failed_phase is None:
            raise SWEGymVerifierError("SWE-Gym phase failure has no failed phase")
        result["failure_stage"] = failed_phase or "evaluation"
        log = str(result.get("compiler_diagnostics") or "") + '\n' + str(result.get("test_log_tail") or "")
        # Require a compiler diagnostic in a file actually changed by the
        # candidate. A missing compiler/dependency or a generic pip failure is
        # insufficient to blame the patch.
        compiler_paths = set(re.findall(
            r"(?:^|\n)[ \t]*(?:/testbed/)?([^\s:]+\.pyx):\d+:\d+: (?:undeclared name not builtin:|Cannot assign type |Type is not specific)", log))
        implicated = sorted(compiler_paths.intersection(candidate_patch_paths))
        if (failed_phase == "prepare" and implicated
                and ("Cython.Compiler.Errors.CompileError:" in log or "Error compiling Cython file:" in log)
                and reason != "fresh_eval_timeout"):
            result.update(failure_reason="patch_build_failure",
                          original_failure_reason=reason,
                          patch_build_error_paths=implicated)
        else:
            result.update(environment_failure=True,
                          scoring_policy="environment_failure_zero")
    validate_verifier_result(result, allow_environment_failure=True)
    return result


def is_scored_environment_failure(diag):
    return (isinstance(diag, dict) and diag.get("resolved") is False
            and diag.get("environment_failure") is True
            and diag.get("scoring_policy") == "environment_failure_zero"
            and diag.get("failure_reason") in ENVIRONMENT_FAILURE_REASONS
            and not diag.get("parse_failure"))

class SWEGymVerifierError(RuntimeError):
    # Re-sampling cannot repair deterministic verifier/schema failures.
    retryable = False


def validate_verifier_result(diag, *, allow_environment_failure=False):
    if not isinstance(diag, dict) or type(diag.get("resolved")) is not bool:
        raise SWEGymVerifierError("SWE-Gym verifier result missing or malformed")
    reason = str(diag.get("failure_reason") or "")
    # Only explicit runtime failures emitted by the environment adapter may
    # become zero rewards. Offline environment admission stays strict by default.
    if allow_environment_failure and is_scored_environment_failure(diag):
        return diag
    if diag.get("environment_failure") or diag.get("scoring_policy"):
        raise SWEGymVerifierError("SWE-Gym environment failure is invalid or requires explicit zero scoring")
    infrastructure = {
        "missing_reward_marker", "invalid_reward_marker", "missing_official_log_parser",
        "patch_extract_failed", "verifier_error", "after_agent_timeout",
        "fresh_eval_timeout", "fresh_eval_request_failed", "fresh_eval_exception", "evaluation_phase_failure",
    }
    if (diag.get("parse_failure") or diag.get("error") or diag.get("timed_out")
            or reason in infrastructure or reason.startswith("official_log_parser_exception:")):
        raise SWEGymVerifierError(f"SWE-Gym verifier infrastructure failure: {reason or 'unclassified'}")
    # Empty test output may be caused by a model's import/syntax error. An
    # explicit parser exception is infrastructure; a nonzero test exit is a
    # legitimate failed patch. Missing tests with a successful exit is broken.
    if reason == "test_log_parse_failure" and (diag.get("verifier_protocol") == "phased-pytest-v2" or not diag.get("eval_returncode")):
        raise SWEGymVerifierError("SWE-Gym verifier emitted no parsed tests with a successful/unknown exit")
    return diag
