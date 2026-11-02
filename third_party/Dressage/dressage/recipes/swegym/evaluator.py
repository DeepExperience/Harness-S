"""Fresh-sandbox SWE-Gym evaluation for blackbox rollouts."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import shlex
import time
import uuid
import zlib
from pathlib import Path
from typing import Any

import httpx

from dressage.recipes.swegym.patches import extract_patch_command, prepare_patch, is_protected_path, changed_patch_paths
from dressage.config import proxy_url
from dressage.paddock.blackbox.execute_hooks import (
    BlackboxExecuteCmd,
    BlackboxExecuteCmdSchedule,
    execute_blackbox_cmds_for_stage,
    maybe_await,
)
from dressage.paddock.lifecycle import terminate_paddock_best_effort
from dressage.recipes.swegym.verifier_slots import limited_verifier
from dressage.recipes.swegym.deadlines import bounded, seconds
from dressage.recipes.swegym.eval_artifacts import save_eval_logs
from dressage.recipes.swegym.diagnostics import ENVIRONMENT_FAILURE_REASONS

logger = logging.getLogger(__name__)

REWARD_MARKER = "DRESSAGE_SWEGYM_REWARD_JSON="
_POLL_PENDING_MARKER = "__DRESSAGE_SWEGYM_EVAL_PENDING__"
_POLL_DONE_MARKER = "__DRESSAGE_SWEGYM_EVAL_DONE__="
_POLL_TRUNCATED_MARKER = "__DRESSAGE_SWEGYM_EVAL_TRUNCATED__"
_DEFAULT_POLL_OUTPUT_LIMIT = 16384


def use_fresh_swegym_eval(metadata: dict[str, Any]) -> bool:
    if str(metadata.get("reward_fn") or "") != "swegym_harness_marker":
        return False
    value = metadata.get("swegym_fresh_eval")
    if value is None:
        value = os.environ.get("DRESSAGE_SWEGYM_FRESH_EVAL", "1")
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@limited_verifier
async def execute_fresh_swegym_eval(
    paddock: Any,
    source_state: Any,
    metadata: dict[str, Any],
    *,
    schedule: BlackboxExecuteCmdSchedule,
    source_session_id: str,
    instance_id: str,
    env_type: str | None,
    env_args: dict[str, Any],
    blackbox_type: str,
    backend_options: Any,
) -> None:
    """Extract the agent patch and run the configured hook in a clean sandbox."""

    workdir = str(metadata.get("workdir") or "/testbed")
    base_commit = str(metadata.get("base_commit") or "").strip()
    patch_timeout = _positive_float(
        metadata.get("swegym_patch_timeout_sec"),
        env_name="DRESSAGE_SWEGYM_PATCH_TIMEOUT_SEC",
        default=120.0,
    )
    eval_session_id = f"swegym-eval-{uuid.uuid4().hex}"
    eval_state = None
    metadata["swegym_eval"] = {
        "mode": "fresh_sandbox",
        "source_session_id": source_session_id,
        "eval_session_id": eval_session_id,
        "base_commit": base_commit or None,
    }

    try:
        patch_result = await _extract_source_patch_with_retry(
            paddock,
            source_state,
            metadata,
            session_id=source_session_id,
            cmd=_extract_patch_command(workdir, base_commit=base_commit),
            timeout=patch_timeout,
        )
        if _result_failed(patch_result):
            _record_failure(metadata, "patch_extract_failed", patch_result)
            return
        patch = _stdout(patch_result)
        metadata["swegym_eval"]["patch_bytes"] = len(patch.encode())
        _save_patch_artifact(metadata, "original", patch)
        if not patch.strip():
            _record_failure(metadata, "empty_generation")
            return
        patch, ignored, prohibited_paths = prepare_patch(patch)
        _save_patch_artifact(metadata, "graded", patch)
        metadata["swegym_eval"]["ignored_reproduction_files"] = ignored
        metadata["swegym_eval"]["graded_patch_bytes"] = len(patch.encode())
        metadata["swegym_eval"]["graded_patch_paths"] = changed_patch_paths(patch)
        if prohibited_paths:
            metadata["swegym_eval"]["prohibited_paths"] = prohibited_paths
            _record_failure(metadata, "prohibited_test_modification")
            return

        if not patch.strip():
            _record_failure(metadata, "empty_generation")
            return

        async def prepare():
            nonlocal eval_state
            eval_state = await maybe_await(
                paddock.init(eval_session_id, env_type, dict(env_args))
            )
            await maybe_await(
                paddock.register_agent(
                    eval_state,
                    instance_id=instance_id,
                    session_id=eval_session_id,
                    router_url=proxy_url(),
                    blackbox_type=blackbox_type,
                    backend_options=backend_options,
                )
            )
            # Replay the same repository/environment preconditions used by the
            # agent and standalone validation, before applying the model patch.
            await execute_blackbox_cmds_for_stage(
                paddock, eval_state, metadata, schedule=schedule,
                session_id=eval_session_id, stage="before_agent",
            )
            apply_result = await maybe_await(
                paddock.execute_cmd(
                    eval_state,
                    session_id=eval_session_id,
                    cmd=_apply_patch_command(
                        patch,
                        workdir,
                        base_commit=base_commit,
                    ),
                    timeout=patch_timeout,
                )
            )
            if _result_failed(apply_result):
                _record_failure(metadata, "failed_apply_patch", apply_result)
                return False

            return True

        phase_started = time.monotonic()
        try:
            ready = await bounded(prepare(), "prepare", timeout=seconds("prepare") or 900)
        finally:
            metadata.setdefault("swegym_eval_timings", {})["prepare_seconds"] = time.monotonic() - phase_started
        if not ready:
            return

        phase_started = time.monotonic()
        try:
            await bounded(_execute_after_agent_commands(
                paddock,
                eval_state,
                metadata,
                schedule=schedule,
                session_id=eval_session_id,
            ), "verifier", timeout=sum(c.timeout or 1260 for c in schedule.get("after_agent", ())) + 60)
        finally:
            metadata.setdefault("swegym_eval_timings", {})["grading_seconds"] = time.monotonic() - phase_started

    except Exception as exc:
        from dressage.rollout.case_failure import is_global_failure
        if is_global_failure(exc, contract_errors=False):
            raise
        # This boundary is only patch extraction/preparation/scoring, after
        # the policy has run. Preserve its real trajectory on any case error.
        # asyncio cancellation remains a BaseException and propagates.
        logger.warning(
            "fresh SWE-Gym evaluation failed for instance=%s: %s",
            instance_id,
            exc,
            exc_info=True,
        )
        _record_failure(
            metadata,
            "fresh_eval_exception",
            error=_source_patch_exception_diagnostics(exc),
        )
    finally:
        if eval_state is not None:
            await save_eval_logs(paddock, eval_state, metadata, eval_session_id)
            await terminate_paddock_best_effort(
                paddock,
                session_id=eval_session_id,
                env_args=dict(env_args),
            )


async def _execute_after_agent_commands(
    paddock: Any,
    state: Any,
    metadata: dict[str, Any],
    *,
    schedule: BlackboxExecuteCmdSchedule,
    session_id: str,
) -> None:
    """Run the long official grade without one long-lived HTTP request."""

    for command in schedule.get("after_agent", ()):
        if command.name == "swegym_harness":
            await _execute_polled_swegym_harness(
                paddock,
                state,
                metadata,
                session_id=session_id,
                command=command,
            )
            continue
        await execute_blackbox_cmds_for_stage(
            paddock,
            state,
            metadata,
            schedule={"before_agent": (), "after_agent": (command,)},
            session_id=session_id,
            stage="after_agent",
        )


async def _execute_polled_swegym_harness(
    paddock: Any,
    state: Any,
    metadata: dict[str, Any],
    *,
    session_id: str,
    command: BlackboxExecuteCmd,
) -> None:
    """Start the grade in the background and retrieve its result by polling."""

    timeout = command.timeout or 1260.0
    poll_interval = _positive_float(
        metadata.get("swegym_eval_poll_interval_sec"),
        env_name="DRESSAGE_SWEGYM_EVAL_POLL_INTERVAL_SEC",
        default=15.0,
    )
    token = uuid.uuid4().hex
    output_path = f"/tmp/dressage_swegym_eval_{token}.out"
    done_path = f"/tmp/dressage_swegym_eval_{token}.done"
    script_path = f"/tmp/dressage_swegym_eval_{token}.sh"
    output_limit = _DEFAULT_POLL_OUTPUT_LIMIT
    start_cmd = _background_eval_start_command(
        command.cmd,
        token=token,
        output_path=output_path,
        done_path=done_path,
        script_path=script_path,
    )
    start_result = await maybe_await(
        paddock.execute_cmd(
            state,
            session_id=session_id,
            cmd=start_cmd,
            timeout=60.0,
        )
    )
    if _result_failed(start_result) or not _stdout(start_result).strip():
        _record_failure(metadata, "fresh_eval_start_failed", start_result)
        return

    poll_cmd = (
        f"if test -f {shlex.quote(done_path)}; then "
        f"rc=$(cat {shlex.quote(done_path)}); "
        f"printf '{_POLL_DONE_MARKER}%s\\n' \"$rc\"; "
        f"size=$(wc -c < {shlex.quote(output_path)}); "
        f'if test "$size" -gt {output_limit}; then '
        f"printf '{_POLL_TRUNCATED_MARKER}\\n'; "
        f"tail -c {output_limit} {shlex.quote(output_path)}; "
        f"else cat {shlex.quote(output_path)}; fi; "
        f"else printf '{_POLL_PENDING_MARKER}\\n'; fi"
    )
    deadline = time.monotonic() + timeout
    attempts = 0
    poll_errors: list[str] = []
    poll_output = ""
    while (remaining := deadline - time.monotonic()) > 0:
        attempts += 1
        try:
            poll_result = await maybe_await(
                paddock.execute_cmd(
                    state,
                    session_id=session_id,
                    cmd=poll_cmd,
                    timeout=min(60.0, remaining),
                )
            )
            if _result_failed(poll_result):
                _record_failure(metadata, "fresh_eval_poll_failed", poll_result)
                return
            poll_output = _stdout(poll_result)
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            if not _is_retryable_request_error(exc):
                raise
            poll_errors.append(f"{type(exc).__module__}.{type(exc).__name__}: {exc}")
            logger.warning(
                "SWE-Gym evaluator poll failed; retrying: session_id=%s "
                "attempt=%s error=%s",
                session_id,
                attempts,
                exc,
            )
        else:
            if poll_output.startswith(_POLL_DONE_MARKER):
                break
            if not poll_output.startswith(_POLL_PENDING_MARKER):
                raise RuntimeError(
                    "unexpected background evaluator poll response: "
                    f"{poll_output[:500]!r}"
                )

        await asyncio.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
    else:
        raise TimeoutError(
            f"background SWE-Gym evaluator did not finish within {timeout:.1f}s; "
            f"poll_attempts={attempts} poll_errors={poll_errors[-5:]!r}"
        )

    header, _, stdout = poll_output.partition("\n")
    returncode = int(header[len(_POLL_DONE_MARKER) :].strip())
    truncated_prefix = _POLL_TRUNCATED_MARKER + "\n"
    stdout_truncated = stdout.startswith(truncated_prefix)
    if stdout_truncated:
        stdout = stdout[len(truncated_prefix) :]
    diagnostics = {
        "background_polled": True,
        "poll_attempts": attempts,
        "poll_errors": poll_errors[-5:],
    }
    metadata.setdefault("execute_cmds", []).append(
        {
            "stage": "after_agent",
            "name": command.name,
            "cmd": command.cmd,
            "timeout": command.timeout,
            "required": command.required,
            "cmd_result": {
                "stdout": stdout,
                "stderr": "",
                "returncode": returncode,
                "timed_out": False,
                "stdout_truncated": stdout_truncated,
                **diagnostics,
            },
        }
    )
    metadata.setdefault("swegym_eval", {}).update(diagnostics)


def _background_eval_start_command(
    command: str,
    *,
    token: str,
    output_path: str,
    done_path: str,
    script_path: str,
) -> str:
    """Build an argv-safe launcher for a potentially very large evaluator.

    The evaluator command contains the compressed model patch and can exceed
    Linux's per-argument limit.  Writing it to a script through a heredoc keeps
    the payload out of the nested ``nohup`` argv.  Run the command in a
    subshell so an embedded ``set -e`` cannot skip creation of the done marker.
    """

    heredoc_marker = f"__DRESSAGE_SWEGYM_RUNNER_{token.upper()}__"
    runner = (
        "#!/usr/bin/env bash\n"
        "(\n"
        f"{command}\n"
        ")\n"
        "rc=$?\n"
        f"printf '%s\\n' \"$rc\" > {shlex.quote(done_path)}\n"
        "exit \"$rc\"\n"
    )
    return (
        f"rm -f {shlex.quote(output_path)} {shlex.quote(done_path)} "
        f"{shlex.quote(script_path)}\n"
        f"cat > {shlex.quote(script_path)} <<'{heredoc_marker}'\n"
        f"{runner}"
        f"{heredoc_marker}\n"
        f"chmod 700 {shlex.quote(script_path)}\n"
        f"nohup bash {shlex.quote(script_path)} "
        f"> {shlex.quote(output_path)} 2>&1 < /dev/null & echo $!"
    )


def _extract_patch_command(workdir: str, *, base_commit: str = "") -> str:
    return extract_patch_command(workdir, base_commit)


def _save_patch_artifact(metadata: dict[str, Any], kind: str, patch: str) -> None:
    """Preserve exact patches without bloating the model sample metadata."""
    info = metadata["swegym_eval"]
    info[f"{kind}_patch_sha256"] = hashlib.sha256(patch.encode()).hexdigest()
    run_root = os.environ.get("HAPT_JOINT_RUN_ROOT")
    if not run_root:
        return
    # A generated evaluator UUID names the artifact; task-provided paths are
    # never interpreted as filesystem locations.
    key = hashlib.sha256(str(info["eval_session_id"]).encode()).hexdigest()
    directory = Path(run_root) / "training-audit/patches" / key
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{kind}.patch"
    temporary = directory / f".{kind}-{uuid.uuid4().hex}.tmp"
    temporary.write_text(patch)
    temporary.replace(path)
    info[f"{kind}_patch_path"] = str(path)


async def _extract_source_patch_with_retry(
    paddock: Any,
    state: Any,
    metadata: dict[str, Any],
    *,
    session_id: str,
    cmd: str,
    timeout: float,
) -> Any:
    """Retry transient source-sandbox conflicts before grading the patch."""

    max_attempts = _positive_int(
        metadata.get("swegym_patch_extract_max_attempts"),
        env_name="DRESSAGE_SWEGYM_PATCH_EXTRACT_MAX_ATTEMPTS",
        default=5,
    )
    retry_delay = _positive_float(
        metadata.get("swegym_patch_extract_retry_delay_sec"),
        env_name="DRESSAGE_SWEGYM_PATCH_EXTRACT_RETRY_DELAY_SEC",
        default=2.0,
    )
    errors: list[dict[str, Any]] = []

    for attempt in range(1, max_attempts + 1):
        try:
            result = await maybe_await(
                paddock.execute_cmd(
                    state,
                    session_id=session_id,
                    cmd=cmd,
                    timeout=timeout,
                )
            )
        except Exception as exc:
            error = _source_patch_exception_diagnostics(exc)
            errors.append(error)
            metadata["swegym_eval"].update(
                {
                    "patch_extract_attempts": attempt,
                    "patch_extract_errors": errors[-5:],
                }
            )
            if attempt >= max_attempts or not error["retryable"]:
                raise
            delay = retry_delay * (2 ** (attempt - 1))
            logger.warning(
                "source patch extraction failed transiently; retrying: "
                "session_id=%s attempt=%s/%s delay=%.1fs status=%s body=%r error=%s",
                session_id,
                attempt,
                max_attempts,
                delay,
                error.get("status_code"),
                error.get("response_body"),
                error["message"],
            )
            await asyncio.sleep(delay)
            continue

        metadata["swegym_eval"].update(
            {
                "patch_extract_attempts": attempt,
                "patch_extract_errors": errors[-5:],
            }
        )
        return result

    raise AssertionError("unreachable source patch extraction retry state")


def _source_patch_exception_diagnostics(exc: Exception) -> dict[str, Any]:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    response_body = ""
    if response is not None:
        try:
            response_body = str(response.text or "")[-2000:]
        except Exception:
            response_body = "<unavailable>"
    retryable = _is_retryable_request_error(exc)
    return {
        "type": f"{type(exc).__module__}.{type(exc).__name__}",
        "message": str(exc),
        "status_code": status_code,
        "response_body": response_body,
        "retryable": retryable,
    }


def _is_retryable_request_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code == 409:
        try:
            payload = response.json()
        except (ValueError, AttributeError):
            payload = {}
        if isinstance(payload, dict):
            details = payload.get("details") or {}
            if (payload.get("error") == "bound_session_mismatch"
                    or (isinstance(details, dict) and details.get("state") in {"desynced", "aborted"})):
                # These states cannot heal with exponential backoff.
                return False
    return status_code in {408, 409, 429, 500, 502, 503, 504} or isinstance(exc, (httpx.TransportError, TimeoutError))


def is_environment_request_error(exc: BaseException) -> bool:
    """Runtime transport/session failures, excluding invalid API configuration."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return (isinstance(exc, (httpx.TransportError, TimeoutError))
            or status in {404, 408, 409, 429}
            or isinstance(status, int) and 500 <= status < 600)


def record_agent_environment_exception(metadata: dict[str, Any], exc: Exception) -> None:
    """Retain a runtime failure while harvesting genuine existing policy tokens."""
    _record_failure(metadata, "agent_environment_exception",
                    error=_source_patch_exception_diagnostics(exc))


def _apply_patch_command(
    patch: str,
    workdir: str,
    *,
    base_commit: str = "",
) -> str:
    patch_b64 = base64.b64encode(zlib.compress(patch.encode(), level=9)).decode()
    patch_path = "/tmp/dressage_swegym_model.patch"
    heredoc_marker = "__DRESSAGE_SWEGYM_PATCH_B64_EOF__"
    decoder = (
        "import base64,zlib,sys;"
        "open(sys.argv[1],'wb').write("
        "zlib.decompress(base64.b64decode(sys.stdin.buffer.read())))"
    )
    prepare_worktree = f"cd {shlex.quote(workdir)}"
    if base_commit:
        # SWE-Gym images can contain build-time tracked/untracked changes.
        # Applying a patch that includes that baseline dirt onto another dirty
        # image conflicts (for example MONAI-6924 requirements-dev.txt). Reset
        # the disposable evaluation sandbox to the dataset baseline first.
        base = shlex.quote(base_commit)
        prepare_worktree += f" && git reset --hard {base} && git clean -fd"
    # Do not put patch_b64 in argv.  Linux applies MAX_ARG_STRLEN to each
    # execve argument (typically 128 KiB), independently of the much larger
    # aggregate ARG_MAX.  Feed it through a quoted heredoc instead; the Docker
    # provider also streams the resulting large shell program over stdin.
    return (
        "set -e\n"
        f"python3 -c {shlex.quote(decoder)} {shlex.quote(patch_path)} "
        f"<<'{heredoc_marker}'\n"
        f"{patch_b64}\n"
        f"{heredoc_marker}\n"
        f"{prepare_worktree}"
        f" && git apply --binary -v {shlex.quote(patch_path)}"
    )


def _stdout(result: Any) -> str:
    if not isinstance(result, dict):
        raise TypeError(f"execute_cmd returned {type(result).__name__}; expected dict")
    stdout = result.get("stdout", "")
    if not isinstance(stdout, str):
        raise TypeError("execute_cmd result.stdout must be a string")
    return stdout


def _prohibited_patch_paths(patch: str) -> list[str]:
    return prepare_patch(patch)[2]


def _is_test_or_harness_path(path: str) -> bool:
    return is_protected_path(path)


def _result_failed(result: Any) -> bool:
    if not isinstance(result, dict):
        raise TypeError(f"execute_cmd returned {type(result).__name__}; expected dict")
    return result.get("returncode") != 0 or result.get("timed_out") is True


def _record_failure(
    metadata: dict[str, Any],
    reason: str,
    result: Any = None,
    *,
    error: dict[str, Any] | None = None,
) -> None:
    diag = {
        "resolved": False,
        "failure_reason": reason,
        "harness": "swegym",
        "fresh_sandbox": True,
    }
    if isinstance(result, dict):
        diag["returncode"] = result.get("returncode")
        diag["timed_out"] = bool(result.get("timed_out"))
        stderr = str(result.get("stderr") or "")
        if stderr:
            diag["stderr_tail"] = stderr[-2000:]
    if error is not None:
        diag["error"] = error
    if diag.get("timed_out") and reason not in ENVIRONMENT_FAILURE_REASONS:
        diag["operation_failure_reason"] = reason
        reason = "fresh_eval_timeout"
        diag["failure_reason"] = reason
    if reason in ENVIRONMENT_FAILURE_REASONS:
        diag.update(environment_failure=True, scoring_policy="environment_failure_zero")
    marker = REWARD_MARKER + json.dumps(diag, ensure_ascii=False, separators=(",", ":"))
    metadata.setdefault("execute_cmds", []).append(
        {
            "stage": "after_agent",
            "name": "swegym_harness",
            "cmd": "<fresh-sandbox-evaluator>",
            "timeout": None,
            "required": False,
            "cmd_result": {
                "stdout": marker,
                "stderr": "",
                "returncode": 0,
                "timed_out": False,
            },
        }
    )
    metadata.setdefault("swegym_eval", {})["failure_reason"] = reason


def _positive_float(value: Any, *, env_name: str, default: float) -> float:
    raw = value if value not in (None, "") else os.environ.get(env_name, default)
    try:
        parsed = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_name} must be a positive number; got {raw!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{env_name} must be positive; got {raw!r}")
    return parsed


def _positive_int(value: Any, *, env_name: str, default: int) -> int:
    raw = value if value not in (None, "") else os.environ.get(env_name, default)
    try:
        parsed = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_name} must be a positive integer; got {raw!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{env_name} must be positive; got {raw!r}")
    return parsed
