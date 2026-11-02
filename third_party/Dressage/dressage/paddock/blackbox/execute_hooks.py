"""Execute-cmd hooks for blackbox rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import base64
from enum import Enum
import inspect
import logging
from pathlib import Path
from typing import Any, TypeAlias

import httpx

from dressage.paddock.blackbox.common.command import build_execute_cmd_payload

logger = logging.getLogger(__name__)

EXECUTE_CMD_STAGES = ("before_agent", "after_agent")
EXECUTE_CMD_STDIO_METADATA_LIMIT = 4096

_ALLOWED_STAGE_SET = frozenset(EXECUTE_CMD_STAGES)
_ALLOWED_COMMAND_KEYS = frozenset(("name", "cmd", "timeout", "required", "stdout_limit"))
_REQUIRED_COMMAND_KEYS = frozenset(("name", "cmd", "required"))


@dataclass(frozen=True)
class BlackboxExecuteCmd:
    stage: str
    name: str
    cmd: str
    timeout: float | None
    required: bool
    stdout_limit: int | None = None


BlackboxExecuteCmdSchedule: TypeAlias = dict[str, tuple[BlackboxExecuteCmd, ...]]


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def parse_blackbox_execute_cmds(value: Any) -> BlackboxExecuteCmdSchedule:
    if value is None:
        return _empty_schedule()
    if not isinstance(value, dict):
        raise ValueError(
            "metadata.blackbox_execute_cmds must be a dict with before_agent "
            "and after_agent command lists"
        )

    unknown_stages = sorted(
        str(stage) for stage in value if stage not in _ALLOWED_STAGE_SET
    )
    if unknown_stages:
        raise ValueError(
            "metadata.blackbox_execute_cmds contains unsupported stage(s): "
            + ", ".join(unknown_stages)
        )

    schedule: BlackboxExecuteCmdSchedule = {}
    for stage in EXECUTE_CMD_STAGES:
        raw_commands = value.get(stage, [])
        if not isinstance(raw_commands, list):
            raise ValueError(
                f"metadata.blackbox_execute_cmds.{stage} must be a list"
            )
        schedule[stage] = tuple(
            _parse_command(stage, index, raw_command)
            for index, raw_command in enumerate(raw_commands)
        )
    return schedule


async def execute_blackbox_cmds_for_stage(
    paddock: Any,
    state: Any,
    metadata: dict[str, Any],
    *,
    schedule: BlackboxExecuteCmdSchedule,
    session_id: str,
    stage: str,
) -> None:
    if stage not in _ALLOWED_STAGE_SET:
        raise ValueError(f"unsupported execute_cmd stage: {stage}")

    for command in schedule.get(stage, ()):
        await _execute_blackbox_cmd(
            paddock,
            state,
            metadata,
            session_id=session_id,
            command=command,
        )


def truncated_metadata_text(
    value: Any,
    *,
    limit: int | None = None,
) -> tuple[str, bool]:
    text = "" if value is None else str(value)
    if limit is None or len(text) <= limit:
        return text, False
    return text[:limit], True


def _empty_schedule() -> BlackboxExecuteCmdSchedule:
    return {stage: () for stage in EXECUTE_CMD_STAGES}


def _parse_command(stage: str, index: int, value: Any) -> BlackboxExecuteCmd:
    path = f"metadata.blackbox_execute_cmds.{stage}[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a dict")

    unknown_keys = sorted(str(key) for key in value if key not in _ALLOWED_COMMAND_KEYS)
    if unknown_keys:
        raise ValueError(
            f"{path} contains unsupported key(s): " + ", ".join(unknown_keys)
        )

    missing_keys = sorted(_REQUIRED_COMMAND_KEYS - set(value))
    if missing_keys:
        raise ValueError(
            f"{path} missing required key(s): " + ", ".join(missing_keys)
        )

    name = value["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{path}.name must be a non-empty string")

    required = value["required"]
    if not isinstance(required, bool):
        raise ValueError(f"{path}.required must be a bool")

    cmd = value["cmd"]
    if isinstance(cmd, str) and "\n" in cmd and "\x00" not in cmd:
        # The API uses /bin/sh, while task programs can require Bash. Base64
        # keeps transport POSIX-compatible and preserves here-documents.
        # printf is a shell builtin; no large program enters a nested exec argv.
        encoded = base64.b64encode(cmd.encode()).decode("ascii")
        chunks = " ".join("'" + encoded[i:i + 8192] + "'" for i in range(0, len(encoded), 8192))
        cmd = "printf %s " + chunks + " | base64 -d | bash -l -s"
    payload = build_execute_cmd_payload(
        cmd=cmd,
        timeout=value.get("timeout"),
    )
    return BlackboxExecuteCmd(
        stage=stage,
        name=name.strip(),
        cmd=payload["cmd"],
        timeout=payload["timeout"],
        required=required,
        stdout_limit=value.get("stdout_limit"),
    )


async def _execute_blackbox_cmd(
    paddock: Any,
    state: Any,
    metadata: dict[str, Any],
    *,
    session_id: str,
    command: BlackboxExecuteCmd,
) -> None:
    execute_cmd = getattr(paddock, "execute_cmd", None)
    if execute_cmd is None:
        raise TypeError(f"{type(paddock).__name__} does not implement execute_cmd")

    try:
        cmd_result = await maybe_await(
            execute_cmd(
                state,
                session_id=session_id,
                cmd=command.cmd,
                timeout=command.timeout,
            )
        )
    except Exception as exc:
        record = {
            "stage": command.stage,
            "name": command.name,
            "cmd": command.cmd,
            "timeout": command.timeout,
            "required": command.required,
            "cmd_error": {
                "type": f"{type(exc).__module__}.{type(exc).__name__}",
                "message": str(exc),
                "summary": _exception_summary(exc),
            },
            "http": _http_exception_details(exc),
        }
        metadata.setdefault("execute_cmds", []).append(record)

        if command.required:
            raise

        logger.warning(
            "optional blackbox execute_cmd failed; continuing rollout: "
            "stage=%s name=%s session_id=%s error=%s",
            command.stage,
            command.name,
            session_id,
            _exception_summary(exc),
        )
        return

    result_summary = _execute_cmd_result_summary(cmd_result, stdout_limit=command.stdout_limit)
    record = {
        "stage": command.stage,
        "name": command.name,
        "cmd": command.cmd,
        "timeout": command.timeout,
        "required": command.required,
        "cmd_result": result_summary,
    }
    metadata.setdefault("execute_cmds", []).append(record)

    if command.required and _execute_cmd_failed(result_summary):
        raise RuntimeError(
            "required execute_cmd failed: "
            f"name={command.name} returncode={result_summary.get('returncode')} "
            f"timed_out={result_summary.get('timed_out')}"
        )


def _execute_cmd_result_summary(
    result: Any,
    *,
    stdout_limit: int | None = None,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raw, raw_truncated = truncated_metadata_text(result, limit=stdout_limit)
        return {"raw": raw, "raw_truncated": raw_truncated}

    summary = {str(key): _json_safe(value) for key, value in result.items()}
    stdout, stdout_truncated = truncated_metadata_text(summary.get("stdout"), limit=stdout_limit)
    stderr, stderr_truncated = truncated_metadata_text(summary.get("stderr"), limit=stdout_limit)
    summary["stdout"] = stdout
    summary["stderr"] = stderr
    summary["stdout_truncated"] = stdout_truncated
    summary["stderr_truncated"] = stderr_truncated
    return summary


def _execute_cmd_failed(result: dict[str, Any]) -> bool:
    return result.get("returncode") != 0 or result.get("timed_out") is True


def _exception_summary(exc: BaseException) -> str:
    return " ".join(str(exc).splitlines()) or type(exc).__name__


def _http_exception_details(exc: BaseException) -> dict[str, Any] | None:
    if not isinstance(exc, httpx.HTTPStatusError):
        return None

    request = exc.request
    response = exc.response
    return {
        "request": {
            "method": request.method,
            "url": str(request.url),
            "headers": _headers_for_log(request.headers),
            "body": _bytes_for_log(request.content),
        },
        "response": {
            "status_code": response.status_code,
            "reason_phrase": response.reason_phrase,
            "headers": _headers_for_log(response.headers),
            "body": response.text,
        },
    }


def _headers_for_log(headers: httpx.Headers) -> dict[str, str]:
    redacted = {}
    sensitive = {"authorization", "api-key", "x-api-key", "cookie", "set-cookie"}
    for key, value in headers.items():
        redacted[key] = "<redacted>" if key.lower() in sensitive else value
    return redacted


def _bytes_for_log(data: bytes | None) -> str | None:
    if not data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)
