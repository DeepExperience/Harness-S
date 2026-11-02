"""Command payload helpers for blackbox session execution."""

from __future__ import annotations

from typing import Any


def build_execute_cmd_payload(
    *,
    cmd: str,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Validate and build the execute_cmd request payload."""
    normalized_cmd = normalize_execute_cmd(cmd)
    normalized_timeout = normalize_execute_cmd_timeout(timeout)
    return {"cmd": normalized_cmd, "timeout": normalized_timeout}


def normalize_execute_cmd(cmd: str) -> str:
    if not isinstance(cmd, str):
        raise TypeError("cmd must be a string")

    normalized = cmd.strip()
    if not normalized:
        raise ValueError("cmd must be a non-empty string")

    for forbidden in ("\n", "\r", "\x00"):
        if forbidden in normalized:
            raise ValueError("cmd must be a single line without NUL bytes")

    return normalized


def normalize_execute_cmd_timeout(timeout: float | None) -> float | None:
    if timeout is None:
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError("timeout must be a positive number or None")
    timeout = float(timeout)
    if timeout <= 0:
        raise ValueError("timeout must be > 0")
    return timeout
