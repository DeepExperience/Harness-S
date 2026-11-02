"""Exact structured DSH terminal classification, shared by training and HE."""
import asyncio
import json
import re


async def run_cli_and_drain(runner, proxy, *, agent_timeout, drain_timeout=30.0):
    """Keep execution deadlines separate from auxiliary request draining."""
    execution_failure = None
    drain_error = None
    try:
        returncode, stdout, stderr = await runner
    except asyncio.TimeoutError:
        returncode, stdout, stderr = 124, "", f"harness timeout after {agent_timeout:.1f}s"
        execution_failure = {"kind": "agent_execution_timeout", "stage": "agent",
                             "timeout_seconds": agent_timeout}
    if returncode == 124 and stderr.startswith("harness timeout after "):
        execution_failure = {"kind": "agent_execution_timeout", "stage": "agent",
                             "timeout_seconds": agent_timeout}
    try:
        await proxy.drain_turn(timeout=drain_timeout)
    except asyncio.TimeoutError:
        # Preserve the actual process exit, stdout and diagnostic. A finished
        # agent is not a timed-out agent just because an orphaned request has
        # not drained. The caller still performs bounded session cleanup.
        drain_error = {"kind": "proxy_drain_timeout", "stage": "cleanup",
                       "timeout_seconds": drain_timeout}
    return returncode, stdout, stderr, execution_failure, drain_error


def classify_cli_termination(returncode, trace, execution_failure=None):
    """Share CLI exit semantics across training, validation and HE runners.

    Only a durable, well-formed final budget event permits harvesting after
    exit 1. Signals/timeouts and unknown process failures never get excused by
    an earlier event. External cancellation remains owned by the caller.
    """
    trace = trace if isinstance(trace, dict) else {}
    if isinstance(execution_failure, dict) and execution_failure.get("kind") == "agent_execution_timeout":
        return {"kind": "agent_execution_timeout", "score_patch": False}
    if returncode == 0:
        return {"kind": "completed", "score_patch": True}
    reason = budget_stop_reason(trace.get("terminal_reasons"))
    if returncode == 1 and reason and not trace.get("malformed_jsonl_lines", 0):
        return {"kind": reason, "score_patch": True}
    reasons = trace.get("terminal_reasons")
    terminal = reasons[-1] if isinstance(reasons, list) and reasons else None
    if isinstance(terminal, dict) and terminal.get("kind") == "error":
        error = terminal.get("error")
        if (returncode == 1 and not trace.get("malformed_jsonl_lines", 0)
                and isinstance(error,dict) and error.get('code')=='EMPTY_RESPONSE'
                and 'completed response with no content' in str(error.get('message',''))):
            return {'kind':'model_stopped_empty','score_patch':True}
        if isinstance(error, dict) and error.get("code") == "TIMEOUT":
            return {"kind": "model_request_timeout", "score_patch": False}
    return {"kind": "harness_process_exit" if trace.get("event_counts") else "harness_startup_failure",
            "score_patch": False}


def budget_stop_reason(reasons):
    # An earlier budget event cannot excuse a later transport/process failure.
    if not isinstance(reasons, list) or not reasons or not isinstance(reasons[-1], dict):
        return None
    reason = reasons[-1]
    allowed = {"max-tokens": "max_tokens_exceeded", "max-steps": "max_steps_exceeded",
               "max_tokens_exceeded": "max_tokens_exceeded", "max_steps_exceeded": "max_steps_exceeded",
               "context_overflow": "context_overflow"}
    kind = reason.get("kind")
    if kind in allowed:
        return allowed[kind]
    error = reason.get("error")
    if kind != "error" or not isinstance(error, dict):
        return None
    message = error.get("message")
    if not isinstance(message, str):
        return None
    if error.get("code") == "CONTEXT_WINDOW_EXCEEDED":
        return "context_overflow"
    try:
        payload = json.loads(message[message.index("{"):])
    except (ValueError, TypeError):
        # PIAI also emits a JSON-string response body for HTTP 413. Only a
        # terminal error with this exact transport shape is a budget stop.
        return "context_overflow" if re.fullmatch(
            r'(?:PI_AI_ERROR:\s*)?413\s+"context_overflow"', message.strip()) else None
    return allowed.get(payload.get("code")) if isinstance(payload, dict) else None
