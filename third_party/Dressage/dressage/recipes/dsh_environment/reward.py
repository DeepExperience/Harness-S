"""Official replay rewards for black-box ALFWorld and AppWorld rollouts."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any

from dressage.reward import register_reward


LOGGER = logging.getLogger(__name__)
PUBLIC_PREFIX = "mcp__environment__"


def _arguments(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _executed_tool_call_ids(messages: list[Any]) -> set[str]:
    """Return tool-call ids that have a matching result in the transcript.

    Claude Code may emit more tool calls than it actually executes when a
    response is truncated or the task becomes terminal.  Replaying every call
    from the assistant payload changes the environment trajectory and can turn
    a 30-step ALFWorld episode into hundreds of synthetic actions.  Tool result
    messages are the durable acknowledgement that a call reached the MCP
    server, so use them as the execution ledger when ids are available.
    """

    executed: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if call_id:
                executed.add(str(call_id))
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            call_id = block.get("tool_use_id")
            if call_id:
                executed.add(str(call_id))
    return executed


def extract_environment_actions(messages: Any) -> list[dict[str, Any]]:
    """Extract only genuine native MCP calls from proxy-recorded messages."""

    if not isinstance(messages, list):
        return []
    executed_call_ids = _executed_tool_call_ids(messages)
    seen_call_ids: set[str] = set()
    actions: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if not isinstance(function, dict):
                    continue
                public_name = str(function.get("name") or "")
                if not public_name.startswith(PUBLIC_PREFIX):
                    continue
                call_id = str(call.get("id") or "")
                if executed_call_ids and call_id not in executed_call_ids:
                    continue
                if call_id and call_id in seen_call_ids:
                    continue
                arguments = _arguments(function.get("arguments"))
                if arguments is None:
                    continue
                if call_id:
                    seen_call_ids.add(call_id)
                actions.append(
                    {
                        "name": public_name[len(PUBLIC_PREFIX) :],
                        "arguments": arguments,
                    }
                )
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                public_name = str(block.get("name") or "")
                if not public_name.startswith(PUBLIC_PREFIX):
                    continue
                call_id = str(block.get("id") or "")
                if executed_call_ids and call_id not in executed_call_ids:
                    continue
                if call_id and call_id in seen_call_ids:
                    continue
                arguments = _arguments(block.get("input"))
                if arguments is None:
                    continue
                if call_id:
                    seen_call_ids.add(call_id)
                actions.append(
                    {
                        "name": public_name[len(PUBLIC_PREFIX) :],
                        "arguments": arguments,
                    }
                )
    return actions


def _task_options(metadata: dict[str, Any], expected_kind: str) -> dict[str, Any]:
    backend = metadata.get("backend_options")
    if not isinstance(backend, dict):
        raise ValueError("metadata.backend_options is missing")
    task = backend.get("task_environment")
    if not isinstance(task, dict):
        raise ValueError("backend_options.task_environment is missing")
    if task.get("kind") != expected_kind:
        raise ValueError(
            f"expected task kind {expected_kind!r}, got {task.get('kind')!r}"
        )
    return task


def _replay(
    sample: Any, *, expected_kind: str
) -> tuple[float, dict[str, Any]]:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError("sample metadata is missing")
    task = _task_options(metadata, expected_kind)
    actions = extract_environment_actions(metadata.get("messages"))
    if not actions:
        return 0.0, {
            "success": False,
            "score": 0.0,
            "failure_reason": "no_environment_actions",
            "action_count": 0,
        }

    python_executable = str(task.get("python_executable") or "")
    server_script = str(task.get("server_script") or "")
    if not python_executable or not server_script:
        raise ValueError("task_environment verifier executable/script is missing")

    env = os.environ.copy()
    env.update(
        {
            "DSH_TASK_KIND": expected_kind,
            "DSH_TASK_MAX_STEPS": str(task.get("max_steps") or 30),
            "DSH_TASK_REWARD_MODE": str(task.get("reward_mode") or "binary"),
            "DSH_TASK_EXPERIMENT_NAME": f"verify-{metadata.get('session_id', 'sample')}",
            "DSH_TASK_RESULT_FILE": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    if task.get("game_file"):
        env["DSH_TASK_GAME_FILE"] = str(task["game_file"])
    if task.get("task_id"):
        env["DSH_TASK_ID"] = str(task["task_id"])
    if task.get("data_root"):
        env["DSH_TASK_DATA_ROOT"] = str(task["data_root"])

    completed = subprocess.run(
        [python_executable, server_script, "--replay"],
        input=json.dumps({"actions": actions}, ensure_ascii=False),
        text=True,
        capture_output=True,
        timeout=max(180, int(task.get("max_steps") or 30) * 30),
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"verifier exited {completed.returncode}: {completed.stderr[-2000:]}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("verifier returned no JSON result")
    result = json.loads(lines[-1])
    if not isinstance(result, dict):
        raise RuntimeError("verifier result is not an object")
    result["action_count"] = len(actions)
    reward_mode = str(task.get("reward_mode") or "binary")
    reward = (
        float(result.get("score", 0.0))
        if reward_mode == "fractional"
        else (1.0 if result.get("success") is True else 0.0)
    )
    return max(0.0, min(1.0, reward)), result


def _reward(sample: Any, expected_kind: str) -> float:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        return 0.0
    try:
        reward, diagnostics = _replay(sample, expected_kind=expected_kind)
    except Exception as exc:
        reward = 0.0
        diagnostics = {
            "success": False,
            "score": 0.0,
            "failure_reason": "verifier_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        LOGGER.exception(
            "%s reward verification failed for instance=%s",
            expected_kind,
            metadata.get("instance_id", "?"),
        )
    diagnostics["reward"] = reward
    metadata[f"{expected_kind}_blackbox_reward"] = diagnostics
    # Preserve the original diagnostic key for existing DSH datasets.
    metadata[f"{expected_kind}_dsh_reward"] = diagnostics
    LOGGER.info(
        "%s DSH reward=%.3f success=%s actions=%s instance=%s",
        expected_kind,
        reward,
        diagnostics.get("success"),
        diagnostics.get("action_count", 0),
        metadata.get("instance_id", "?"),
    )
    return reward


@register_reward("alfworld_dsh")
def alfworld_dsh(sample: Any, *, args: Any = None, **kwargs: Any) -> float:
    del args, kwargs
    return _reward(sample, "alfworld")


@register_reward("appworld_dsh")
def appworld_dsh(sample: Any, *, args: Any = None, **kwargs: Any) -> float:
    del args, kwargs
    return _reward(sample, "appworld")


@register_reward("alfworld_blackbox")
def alfworld_blackbox(sample: Any, *, args: Any = None, **kwargs: Any) -> float:
    del args, kwargs
    return _reward(sample, "alfworld")


@register_reward("appworld_blackbox")
def appworld_blackbox(sample: Any, *, args: Any = None, **kwargs: Any) -> float:
    del args, kwargs
    return _reward(sample, "appworld")
