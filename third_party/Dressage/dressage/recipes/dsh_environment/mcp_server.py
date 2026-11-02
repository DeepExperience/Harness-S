#!/usr/bin/env python3
"""Dependency-free MCP stdio bridge for ALFWorld and AppWorld.

The protocol layer deliberately uses only the Python standard library so the
same file can run under the separate ALFWorld and AppWorld conda environments.
Task packages are imported lazily after the selected environment is known.

Normal mode is configured by ``DSH_TASK_*`` variables and serves newline-
delimited MCP JSON-RPC on stdin/stdout. ``--replay`` executes a recorded action
sequence from JSON stdin and prints one verifier result; it is used by the
Dressage reward function to avoid trusting rollout-side status markers.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import sys
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "dressage-dsh-task-environment", "version": "1.0.0"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _install_appworld_request_log_encoding() -> None:
    """Keep non-UTF8 request bytes in logs without changing API execution."""
    from appworld.requester import RequestTracker
    if getattr(RequestTracker.save, '_dressage_binary_log_encoding', False):
        return
    original_save = RequestTracker.save

    def save(self, file_path):
        try:
            return original_save(self, file_path)
        except UnicodeDecodeError:
            from appworld.common.utils import write_jsonl
            from fastapi.encoders import jsonable_encoder
            requests = [jsonable_encoder(request, custom_encoder={bytes: lambda value:
                {'__appworld_log_type__': 'bytes', 'hex': value.hex()}})
                for request in self.requests]
            write_jsonl(requests, file_path, silent=True)

    save._dressage_binary_log_encoding = True
    RequestTracker.save = save


def _atomic_json(path_text: str, payload: dict[str, Any]) -> None:
    if not path_text:
        return
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _text_result(payload: dict[str, Any], *, is_error: bool = False) -> dict[str, Any]:
    safe = _json_safe(payload)
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(safe, ensure_ascii=False, sort_keys=True),
            }
        ],
        "structuredContent": safe,
        "isError": is_error,
    }


class ALFWorldEnvironment:
    kind = "alfworld"

    def __init__(self) -> None:
        game_file = os.environ.get("DSH_TASK_GAME_FILE", "").strip()
        if not game_file or not Path(game_file).is_file():
            raise ValueError(f"ALFWorld game file is missing: {game_file!r}")
        self.game_file = game_file
        self.max_steps = _env_int("DSH_TASK_MAX_STEPS", 30)
        self.result_file = os.environ.get("DSH_TASK_RESULT_FILE", "")
        self.steps = 0
        self.invalid_actions = 0
        self.done = False
        self.won = False

        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

        request_infos = textworld.EnvInfos(
            won=True,
            admissible_commands=True,
            description=True,
            inventory=True,
        )
        env_id = textworld.gym.register_game(
            game_file,
            request_infos=request_infos,
            max_episode_steps=self.max_steps,
            wrappers=[AlfredDemangler(shuffle=False)],
            asynchronous=False,
        )
        self.env = textworld.gym.make(env_id)
        observation, infos = self.env.reset()
        self.observation = str(observation or "")
        self.admissible = self._commands(infos)
        self._record()

    @staticmethod
    def _commands(infos: dict[str, Any]) -> list[str]:
        return [
            str(command)
            for command in (infos.get("admissible_commands") or [])
            if str(command) != "help"
        ]

    def tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "observe",
                "description": (
                    "Return the current ALFWorld observation and exact admissible "
                    "commands without changing state. Call this first."
                ),
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "step",
                "description": (
                    "Execute exactly one command from the latest admissible_commands "
                    "list and return the next official observation."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "One exact currently admissible command.",
                        }
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        ]

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "observation": self.observation[:6000],
            "admissible_commands": self.admissible,
            "steps": self.steps,
            "invalid_actions": self.invalid_actions,
            "done": self.done,
            "won": self.won,
        }

    def _record(self) -> None:
        _atomic_json(self.result_file, self.snapshot())

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "observe":
            return _text_result(self.snapshot())
        if name != "step":
            return _text_result({"error": f"unknown tool: {name}"}, is_error=True)
        if self.done or self.steps >= self.max_steps:
            self.done = True
            self._record()
            return _text_result(
                {"error": "episode is already terminal", **self.snapshot()},
                is_error=True,
            )
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            self.invalid_actions += 1
            self._record()
            return _text_result(
                {"error": "command must be a non-empty string", **self.snapshot()},
                is_error=True,
            )
        command = command.strip()
        if command not in self.admissible:
            self.invalid_actions += 1
            self._record()
            return _text_result(
                {
                    "error": "command is not currently admissible; state unchanged",
                    "rejected_command": command,
                    **self.snapshot(),
                }
            )
        observation, _reward, done, infos = self.env.step(command)
        self.steps += 1
        self.observation = str(observation or "")
        self.admissible = self._commands(infos)
        self.won = bool(infos.get("won", False))
        self.done = bool(done or self.won or self.steps >= self.max_steps)
        self._record()
        return _text_result(self.snapshot())

    def replay(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        for action in actions:
            if self.done:
                break
            if action.get("name") != "step":
                continue
            self.call("step", action.get("arguments") or {})
        return {
            "success": self.won,
            "score": 1.0 if self.won else 0.0,
            "steps": self.steps,
            "invalid_actions": self.invalid_actions,
            "done": self.done,
        }

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


class AppWorldEnvironment:
    kind = "appworld"

    def __init__(self) -> None:
        task_id = os.environ.get("DSH_TASK_ID", "").strip()
        data_root = os.environ.get("DSH_TASK_DATA_ROOT", "").strip()
        if not task_id:
            raise ValueError("DSH_TASK_ID is required for AppWorld")
        if not data_root or not Path(data_root).is_dir():
            raise ValueError(f"AppWorld data root is missing: {data_root!r}")
        self.task_id = task_id
        self.data_root = data_root
        self.max_steps = _env_int("DSH_TASK_MAX_STEPS", 30)
        self.result_file = os.environ.get("DSH_TASK_RESULT_FILE", "")
        self.steps = 0
        self.done = False

        # AppWorld construction can take longer than the MCP SDK's fixed
        # 60-second initialize/tools-list deadline on a cold shared filesystem.
        # Keep protocol discovery dependency-free and initialize the world on
        # the first actual execute/replay action instead.
        self.world: Any | None = None
        self.runtime_root: Path | None = None
        self._record()

    def _ensure_world(self) -> None:
        if self.world is not None:
            return

        from appworld import AppWorld, update_root
        _install_appworld_request_log_encoding()

        # AppWorld keeps immutable task/base DB data and mutable experiment
        # outputs under one logical root. The shared dataset is mounted
        # read-only in rollout sandboxes, so give each episode a private root
        # whose `data` entry points at the immutable source.
        self.runtime_root = Path(tempfile.mkdtemp(prefix="dressage-appworld-"))
        (self.runtime_root / "data").symlink_to(
            Path(self.data_root) / "data", target_is_directory=True
        )
        (self.runtime_root / "experiments" / "outputs").mkdir(
            parents=True, exist_ok=True
        )
        (self.runtime_root / "experiments" / "configs").mkdir(
            parents=True, exist_ok=True
        )
        update_root(str(self.runtime_root))
        experiment_base = os.environ.get(
            "DSH_TASK_EXPERIMENT_NAME", "dressage"
        ).strip()
        experiment_name = f"{experiment_base[:48]}-{uuid.uuid4().hex[:10]}"
        self.world = AppWorld(
            self.task_id,
            experiment_name=experiment_name,
            load_ground_truth=True,
            ground_truth_mode="minimal",
            max_interactions=self.max_steps,
            # Match AppWorld's public execution contract. Some normal tasks
            # need more than 100 API calls when pagination is correctly batched
            # into one Python block.
            max_api_calls_per_interaction=1000,
            timeout_seconds=120,
            show_api_response_schemas=True,
        )
        self._record()

    def tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "execute",
                "description": (
                    "Execute Python in the persistent AppWorld REPL. Use `apis` to "
                    "inspect API documentation and call apps; `apis` is already "
                    "bound and must not be imported. Variables persist across calls, "
                    "and one block may batch related calls, loops, and pagination. "
                    "Print concise values you need to observe. The task is complete "
                    "only after the block calls `apis.supervisor.complete_task(...)`."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute in AppWorld.",
                        }
                    },
                    "required": ["code"],
                    "additionalProperties": False,
                },
            }
        ]

    def snapshot(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "task_id": self.task_id,
            "steps": self.steps,
            "remaining_steps": max(0, self.max_steps - self.steps),
            "done": self.done,
        }

    def _record(self) -> None:
        _atomic_json(self.result_file, self.snapshot())

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name != "execute":
            return _text_result({"error": f"unknown tool: {name}"}, is_error=True)
        if self.done or self.steps >= self.max_steps:
            self.done = True
            self._record()
            return _text_result(
                {"error": "episode is already terminal", **self.snapshot()},
                is_error=True,
            )
        code = arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            return _text_result(
                {"error": "code must be a non-empty string", **self.snapshot()},
                is_error=True,
            )
        self._ensure_world()
        assert self.world is not None
        output = str(self.world.execute(code))
        self.steps += 1
        self.done = bool(self.world.task_completed() or self.steps >= self.max_steps)
        self._record()
        return _text_result(
            {"output": output[:16000], **self.snapshot()},
            is_error=output.startswith("Execution failed."),
        )

    def replay(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        for action in actions:
            if self.done:
                break
            if action.get("name") != "execute":
                continue
            self.call("execute", action.get("arguments") or {})
        self._ensure_world()
        assert self.world is not None
        tracker = self.world.evaluate()
        evaluation = tracker.to_dict(stats_only=False)
        # Host replay only: never expose these fields from live call()/snapshot().
        # Minimal ground truth includes evaluation code, not the oracle solution.
        diagnostics = {"missing": []}
        ground_truth = getattr(getattr(self.world, "task", None), "ground_truth", None)
        code = getattr(ground_truth, "evaluation_code", None)
        if isinstance(code, str) and code:
            diagnostics["evaluation_code"] = code
        else:
            diagnostics["missing"].append("Executed AppWorld evaluation code unavailable")
        success = bool(evaluation.get("success", False))
        passes = len(getattr(tracker, "passes", []) or [])
        failures = len(getattr(tracker, "failures", []) or [])
        if not passes and not failures:
            num_tests = int(evaluation.get("num_tests", 0) or 0)
            if num_tests:
                passes = num_tests if success else 0
                failures = 0 if success else num_tests
        denominator = passes + failures
        score = (passes / denominator) if denominator else (1.0 if success else 0.0)
        return {
            "success": success,
            "score": score,
            "steps": self.steps,
            "done": self.done,
            "evaluation": evaluation,
            "scorer_diagnostics": diagnostics,
        }

    def close(self) -> None:
        if self.world is not None:
            try:
                self.world.close()
            except Exception:
                pass
        if self.runtime_root is not None:
            shutil.rmtree(self.runtime_root, ignore_errors=True)


def make_environment() -> ALFWorldEnvironment | AppWorldEnvironment:
    kind = os.environ.get("DSH_TASK_KIND", "").strip().lower()
    if kind == "alfworld":
        return ALFWorldEnvironment()
    if kind == "appworld":
        return AppWorldEnvironment()
    raise ValueError(f"unsupported DSH_TASK_KIND: {kind!r}")


def _response(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _dispatch(environment: Any, message: dict[str, Any]) -> dict[str, Any] | None:
    request_id = message.get("id")
    method = message.get("method")
    if request_id is None:
        return None
    if method == "initialize":
        requested = (message.get("params") or {}).get("protocolVersion")
        return _response(
            request_id,
            {
                "protocolVersion": requested or PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
            },
        )
    if method == "ping":
        return _response(request_id, {})
    if method == "tools/list":
        return _response(request_id, {"tools": environment.tools()})
    if method == "tools/call":
        params = message.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return _error(request_id, -32602, "invalid tools/call parameters")
        return _response(request_id, environment.call(name, arguments))
    return _error(request_id, -32601, f"method not found: {method}")


def serve() -> int:
    environment = make_environment()
    atexit.register(environment.close)
    try:
        for raw_line in sys.stdin.buffer:
            try:
                message = json.loads(raw_line)
                if not isinstance(message, dict):
                    raise ValueError("JSON-RPC message must be an object")
                result = _dispatch(environment, message)
            except Exception as exc:
                request_id = (
                    message.get("id")
                    if isinstance(locals().get("message"), dict)
                    else None
                )
                result = _error(request_id, -32603, f"{type(exc).__name__}: {exc}")
                traceback.print_exc(file=sys.stderr)
            if result is not None:
                sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
                sys.stdout.flush()
    finally:
        # AppWorld's official evaluation is intentionally not returned to the
        # model. The independent replay verifier computes reward after rollout.
        environment.close()
    return 0


def replay() -> int:
    payload = json.load(sys.stdin)
    actions = payload.get("actions") if isinstance(payload, dict) else None
    if not isinstance(actions, list):
        raise ValueError("replay input requires an actions list")
    environment = make_environment()
    try:
        result = environment.replay(actions)
    finally:
        environment.close()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def main() -> int:
    try:
        if len(sys.argv) == 2 and sys.argv[1] == "--replay":
            return replay()
        if len(sys.argv) != 1:
            raise ValueError(f"unsupported arguments: {sys.argv[1:]}")
        return serve()
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
