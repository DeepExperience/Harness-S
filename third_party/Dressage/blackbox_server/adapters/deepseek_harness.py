"""Dressage blackbox adapter for project-frozen DeepSeek Harness compositions."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import signal
import socket
import sys
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

import httpx
import uvicorn
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.requests import Request

from blackbox_server.adapters.base import (
    BackendAdapter,
    BackendCapabilities,
    BackendProcessError,
    BackendProtocolError,
    BackendTransportError,
    backend_context_overflow_from_proxy_payload,
)
from blackbox_server.core.models import (
    AdapterResponse,
    BindingContext,
    Message,
    ProxyOptions,
    SessionContext,
    TraceEvent,
    TurnContext,
    TurnUsage,
    utcnow,
)
from blackbox_server.proxy.rollout_llm_proxy import RolloutLLMProxy


LOGGER = logging.getLogger(__name__)
_STDOUT_LIMIT = 100 * 1024 * 1024
_GATEWAY_ENV_KEYS = (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "QWEN_LOCAL_API_KEY",
    "QWEN_LOCAL_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "DSH_HOME",
    "DSH_SESSION_ROOT",
    "DSH_CLAW_MCP_ENABLED",
    "DSH_CLAW_MCP_PYTHON",
    "DSH_CLAW_MCP_SERVER",
    "DSH_CLAW_TASK_ROOT",
    "DSH_CLAW_TOOL_SPECS_DIR",
    "DSH_CLAW_TRACE_DIR",
    "DSH_CLAW_SYSTEM_PROMPT_FILE",
    "HAPT_RUNTIME_ROOT",
)
_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


class _BackgroundUvicornServer(uvicorn.Server):
    @contextlib.contextmanager
    def capture_signals(self):
        yield


def _normalize_qwen_reasoning(payload: Any) -> Any:
    """Expose vLLM's `reasoning` response field under DeepSeek's wire name."""
    if not isinstance(payload, dict):
        return payload
    for choice in payload.get("choices", ()):
        if not isinstance(choice, dict):
            continue
        for key in ("delta", "message"):
            message = choice.get(key)
            if (
                isinstance(message, dict)
                and "reasoning" in message
                and "reasoning_content" not in message
            ):
                message["reasoning_content"] = message.pop("reasoning")
    return payload


def _normalize_sse_frame(frame: bytes) -> bytes:
    normalized: list[bytes] = []
    for line in frame.splitlines():
        if not line.startswith(b"data:"):
            normalized.append(line)
            continue
        raw = line[5:].strip()
        if raw == b"[DONE]":
            normalized.append(line)
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            normalized.append(line)
            continue
        normalized.append(
            b"data: "
            + json.dumps(
                _normalize_qwen_reasoning(payload),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    return b"\n".join(normalized)


class QwenDeepSeekRolloutProxy(RolloutLLMProxy):
    """Narrow Qwen-to-DeepSeek transport shim; never synthesizes policy tokens."""

    async def _handle_proxy(self, request: Request, path: str) -> Any:
        if self._is_chat_completion(request.method, path):
            raw_body = await request.body()
            try:
                payload = json.loads(raw_body)
            except json.JSONDecodeError:
                payload = None
            thinking = payload.get("thinking") if isinstance(payload, dict) else None
            if isinstance(thinking, dict) and thinking.get("type") in {
                "enabled",
                "disabled",
            }:
                kwargs = payload.get("chat_template_kwargs")
                kwargs = dict(kwargs) if isinstance(kwargs, dict) else {}
                kwargs["enable_thinking"] = thinking["type"] == "enabled"
                payload["chat_template_kwargs"] = kwargs
                rewritten = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                delivered = False

                async def receive() -> dict[str, Any]:
                    nonlocal delivered
                    if delivered:
                        return {"type": "http.request", "body": b"", "more_body": False}
                    delivered = True
                    return {
                        "type": "http.request",
                        "body": rewritten,
                        "more_body": False,
                    }

                request = Request(request.scope, receive)
        return await super()._handle_proxy(request, path)

    async def _stream_proxy(self, *args: Any, **kwargs: Any) -> Any:
        response = await super()._stream_proxy(*args, **kwargs)
        if response.status_code >= 400 or not hasattr(response, "body_iterator"):
            return response
        original_iterator = response.body_iterator

        async def normalized_iterator():
            buffer = b""
            async for chunk in original_iterator:
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")
                buffer += chunk
                buffer = buffer.replace(b"\r\n", b"\n")
                while b"\n\n" in buffer:
                    frame, buffer = buffer.split(b"\n\n", 1)
                    yield _normalize_sse_frame(frame) + b"\n\n"
            if buffer:
                yield _normalize_sse_frame(buffer)

        response.body_iterator = normalized_iterator()
        return response


class DeepSeekHarnessModelOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default="Qwen3.5-4B", min_length=1)
    context_window: int = Field(default=65536, gt=0)
    max_tokens: int = Field(default=8192, gt=0)


class DeepSeekHarnessRuntimeOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    python_executable: str = Field(default_factory=lambda: sys.executable)
    dsh_executable: str = "/opt/hapt-runtime/runtime/dsh-v2/node_modules/.bin/dsh"
    jsonrpc_executable: str = (
        "/opt/hapt-runtime/runtime/dsh-jsonrpc-v2/"
        "dsh-jsonrpc-agent-pkg-linux-x64"
    )
    minimal_composition: str = "/opt/hapt-runtime/configs/dsh/minimal.cordis.yml"
    model_overlay: str = "/opt/hapt-runtime/configs/dsh/model-proxy-piai.cordis.yml"
    core_patch: str = "/opt/hapt-runtime/configs/dsh/standard-core-native.cordis.yml"
    environment_patch: str = (
        "/opt/hapt-runtime/configs/dsh/task-environment-native.cordis.yml"
    )
    # Optional, immutable experiment-owned layer applied after the frozen task
    # environment.  Keeping it in the typed runtime schema lets rollout rows
    # bind an exact Harness version without allowing arbitrary environment
    # variables or changing the MCP/reward plane.
    harness_overlay: str | None = None
    sweagent_python: str | None = None
    sweagent_runner: str | None = None
    sweagent_config: str | None = None
    sweagent_source_root: str | None = None
    sweagent_lock: str | None = None


class DeepSeekHarnessTaskEnvironmentOptions(BaseModel):
    """Task environment launched by the DSH MCP client.

    The schema is intentionally concrete rather than accepting an arbitrary
    environment mapping: dataset rows may select a task, but cannot inject
    unrelated credentials into the Harness subprocess.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["alfworld", "appworld", "tau2", "terminal", "dbbench"]
    python_executable: str = Field(min_length=1)
    server_script: str = Field(min_length=1)
    game_file: str | None = None
    task_id: str | None = None
    data_root: str | None = None
    max_steps: int = Field(default=30, gt=0, le=200)
    reward_mode: Literal["binary", "fractional"] = "binary"

    @model_validator(mode="after")
    def validate_task_identity(self) -> "DeepSeekHarnessTaskEnvironmentOptions":
        if self.kind == "alfworld":
            if not self.game_file:
                raise ValueError("ALFWorld task_environment requires game_file")
            if self.task_id or self.data_root:
                raise ValueError(
                    "ALFWorld task_environment does not accept task_id/data_root"
                )
            if self.reward_mode != "binary":
                raise ValueError("ALFWorld only supports binary reward")
        elif self.kind in ("tau2", "terminal", "dbbench"):
            if not self.task_id or not self.data_root or self.game_file or self.reward_mode != "binary":
                raise ValueError("Tau2 requires a task identity and episode endpoint, with binary reward")
        else:
            if not self.task_id or not self.data_root:
                raise ValueError(
                    "AppWorld task_environment requires task_id and data_root"
                )
            if self.game_file:
                raise ValueError("AppWorld task_environment does not accept game_file")
        return self


class DeepSeekHarnessClawEnvironmentOptions(BaseModel):
    """Explicit task-scoped paths for a staged Dressage-Claw episode."""

    model_config = ConfigDict(extra="forbid")

    mcp_enabled: bool = False
    python_executable: str = Field(default="python3", min_length=1)
    server_script: str = Field(
        default="/tmp/omni_task/runtime/claw_tools_mcp.py", min_length=1
    )
    task_root: str = Field(default="/tmp/omni_task", min_length=1)
    tool_specs_dir: str = Field(default="/tmp/omni_task/tools", min_length=1)
    trace_dir: str = Field(default="/tmp/omni_task/trace", min_length=1)
    system_prompt_file: str | None = None


class DeepSeekHarnessBackendOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    harness_id: Literal[
        "minimal",
        "standard-core-native",
        "standard",
        "task-environment-native",
        "dressage-claw-native",
        "sweagent-native",
    ]
    working_directory: str = "/testbed"
    model: DeepSeekHarnessModelOptions = Field(
        default_factory=DeepSeekHarnessModelOptions
    )
    thinking: Literal["enabled", "disabled"] = "enabled"
    reasoning_effort: Literal["off", "low", "medium", "high"] = "high"
    permission_mode: Literal["workspace-write", "danger-full-access"] = (
        "danger-full-access"
    )
    runtime: DeepSeekHarnessRuntimeOptions = Field(
        default_factory=DeepSeekHarnessRuntimeOptions
    )
    task_environment: DeepSeekHarnessTaskEnvironmentOptions | None = None
    claw_environment: DeepSeekHarnessClawEnvironmentOptions | None = None
    proxy: ProxyOptions = Field(default_factory=ProxyOptions)

    @model_validator(mode="after")
    def validate_task_environment(self) -> "DeepSeekHarnessBackendOptions":
        has_task_environment = self.task_environment is not None
        is_task_harness = self.harness_id == "task-environment-native"
        if has_task_environment != is_task_harness:
            raise ValueError(
                "task_environment must be set exactly when harness_id is "
                "'task-environment-native'"
            )
        has_claw_environment = self.claw_environment is not None
        is_claw_harness = self.harness_id == "dressage-claw-native"
        if has_claw_environment != is_claw_harness:
            raise ValueError(
                "claw_environment must be set exactly when harness_id is "
                "'dressage-claw-native'"
            )
        return self


class DeepSeekHarnessAdapter(BackendAdapter):
    """One-shot DSH adapter with a Dressage-owned inference gateway."""

    def __init__(self) -> None:
        self._binding_context: BindingContext | None = None
        self._options: DeepSeekHarnessBackendOptions | None = None
        self._proxy: QwenDeepSeekRolloutProxy | None = None
        self._proxy_port: int | None = None
        self._proxy_server: uvicorn.Server | None = None
        self._proxy_task: asyncio.Task[Any] | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._process_group_id: int | None = None
        self._process_termination_lock = asyncio.Lock()
        self._stdout_handle = None
        self._stderr_handle = None

    async def initialize(self, binding_context: BindingContext) -> None:
        self._binding_context = binding_context
        self._options = self._parse_options(binding_context.binding.backend_options)
        self._prepare_runtime_dirs(Path(binding_context.binding.runtime_dir))
        self._validate_runtime(self._options)
        await self._start_proxy(binding_context, self._options)

    async def send_message(
        self,
        session_context: SessionContext,
        turn_context: TurnContext,
        new_messages: list[Message],
    ) -> AdapterResponse:
        if not await self.health():
            raise BackendProcessError("deepseek_harness backend is not healthy.")
        if session_context.backend_session_id is not None:
            raise BackendProtocolError(
                "deepseek_harness is a one-shot backend and cannot resume a session."
            )
        if len(new_messages) != 1 or new_messages[0].role != "user":
            raise BackendProtocolError(
                "deepseek_harness expects exactly one user message."
            )
        user_text = new_messages[0].content
        if user_text is None or not user_text.strip():
            raise BackendProtocolError("deepseek_harness requires non-empty user content.")

        backend_session_id = self._stable_backend_session_id()
        if self._proxy is not None:
            await self._proxy.open_turn(
                turn_context.turn_id, backend_session_id=backend_session_id
            )
        success = False
        try:
            task = asyncio.create_task(
                self._run_turn(
                    user_text=user_text,
                    backend_session_id=backend_session_id,
                    turn_context=turn_context,
                )
            )
            try:
                result = await asyncio.wait_for(
                    self._await_backend_task_or_proxy_max_steps(
                        task,
                        session_context=session_context,
                        proxy=self._proxy,
                    ),
                    timeout=turn_context.deadline_seconds,
                )
            except asyncio.TimeoutError:
                await self._terminate_active_process()
                raise
            except BackendTransportError as exc:
                await self._raise_if_proxy_context_overflow()
                await self._raise_if_proxy_rollout_invalidated()
                await self._raise_with_proxy_failed_upstream(exc)
            if self._proxy is not None:
                try:
                    await self._proxy.drain_turn(timeout=30.0)
                except asyncio.TimeoutError as exc:
                    raise BackendTransportError("Proxy drain cleanup timed out after 30s.") from exc
                await self._raise_if_proxy_context_overflow()
                await self._raise_if_proxy_rollout_invalidated()
            session_context.backend_session_id = backend_session_id
            success = True
            return result
        finally:
            if not success:
                await self._terminate_active_process()
            if self._proxy is not None:
                with contextlib.suppress(asyncio.TimeoutError):
                    await self._proxy.drain_turn(timeout=None if success else 2.0)
                await self._proxy.clear_turn()

    async def abort_session(self, session_context: SessionContext) -> bool:
        del session_context
        terminated = await self._terminate_active_process()
        if self._proxy is not None:
            with contextlib.suppress(Exception):
                await self._proxy.clear_turn()
        return terminated

    async def health(self) -> bool:
        if self._options is None:
            return False
        try:
            self._validate_runtime(self._options)
        except (BackendProcessError, BackendProtocolError):
            return False
        return self._proxy_task is not None and not self._proxy_task.done()

    async def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            chat=True,
            abort=True,
            pause_resume=True,
            stream=False,
            multi_message_input=False,
            system_message=False,
            history_injection=False,
        )

    async def pause(
        self,
        *,
        reason: str = "weight_update",
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        if self._proxy is None:
            return {"status": "not_started", "reason": reason, "quiesced": True}
        return await self._proxy.pause(reason=reason, timeout_seconds=timeout_seconds)

    async def resume(
        self,
        *,
        version: str | None = None,
        reason: str = "weight_update",
    ) -> dict[str, Any]:
        if self._proxy is None:
            return {"status": "not_started", "reason": reason, "version": version}
        return await self._proxy.resume(version=version, reason=reason)

    def pause_state(self) -> dict[str, Any]:
        if self._proxy is None:
            return {"paused": False, "http_inflight_requests": 0}
        return self._proxy.pause_state()

    async def shutdown(self) -> None:
        await self._terminate_active_process()
        if self._proxy_server is not None:
            self._proxy_server.should_exit = True
        if self._proxy_task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._proxy_task, timeout=5.0)
        self._proxy_task = None
        self._proxy_server = None
        self._proxy = None
        self._proxy_port = None
        for name in ("_stdout_handle", "_stderr_handle"):
            handle = getattr(self, name)
            if handle is not None:
                with contextlib.suppress(Exception):
                    handle.close()
                setattr(self, name, None)

    def _parse_options(
        self, backend_options: dict[str, Any]
    ) -> DeepSeekHarnessBackendOptions:
        try:
            return DeepSeekHarnessBackendOptions.model_validate(backend_options)
        except Exception as exc:
            raise BackendProtocolError(
                f"Invalid deepseek_harness backend_options: {exc}"
            ) from exc

    def _prepare_runtime_dirs(self, runtime_dir: Path) -> None:
        for path in (
            runtime_dir / "home",
            runtime_dir / "logs",
            runtime_dir / "run",
            runtime_dir / "tmp",
            runtime_dir / "prompts",
            runtime_dir / "sessions",
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _validate_runtime(self, options: DeepSeekHarnessBackendOptions) -> None:
        if self._resolve_executable(options.runtime.python_executable) is None:
            raise BackendProcessError(
                f"Python executable not found: {options.runtime.python_executable}"
            )
        required = []
        if options.harness_id == "sweagent-native":
            required = [options.runtime.sweagent_runner, options.runtime.sweagent_config,
                        options.runtime.sweagent_lock]
            if (not options.runtime.sweagent_python
                    or self._resolve_executable(options.runtime.sweagent_python) is None):
                raise BackendProcessError("frozen SWE-agent Python executable missing")
            if any(not path or not Path(path).is_file() for path in required):
                raise BackendProtocolError("frozen SWE-agent runner/config missing")
            if not options.runtime.sweagent_source_root or not Path(options.runtime.sweagent_source_root, "tools").is_dir():
                raise BackendProtocolError("frozen SWE-agent source/tool root missing")
        elif options.harness_id == "minimal":
            if self._resolve_executable(options.runtime.jsonrpc_executable) is None:
                raise BackendProcessError(
                    "DeepSeek Harness JSON-RPC executable not found: "
                    f"{options.runtime.jsonrpc_executable}"
                )
            required.append(options.runtime.minimal_composition)
        else:
            if self._resolve_executable(options.runtime.dsh_executable) is None:
                raise BackendProcessError(
                    f"DeepSeek Harness executable not found: {options.runtime.dsh_executable}"
                )
            required.append(options.runtime.model_overlay)
            if options.harness_id in {
                "standard-core-native",
                "task-environment-native",
                "dressage-claw-native",
            }:
                required.append(options.runtime.core_patch)
        if options.harness_id in {
            "task-environment-native",
            "dressage-claw-native",
        }:
            required.append(options.runtime.environment_patch)
            if options.runtime.harness_overlay:
                required.append(options.runtime.harness_overlay)
        missing = [path for path in required if not Path(path).is_file()]
        if missing:
            raise BackendProtocolError(
                f"DeepSeek Harness frozen composition files missing: {missing}"
            )
        if not Path(options.working_directory).is_dir():
            raise BackendProtocolError(
                f"DeepSeek Harness working_directory does not exist: {options.working_directory}"
            )
        task = options.task_environment
        if task is not None:
            if self._resolve_executable(task.python_executable) is None:
                raise BackendProcessError(
                    f"Task-environment Python executable not found: {task.python_executable}"
                )
            if not Path(task.server_script).is_file():
                raise BackendProtocolError(
                    f"Task-environment MCP server missing: {task.server_script}"
                )
            if task.game_file and not Path(task.game_file).is_file():
                raise BackendProtocolError(
                    f"ALFWorld game file missing: {task.game_file}"
                )
            if task.kind not in ("tau2", "terminal", "dbbench") and task.data_root and not Path(task.data_root).is_dir():
                raise BackendProtocolError(
                    f"AppWorld data root missing: {task.data_root}"
                )
        claw = options.claw_environment
        if claw is not None:
            if claw.mcp_enabled:
                if self._resolve_executable(claw.python_executable) is None:
                    raise BackendProcessError(
                        f"Dressage-Claw MCP Python executable not found: {claw.python_executable}"
                    )
                if not Path(claw.server_script).is_file():
                    raise BackendProtocolError(
                        f"Dressage-Claw MCP server missing: {claw.server_script}"
                    )
                if not Path(claw.tool_specs_dir).is_dir():
                    raise BackendProtocolError(
                        f"Dressage-Claw tool specs missing: {claw.tool_specs_dir}"
                    )
            if not Path(claw.task_root).is_dir():
                raise BackendProtocolError(
                    f"Dressage-Claw task root missing: {claw.task_root}"
                )
            if claw.system_prompt_file and not Path(claw.system_prompt_file).is_file():
                raise BackendProtocolError(
                    "Dressage-Claw task system prompt missing: "
                    f"{claw.system_prompt_file}"
                )

    async def _run_turn(
        self,
        *,
        user_text: str,
        backend_session_id: str,
        turn_context: TurnContext,
    ) -> AdapterResponse:
        if self._binding_context is None or self._options is None:
            raise BackendProcessError("deepseek_harness adapter is not initialized.")
        if self._proxy_port is None:
            raise BackendProcessError("rollout proxy is not initialized.")
        runtime_dir = Path(self._binding_context.binding.runtime_dir)
        task_file = runtime_dir / "prompts" / f"{turn_context.turn_id}.txt"
        task_file.write_text(user_text, encoding="utf-8")
        args = self._runner_args(task_file, backend_session_id, runtime_dir)
        env = self._build_env(runtime_dir)
        logs_dir = runtime_dir / "logs"
        if self._stdout_handle is None:
            self._stdout_handle = open(logs_dir / "deepseek_harness.stdout.log", "ab")
        if self._stderr_handle is None:
            self._stderr_handle = open(logs_dir / "deepseek_harness.stderr.log", "ab")
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                cwd=self._options.working_directory,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_STDOUT_LIMIT,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise BackendProcessError(f"DeepSeek Harness launcher not found: {args[0]}") from exc
        self._process = process
        self._process_group_id = process.pid
        (runtime_dir / "run" / "deepseek_harness.pid").write_text(
            str(process.pid), encoding="utf-8"
        )
        try:
            stdout, stderr = await process.communicate()
            returncode = await process.wait()
        except asyncio.CancelledError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
            raise
        finally:
            if self._process is process:
                self._process = None
                self._process_group_id = None
        self._stdout_handle.write(stdout)
        self._stdout_handle.flush()
        self._stderr_handle.write(stderr)
        self._stderr_handle.flush()
        if returncode != 0:
            raise BackendTransportError(
                f"deepseek_harness exited with code {returncode}; "
                f"stderr tail: {stderr.decode('utf-8', errors='replace')[-4000:]}"
            )
        payload = self._parse_runner_payload(stdout)
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        usage_data = metrics.get("usage") if isinstance(metrics.get("usage"), dict) else {}
        final_response = str(payload.get("final_response") or "")
        trace = TraceEvent(
            turn_id=turn_context.turn_id,
            seq=1,
            source=("sweagent" if self._options.harness_id == "sweagent-native" else "deepseek_harness"),
            event_type=("sweagent/turn" if self._options.harness_id == "sweagent-native" else "dsh/turn"),
            payload={
                "agent_backend": self._options.harness_id,
                "harness_id": self._options.harness_id,
                "harness_config": payload.get("harness_config") or {},
                "metrics": metrics,
                "tool_schema_audit": self._tool_schema_audit(logs_dir),
                "runner_stderr": payload.get("stderr") or "",
                "termination": payload.get("termination", "normal"),
                "original_returncode": payload.get("original_returncode", 0),
            },
            created_at=utcnow(),
        )
        return AdapterResponse(
            outputs=[Message(role="assistant", content=final_response)],
            trace_events=[trace],
            usage=TurnUsage(
                total_tokens=int(usage_data.get("inputTokens", 0))
                + int(usage_data.get("outputTokens", 0)),
                input_tokens=int(usage_data.get("inputTokens", 0)),
                output_tokens=int(usage_data.get("outputTokens", 0)),
                reasoning_tokens=int(usage_data.get("reasoningTokens", 0)),
                steps=int(metrics.get("event_counts", {}).get("turn/end", 0)),
                tool_calls=int(metrics.get("tool_calls", 0)),
            ),
            backend_session_id=backend_session_id,
        )

    def _runner_args(
        self, task_file: Path, backend_session_id: str, runtime_dir: Path
    ) -> list[str]:
        assert self._options is not None
        options = self._options
        if options.harness_id == "sweagent-native":
            assert self._proxy_port is not None
            assert options.runtime.sweagent_python and options.runtime.sweagent_runner and options.runtime.sweagent_config and options.runtime.sweagent_source_root
            return [options.runtime.sweagent_python, options.runtime.sweagent_runner,
                    "--task-file", str(task_file),
                    "--output", str(runtime_dir / "sweagent"),
                    "--instance-id", self._binding_context.binding.bound_instance_id,
                    "--model", options.model.id,
                    "--api-base", f"http://127.0.0.1:{self._proxy_port}/v1",
                    "--session-id", backend_session_id,
                    "--seed", str(options.proxy.sampling_seed_base or 42),
                    "--temperature", str(options.proxy.sampling_temperature or 0.),
                    "--top-p", str(options.proxy.sampling_top_p or 1.),
                    "--max-calls", str(options.proxy.max_steps or 80),
                    "--max-tokens", str(options.model.max_tokens),
                    "--context-window", str(options.model.context_window),
                    "--config", options.runtime.sweagent_config,
                    "--source-root", options.runtime.sweagent_source_root]
        args = [
            options.runtime.python_executable,
            # The runner starts in the task checkout. Prevent repositories
            # such as pydantic from shadowing control-runtime dependencies.
            "-P",
            "-m",
            "blackbox_server.adapters.deepseek_harness_runner",
            "--harness-id",
            options.harness_id,
            "--task-file",
            str(task_file),
            "--workspace",
            options.working_directory,
            "--session-root",
            str(runtime_dir / "sessions"),
            "--session-id",
            backend_session_id,
            "--model",
            options.model.id,
            "--max-tokens",
            str(options.model.max_tokens),
        ]
        if options.harness_id == "minimal":
            args.extend(
                [
                    "--minimal-composition",
                    options.runtime.minimal_composition,
                    "--jsonrpc-executable",
                    options.runtime.jsonrpc_executable,
                ]
            )
        else:
            args.extend(
                [
                    "--dsh-executable",
                    options.runtime.dsh_executable,
                    "--model-overlay",
                    options.runtime.model_overlay,
                ]
            )
            if options.harness_id in {
                "standard-core-native",
                "task-environment-native",
                "dressage-claw-native",
            }:
                args.extend(["--core-patch", options.runtime.core_patch])
            if options.harness_id in {
                "task-environment-native",
                "dressage-claw-native",
            }:
                args.extend(
                    ["--environment-patch", options.runtime.environment_patch]
                )
                if options.runtime.harness_overlay:
                    args.extend(
                        ["--harness-overlay", options.runtime.harness_overlay]
                    )
        return args

    def _build_env(self, runtime_dir: Path) -> dict[str, str]:
        assert self._options is not None and self._proxy_port is not None
        options = self._options
        env = os.environ.copy()
        for key in (*_GATEWAY_ENV_KEYS, *_PROXY_ENV_KEYS):
            env.pop(key, None)
        proxy_url = f"http://127.0.0.1:{self._proxy_port}/v1"
        env.update(
            {
                "HOME": str(runtime_dir / "home"),
                "TMPDIR": str(runtime_dir / "tmp"),
                "DEEPSEEK_API_KEY": "dressage-local",
                "DEEPSEEK_BASE_URL": proxy_url,
                "QWEN_LOCAL_API_KEY": "dressage-local",
                "QWEN_LOCAL_BASE_URL": proxy_url,
                "DSH_MODEL": options.model.id,
                "DSH_CWD": options.working_directory,
                "DSH_CONTEXT_WINDOW": str(options.model.context_window),
                "DSH_MAX_TOKENS": str(options.model.max_tokens),
                "DSH_PERMISSION_MODE": options.permission_mode,
                "DSH_THINKING": options.thinking,
                "DSH_REASONING_EFFORT": options.reasoning_effort,
                "DSH_TOOLS_MODE": "native",
                "DSH_TELEMETRY_DISABLED": "1",
                "DSH_HOME": str(runtime_dir / "home" / ".dsh"),
                "DSH_SESSION_ROOT": str(runtime_dir / "sessions"),
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        task = options.task_environment
        if task is not None:
            task_result_file = runtime_dir / "run" / "task_environment_result.json"
            env.update(
                {
                    "DSH_TASK_KIND": task.kind,
                    "DSH_TASK_PYTHON": task.python_executable,
                    "DSH_TASK_SERVER": task.server_script,
                    "DSH_TASK_MAX_STEPS": str(task.max_steps),
                    "DSH_TASK_REWARD_MODE": task.reward_mode,
                    "DSH_TASK_RESULT_FILE": str(task_result_file),
                    "DSH_TASK_EXPERIMENT_NAME": self._stable_backend_session_id(),
                }
            )
            if task.game_file:
                env["DSH_TASK_GAME_FILE"] = task.game_file
            if task.task_id:
                env["DSH_TASK_ID"] = task.task_id
            if task.data_root:
                env["DSH_TASK_DATA_ROOT"] = task.data_root
        claw = options.claw_environment
        if claw is not None:
            env.update(
                {
                    "HAPT_RUNTIME_ROOT": "/opt/hapt-runtime",
                    "DSH_CLAW_MCP_ENABLED": "1" if claw.mcp_enabled else "0",
                    "DSH_CLAW_MCP_PYTHON": claw.python_executable,
                    "DSH_CLAW_MCP_SERVER": claw.server_script,
                    "DSH_CLAW_TASK_ROOT": claw.task_root,
                    "DSH_CLAW_TOOL_SPECS_DIR": claw.tool_specs_dir,
                    "DSH_CLAW_TRACE_DIR": claw.trace_dir,
                }
            )
            if claw.system_prompt_file:
                env["DSH_CLAW_SYSTEM_PROMPT_FILE"] = claw.system_prompt_file
        return {str(key): str(value) for key, value in env.items()}

    def _parse_runner_payload(self, stdout: bytes) -> dict[str, Any]:
        text = stdout.decode("utf-8", errors="replace")
        for line in reversed(text.splitlines()):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "final_response" in payload:
                return payload
        raise BackendProtocolError(
            f"deepseek_harness launcher returned no result payload; stdout tail: {text[-2000:]}"
        )

    def _tool_schema_audit(self, logs_dir: Path) -> list[dict[str, Any]]:
        audits: list[dict[str, Any]] = []
        for path in sorted(logs_dir.glob("tool_summary.*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            audits.append(
                {
                    "step": payload.get("step"),
                    "request_kind": payload.get("request_kind"),
                    "tool_count": payload.get("raw_tool_count"),
                    "tool_names": payload.get("raw_tool_names") or [],
                }
            )
        return audits

    async def _start_proxy(
        self,
        binding_context: BindingContext,
        options: DeepSeekHarnessBackendOptions,
    ) -> None:
        upstream_origin = self._resolve_upstream_origin(
            binding_context.binding.router_base_url
        )
        self._proxy_port = self._find_free_port()
        self._proxy = QwenDeepSeekRolloutProxy(
            upstream_origin=upstream_origin,
            router_api_path=binding_context.binding.router_api_path,
            bound_session_id=binding_context.binding.bound_session_id,
            bound_instance_id=binding_context.binding.bound_instance_id,
            sticky_header_name=options.proxy.sticky_header_name,
            max_steps=options.proxy.max_steps,
            default_temperature=options.proxy.default_temperature,
            model_override=options.model.id,
            **options.proxy.sampling_kwargs(),
            debug_log_dir=Path(binding_context.binding.runtime_dir) / "logs",
        )
        config = uvicorn.Config(
            self._proxy.app,
            host="127.0.0.1",
            port=self._proxy_port,
            log_level="warning",
        )
        self._proxy_server = _BackgroundUvicornServer(config)
        self._proxy_task = asyncio.create_task(self._proxy_server.serve())
        await self._wait_for_proxy()
        run_dir = Path(binding_context.binding.runtime_dir) / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "proxy.port").write_text(str(self._proxy_port), encoding="utf-8")

    async def _wait_for_proxy(self, timeout: float = 5.0) -> None:
        assert self._proxy_port is not None
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"http://127.0.0.1:{self._proxy_port}/__proxy_health",
                        timeout=0.5,
                    )
                    if response.status_code == 200:
                        return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.1)
        raise BackendProcessError("Timed out waiting for rollout proxy startup.")

    async def _raise_if_proxy_context_overflow(self) -> None:
        if self._proxy is None:
            return
        payload = await self._proxy.consume_context_overflow_error()
        typed = backend_context_overflow_from_proxy_payload(payload)
        if typed is not None:
            raise typed

    async def _raise_if_proxy_rollout_invalidated(self) -> None:
        if self._proxy is None:
            return
        payload = await self._proxy.consume_rollout_invalidated_error()
        if payload is not None:
            raise BackendTransportError(
                f"Dressage proxy {payload.get('error') or 'rollout_invalidated'}: "
                f"{payload.get('message') or 'rollout was invalidated'}"
            )

    async def _raise_with_proxy_failed_upstream(
        self, exc: BackendTransportError
    ) -> None:
        if self._proxy is None:
            raise exc
        payload = await self._proxy.consume_failed_upstream_error()
        if payload is None:
            raise exc
        raise BackendTransportError(
            f"{exc}; upstream error: {payload.get('message') or payload}"
        ) from exc

    async def _terminate_active_process(self) -> bool:
        async with self._process_termination_lock:
            process = self._process
            if process is None or process.returncode is not None:
                if self._process is process:
                    self._process = None
                    self._process_group_id = None
                return False
            process_group_id = self._process_group_id
            if process_group_id is not None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process_group_id, signal.SIGTERM)
            else:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                if process_group_id is not None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process_group_id, signal.SIGKILL)
                else:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    LOGGER.warning(
                        "DeepSeek Harness process pid=%s did not reap after SIGKILL",
                        getattr(process, "pid", None),
                    )
            finally:
                if self._process is process:
                    self._process = None
                    self._process_group_id = None
            return True

    def _stable_backend_session_id(self) -> str:
        if self._binding_context is None:
            raise BackendProcessError("deepseek_harness binding is not initialized.")
        binding = self._binding_context.binding
        return str(
            uuid5(
                NAMESPACE_URL,
                f"{binding.bound_instance_id}:{binding.bound_session_id}",
            )
        )

    def _resolve_upstream_origin(self, router_base_url: str) -> str:
        raw = router_base_url if "://" in router_base_url else f"http://{router_base_url}"
        parsed = urlparse(raw)
        if not parsed.netloc:
            raise BackendProtocolError(f"Invalid router_base_url: {router_base_url}")
        return f"{parsed.scheme or 'http'}://{parsed.netloc}"

    def _find_free_port(self) -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with contextlib.closing(sock):
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            return int(sock.getsockname()[1])

    def _resolve_executable(self, executable: str) -> str | None:
        if os.path.sep in executable:
            return executable if Path(executable).is_file() else None
        return shutil.which(executable)
