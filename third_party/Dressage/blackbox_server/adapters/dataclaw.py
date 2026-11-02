from __future__ import annotations

import asyncio
import json
import os
import signal
import shutil
from pathlib import Path
from typing import Any

import httpx
from pydantic import ConfigDict, Field

from blackbox_server.adapters.base import (
    BackendProcessError,
    BackendProtocolError,
)
from blackbox_server.adapters.openclaw import (
    OpenClawAdapter,
    OpenClawBackendOptions,
    OpenClawCompactionOptions,
)
from blackbox_server.core.models import BindingContext, SessionContext


class DataclawBackendOptions(OpenClawBackendOptions):
    """OpenClaw-compatible options for the internal dataclaw distribution.

    The real upstream credential stays in the BlackboxServer environment.  The
    generated dataclaw config only contains ``api_key``, which is a local proxy
    placeholder and is never forwarded upstream.
    """

    model_config = ConfigDict(extra="forbid")

    upstream_api_key_env: str = "OPENAI_API_KEY"
    startup_health_timeout: float = Field(default=240.0, gt=0)
    session_key_override: str | None = Field(default=None, min_length=1, max_length=512)
    plugin_paths: list[str] = Field(default_factory=list)
    skills_source_dir: str | None = None


class DataclawAdapter(OpenClawAdapter):
    """Dressage blackbox adapter for the dataclaw OpenClaw fork."""

    def _parse_options(self, backend_options: dict[str, Any]) -> DataclawBackendOptions:
        try:
            return DataclawBackendOptions.model_validate(backend_options)
        except Exception as exc:
            raise BackendProtocolError(f"Invalid dataclaw backend_options: {exc}") from exc

    def _ensure_backend_session_id(self, session_context: SessionContext) -> str:
        if session_context.backend_session_id:
            return session_context.backend_session_id
        options = self._options
        if isinstance(options, DataclawBackendOptions) and options.session_key_override:
            session_context.backend_session_id = options.session_key_override
            return options.session_key_override
        return super()._ensure_backend_session_id(session_context)

    def _prepare_workspace(self, binding_context: BindingContext) -> None:
        super()._prepare_workspace(binding_context)
        options = self._options
        if not isinstance(options, DataclawBackendOptions) or not options.skills_source_dir:
            return
        source = Path(options.skills_source_dir).resolve()
        if not source.is_dir():
            raise BackendProtocolError(f"dataclaw skills_source_dir is not a directory: {source}")
        runtime_dir = Path(binding_context.binding.runtime_dir)
        target = runtime_dir / "home" / ".openclaw" / "workspace" / "skills"
        target.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            if child.is_dir():
                shutil.copytree(child, target / child.name, dirs_exist_ok=True)

    def _build_compaction_config(
        self,
        options: DataclawBackendOptions,
    ) -> dict[str, Any]:
        # dataclaw validates this object strictly.  Its 2026.3.8 schema does
        # not accept OpenClaw's midTurnPrecheck/truncateAfterCompaction/
        # notifyUser fields, so only emit the shared keys.
        config: dict[str, Any] = {"mode": "safeguard"}
        compaction: OpenClawCompactionOptions | None = options.compaction
        if compaction is None:
            return config
        shared = {
            "reserveTokens": compaction.reserve_tokens,
            "reserveTokensFloor": compaction.reserve_tokens_floor,
            "keepRecentTokens": compaction.keep_recent_tokens,
            "postCompactionSections": compaction.post_compaction_sections,
            "model": compaction.model,
        }
        config.update({key: value for key, value in shared.items() if value is not None})
        return config

    def _build_openclaw_config(
        self,
        binding_context: BindingContext,
        options: DataclawBackendOptions,
        *,
        permission_mode: str = "dual",
    ) -> dict[str, Any]:
        config = super()._build_openclaw_config(
            binding_context,
            options,
            permission_mode=permission_mode,
        )
        provider = config["models"]["providers"][options.provider_id]
        model = provider["models"][0]
        model["contextWindow"] = options.context_window
        # dataclaw's strict 2026.3.8 schema supports openai/zai/qwen here;
        # Dressage's upstream OpenClaw adapter emits the legacy "deepseek"
        # value.  GLM-5.3 is exposed through an OpenAI-compatible endpoint.
        model["compat"]["thinkingFormat"] = "openai"
        # dataclaw does not define OpenClaw's permission config fields.  The
        # deny-question policy is still enforced by the injected AGENTS.md.
        config.pop("permission", None)
        config["agents"]["defaults"].pop("permission", None)
        if options.plugin_paths:
            resolved: list[str] = []
            allowed: list[str] = []
            for raw in options.plugin_paths:
                path = Path(raw).resolve()
                if not path.is_dir():
                    raise BackendProtocolError(f"dataclaw plugin path is not a directory: {path}")
                manifest = path / "openclaw.plugin.json"
                try:
                    plugin_id = json.loads(manifest.read_text(encoding="utf-8"))["id"]
                except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise BackendProtocolError(
                        f"dataclaw plugin has no valid openclaw.plugin.json id: {path}"
                    ) from exc
                if not isinstance(plugin_id, str) or not plugin_id.strip():
                    raise BackendProtocolError(f"dataclaw plugin id is empty: {path}")
                resolved.append(str(path))
                allowed.append(plugin_id)
            config["plugins"] = {
                "enabled": True,
                "allow": allowed,
                "load": {"paths": resolved},
            }
        return config

    async def _start_proxy(
        self,
        binding_context: BindingContext,
        options: DataclawBackendOptions,
    ) -> None:
        upstream_key = os.environ.get(options.upstream_api_key_env, "").strip()
        if not upstream_key:
            raise BackendProtocolError(
                f"dataclaw upstream credential is missing: {options.upstream_api_key_env}"
            )
        await super()._start_proxy(binding_context, options)
        if self._proxy is None:
            raise BackendProcessError("dataclaw rollout proxy did not initialize")
        self._proxy.upstream_headers = {"Authorization": f"Bearer {upstream_key}"}
        self._proxy._static_upstream_header_names = {"authorization"}
        self._proxy.model_override = options.model_id
        self._proxy.debug_log_dir = Path(binding_context.binding.runtime_dir) / "logs" / "proxy"

    async def health(self) -> bool:
        if self._process is None or self._client is None:
            return False
        if self._process.returncode is not None or self._gateway_token is None:
            return False
        try:
            response = await self._client.get("/readyz", timeout=2.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def _wait_until_healthy(self) -> None:
        assert self._binding_context is not None
        options = self._options
        timeout = self._binding_context.effective_config.health_check_timeout
        if isinstance(options, DataclawBackendOptions):
            timeout = max(timeout, options.startup_health_timeout)
        deadline = asyncio.get_running_loop().time() + timeout
        interval = self._binding_context.effective_config.health_check_interval
        while asyncio.get_running_loop().time() < deadline:
            if self._process is not None and self._process.returncode is not None:
                stderr_tail = self._openclaw_stderr_tail()
                message = f"dataclaw exited early with code {self._process.returncode}"
                if stderr_tail:
                    message = f"{message}; stderr tail: {stderr_tail}"
                raise BackendProcessError(message)
            if await self.health():
                return
            await asyncio.sleep(interval)
        raise BackendProcessError("Timed out waiting for dataclaw health check.")

    async def _stop_gateway_process(self) -> None:
        # dataclaw's launcher can leave the actual gateway child alive after
        # its wrapper exits on SIGTERM.  Retain the process group id and reap
        # any such child after the shared OpenClaw shutdown path runs.
        process_group_id = self._process_group_id
        await super()._stop_gateway_process()
        if process_group_id is not None:
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
