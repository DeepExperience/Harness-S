"""Paddock implementation for whitebox tool rollouts."""

from __future__ import annotations

from typing import Any

from dressage.paddock.interface import WhiteboxPaddock
from dressage.paddock.whitebox.tools import WhiteboxToolAdapter
from dressage.sandbox import SandboxLease, SandboxSpec
from dressage.sandbox.factory import create_sandbox_provider_from_env
from dressage.sandbox.provider import SandboxProvider


class WhiteboxToolPaddock(WhiteboxPaddock):
    """Run whitebox tool calls directly through a sandbox provider."""

    def __init__(
        self,
        *,
        provider: SandboxProvider | None = None,
        tool_adapter: WhiteboxToolAdapter | None = None,
    ) -> None:
        self._provider = provider or create_sandbox_provider_from_env()
        self._tools = tool_adapter or WhiteboxToolAdapter(self._provider)
        self._leases: dict[str, SandboxLease] = {}

    async def init(
        self,
        traj_id: str,
        env_type: str | None = None,
        env_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> SandboxLease:
        env_args = {**(env_args or {}), **kwargs}
        lease = await self._provider.create(
            SandboxSpec(
                trajectory_id=traj_id,
                env_type=env_type,
                env_args=env_args,
                timeout_sec=env_args.get("sandbox_timeout_sec"),
                metadata={"paddock_mode": "whitebox"},
            )
        )
        self._leases[traj_id] = lease
        return lease

    async def terminate(
        self,
        traj_id: str,
        env_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del env_args, kwargs
        lease = self._leases.pop(traj_id, None)
        if lease is None:
            return {"terminated": False, "trajectory_id": traj_id, "missing": True}
        return await self._provider.terminate(lease)

    async def tool_call(
        self,
        traj_id: str,
        tool_id: str,
        tool_args: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        lease = self._resolve_lease(traj_id)
        return await self._tools.tool_call(lease, tool_id, tool_args)

    def _resolve_lease(self, traj_id: str) -> SandboxLease:
        try:
            return self._leases[traj_id]
        except KeyError as exc:
            raise KeyError(f"sandbox lease not found for trajectory_id={traj_id}") from exc
