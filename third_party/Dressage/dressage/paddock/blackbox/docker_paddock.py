"""Docker blackbox paddock with network-none fresh evaluator support."""

from __future__ import annotations

from typing import Any

from dressage.paddock.blackbox.common.state import SandboxState
from dressage.paddock.blackbox.paddock import BlackboxAgentPaddock
from dressage.sandbox import SandboxSpec
from dressage.sandbox.local.docker import DockerSandboxProvider


class DockerBlackboxAgentPaddock(BlackboxAgentPaddock):
    """Use BBS for agents and direct docker-exec for fresh evaluators."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("provider", DockerSandboxProvider())
        super().__init__(**kwargs)

    async def init(
        self,
        traj_id: str,
        env_type: str | None = None,
        env_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> SandboxState:
        if not traj_id.startswith("swegym-eval-"):
            return await super().init(traj_id, env_type, env_args, **kwargs)
        combined_args = {**(env_args or {}), **kwargs}
        lease = await self._provider.create(
            SandboxSpec(
                trajectory_id=traj_id,
                env_type=env_type,
                env_args=combined_args,
                timeout_sec=combined_args.get("sandbox_timeout_sec"),
                metadata={"paddock_mode": "blackbox", "role": "evaluator"},
            )
        )
        state = SandboxState(
            trajectory_id=traj_id,
            sandbox_url=f"docker://{lease.sandbox_id}",
            sandbox_id=lease.sandbox_id,
            raw_register_response={
                "provider": lease.provider,
                "sandbox_id": lease.sandbox_id,
                "metadata": lease.metadata,
                "evaluator_only": True,
            },
        )
        self._leases[traj_id] = lease
        self._states[traj_id] = state
        return state

    async def register_agent(
        self,
        state: SandboxState | str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        resolved = self._resolve_state(state)
        lease = self._leases.get(resolved.trajectory_id)
        if lease is not None and lease.metadata.get("evaluator_only"):
            return {
                "status": "not_required",
                "evaluator_only": True,
                "network": "none",
            }
        return await super().register_agent(resolved, **kwargs)

    async def execute_cmd(
        self,
        state: SandboxState | str,
        *,
        session_id: str,
        cmd: str,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        resolved = self._resolve_state(state)
        lease = self._leases.get(resolved.trajectory_id)
        if lease is not None and lease.metadata.get("evaluator_only"):
            result = await self._provider.run_command(
                lease,
                cmd,
                timeout=timeout,
            )
            return result.to_dict()
        return await super().execute_cmd(
            resolved,
            session_id=session_id,
            cmd=cmd,
            timeout=timeout,
        )
