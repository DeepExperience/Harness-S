"""Sandbox provider protocol.

A provider owns sandbox lifecycle and low-level sandbox capabilities only.  It
must not know about Dressage blackbox-agent registration or message protocols.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from dressage.sandbox.types import CommandResult, SandboxEndpoint, SandboxLease, SandboxSpec


@runtime_checkable
class SandboxProvider(Protocol):
    """Provider-neutral sandbox capability surface."""

    name: str

    async def create(self, spec: SandboxSpec) -> SandboxLease:
        """Create or acquire a sandbox lease for one trajectory."""

    async def terminate(self, lease: SandboxLease | str) -> dict[str, Any]:
        """Terminate or release a sandbox lease."""

    async def get_public_url(
        self,
        lease: SandboxLease,
        *,
        port: int,
        service_name: str | None = None,
    ) -> SandboxEndpoint:
        """Return a public endpoint for a service/port in the sandbox."""

    async def run_command(
        self,
        lease: SandboxLease,
        command: str | list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        stdin: str | bytes | None = None,
    ) -> CommandResult:
        """Execute a command in the sandbox."""

    async def read_file(
        self,
        lease: SandboxLease,
        path: str,
        *,
        encoding: str | None = "utf-8",
        max_bytes: int | None = None,
    ) -> str | bytes:
        """Read a file from the sandbox."""

    async def write_file(
        self,
        lease: SandboxLease,
        path: str,
        content: str | bytes,
        *,
        encoding: str | None = "utf-8",
        append: bool = False,
    ) -> dict[str, Any]:
        """Write a file in the sandbox."""
