"""Shared sandbox provider data types.

These types intentionally do not contain blackbox-agent concepts such as
``register_agent`` or ``call_agent``.  They describe only sandbox lifecycle,
network endpoints, filesystem access, and command execution capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


SandboxCapability = Literal["command", "file", "public_url"]


@dataclass(frozen=True)
class SandboxServiceSpec:
    """A service expected to be reachable inside a sandbox."""

    name: str
    port: int
    health_path: str = "/health"


@dataclass(frozen=True)
class SandboxSpec:
    """Sandbox creation request shared by local and remote providers."""

    trajectory_id: str
    env_type: str | None = None
    env_args: dict[str, Any] = field(default_factory=dict)
    services: tuple[SandboxServiceSpec, ...] = ()
    timeout_sec: float | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SandboxEndpoint:
    """Public endpoint for a sandbox service.

    ``headers`` travels with the endpoint because remote providers may expose a
    URL that requires authentication or provider-specific forwarding headers.
    """

    url: str
    headers: dict[str, str] = field(default_factory=dict)

    def normalized(self) -> "SandboxEndpoint":
        return SandboxEndpoint(url=self.url.rstrip("/"), headers=dict(self.headers))


@dataclass
class SandboxLease:
    """Provider-neutral sandbox lease returned from ``SandboxProvider.create``."""

    trajectory_id: str
    provider: str
    sandbox_id: str | None = None
    endpoints: dict[str, SandboxEndpoint] = field(default_factory=dict)
    capabilities: set[SandboxCapability] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: Any | None = None

    def endpoint(self, service_name: str) -> SandboxEndpoint:
        try:
            return self.endpoints[service_name].normalized()
        except KeyError as exc:
            raise KeyError(
                f"sandbox lease for trajectory_id={self.trajectory_id!r} has no "
                f"endpoint named {service_name!r}"
            ) from exc


@dataclass(frozen=True)
class CommandResult:
    """Result of executing a command through a sandbox provider."""

    cmd: str | list[str]
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    timed_out: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cmd": self.cmd,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "metadata": dict(self.metadata),
        }
