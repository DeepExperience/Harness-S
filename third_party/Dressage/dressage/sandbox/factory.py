"""Factory for public sandbox provider names."""

from __future__ import annotations

import os

from dressage.config import sandbox_provider
from dressage.sandbox.provider import SandboxProvider

_PUBLIC_PROVIDERS = {"docker", "e2b", "local_bwrap"}


def create_sandbox_provider_from_env(provider_name: str | None = None) -> SandboxProvider:
    """Create the configured sandbox provider.

    Only provider names are public. Ray pool, router, single-process debug, and
    other implementation details are not provider names.
    """

    name = (provider_name or os.environ.get("DRESSAGE_SANDBOX_PROVIDER") or sandbox_provider()).strip().lower()
    if name == "e2b":
        from dressage.sandbox.remote.e2b import E2BSandboxProvider

        return E2BSandboxProvider()
    if name == "local_bwrap":
        from dressage.sandbox.local.bwrap import LocalBwrapSandboxProvider

        return LocalBwrapSandboxProvider()
    if name == "docker":
        from dressage.sandbox.local.docker import DockerSandboxProvider

        return DockerSandboxProvider()
    expected = "|".join(sorted(_PUBLIC_PROVIDERS))
    raise ValueError(
        f"unsupported DRESSAGE_SANDBOX_PROVIDER={name!r}; expected one of {expected}"
    )
