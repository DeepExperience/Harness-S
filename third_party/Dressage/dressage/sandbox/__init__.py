"""Sandbox provider abstractions."""

from dressage.sandbox.factory import create_sandbox_provider_from_env
from dressage.sandbox.provider import SandboxProvider
from dressage.sandbox.types import (
    CommandResult,
    SandboxEndpoint,
    SandboxLease,
    SandboxServiceSpec,
    SandboxSpec,
)

__all__ = [
    "CommandResult",
    "SandboxEndpoint",
    "SandboxLease",
    "SandboxProvider",
    "SandboxServiceSpec",
    "SandboxSpec",
    "create_sandbox_provider_from_env",
]
