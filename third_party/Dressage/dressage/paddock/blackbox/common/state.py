"""Shared blackbox sandbox lease state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SandboxState:
    trajectory_id: str
    sandbox_url: str
    sandbox_id: str | None = None
    raw_register_response: dict[str, Any] = field(default_factory=dict)
