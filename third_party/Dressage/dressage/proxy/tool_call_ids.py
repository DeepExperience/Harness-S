"""Tool call id helpers for provider-compatible replay handling."""

from __future__ import annotations

import re
import uuid
from typing import Any

_OPENCLAW_STRICT_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]")


def new_openai_tool_call_id() -> str:
    return f"call{uuid.uuid4().hex[:8]}"


def canonicalize_openclaw_tool_call_id(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return _OPENCLAW_STRICT_NON_ALNUM_RE.sub("", value)
