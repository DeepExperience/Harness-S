"""Shared blackbox Paddock utility helpers."""

from __future__ import annotations

import logging
import os
import random
from urllib.parse import urlparse

from dressage.config import proxy_url as default_proxy_url

logger = logging.getLogger(__name__)

LOCAL_PROXY_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _require_public_proxy_url(proxy_public_url: str | None = None) -> str:
    value = proxy_public_url or default_proxy_url()
    if not value:
        raise ValueError(
            "DRESSAGE_PROXY_URL must be set to a sandbox-reachable proxy URL"
        )
    return _validate_public_proxy_url(value)


def _validate_public_proxy_url(value: str) -> str:
    url = value.rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(
            "DRESSAGE_PROXY_URL must be an http(s) URL with a host, "
            f"got: {value!r}"
        )
    host = parsed.hostname.lower()
    if host in LOCAL_PROXY_HOSTS or host.startswith("127."):
        raise ValueError(
            "DRESSAGE_PROXY_URL must be reachable from the sandbox and "
            f"cannot use a local-only host: {value!r}"
        )
    return url


def _env_int(name: str, default: int, *, min_value: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("invalid %s=%r; falling back to %d", name, value, default)
        return default
    if parsed < min_value:
        logger.warning("invalid %s=%r; falling back to %d", name, value, default)
        return default
    return parsed


def _env_float(name: str, default: float, *, min_value: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError:
        logger.warning("invalid %s=%r; falling back to %.3f", name, value, default)
        return default
    if parsed < min_value:
        logger.warning("invalid %s=%r; falling back to %.3f", name, value, default)
        return default
    return parsed


def _jittered_delay(delay: float, jitter_fraction: float) -> float:
    if delay <= 0 or jitter_fraction <= 0:
        return delay
    return delay + random.uniform(0.0, delay * jitter_fraction)


def _exception_summary(exc: BaseException) -> str:
    return " ".join(str(exc).splitlines()) or type(exc).__name__
