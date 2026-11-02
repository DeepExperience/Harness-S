"""Sample-oriented reward function registry."""

from __future__ import annotations

import importlib
import os
from typing import Any, Callable

from dressage.reward.execution import execute_reward

_REWARD_REGISTRY: dict[str, Callable] = {}
_LOADED_REWARD_MODULES: set[str] = set()


def register_reward(name: str):
    """Register a reward function with signature ``fn(sample, *, args=None)``."""
    def decorator(fn: Callable) -> Callable:
        if name in _REWARD_REGISTRY:
            raise ValueError(f"Reward function '{name}' is already registered")
        _REWARD_REGISTRY[name] = fn
        return fn
    return decorator


def get_reward_fn(name: str) -> Callable:
    """Look up a reward function by name. Raises KeyError if not found."""
    if name not in _REWARD_REGISTRY:
        raise KeyError(
            f"Reward function '{name}' not found. "
            f"Available: {list(_REWARD_REGISTRY.keys())}"
        )
    return _REWARD_REGISTRY[name]


def list_reward_fns() -> list[str]:
    """List all registered reward function names."""
    return list(_REWARD_REGISTRY.keys())


def load_reward_modules(modules: str | list[str] | tuple[str, ...] | None = None) -> None:
    """Import modules containing ``@register_reward`` declarations.

    When ``modules`` is omitted, ``DRESSAGE_REWARD_MODULES`` is used. The value
    may be a comma-separated string or an iterable of module paths.
    """
    if modules is None:
        modules = os.environ.get("DRESSAGE_REWARD_MODULES", "")
    if isinstance(modules, str):
        module_names = [item.strip() for item in modules.split(",") if item.strip()]
    else:
        module_names = [str(item).strip() for item in modules if str(item).strip()]

    for module_name in module_names:
        if module_name in _LOADED_REWARD_MODULES:
            continue
        importlib.import_module(module_name)
        _LOADED_REWARD_MODULES.add(module_name)


async def call_reward_fn(
    name: str,
    sample: Any,
    *,
    args: Any | None = None,
    **kwargs: Any,
) -> float:
    """Execute a registered reward without blocking rollout scheduling."""
    fn = get_reward_fn(name)
    return await execute_reward(name, fn, sample, args=args, **kwargs)
