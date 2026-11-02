"""Reward system — per-sample reward functions with registry."""

from dressage.reward.registry import (
    call_reward_fn,
    get_reward_fn,
    list_reward_fns,
    load_reward_modules,
    register_reward,
)

import dressage.reward.helpers  # noqa: F401  — registers built-in reward functions

__all__ = [
    "call_reward_fn",
    "get_reward_fn",
    "list_reward_fns",
    "load_reward_modules",
    "register_reward",
]
