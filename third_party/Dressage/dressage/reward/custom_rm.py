"""Slime custom reward model entrypoint backed by Dressage reward registry."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from dressage.reward import call_reward_fn, load_reward_modules

_MODULES_LOADED = False


def _reward_name(sample: Any) -> str:
    metadata = getattr(sample, "metadata", None)
    if isinstance(metadata, dict):
        return str(metadata.get("reward_fn") or "default")
    return "default"


async def _score_one(args: Any, sample: Any, **kwargs: Any) -> float:
    global _MODULES_LOADED
    if not _MODULES_LOADED:
        load_reward_modules()
        config_path = getattr(args, "mopd_teacher_config", None) or os.environ.get(
            "DRESSAGE_MOPD_TEACHER_CONFIG"
        )
        if config_path:
            from dressage.rollout.mopd import load_mopd_config

            load_reward_modules(load_mopd_config(str(config_path)).reward_modules)
        _MODULES_LOADED = True
    return await call_reward_fn(_reward_name(sample), sample, args=args, **kwargs)


async def custom_rm(args: Any, sample_or_samples: Any, **kwargs: Any) -> float | list[float]:
    """Reward function loaded through Slime ``--custom-rm-path``.

    Slime calls this after custom generation when ``sample.reward`` is still
    unset. This function intentionally returns rewards to Slime instead of
    writing to proxy state.
    """
    if isinstance(sample_or_samples, list):
        tasks = [asyncio.create_task(_score_one(args, sample, **kwargs)) for sample in sample_or_samples]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            # Join sibling segment scorers before the trajectory boundary
            # records zero or propagates shutdown. Thread-backed graders drain
            # on cancellation, so no late writes race the retained Sample.
            for task in tasks:
                if not task.done():task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
    return await _score_one(args, sample_or_samples, **kwargs)
