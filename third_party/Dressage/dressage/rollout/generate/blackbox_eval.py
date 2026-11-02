"""Evaluation-safe wrapper around the immutable black-box dispatcher."""

from __future__ import annotations

from typing import Any

from dressage.rollout.generate.blackbox_dispatch import generate as _generate


async def generate(
    args: Any,
    sample: Any,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Any:
    """Use the training-identical dispatcher under Slime's eval protocol.

    ``blackbox_dispatch.generate`` rejects evaluation mode because its output
    is multi-segment training data. AppWorld evaluation intentionally needs
    that same immutable Harness execution and official reward path; the
    task-level eval logger collapses the returned segments to one terminal
    outcome per task.
    """

    del evaluation
    return await _generate(args, sample, sampling_params, evaluation=False)
