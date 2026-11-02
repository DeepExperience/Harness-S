"""Built-in Sample-oriented reward functions."""

from __future__ import annotations

from typing import Any

from dressage.reward.registry import register_reward


def _metadata(sample: Any) -> dict:
    metadata = getattr(sample, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _label(sample: Any) -> str:
    label = getattr(sample, "label", None)
    if label is None:
        label = _metadata(sample).get("label", "")
    return "" if label is None else str(label).strip()


def _response(sample: Any) -> str:
    return "" if getattr(sample, "response", None) is None else str(sample.response)


@register_reward("exact_match")
def exact_match(sample: Any, *, args: Any | None = None, **_: Any) -> float:
    """Return 1.0 when the response exactly matches the label."""
    del args
    label = _label(sample)
    if not label:
        return 0.0
    return 1.0 if _response(sample).strip() == label else 0.0


@register_reward("contains_label")
def contains_label(sample: Any, *, args: Any | None = None, **_: Any) -> float:
    """Return 1.0 when the label appears in the response."""
    del args
    label = _label(sample)
    if not label:
        return 0.0
    return 1.0 if label in _response(sample) else 0.0


@register_reward("constant")
def constant(sample: Any, *, args: Any | None = None, **_: Any) -> float:
    """Return ``sample.metadata['constant_reward']`` for tests and smoke runs."""
    del args
    return float(_metadata(sample).get("constant_reward", 0.0))


@register_reward("metadata_score")
def metadata_score(sample: Any, *, args: Any | None = None, **_: Any) -> float:
    """Return a score embedded in sample metadata."""
    del args
    metadata = _metadata(sample)
    if "reward" in metadata:
        return float(metadata["reward"])
    if "score" in metadata:
        return float(metadata["score"])
    return 0.0


@register_reward("default")
def default_reward(sample: Any, *, args: Any | None = None, **kwargs: Any) -> float:
    """Default to label containment; unlabeled samples receive zero reward."""
    return contains_label(sample, args=args, **kwargs)
