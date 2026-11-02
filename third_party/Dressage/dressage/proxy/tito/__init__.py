"""Incremental tokenization helpers for concat trajectory building."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .tito_tokenizer import Qwen35TITOTokenizer, Qwen36TITOTokenizer


_TEMPLATE_FILES = {
    "qwen3_5": "qwen3_5_fixed.jinja",
}
_NATIVE_TEMPLATE_MODELS = {"qwen3_6"}
SUPPORTED_TITO_MODELS = tuple(sorted(set(_TEMPLATE_FILES) | _NATIVE_TEMPLATE_MODELS))


def load_fixed_template(model_type: str) -> str:
    """Load the fixed chat template for a supported TITO model type."""

    try:
        filename = _TEMPLATE_FILES[model_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported TITO model type: {model_type!r}") from exc

    template_path = Path(__file__).with_name("templates") / filename
    template = template_path.read_text(encoding="utf-8")
    if not template.strip():
        raise ValueError(f"TITO chat template is empty: {template_path}")
    return template


def configure_tito_chat_template(tokenizer: Any, *, model_type: str) -> None:
    """Configure the tokenizer chat template for a supported TITO model type."""

    if model_type in _TEMPLATE_FILES:
        tokenizer.chat_template = load_fixed_template(model_type)
        return
    if model_type in _NATIVE_TEMPLATE_MODELS:
        template = getattr(tokenizer, "chat_template", None)
        if not isinstance(template, str) or not template.strip():
            raise ValueError(f"TITO model {model_type!r} requires tokenizer.chat_template")
        return
    raise ValueError(f"Unsupported TITO model type: {model_type!r}")


def create_tito_tokenizer(tokenizer: Any, *, model_type: str) -> Qwen35TITOTokenizer:
    """Create the TITO tokenizer for a supported model type."""

    if model_type == "qwen3_5":
        return Qwen35TITOTokenizer(tokenizer)
    if model_type == "qwen3_6":
        return Qwen36TITOTokenizer(tokenizer)
    raise ValueError(f"Unsupported TITO model type: {model_type!r}")


__all__ = [
    "Qwen35TITOTokenizer",
    "Qwen36TITOTokenizer",
    "SUPPORTED_TITO_MODELS",
    "configure_tito_chat_template",
    "create_tito_tokenizer",
    "load_fixed_template",
]
