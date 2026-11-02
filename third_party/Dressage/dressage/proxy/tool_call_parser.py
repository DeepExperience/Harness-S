"""Tool call parsing helpers and parser orchestration for proxy responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

from .tool_call_ids import new_openai_tool_call_id

ToolCallParser = Callable[[str], tuple[str | None, list[dict] | None]]

_HERMES_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)
_QWEN_TOOL_CALL_BLOCK_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
)
_QWEN_FUNCTION_PATTERN = re.compile(
    r"<function=([^>\s]+)>\s*(.*?)\s*</function>", re.DOTALL
)
_QWEN_PARAMETER_PATTERN = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL
)


def _strip_blank_content(text: str | None) -> str | None:
    if text is None:
        return None
    stripped = text.strip()
    return stripped or None


def _build_openai_tool_call(
    *,
    index: int,
    name: str,
    arguments: Any,
) -> dict[str, Any]:
    return {
        "id": new_openai_tool_call_id(),
        "type": "function",
        "index": index,
        "function": {
            "name": str(name),
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _clean_text_without_spans(text: str, spans: list[tuple[int, int]]) -> str | None:
    if not spans:
        return text
    pieces: list[str] = []
    cursor = 0
    for start, end in spans:
        if cursor < start:
            pieces.append(text[cursor:start])
        cursor = end
    if cursor < len(text):
        pieces.append(text[cursor:])
    return _strip_blank_content("".join(pieces))


def parse_hermes_tool_calls(text: str) -> tuple[str | None, list[dict] | None]:
    """Extract Hermes-style ``<tool_call>`` JSON blocks from raw text."""

    tool_calls: list[dict[str, Any]] = []
    parsed_spans: list[tuple[int, int]] = []
    for match in _HERMES_TOOL_CALL_PATTERN.finditer(text):
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        tool_calls.append(
            _build_openai_tool_call(
                index=len(tool_calls),
                name=parsed.get("name", ""),
                arguments=parsed.get("arguments", {}),
            )
        )
        parsed_spans.append(match.span())

    if not tool_calls:
        return text, None
    return _clean_text_without_spans(text, parsed_spans), tool_calls


def _parse_qwen_parameter_value(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_qwen3_5_tool_calls(text: str) -> tuple[str | None, list[dict] | None]:
    """Extract Qwen 3.5 XML-style tool calls from raw text."""

    tool_calls: list[dict[str, Any]] = []
    parsed_spans: list[tuple[int, int]] = []

    for block_match in _QWEN_TOOL_CALL_BLOCK_PATTERN.finditer(text):
        block_body = block_match.group(1)
        parsed_in_block: list[dict[str, Any]] = []
        for function_match in _QWEN_FUNCTION_PATTERN.finditer(block_body):
            function_name = function_match.group(1).strip()
            if not function_name:
                continue
            parameters: dict[str, Any] = {}
            function_body = function_match.group(2)
            for parameter_match in _QWEN_PARAMETER_PATTERN.finditer(function_body):
                parameter_name = parameter_match.group(1).strip()
                if not parameter_name:
                    continue
                parameters[parameter_name] = _parse_qwen_parameter_value(
                    parameter_match.group(2)
                )
            parsed_in_block.append(
                _build_openai_tool_call(
                    index=len(tool_calls) + len(parsed_in_block),
                    name=function_name,
                    arguments=parameters,
                )
            )

        if not parsed_in_block:
            continue
        parsed_spans.append(block_match.span())
        tool_calls.extend(parsed_in_block)

    if not tool_calls:
        return text, None
    return _clean_text_without_spans(text, parsed_spans), tool_calls


@dataclass(frozen=True)
class ToolCallParserSpec:
    local_parser: ToolCallParser | None = None
    sglang_tool_call_parser_name: str | None = None


class ModelToolCallParserRegistry:
    """Resolve model tool-call types into local and SGLang parser capabilities."""

    def __init__(self):
        self._specs: dict[str, ToolCallParserSpec] = {}

    def register(self, model_tool_call_type: str, spec: ToolCallParserSpec) -> None:
        self._specs[model_tool_call_type] = spec

    def resolve(self, model_tool_call_type: str | None) -> ToolCallParserSpec | None:
        if model_tool_call_type is None:
            return None
        return self._specs.get(model_tool_call_type)


def create_default_tool_call_parser_registry() -> ModelToolCallParserRegistry:
    registry = ModelToolCallParserRegistry()
    registry.register("hermes", ToolCallParserSpec(local_parser=parse_hermes_tool_calls))
    registry.register(
        "qwen3_5",
        ToolCallParserSpec(
            local_parser=parse_qwen3_5_tool_calls,
            sglang_tool_call_parser_name="qwen3_coder",
        ),
    )
    return registry


class ProxyToolCallParser:
    """Parse raw SGLang text into OpenAI-style assistant content and tool calls."""

    def __init__(
        self,
        sglang_client: Any,
        *,
        model_tool_call_type: str | None,
        backend: Literal["local", "sglang_api", "hybrid"],
        registry: ModelToolCallParserRegistry | None = None,
        legacy_local_parser: ToolCallParser | None = None,
    ):
        self._sglang_client = sglang_client
        self._model_tool_call_type = model_tool_call_type
        self._backend = backend
        self._registry = registry or create_default_tool_call_parser_registry()
        self._legacy_local_parser = legacy_local_parser

    def _resolve_spec(self) -> ToolCallParserSpec | None:
        return self._registry.resolve(self._model_tool_call_type)

    def _resolve_local_parser(self) -> ToolCallParser | None:
        if self._legacy_local_parser is not None:
            return self._legacy_local_parser
        spec = self._resolve_spec()
        return None if spec is None else spec.local_parser

    def _resolve_sglang_tool_call_parser_name(self) -> str | None:
        spec = self._resolve_spec()
        return None if spec is None else spec.sglang_tool_call_parser_name

    @staticmethod
    def _normalize_sglang_call_arguments(call: Any) -> str | None:
        if not isinstance(call, dict):
            return None
        arguments = call.get("parameters", call.get("arguments"))
        if isinstance(arguments, str):
            return arguments
        if arguments is None:
            return json.dumps({}, ensure_ascii=False)
        if isinstance(arguments, (dict, list)):
            return json.dumps(arguments, ensure_ascii=False)
        return json.dumps(arguments, ensure_ascii=False)

    def _normalize_sglang_result(
        self,
        parsed: dict[str, Any] | None,
    ) -> tuple[str | None, list[dict] | None] | None:
        if not isinstance(parsed, dict):
            return None
        if "normal_text" not in parsed or "calls" not in parsed:
            return None
        calls = parsed.get("calls")
        if not isinstance(calls, list):
            return None

        tool_calls: list[dict[str, Any]] = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            if not isinstance(name, str) or not name:
                continue
            arguments = self._normalize_sglang_call_arguments(call)
            if arguments is None:
                continue
            tool_calls.append(
                {
                    "id": new_openai_tool_call_id(),
                    "type": "function",
                    "index": len(tool_calls),
                    "function": {"name": name, "arguments": arguments},
                }
            )

        if not tool_calls:
            return None

        normal_text = parsed.get("normal_text")
        if normal_text is None:
            content = None
        else:
            content = str(normal_text)
        return _strip_blank_content(content), tool_calls

    async def _parse_with_sglang_api(
        self,
        raw_text: str,
        tools: list[dict] | None,
        *,
        routing_key: str | None,
    ) -> tuple[str | None, list[dict] | None] | None:
        if not tools:
            return None
        parser_name = self._resolve_sglang_tool_call_parser_name()
        if parser_name is None:
            return None
        try:
            parsed = await self._sglang_client.parse_function_call(
                raw_text,
                tool_call_parser=parser_name,
                tools=tools,
                routing_key=routing_key,
            )
        except Exception:
            return None
        return self._normalize_sglang_result(parsed)

    def _parse_locally(
        self,
        raw_text: str,
    ) -> tuple[str | None, list[dict] | None]:
        local_parser = self._resolve_local_parser()
        if local_parser is None:
            return raw_text, None
        return local_parser(raw_text)

    async def parse(
        self,
        raw_text: str,
        tools: list[dict] | None,
        *,
        routing_key: str | None = None,
    ) -> tuple[str | None, list[dict] | None]:
        if self._legacy_local_parser is not None:
            return self._parse_locally(raw_text)

        if self._backend == "local":
            return self._parse_locally(raw_text)

        if self._backend == "sglang_api":
            return (
                await self._parse_with_sglang_api(
                    raw_text, tools, routing_key=routing_key
                )
            ) or (raw_text, None)

        parsed = await self._parse_with_sglang_api(
            raw_text, tools, routing_key=routing_key
        )
        if parsed is not None:
            return parsed
        return self._parse_locally(raw_text)
