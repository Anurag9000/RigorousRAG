"""Validated provider boundary over the research-agent implementation.

The complete tool dispatch surface remains in ``search_agent_legacy``. This module
normalizes process-wide budgets and ensures only bounded, locally constructed tool
calls, evidence objects, and provider text enter the reasoning conversation.
"""

from __future__ import annotations

import itertools
import json
import math
import operator
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tools.config import bounded_int_env

_TOOL_WORKERS = bounded_int_env(
    "MAX_CONCURRENT_TOOL_WORKERS",
    32,
    minimum=1,
    maximum=256,
    write_back=True,
)
for _name, _default, _minimum, _maximum in (
    ("MAX_TOOL_ARGUMENT_CHARS", 50_000, 1000, 500_000),
    ("MAX_TOOL_RESULT_CHARS", 30_000, 1000, 200_000),
    ("MAX_EVIDENCE_SOURCES", 100, 1, 500),
    ("MAX_RESPONSE_TOKENS", 2000, 128, 16_000),
):
    bounded_int_env(
        _name,
        _default,
        minimum=_minimum,
        maximum=_maximum,
        write_back=True,
    )
_PENDING_TOOLS = bounded_int_env(
    "MAX_PENDING_TOOL_TASKS",
    64,
    minimum=_TOOL_WORKERS,
    maximum=4096,
    write_back=True,
)
os.environ["MAX_PENDING_TOOL_TASKS"] = str(max(_PENDING_TOOLS, _TOOL_WORKERS))

import search_agent_legacy as _implementation

from tools.security import normalize_owner_id

if not hasattr(_implementation, "_boundary_original_validate_schema_value"):
    _implementation._boundary_original_validate_schema_value = (
        _implementation._validate_schema_value
    )
if not hasattr(_implementation, "_boundary_original_ToolExecution"):
    _implementation._boundary_original_ToolExecution = _implementation.ToolExecution
if not hasattr(_implementation, "_boundary_original_SearchAgent"):
    _implementation._boundary_original_SearchAgent = _implementation.SearchAgent

_original_validate_schema_value = (
    _implementation._boundary_original_validate_schema_value
)
_original_tool_execution = _implementation._boundary_original_ToolExecution
_original_search_agent = _implementation._boundary_original_SearchAgent
_MAX_IDENTIFIER_CHARS = 200
_MAX_PROVIDER_FIELD_CHARS = 4096
_INVALID_ARGUMENTS = "__INVALID_TOOL_ARGUMENTS__"


def _safe_text(value: Any, *, limit: int, default: str = "") -> str:
    try:
        rendered = str(value if value is not None else default)
    except Exception:
        rendered = default
    return rendered[:limit]


def _safe_getattr(value: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(value, name, default)
    except Exception:
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    try:
        return bool(value)
    except Exception:
        return default


def _clean_identifier(value: Any, *, default: str) -> str:
    rendered = _safe_text(value, limit=_MAX_IDENTIFIER_CHARS, default=default)
    rendered = " ".join(
        rendered.replace("\x00", " ").replace("\r", " ").replace("\n", " ").split()
    )
    return rendered[:_MAX_IDENTIFIER_CHARS] or default


def _optional_provider_value(value: Any, label: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    if value == "":
        return None
    rendered = value.strip()
    if len(rendered) > _MAX_PROVIDER_FIELD_CHARS:
        raise ValueError(
            f"{label} may contain at most {_MAX_PROVIDER_FIELD_CHARS:,} characters."
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in rendered):
        raise ValueError(f"{label} contains invalid control characters.")
    return rendered or None


def _required_model(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("model must be a string.")
    selected = value.strip()
    if (
        not selected
        or len(selected) > _MAX_IDENTIFIER_CHARS
        or any(ord(character) < 32 or ord(character) == 127 for character in selected)
    ):
        raise ValueError("model must contain between 1 and 200 valid characters.")
    return selected


def _strict_int(value: Any, label: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        parsed = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return parsed


def _finite_timeout(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{label} must be a finite positive number.")
    return min(parsed, 300.0)


def _validate_schema_value(
    value: Any,
    schema: Any,
    path: str,
    depth: int = 0,
) -> None:
    """Reject non-finite JSON numbers before normal schema recursion."""

    expected = schema.get("type") if isinstance(schema, dict) else None
    if expected == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"{path} must be a finite number.")
    _original_validate_schema_value(value, schema, path, depth)


@dataclass(frozen=True)
class _SafeToolFunction:
    name: str
    arguments: str


@dataclass(frozen=True)
class _SafeToolCall:
    id: str
    function: _SafeToolFunction


def _sanitize_tool_call(raw_call: Any, index: int) -> _SafeToolCall:
    call_id = _clean_identifier(
        _safe_getattr(raw_call, "id"),
        default=f"call-{index + 1}",
    )
    function = _safe_getattr(raw_call, "function")
    name = _clean_identifier(
        _safe_getattr(function, "name"),
        default="unknown",
    )
    raw_arguments = _safe_getattr(function, "arguments")
    arguments = (
        raw_arguments
        if isinstance(raw_arguments, str)
        and len(raw_arguments) <= _implementation._MAX_TOOL_ARGUMENT_CHARS
        and "\x00" not in raw_arguments
        else _INVALID_ARGUMENTS
    )
    return _SafeToolCall(
        id=call_id,
        function=_SafeToolFunction(name=name, arguments=arguments),
    )


def _bounded_tool_calls(raw_calls: Any, maximum: int) -> Tuple[List[_SafeToolCall], bool]:
    limit = _strict_int(maximum, "maximum", minimum=0, maximum=64)
    if raw_calls is None:
        return [], False
    if isinstance(raw_calls, list) and not raw_calls:
        return [], False
    if isinstance(raw_calls, (str, bytes, bytearray)):
        return [], True
    try:
        values = list(itertools.islice(iter(raw_calls), limit + 1))
    except Exception:
        return [], True
    overflow = len(values) > limit
    return [
        _sanitize_tool_call(call, index)
        for index, call in enumerate(values[:limit])
    ], overflow


class _ToolExecutionBoundary(_original_tool_execution):
    """Bound provider-controlled identifiers, content, citations, and telemetry."""

    def __init__(
        self,
        tool_call_id: str,
        tool_name: str,
        content: str,
        citations: Optional[Sequence[Any]] = None,
        success: bool = True,
        error_type: Optional[str] = None,
        duration: float = 0.0,
    ) -> None:
        try:
            elapsed = float(duration)
        except (TypeError, ValueError, OverflowError):
            elapsed = 0.0
        if not math.isfinite(elapsed):
            elapsed = 0.0
        if citations is None or isinstance(citations, (str, bytes, bytearray)):
            bounded_citations: List[Any] = []
        else:
            try:
                bounded_citations = [
                    citation
                    for citation in itertools.islice(
                        iter(citations),
                        _implementation._MAX_EVIDENCE_SOURCES,
                    )
                    if isinstance(citation, _implementation.Citation)
                ]
            except Exception:
                bounded_citations = []
        super().__init__(
            tool_call_id=_clean_identifier(tool_call_id, default="unknown"),
            tool_name=_clean_identifier(tool_name, default="unknown"),
            content=_safe_text(
                content,
                limit=_implementation._MAX_TOOL_RESULT_CHARS + 1000,
            ),
            citations=bounded_citations,
            success=_safe_bool(success),
            error_type=(
                _clean_identifier(error_type, default="Error")
                if error_type is not None
                else None
            ),
            duration=max(elapsed, 0.0),
        )


class _SearchAgentBoundary(_original_search_agent):
    """Research agent with sanitized provider calls and authoritative evidence."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        owner_id: str = "default_user",
        *,
        request_timeout: float = 60.0,
        max_turns: int = 8,
        max_tool_calls: int = 24,
        tool_timeout: float = 45.0,
        max_response_tokens: Optional[int] = None,
    ) -> None:
        selected_model = _required_model(model)
        if not isinstance(owner_id, str):
            raise ValueError("owner_id must be a string.")
        owner = normalize_owner_id(owner_id)
        selected_key = _optional_provider_value(
            api_key if api_key is not None else os.getenv("OPENAI_API_KEY"),
            "api_key",
        )
        selected_base_url = _optional_provider_value(
            base_url if base_url is not None else os.getenv("OPENAI_BASE_URL"),
            "base_url",
        )
        turns = _strict_int(max_turns, "max_turns", minimum=1, maximum=20)
        tool_calls = _strict_int(
            max_tool_calls,
            "max_tool_calls",
            minimum=1,
            maximum=64,
        )
        response_tokens = (
            None
            if max_response_tokens is None
            else _strict_int(
                max_response_tokens,
                "max_response_tokens",
                minimum=128,
                maximum=16_000,
            )
        )
        super().__init__(
            model=selected_model,
            api_key=selected_key,
            base_url=selected_base_url,
            owner_id=owner,
            request_timeout=_finite_timeout(request_timeout, "request_timeout"),
            max_turns=turns,
            max_tool_calls=tool_calls,
            tool_timeout=_finite_timeout(tool_timeout, "tool_timeout"),
            max_response_tokens=response_tokens,
        )

    def run(self, query: str) -> _implementation.AgentAnswer:
        started = time.monotonic()
        if not isinstance(query, str):
            raise ValueError("query must be a string.")
        query = query.strip()
        if not query:
            return _implementation.AgentAnswer(answer="The query is empty.")
        if len(query) > 20_000:
            return _implementation.AgentAnswer(
                answer="The query exceeds the 20,000-character limit."
            )
        if self.client is None:
            answer = self._fallback_answer(query)
            _implementation.log_agent_run(
                query,
                time.monotonic() - started,
                len(answer.citations),
                success=not answer.answer.lower().startswith("error"),
                owner_id=self.owner_id,
            )
            return answer

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _implementation.SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        evidence: List[_implementation.Citation] = []
        seen_evidence: Dict[Tuple[str, str, str], str] = {}
        total_tool_calls = 0
        final_text: Optional[str] = None
        warnings: List[str] = []
        try:
            for _ in range(self.max_turns):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=_implementation.TOOLS_SCHEMA,
                    tool_choice="auto",
                    temperature=0.1,
                    max_tokens=self.max_response_tokens,
                )
                choices = _safe_getattr(response, "choices")
                if not isinstance(choices, Sequence) or isinstance(
                    choices, (str, bytes, bytearray)
                ) or not choices:
                    raise ValueError("The model provider returned no usable choice.")
                message = _safe_getattr(choices[0], "message")
                if message is None:
                    raise ValueError("The model provider returned no usable message.")
                content = _safe_text(
                    _safe_getattr(message, "content", "") or "",
                    limit=_implementation._MAX_FINAL_ANSWER_CHARS,
                )
                remaining = max(self.max_tool_calls - total_tool_calls, 0)
                tool_calls, overflow = _bounded_tool_calls(
                    _safe_getattr(message, "tool_calls"),
                    remaining,
                )
                if overflow:
                    warnings.append("The per-request tool-call budget was reached.")
                assistant_message: Dict[str, Any] = {
                    "role": "assistant",
                    "content": content,
                }
                if tool_calls:
                    assistant_message["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in tool_calls
                    ]
                messages.append(assistant_message)
                if not tool_calls:
                    if overflow:
                        break
                    final_text = self._parse_final_text(content)
                    break

                total_tool_calls += len(tool_calls)
                for execution in self._execute_tools(tool_calls):
                    relabelled = self._register_citations(
                        execution.citations,
                        evidence,
                        seen_evidence,
                    )
                    result_text = _safe_text(
                        execution.content,
                        limit=_implementation._MAX_TOOL_RESULT_CHARS,
                    )
                    payload: Dict[str, Any] = {
                        "ok": bool(execution.success),
                        "tool": execution.tool_name,
                        "result": result_text,
                        "citations": [
                            citation.model_dump(exclude_none=True)
                            for citation in relabelled
                        ],
                    }
                    if execution.error_type:
                        payload["error_type"] = execution.error_type
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": execution.tool_call_id,
                            "content": json.dumps(
                                payload,
                                ensure_ascii=False,
                                allow_nan=False,
                            ),
                        }
                    )
                    _implementation.log_tool_call(
                        execution.tool_name,
                        execution.duration,
                        execution.success,
                        error_type=execution.error_type,
                    )
                if len(evidence) >= _implementation._MAX_EVIDENCE_SOURCES and (
                    not warnings
                    or warnings[-1] != "The evidence-source budget was reached."
                ):
                    warnings.append("The evidence-source budget was reached.")
                if overflow:
                    break

            if not final_text:
                final_text = (
                    "The research agent could not complete a supported synthesis "
                    "within the configured reasoning budget."
                )
                warnings.append("Reasoning budget exhausted before a final synthesis.")
            if len(final_text) > _implementation._MAX_FINAL_ANSWER_CHARS:
                final_text = final_text[:_implementation._MAX_FINAL_ANSWER_CHARS]
                warnings.append(
                    "The final answer was truncated by the response-size limit."
                )
            selected = self._citations_used_by_answer(final_text, evidence)
            answer = _implementation.AgentAnswer(
                answer=final_text,
                citations=selected,
                warnings=warnings,
                metadata={
                    "model": self.model,
                    "tool_calls": total_tool_calls,
                    "evidence_retrieved": len(evidence),
                },
            )
            audit = _implementation.audit_hallucination(answer)
            if audit.startswith("⚠️"):
                answer.warnings.append(audit)
            _implementation.log_agent_run(
                query,
                time.monotonic() - started,
                len(answer.citations),
                success=True,
                owner_id=self.owner_id,
            )
            return answer
        except Exception as exc:
            _implementation.log_agent_run(
                query,
                time.monotonic() - started,
                0,
                success=False,
                owner_id=self.owner_id,
            )
            return _implementation.AgentAnswer(
                answer=(
                    "The research request failed before a reliable answer could be "
                    "produced. Retry after checking the configured model provider."
                ),
                warnings=[f"Request failed ({type(exc).__name__})."],
                metadata={"model": self.model},
            )

    def _execute_tools(self, tool_calls: Sequence[Any]) -> List[ToolExecution]:
        safe_calls, _overflow = _bounded_tool_calls(tool_calls, self.max_tool_calls)
        return super()._execute_tools(safe_calls)

    def _execute_tool(self, tool_call: Any) -> ToolExecution:
        return super()._execute_tool(_sanitize_tool_call(tool_call, 0))

    def _dispatch(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Tuple[str, List[_implementation.Citation]]:
        if tool_name == "search_handbook":
            text = _implementation.search_handbook(**arguments)
            if not text or text.strip() == "No handbook passage matched the query.":
                return "No handbook evidence matched the query.", []
            return text, [
                _implementation.Citation(
                    label="[1]",
                    title="RigorousRAG internal handbook",
                    url="local://handbook",
                    source_type="handbook",
                    snippet=text,
                    source_id="handbook",
                )
            ]
        return super()._dispatch(tool_name, arguments)

    @staticmethod
    def _content_with_embedded_citations(
        raw: str,
    ) -> Tuple[str, List[_implementation.Citation]]:
        if not isinstance(raw, str):
            return "Scientific tool returned an invalid response.", []
        if len(raw) > _implementation._MAX_TOOL_RESULT_CHARS * 4:
            return "Scientific tool response exceeded the server limit.", []
        try:
            parsed = json.loads(
                raw,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"Non-standard JSON constant {value}")
                ),
            )
        except Exception:
            return "Scientific tool returned an invalid response.", []
        if not isinstance(parsed, dict):
            return "Scientific tool returned an invalid response.", []
        citation_payloads = parsed.pop("citations", [])
        citations: List[_implementation.Citation] = []
        if isinstance(citation_payloads, list):
            for payload in citation_payloads[:_implementation._MAX_EVIDENCE_SOURCES]:
                if not isinstance(payload, dict):
                    continue
                try:
                    citations.append(_implementation.Citation(**payload))
                except Exception:
                    continue
        return (
            json.dumps(parsed, ensure_ascii=False, allow_nan=False)[
                :_implementation._MAX_TOOL_RESULT_CHARS
            ],
            citations,
        )

    @staticmethod
    def _parse_final_text(content: str) -> str:
        cleaned = _safe_text(
            content,
            limit=_implementation._MAX_FINAL_ANSWER_CHARS,
        ).strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            parsed = json.loads(
                cleaned,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"Non-standard JSON constant {value}")
                ),
            )
            if isinstance(parsed, dict) and isinstance(parsed.get("answer"), str):
                return parsed["answer"].strip()[:_implementation._MAX_FINAL_ANSWER_CHARS]
        except Exception:
            pass
        return cleaned or "No answer was produced."

    @staticmethod
    def _register_citations(
        incoming: Sequence[_implementation.Citation],
        registry: List[_implementation.Citation],
        seen: Dict[Tuple[str, str, str], str],
    ) -> List[_implementation.Citation]:
        if isinstance(incoming, (str, bytes, bytearray)):
            return []
        try:
            values: Iterable[Any] = itertools.islice(
                iter(incoming),
                _implementation._MAX_EVIDENCE_SOURCES,
            )
        except Exception:
            return []
        selected: List[_implementation.Citation] = []
        for citation in values:
            if not isinstance(citation, _implementation.Citation):
                continue
            key = (
                citation.source_id or citation.url,
                citation.doc_id or "",
                citation.quote or citation.snippet or "",
            )
            existing_label = seen.get(key)
            if existing_label:
                existing = next(
                    (item for item in registry if item.label == existing_label),
                    None,
                )
                if existing is not None:
                    selected.append(existing)
                continue
            if len(registry) >= _implementation._MAX_EVIDENCE_SOURCES:
                continue
            copy = citation.model_copy(deep=True)
            copy.label = f"[{len(registry) + 1}]"
            registry.append(copy)
            seen[key] = copy.label
            selected.append(copy)
        return selected

    def _expansion_model(self) -> str:
        configured = os.getenv("RETRIEVAL_EXPANSION_MODEL")
        if configured:
            try:
                return _required_model(configured)
            except ValueError:
                pass
        if self.base_url:
            return self.model
        return "gpt-4o-mini"


if not hasattr(_implementation, "_boundary_public_ToolExecution"):
    _implementation._boundary_public_ToolExecution = _ToolExecutionBoundary
if not hasattr(_implementation, "_boundary_public_SearchAgent"):
    _implementation._boundary_public_SearchAgent = _SearchAgentBoundary
ToolExecution = _implementation._boundary_public_ToolExecution
SearchAgent = _implementation._boundary_public_SearchAgent
ToolExecution.__name__ = "ToolExecution"
ToolExecution.__qualname__ = "ToolExecution"
SearchAgent.__name__ = "SearchAgent"
SearchAgent.__qualname__ = "SearchAgent"

_implementation._validate_schema_value = _validate_schema_value
_implementation.ToolExecution = ToolExecution
_implementation.SearchAgent = SearchAgent
_implementation.__doc__ = __doc__
sys.modules[__name__] = _implementation
