"""Validated provider boundary over the research-agent implementation.

The complete reasoning and tool loop remains in ``search_agent_legacy``. This module
normalizes process-wide budgets before importing it, hardens caller/provider values,
and prevents empty local lookups from becoming evidence.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

_original_validate_schema_value = _implementation._validate_schema_value
_original_tool_execution = _implementation.ToolExecution


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


def _bounded_direct_int(value: Any, label: str, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    return max(minimum, min(parsed, maximum))


class ToolExecution(_original_tool_execution):
    """Bound provider-controlled identifiers and execution telemetry."""

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
        super().__init__(
            tool_call_id=str(tool_call_id or "unknown").strip()[:200] or "unknown",
            tool_name=str(tool_name or "unknown").strip()[:200] or "unknown",
            content=str(content or "")[:_implementation._MAX_TOOL_RESULT_CHARS + 1000],
            citations=list(citations or [])[:_implementation._MAX_EVIDENCE_SOURCES],
            success=bool(success),
            error_type=(str(error_type).strip()[:200] if error_type else None),
            duration=max(elapsed, 0.0),
        )


class SearchAgent(_implementation.SearchAgent):
    """Research agent with validated direct-construction and evidence parameters."""

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
        selected_model = str(model or "").strip()
        if not selected_model or len(selected_model) > 200:
            raise ValueError("model must contain between 1 and 200 characters.")
        owner = normalize_owner_id(owner_id)
        try:
            provider_timeout = float(request_timeout)
            per_tool_timeout = float(tool_timeout)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Agent timeouts must be numeric.") from exc
        if not math.isfinite(provider_timeout) or provider_timeout <= 0:
            raise ValueError("request_timeout must be a finite positive number.")
        if not math.isfinite(per_tool_timeout) or per_tool_timeout <= 0:
            raise ValueError("tool_timeout must be a finite positive number.")
        turns = _bounded_direct_int(max_turns, "max_turns", minimum=1, maximum=20)
        tool_calls = _bounded_direct_int(
            max_tool_calls,
            "max_tool_calls",
            minimum=1,
            maximum=64,
        )
        response_tokens = (
            None
            if max_response_tokens is None
            else _bounded_direct_int(
                max_response_tokens,
                "max_response_tokens",
                minimum=128,
                maximum=16_000,
            )
        )
        selected_base_url = str(base_url).strip() if base_url is not None else None
        if selected_base_url is not None and len(selected_base_url) > 4096:
            raise ValueError("base_url may contain at most 4,096 characters.")
        super().__init__(
            model=selected_model,
            api_key=api_key,
            base_url=selected_base_url,
            owner_id=owner,
            request_timeout=min(provider_timeout, 300.0),
            max_turns=turns,
            max_tool_calls=tool_calls,
            tool_timeout=min(per_tool_timeout, 300.0),
            max_response_tokens=response_tokens,
        )

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


_implementation._validate_schema_value = _validate_schema_value
_implementation.ToolExecution = ToolExecution
_implementation.SearchAgent = SearchAgent
_implementation.__doc__ = __doc__
sys.modules[__name__] = _implementation
