"""Validated provider boundary over the research-agent implementation.

The complete reasoning and tool loop remains in ``search_agent_legacy``. This module
hardens values supplied by callers or model providers before the preserved loop uses
or echoes them, and prevents empty local lookups from becoming evidence.
"""

from __future__ import annotations

import math
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
        super().__init__(
            model=selected_model,
            api_key=api_key,
            base_url=base_url,
            owner_id=owner,
            request_timeout=min(provider_timeout, 300.0),
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            tool_timeout=min(per_tool_timeout, 300.0),
            max_response_tokens=max_response_tokens,
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
