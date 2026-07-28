"""Shared Pydantic models for evidence and agent responses."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from tools.privacy import mask_metadata_text, sanitize_metadata_dict

SourceType = Literal[
    "academic_index",
    "uploaded_document",
    "handbook",
    "web_page",
    "web_search",
    "tool_output",
    "unknown",
]


def _safe_text(value: Any, *, limit: int) -> str:
    try:
        rendered = str(value)
    except Exception:
        rendered = f"[UNPRINTABLE_{type(value).__name__}]"
    return mask_metadata_text(rendered).strip()[:limit]


def _bounded_scalar(value: Any, *, string_limit: int) -> Any:
    if isinstance(value, str):
        return mask_metadata_text(value)[:string_limit]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return _safe_text(value, limit=string_limit)


def _bounded_metadata(value: Dict[str, Any], *, max_items: int = 100) -> Dict[str, Any]:
    sanitized = sanitize_metadata_dict(value if isinstance(value, dict) else {})
    bounded: Dict[str, Any] = {}
    for index, (key, item) in enumerate(sanitized.items()):
        if index >= max_items:
            bounded["__truncated_items__"] = True
            break
        safe_key = str(key)[:200]
        if isinstance(item, dict):
            bounded[safe_key] = _bounded_metadata(item, max_items=50)
        elif isinstance(item, list):
            bounded[safe_key] = [
                _bounded_scalar(entry, string_limit=1000)
                for entry in item[:100]
                if isinstance(entry, (str, int, float, bool)) or entry is None
            ]
            if len(item) > 100:
                bounded[safe_key].append("[TRUNCATED_ITEMS]")
        else:
            bounded[safe_key] = _bounded_scalar(item, string_limit=4000)
    return bounded


class Citation(BaseModel):
    """A source selected from an actual tool result."""

    label: str = Field(..., description="Inline citation marker, normally '[1]'.")
    title: str = Field(..., min_length=1, max_length=500, description="Human-readable source title.")
    url: str = Field(..., min_length=1, max_length=4096, description="Public URL or local:// document identifier.")
    source_type: SourceType = Field(default="unknown")
    snippet: Optional[str] = Field(default=None, description="Relevant evidence passage.")
    quote: Optional[str] = Field(default=None, description="Exact supporting excerpt when available.")
    source_id: Optional[str] = Field(default=None, max_length=4096)
    doc_id: Optional[str] = Field(default=None, max_length=200)
    chunk_id: Optional[str] = Field(default=None, max_length=500)
    page_number: Optional[int] = Field(default=None, ge=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        value = str(value or "").strip()
        inner = value[1:-1].strip() if value.startswith("[") and value.endswith("]") else ""
        if not inner or len(value) > 64 or "\n" in value or "\r" in value:
            raise ValueError("Citation labels must use non-empty bracket notation, for example '[1]'.")
        return value

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        bounded = _safe_text(value, limit=500)
        if not bounded:
            raise ValueError("Citation titles may not be empty.")
        return bounded

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        bounded = _safe_text(value, limit=4096)
        if not bounded:
            raise ValueError("Citation URLs may not be empty.")
        return bounded

    @field_validator("source_id", "doc_id", "chunk_id")
    @classmethod
    def strip_identifier_fields(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        limits = {"source_id": 4096, "doc_id": 200, "chunk_id": 500}
        # Pydantic applies declared max lengths before this validator; this second
        # boundary masks credentials and paths while preserving optional emptiness.
        bounded = _safe_text(value, limit=max(limits.values()))
        return bounded or None

    @field_validator("snippet", "quote")
    @classmethod
    def bound_evidence_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        bounded = _safe_text(value, limit=4000)
        return bounded or None

    @field_validator("metadata")
    @classmethod
    def bound_metadata(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return _bounded_metadata(value)


class AgentAnswer(BaseModel):
    """Structured agent response with server-controlled evidence."""

    answer: str = Field(..., min_length=1, max_length=100_000)
    citations: List[Citation] = Field(default_factory=list, max_length=100)
    warnings: List[str] = Field(default_factory=list, max_length=100)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        bounded = _safe_text(value, limit=100_000)
        if not bounded:
            raise ValueError("Agent answers may not be empty.")
        return bounded

    @field_validator("warnings")
    @classmethod
    def bound_warnings(cls, values: List[str]) -> List[str]:
        bounded: List[str] = []
        for value in values[:100]:
            text = _safe_text(value, limit=2000)
            if text:
                bounded.append(text)
        return bounded

    @field_validator("metadata")
    @classmethod
    def bound_answer_metadata(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return _bounded_metadata(value)
