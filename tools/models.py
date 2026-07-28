"""Shared Pydantic models for evidence and agent responses."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from tools.privacy import sanitize_metadata_dict

SourceType = Literal[
    "academic_index",
    "uploaded_document",
    "handbook",
    "web_page",
    "web_search",
    "tool_output",
    "unknown",
]


def _bounded_metadata(value: Dict[str, Any], *, max_items: int = 100) -> Dict[str, Any]:
    sanitized = sanitize_metadata_dict(value)
    bounded: Dict[str, Any] = {}
    for index, (key, item) in enumerate(sanitized.items()):
        if index >= max_items:
            break
        safe_key = str(key)[:200]
        if isinstance(item, str):
            bounded[safe_key] = item[:4000]
        elif isinstance(item, (int, float, bool)) or item is None:
            bounded[safe_key] = item
        elif isinstance(item, list):
            bounded[safe_key] = [
                entry[:1000] if isinstance(entry, str) else entry
                for entry in item[:100]
                if isinstance(entry, (str, int, float, bool)) or entry is None
            ]
        elif isinstance(item, dict):
            bounded[safe_key] = _bounded_metadata(item, max_items=50)
        else:
            bounded[safe_key] = str(item)[:1000]
    return bounded


class Citation(BaseModel):
    """A source selected from an actual tool result."""

    label: str = Field(..., description="Inline citation marker, normally '[1]'.")
    title: str = Field(..., min_length=1, max_length=500, description="Human-readable source title.")
    url: str = Field(..., max_length=4096, description="Public URL or local:// document identifier.")
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
        value = value.strip()
        if not (value.startswith("[") and value.endswith("]") and len(value) <= 64):
            raise ValueError("Citation labels must use bracket notation, for example '[1]'.")
        return value

    @field_validator("title", "url", "source_id", "doc_id", "chunk_id")
    @classmethod
    def strip_identifier_fields(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip()

    @field_validator("snippet", "quote")
    @classmethod
    def bound_evidence_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value[:4000] or None

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

    @field_validator("warnings")
    @classmethod
    def bound_warnings(cls, values: List[str]) -> List[str]:
        return [str(value).strip()[:2000] for value in values if str(value).strip()]

    @field_validator("metadata")
    @classmethod
    def bound_answer_metadata(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return _bounded_metadata(value)
