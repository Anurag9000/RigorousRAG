"""Shared Pydantic models for evidence and agent responses."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

SourceType = Literal[
    "academic_index",
    "uploaded_document",
    "handbook",
    "web_page",
    "web_search",
    "tool_output",
    "unknown",
]


class Citation(BaseModel):
    """A source selected from an actual tool result."""

    label: str = Field(..., description="Inline citation marker, normally '[1]'.")
    title: str = Field(..., min_length=1, description="Human-readable source title.")
    url: str = Field(..., description="Public URL or local:// document identifier.")
    source_type: SourceType = Field(default="unknown")
    snippet: Optional[str] = Field(default=None, description="Relevant evidence passage.")
    quote: Optional[str] = Field(default=None, description="Exact supporting excerpt when available.")
    source_id: Optional[str] = None
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    page_number: Optional[int] = Field(default=None, ge=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        value = value.strip()
        if not (value.startswith("[") and value.endswith("]") and len(value) <= 64):
            raise ValueError("Citation labels must use bracket notation, for example '[1]'.")
        return value

    @field_validator("snippet", "quote")
    @classmethod
    def bound_evidence_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value[:4000] or None


class AgentAnswer(BaseModel):
    """Structured agent response with server-controlled evidence."""

    answer: str = Field(..., min_length=1)
    citations: List[Citation] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
