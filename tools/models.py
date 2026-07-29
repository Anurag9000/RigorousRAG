"""Shared Pydantic models for evidence and agent responses."""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlsplit, urlunsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)

from tools.privacy import mask_metadata_text, sanitize_metadata

SourceType = Literal[
    "academic_index",
    "uploaded_document",
    "handbook",
    "web_page",
    "web_search",
    "tool_output",
    "unknown",
]

_MAX_METADATA_DEPTH = 6
_MAX_METADATA_ITEMS = 100
_MAX_CITATIONS = 100
_MAX_WARNINGS = 100


def _safe_text(value: Any, *, limit: int) -> str:
    if isinstance(value, str):
        rendered = value
    else:
        try:
            rendered = str(value)
        except Exception:
            try:
                type_name = type(value).__name__
            except Exception:
                type_name = "OBJECT"
            rendered = f"[UNPRINTABLE_{type_name}]"
    return mask_metadata_text(rendered).strip()[:limit]


def _bounded_value(
    value: Any,
    *,
    depth: int,
    max_items: int,
    string_limit: int,
) -> Any:
    if depth > _MAX_METADATA_DEPTH:
        return "[TRUNCATED_DEPTH]"
    if isinstance(value, str):
        return value[:string_limit]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        bounded: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                bounded["__truncated_items__"] = True
                break
            bounded[str(key)[:200]] = _bounded_value(
                item,
                depth=depth + 1,
                max_items=min(max_items, 50),
                string_limit=min(string_limit, 4000),
            )
        return bounded
    if isinstance(value, list):
        items = [
            _bounded_value(
                item,
                depth=depth + 1,
                max_items=min(max_items, 50),
                string_limit=min(string_limit, 1000),
            )
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            items.append({"__truncated_items__": True})
        return items
    return _safe_text(value, limit=string_limit)


def _bounded_metadata(value: Any, *, max_items: int = _MAX_METADATA_ITEMS) -> Dict[str, Any]:
    sanitized = sanitize_metadata(value)
    if not isinstance(sanitized, dict):
        return {}
    result = _bounded_value(
        sanitized,
        depth=0,
        max_items=max_items,
        string_limit=4000,
    )
    return result if isinstance(result, dict) else {}


def _safe_citation_url(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Citation URLs must be strings.")
    bounded = _safe_text(value, limit=4096)
    if not bounded:
        raise ValueError("Citation URLs may not be empty.")
    if any(ord(character) < 32 or ord(character) == 127 for character in bounded):
        raise ValueError("Citation URLs may not contain control characters.")
    try:
        parsed = urlsplit(bounded)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Citation URLs must be valid URLs.") from exc
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https", "local"}:
        raise ValueError("Citation URLs must use http, https, or local schemes.")
    if scheme == "local":
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Local citation URLs may not contain credentials.")
        if not parsed.netloc and not parsed.path.strip("/"):
            raise ValueError("Local citation URLs must identify a source.")
        return urlunsplit(
            (scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment)
        )[:4096]
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Public citation URLs must contain a hostname.")
    rendered_host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = rendered_host
    if port is not None:
        netloc = f"{rendered_host}:{port}"
    return urlunsplit(
        (scheme, netloc, parsed.path, parsed.query, parsed.fragment)
    )[:4096]


def _bounded_iterable(value: Any, label: str, maximum: int) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a list.")
    try:
        values = list(itertools.islice(iter(value), maximum + 1))
    except Exception as exc:
        raise ValueError(f"{label} must be iterable.") from exc
    if len(values) > maximum:
        raise ValueError(f"{label} may contain at most {maximum} items.")
    return values


class Citation(BaseModel):
    """A source selected from an actual tool result."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

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

    @field_validator("label", mode="before")
    @classmethod
    def validate_label(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("Citation labels must be strings.")
        bounded = value.strip()
        inner = (
            bounded[1:-1].strip()
            if bounded.startswith("[") and bounded.endswith("]")
            else ""
        )
        if (
            not inner
            or len(bounded) > 64
            or any(ord(character) < 32 or ord(character) == 127 for character in bounded)
            or "[" in inner
            or "]" in inner
        ):
            raise ValueError(
                "Citation labels must use non-empty bracket notation, for example '[1]'."
            )
        return bounded

    @field_validator("title", mode="before")
    @classmethod
    def validate_title(cls, value: Any) -> str:
        bounded = _safe_text(value, limit=500)
        if not bounded:
            raise ValueError("Citation titles may not be empty.")
        return bounded

    @field_validator("url", mode="before")
    @classmethod
    def validate_url(cls, value: Any) -> str:
        return _safe_citation_url(value)

    @field_validator("source_id", "doc_id", "chunk_id", mode="before")
    @classmethod
    def strip_identifier_fields(
        cls,
        value: Any,
        info: ValidationInfo,
    ) -> Optional[str]:
        if value is None:
            return None
        limits = {"source_id": 4096, "doc_id": 200, "chunk_id": 500}
        if not isinstance(value, str):
            raise ValueError(f"{info.field_name} must be a string.")
        bounded = _safe_text(value, limit=limits[info.field_name])
        if any(ord(character) < 32 or ord(character) == 127 for character in bounded):
            raise ValueError(f"{info.field_name} may not contain control characters.")
        return bounded or None

    @field_validator("snippet", "quote", mode="before")
    @classmethod
    def bound_evidence_text(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        bounded = _safe_text(value, limit=4000)
        return bounded or None

    @field_validator("page_number", mode="before")
    @classmethod
    def validate_page_number(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("page_number must be an integer.")
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def bound_metadata(cls, value: Any) -> Dict[str, Any]:
        return _bounded_metadata(value)


class AgentAnswer(BaseModel):
    """Structured agent response with server-controlled evidence."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    answer: str = Field(..., min_length=1, max_length=100_000)
    citations: List[Citation] = Field(default_factory=list, max_length=_MAX_CITATIONS)
    warnings: List[str] = Field(default_factory=list, max_length=_MAX_WARNINGS)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("answer", mode="before")
    @classmethod
    def validate_answer(cls, value: Any) -> str:
        bounded = _safe_text(value, limit=100_000)
        if not bounded:
            raise ValueError("Agent answers may not be empty.")
        return bounded

    @field_validator("citations", mode="before")
    @classmethod
    def bound_citation_iterable(cls, values: Any) -> list[Any]:
        return _bounded_iterable(values, "citations", _MAX_CITATIONS)

    @field_validator("citations")
    @classmethod
    def deduplicate_citations(cls, values: List[Citation]) -> List[Citation]:
        selected: List[Citation] = []
        labels: set[str] = set()
        identities: set[tuple[str, str, str]] = set()
        for citation in values:
            identity = (
                citation.source_id or citation.url,
                citation.doc_id or "",
                citation.quote or citation.snippet or "",
            )
            if citation.label in labels or identity in identities:
                continue
            labels.add(citation.label)
            identities.add(identity)
            selected.append(citation)
        return selected

    @field_validator("warnings", mode="before")
    @classmethod
    def bound_warnings(cls, values: Any) -> List[str]:
        raw_values = _bounded_iterable(values, "warnings", _MAX_WARNINGS)
        bounded: List[str] = []
        for value in raw_values:
            text = _safe_text(value, limit=2000)
            if text:
                bounded.append(text)
        return bounded

    @field_validator("metadata", mode="before")
    @classmethod
    def bound_answer_metadata(cls, value: Any) -> Dict[str, Any]:
        return _bounded_metadata(value)
