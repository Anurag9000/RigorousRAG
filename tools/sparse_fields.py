"""Deterministic conversion of privacy-finalized documents into sparse fields."""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Mapping
from typing import Any

from tools.sparse_index import SparseField

_MAX_SECTIONS = 10_000
_MAX_FIELDS = 20_000
_MAX_FIELD_CHARS = 5_000_000
_MAX_TOTAL_CHARS = 50_000_000
_MAX_METADATA_ITEMS = 64


def _text(value: Any, label: str, *, maximum: int, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip()
    if len(cleaned) > maximum:
        raise ValueError(f"{label} exceeds the character limit.")
    if any(ord(ch) < 32 and ch not in "\t\r\n" for ch in cleaned) or "\x7f" in cleaned:
        raise ValueError(f"{label} contains invalid control characters.")
    if not allow_empty and not cleaned:
        raise ValueError(f"{label} is required.")
    return cleaned


def _page_number(value: Any) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= 1_000_000
    ):
        raise ValueError("page_number must be a positive integer or null.")
    return value


def _safe_metadata(value: Any) -> dict[str, str | int | float | bool | None]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("section metadata must be a mapping.")
    result: dict[str, str | int | float | bool | None] = {}
    try:
        for index, (raw_key, raw_value) in enumerate(value.items()):
            if index >= _MAX_METADATA_ITEMS:
                raise ValueError("section metadata contains too many fields.")
            key = _text(raw_key, "metadata key", maximum=200)
            if raw_value is None or isinstance(raw_value, (bool, int)):
                result[key] = raw_value
            elif isinstance(raw_value, float):
                if raw_value != raw_value or raw_value in (
                    float("inf"),
                    float("-inf"),
                ):
                    raise ValueError("section metadata contains non-finite numbers.")
                result[key] = raw_value
            elif isinstance(raw_value, str):
                result[key] = _text(
                    raw_value,
                    "metadata value",
                    maximum=4000,
                    allow_empty=True,
                )
            else:
                raise ValueError("section metadata contains an unsupported value.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("section metadata is not safely iterable.") from exc
    return result


def _field_type(title: str, metadata: Mapping[str, Any]) -> str:
    explicit = metadata.get("field_type") or metadata.get("content_type")
    if isinstance(explicit, str):
        normalized = explicit.strip().lower().replace("-", "_")
        aliases = {
            "figure": "caption",
            "figure_caption": "caption",
            "table_caption": "caption",
            "bibliography": "reference",
            "references": "reference",
            "methods": "body",
            "method": "body",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in {"abstract", "body", "caption", "table", "reference"}:
            return normalized
    lowered = title.casefold()
    if "abstract" in lowered or "summary" in lowered:
        return "abstract"
    if any(token in lowered for token in ("figure", "caption", "fig.")):
        return "caption"
    if "table" in lowered:
        return "table"
    if any(token in lowered for token in ("reference", "bibliograph", "citation")):
        return "reference"
    return "body"


def _field_id(
    doc_id: str,
    kind: str,
    position: int,
    page: int | None,
    section: str | None,
    text: str,
) -> str:
    digest = hashlib.sha256(
        (
            f"{doc_id}\x00{kind}\x00{position}\x00{page or 0}\x00"
            f"{section or ''}\x00{text}"
        ).encode("utf-8")
    ).hexdigest()[:24]
    return f"{kind}-{position}-{digest}"


def _section_mapping(
    section: Any,
    index: int,
) -> tuple[str, str, int | None, dict[str, Any]]:
    if hasattr(section, "model_dump") and callable(section.model_dump):
        try:
            raw = section.model_dump()
        except Exception as exc:
            raise ValueError("section could not be serialized.") from exc
    elif isinstance(section, Mapping):
        try:
            raw = dict(section)
        except Exception as exc:
            raise ValueError("section could not be copied.") from exc
    else:
        try:
            raw = {
                "title": getattr(section, "title"),
                "content": getattr(section, "content"),
                "page_number": getattr(section, "page_number", None),
                "metadata": getattr(section, "metadata", {}),
            }
        except Exception as exc:
            raise ValueError("section could not be inspected.") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("section must be object-like.")
    title = _text(
        raw.get("title") or f"Section {index + 1}",
        "section title",
        maximum=500,
    )
    content = _text(
        raw.get("content"),
        "section content",
        maximum=_MAX_FIELD_CHARS,
    )
    page = _page_number(raw.get("page_number"))
    metadata = _safe_metadata(raw.get("metadata"))
    return title, content, page, metadata


def build_sparse_fields(
    document: Any,
    *,
    doc_id: str | None = None,
) -> tuple[SparseField, ...]:
    """Build stable title/heading/content fields from a finalized document."""

    identifier = _text(
        doc_id if doc_id is not None else getattr(document, "id", None),
        "doc_id",
        maximum=200,
    )
    title_value = (
        getattr(document, "title", None)
        or getattr(document, "filename", None)
        or identifier
    )
    title = _text(title_value, "document title", maximum=1000)
    raw_sections = getattr(document, "sections", None)
    if raw_sections is None or isinstance(raw_sections, (str, bytes, bytearray)):
        raise ValueError("document sections must be an iterable.")
    try:
        sections = list(itertools.islice(iter(raw_sections), _MAX_SECTIONS + 1))
    except Exception as exc:
        raise ValueError("document sections are not safely iterable.") from exc
    if len(sections) > _MAX_SECTIONS:
        raise ValueError("document has too many sections.")

    fields: list[SparseField] = []
    total_chars = 0

    def append(
        kind: str,
        text: str,
        *,
        page: int | None = None,
        section: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        nonlocal total_chars
        if len(fields) >= _MAX_FIELDS:
            raise ValueError("document produces too many sparse fields.")
        bounded = _text(text, f"{kind} field", maximum=_MAX_FIELD_CHARS)
        total_chars += len(bounded)
        if total_chars > _MAX_TOTAL_CHARS:
            raise ValueError("document sparse fields exceed the total character limit.")
        position = len(fields)
        fields.append(
            SparseField(
                _field_id(identifier, kind, position, page, section, bounded),
                kind,
                bounded,
                position,
                page_number=page,
                section=section,
                metadata=_safe_metadata(metadata),
            )
        )

    append("title", title, metadata={"source": "document_title"})
    for index, raw_section in enumerate(sections):
        section_title, content, page, metadata = _section_mapping(raw_section, index)
        append(
            "heading",
            section_title,
            page=page,
            section=section_title,
            metadata={"section_index": index},
        )
        append(
            _field_type(section_title, metadata),
            content,
            page=page,
            section=section_title,
            metadata={"section_index": index, **metadata},
        )

    if len(fields) == 1:
        body = _text(
            getattr(document, "text", None),
            "document text",
            maximum=_MAX_FIELD_CHARS,
        )
        append("body", body, section="Full Text", metadata={"fallback": True})
    return tuple(fields)


__all__ = ["build_sparse_fields"]
