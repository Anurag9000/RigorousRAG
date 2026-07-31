"""Document-ingestion models with privacy-safe construction and serialization."""

from __future__ import annotations

import itertools
import operator
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from tools.privacy import mask_metadata_text, sanitize_metadata_dict

_MAX_DOCUMENT_TEXT_CHARS = 50_000_000
_MAX_SECTIONS = 10_000
_MAX_METADATA_ITEMS = 1000
_MAX_PAGE_NUMBER = 1_000_000
_MIME_RE = re.compile(r"^[A-Za-z0-9!#$&^_.+-]{1,127}/[A-Za-z0-9!#$&^_.+-]{1,127}$")


def _safe_text(value: Any, *, limit: int, default: str = "") -> str:
    if isinstance(value, str):
        rendered = value
    else:
        try:
            rendered = str(value)
        except Exception:
            rendered = default
    return rendered[:limit]


def _contains_ascii_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _contains_invalid_document_control(value: str) -> bool:
    return any(
        (ord(character) < 32 and character not in "\t\r\n")
        or ord(character) == 127
        for character in value
    )


def _required_text(value: Any, label: str, *, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    bounded = value.strip()
    if not bounded or len(bounded) > limit or _contains_ascii_control(bounded):
        raise ValueError(f"{label} must contain 1-{limit} valid characters.")
    return bounded


def _bounded_sections(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError("sections must be a list of document sections.")
    try:
        sections = list(itertools.islice(iter(value), _MAX_SECTIONS + 1))
    except Exception as exc:
        raise ValueError("sections must be iterable.") from exc
    if len(sections) > _MAX_SECTIONS:
        raise ValueError(f"sections may contain at most {_MAX_SECTIONS} items.")
    return sections


def _exact_page_number(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("page_number must be an integer.")
    try:
        parsed = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("page_number must be an integer.") from exc
    page = int(parsed)
    if not 1 <= page <= _MAX_PAGE_NUMBER:
        raise ValueError(
            f"page_number must be between 1 and {_MAX_PAGE_NUMBER}."
        )
    return page


class DocumentSection(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1, max_length=_MAX_DOCUMENT_TEXT_CHARS)
    page_number: Optional[int] = Field(default=None, ge=1, le=_MAX_PAGE_NUMBER)

    @field_validator("title", mode="before")
    @classmethod
    def mask_section_title(cls, value: Any) -> str:
        bounded = mask_metadata_text(
            _safe_text(value, limit=500, default="Untitled section")
        ).strip()
        return bounded or "Untitled section"

    @field_validator("content", mode="before")
    @classmethod
    def mask_and_validate_content(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("Section content must be a string.")
        if (
            not value
            or len(value) > _MAX_DOCUMENT_TEXT_CHARS
            or _contains_invalid_document_control(value)
        ):
            raise ValueError("Section content must contain valid non-empty text.")
        masked = mask_metadata_text(value)
        if not masked:
            raise ValueError("Section content must contain valid non-empty text.")
        return masked

    @field_validator("page_number", mode="before")
    @classmethod
    def validate_page_number(cls, value: Any) -> Optional[int]:
        return _exact_page_number(value)


class IngestedDocument(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Stable owner-and-content document identifier.",
    )
    filename: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Masked display filename, never a storage path.",
    )
    file_path: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        exclude=True,
        description="Internal source path; excluded from serialization.",
    )
    mime_type: str = Field(..., min_length=1, max_length=200)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    title: Optional[str] = Field(default=None, max_length=1000)
    text: str = Field(default="", max_length=_MAX_DOCUMENT_TEXT_CHARS)
    sections: List[DocumentSection] = Field(default_factory=list, max_length=_MAX_SECTIONS)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", mode="before")
    @classmethod
    def validate_document_id(cls, value: Any) -> str:
        return _required_text(value, "id", limit=200)

    @field_validator("file_path", mode="before")
    @classmethod
    def validate_private_path(cls, value: Any) -> str:
        return _required_text(value, "file_path", limit=4096)

    @field_validator("mime_type", mode="before")
    @classmethod
    def validate_mime_type(cls, value: Any) -> str:
        mime_type = _required_text(value, "mime_type", limit=200).lower()
        if not _MIME_RE.fullmatch(mime_type):
            raise ValueError("mime_type must be a valid type/subtype value.")
        return mime_type

    @field_validator("filename", mode="before")
    @classmethod
    def mask_filename(cls, value: Any) -> str:
        bounded = mask_metadata_text(
            _safe_text(value, limit=500, default="document")
        ).strip()
        return bounded or "document"

    @field_validator("title", mode="before")
    @classmethod
    def mask_title(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        return mask_metadata_text(_safe_text(value, limit=1000)).strip() or None

    @field_validator("text", mode="before")
    @classmethod
    def mask_and_validate_document_text(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("Document text must be a string.")
        if (
            len(value) > _MAX_DOCUMENT_TEXT_CHARS
            or _contains_invalid_document_control(value)
        ):
            raise ValueError("Document text exceeds the valid text boundary.")
        return mask_metadata_text(value)

    @field_validator("sections", mode="before")
    @classmethod
    def bound_section_iterable(cls, value: Any) -> List[Any]:
        return _bounded_sections(value)

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include timezone information.")
        return value.astimezone(timezone.utc)

    @field_validator("metadata", mode="before")
    @classmethod
    def mask_metadata_assignment(cls, value: Any) -> Dict[str, Any]:
        sanitized = sanitize_metadata_dict(value)
        bounded: Dict[str, Any] = {}
        for index, (key, item) in enumerate(sanitized.items()):
            if index >= _MAX_METADATA_ITEMS:
                bounded["__truncated_items__"] = True
                break
            bounded[str(key)[:500]] = item
        return bounded

    @model_validator(mode="after")
    def validate_aggregate_section_text(self) -> "IngestedDocument":
        total = 0
        for section in self.sections:
            total += len(section.content)
            if total > _MAX_DOCUMENT_TEXT_CHARS:
                raise ValueError(
                    "Aggregate semantic section text exceeds the document character limit."
                )
        return self

    @field_serializer("metadata")
    def mask_serialized_metadata(self, value: Dict[str, Any]) -> Dict[str, Any]:
        return self.mask_metadata_assignment(value)


class IngestionResult(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    success: bool
    document: Optional[IngestedDocument] = None
    error: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("success", mode="before")
    @classmethod
    def validate_success(cls, value: Any) -> bool:
        if not isinstance(value, bool):
            raise ValueError("success must be a boolean.")
        return value

    @field_validator("error", mode="before")
    @classmethod
    def mask_error(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        return mask_metadata_text(
            _safe_text(value, limit=2000, default="Document ingestion failed.")
        ).strip() or None

    @model_validator(mode="after")
    def validate_consistency(self) -> "IngestionResult":
        if self.success and self.document is None:
            raise ValueError("Successful ingestion requires a document.")
        if not self.success and self.document is not None:
            raise ValueError("Failed ingestion may not include a document.")
        if self.success and self.error is not None:
            raise ValueError("Successful ingestion may not include an error.")
        if not self.success and self.error is None:
            object.__setattr__(self, "error", "Document ingestion failed.")
        return self
