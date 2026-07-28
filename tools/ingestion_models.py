"""Document-ingestion models with privacy-safe serialization defaults."""

from __future__ import annotations

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


class DocumentSection(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1, max_length=_MAX_DOCUMENT_TEXT_CHARS)
    page_number: Optional[int] = Field(default=None, ge=1, le=1_000_000)

    @field_validator("title")
    @classmethod
    def mask_section_title(cls, value: str) -> str:
        return mask_metadata_text(str(value or "")).strip()[:500] or "Untitled section"


class IngestedDocument(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

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

    @field_validator("id", "file_path", "mime_type")
    @classmethod
    def strip_required_identifiers(cls, value: str) -> str:
        bounded = str(value or "").strip()
        if not bounded:
            raise ValueError("Required ingestion identifiers may not be empty.")
        return bounded

    @field_validator("filename")
    @classmethod
    def mask_filename(cls, value: str) -> str:
        return mask_metadata_text(str(value or "")).strip()[:500] or "document"

    @field_validator("title")
    @classmethod
    def mask_title(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return mask_metadata_text(str(value)).strip()[:1000] or None

    @field_validator("metadata")
    @classmethod
    def mask_metadata_assignment(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = sanitize_metadata_dict(value if isinstance(value, dict) else {})
        bounded: Dict[str, Any] = {}
        for index, (key, item) in enumerate(sanitized.items()):
            if index >= _MAX_METADATA_ITEMS:
                bounded["__truncated_items__"] = True
                break
            bounded[str(key)[:500]] = item
        return bounded

    @field_serializer("metadata")
    def mask_serialized_metadata(self, value: Dict[str, Any]) -> Dict[str, Any]:
        return self.mask_metadata_assignment(value)


class IngestionResult(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    success: bool
    document: Optional[IngestedDocument] = None
    error: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("error")
    @classmethod
    def mask_error(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return mask_metadata_text(str(value)).strip()[:2000] or None

    @model_validator(mode="after")
    def validate_consistency(self) -> "IngestionResult":
        if self.success and self.document is None:
            raise ValueError("Successful ingestion requires a document.")
        if not self.success and self.document is not None:
            raise ValueError("Failed ingestion may not include a document.")
        if self.success and self.error is not None:
            raise ValueError("Successful ingestion may not include an error.")
        if not self.success and self.error is None:
            self.error = "Document ingestion failed."
        return self
