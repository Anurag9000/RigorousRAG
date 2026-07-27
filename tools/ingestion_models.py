"""Document-ingestion models with privacy-safe serialization defaults."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from tools.privacy import mask_metadata_text, sanitize_metadata_dict


class DocumentSection(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    page_number: Optional[int] = Field(default=None, ge=1)

    @field_validator("title")
    @classmethod
    def mask_section_title(cls, value: str) -> str:
        return mask_metadata_text(value).strip()[:500] or "Untitled section"


class IngestedDocument(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(..., description="Stable owner-and-content document identifier.")
    filename: str = Field(..., description="Masked display filename, never a storage path.")
    file_path: str = Field(..., exclude=True, description="Internal source path; excluded from serialization.")
    mime_type: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    title: Optional[str] = None
    text: str
    sections: List[DocumentSection] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("filename")
    @classmethod
    def mask_filename(cls, value: str) -> str:
        return mask_metadata_text(value).strip()[:500] or "document"

    @field_validator("title")
    @classmethod
    def mask_title(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return mask_metadata_text(value).strip()[:1000] or None

    @field_validator("metadata")
    @classmethod
    def mask_metadata_assignment(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        return sanitize_metadata_dict(value)

    @field_serializer("metadata")
    def mask_serialized_metadata(self, value: Dict[str, Any]) -> Dict[str, Any]:
        return sanitize_metadata_dict(value)


class IngestionResult(BaseModel):
    success: bool
    document: Optional[IngestedDocument] = None
    error: Optional[str] = None
