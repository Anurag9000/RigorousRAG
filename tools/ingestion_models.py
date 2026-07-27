"""Document-ingestion models with privacy-safe serialization defaults."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentSection(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    page_number: Optional[int] = Field(default=None, ge=1)


class IngestedDocument(BaseModel):
    id: str = Field(..., description="Stable owner-and-content document identifier.")
    filename: str = Field(..., description="Original display filename, never a storage path.")
    file_path: str = Field(..., exclude=True, description="Internal source path; excluded from serialization.")
    mime_type: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    title: Optional[str] = None
    text: str
    sections: List[DocumentSection] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestionResult(BaseModel):
    success: bool
    document: Optional[IngestedDocument] = None
    error: Optional[str] = None
