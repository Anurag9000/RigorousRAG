from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

class DocumentSection(BaseModel):
    title: str = Field(..., description="Section title or header.")
    content: str = Field(..., description="Text content of the section.")
    page_number: Optional[int] = Field(None, description="Page number where this section starts.")

class IngestedDocument(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the document.")
    filename: str = Field(..., description="Original filename.")
    file_path: str = Field(..., description="Absolute path to the source file.")
    mime_type: str = Field(..., description="MIME type of the file.")
    created_at: datetime = Field(default_factory=datetime.now, description="Ingestion timestamp.")
    
    # Content
    title: Optional[str] = Field(None, description="Extracted title of the document.")
    text: str = Field(..., description="Full raw text content.")
    sections: List[DocumentSection] = Field(default_factory=list, description="Structured sections if detected.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata (e.g., author, date).")

class IngestionResult(BaseModel):
    success: bool
    document: Optional[IngestedDocument] = None
    error: Optional[str] = None
