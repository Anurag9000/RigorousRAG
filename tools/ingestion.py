import os
import mimetypes
from pathlib import Path
from typing import Optional, List, Dict, Any
import fitz  # PyMuPDF
import docx
from tools.ingestion_models import IngestedDocument, DocumentSection, IngestionResult

def detect_mime_type(file_path: str) -> str:
    mime, _ = mimetypes.guess_type(file_path)
    return mime or "application/octet-stream"

import re

def redact_text(text: str) -> str:
    """
    Basic redaction of sensitive fields (emails, phones) for Goal 16.
    """
    # Simple regex for emails
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[REDACTED_EMAIL]', text)
    # Simple regex for phones
    text = re.sub(r'\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}', '[REDACTED_PHONE]', text)
    return text

def extract_academic_metadata(text: str) -> Dict[str, Any]:
    """
    Heuristic extraction of academic metadata from the first few lines of a paper.
    """
    metadata = {}
    head = text[:2000]
    
    # DOI Extraction
    doi_match = re.search(r'10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+', head)
    if doi_match:
        metadata["doi"] = doi_match.group(0)
    
    # Year Extraction
    year_match = re.search(r'\b(19|20)\d{2}\b', head)
    if year_match:
        metadata["year"] = year_match.group(0)
        
    # Title (First non-empty line usually)
    lines = [L.strip() for L in head.split('\n') if L.strip()]
    if lines:
        metadata["extracted_title"] = lines[0][:200]
        
    return metadata

def _chunk_text_semantically(text: str, max_chars: int = 1500) -> List[str]:
    """
    Splits text by double newlines (paragraphs/sections) or single newlines if too large.
    """
    # 1. Split by double newline (paragraphs)
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        if len(current_chunk) + len(p) < max_chars:
            current_chunk += p + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If a single paragraph is larger than max_chars, split it by sentences roughly
            if len(p) > max_chars:
                sentences = re.split(r'(?<=[.!?])\s+', p)
                sub_chunk = ""
                for s in sentences:
                    if len(sub_chunk) + len(s) < max_chars:
                        sub_chunk += s + " "
                    else:
                        chunks.append(sub_chunk.strip())
                        sub_chunk = s + " "
                current_chunk = sub_chunk
            else:
                current_chunk = p + "\n\n"
                
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def ingest_file(file_path: str, owner_id: str = "default_user") -> IngestionResult:
    path = Path(file_path)
    if not path.exists():
        return IngestionResult(success=False, error=f"File not found: {file_path}")

    mime_type = detect_mime_type(file_path)
    
    try:
        # Load and parse
        result = None
        if mime_type == "application/pdf":
            result = _ingest_pdf(path)
        elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            result = _ingest_docx(path)
        elif mime_type.startswith("text/") or path.suffix in [".md", ".txt", ".py"]:
            result = _ingest_text(path, mime_type)
        else:
            return IngestionResult(success=False, error=f"Unsupported file type: {mime_type}")
        
        # Apply Redaction (Goal 16.2)
        if result and result.document:
            result.document.text = redact_text(result.document.text)
            
            # Extract Academic Metadata (New)
            academic_meta = extract_academic_metadata(result.document.text)
            result.document.metadata.update(academic_meta)
            
            # Re-generate sections semantically (New)
            semantic_chunks = _chunk_text_semantically(result.document.text)
            result.document.sections = [
                DocumentSection(title=f"Section {i+1}", content=chunk)
                for i, chunk in enumerate(semantic_chunks)
            ]
            
            # Set ownership (Goal 16.1)
            result.document.metadata["owner_id"] = owner_id
            
        return result
            
    except Exception as e:
        return IngestionResult(success=False, error=str(e))

def _ingest_pdf(path: Path) -> IngestionResult:
    doc = fitz.open(path)
    full_text = []
    sections = []
    
    title = path.stem
    if doc.metadata and doc.metadata.get("title"):
        title = doc.metadata["title"]

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if not text.strip():
            continue
            
        full_text.append(text)
        
        # very basic section detection (placeholder)
        # In a real impl, we'd look for font sizes or bold text
        sections.append(DocumentSection(
            title=f"Page {page_num}",
            content=text,
            page_number=page_num
        ))

    final_text = "\n\n".join(full_text)
    
    return IngestionResult(
        success=True,
        document=IngestedDocument(
            filename=path.name,
            file_path=str(path.absolute()),
            mime_type="application/pdf",
            title=title,
            text=final_text,
            sections=sections,
            metadata=doc.metadata or {}
        )
    )

def _ingest_docx(path: Path) -> IngestionResult:
    doc = docx.Document(path)
    full_text = []
    sections = []
    
    current_section_title = "Start"
    current_section_content = []
    
    properties = doc.core_properties
    title = properties.title or path.stem

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
            
        # Basic heuristic for headings
        if para.style.name.startswith('Heading'):
            # Save previous section
            if current_section_content:
                sections.append(DocumentSection(
                    title=current_section_title,
                    content="\n".join(current_section_content)
                ))
            current_section_title = text
            current_section_content = []
        else:
            current_section_content.append(text)
        
        full_text.append(text)

    # Add last section
    if current_section_content:
        sections.append(DocumentSection(
            title=current_section_title,
            content="\n".join(current_section_content)
        ))

    return IngestionResult(
        success=True,
        document=IngestedDocument(
            filename=path.name,
            file_path=str(path.absolute()),
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            title=title,
            text="\n".join(full_text),
            sections=sections,
            metadata={
                "author": properties.author,
                "created": str(properties.created),
                "modified": str(properties.modified)
            }
        )
    )

def _ingest_text(path: Path, mime_type: str) -> IngestionResult:
    try:
        text = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        # Fallback to latin-1 if utf-8 fails
        text = path.read_text(encoding='latin-1')
        
    return IngestionResult(
        success=True,
        document=IngestedDocument(
            filename=path.name,
            file_path=str(path.absolute()),
            mime_type=mime_type,
            title=path.stem,
            text=text,
            sections=[DocumentSection(title="Full Text", content=text, page_number=1)],
            metadata={}
        )
    )
