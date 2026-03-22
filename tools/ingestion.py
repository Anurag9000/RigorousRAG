"""
Document ingestion pipeline: PDF, DOCX, and plain-text parsing with
font-based section detection, semantic chunking, and comprehensive PII redaction.
"""

import mimetypes
import os
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

import docx
import fitz  # PyMuPDF

from tools.ingestion_models import DocumentSection, IngestedDocument, IngestionResult


# ---------------------------------------------------------------------------
# MIME detection
# ---------------------------------------------------------------------------


def detect_mime_type(file_path: str) -> str:
    """Guess MIME type from file extension; defaults to octet-stream."""
    mime, _ = mimetypes.guess_type(file_path)
    return mime or "application/octet-stream"


# ---------------------------------------------------------------------------
# PII redaction  (Goal 16.2)
# ---------------------------------------------------------------------------


def redact_text(text: str) -> str:
    """
    Redacts a comprehensive set of PII patterns from document text.

    Patterns covered:
      - Email addresses
      - US phone numbers (with/without country code, various separators)
      - International phone numbers (E.164 prefix format)
      - US mailing addresses (number + street name + street type)
      - Social Security Numbers (US)
      - UK National Insurance numbers
      - UK postcodes
      - Credit card numbers (four groups of four digits)
    """
    # --- Emails ---
    text = re.sub(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        "[REDACTED_EMAIL]",
        text,
    )

    # --- US phone: (NXX) NXX-XXXX  or  NXX-NXX-XXXX  or  NXX.NXX.XXXX ---
    text = re.sub(
        r"(?<!\d)"                          # not preceded by digit
        r"(?:\+?1[\s.\-]?)?"               # optional +1 country code
        r"\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}"
        r"(?!\d)",                          # not followed by digit
        "[REDACTED_PHONE]",
        text,
    )

    # --- International phone: +CC followed by digit groups ---
    text = re.sub(
        r"\+\d{1,3}(?:[\s\-]\d+){2,}",
        "[REDACTED_PHONE]",
        text,
    )

    # --- US mailing addresses: 1–5 digits + 1–3 capitalised words + street type ---
    text = re.sub(
        r"\b\d{1,5}\s+"
        r"(?:[A-Z][a-z]+\s+){1,3}"
        r"(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd|"
        r"Lane|Ln|Court|Ct|Way|Place|Pl|Circle|Cir|Highway|Hwy)\.?\b",
        "[REDACTED_ADDRESS]",
        text,
    )

    # --- US Social Security Numbers: NNN-NN-NNNN ---
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]", text)

    # --- UK National Insurance: XX NN NN NN X ---
    text = re.sub(
        r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b",
        "[REDACTED_NI]",
        text,
    )

    # --- UK postcodes: AN NAA or ANN NAA or AAN NAA or AANN NAA ---
    text = re.sub(
        r"\b(?:[A-Z]{1,2}\d{1,2}[A-Z]?)\s\d[A-Z]{2}\b",
        "[REDACTED_POSTCODE]",
        text,
    )

    # --- Credit card numbers: four groups of four digits ---
    text = re.sub(
        r"\b(?:\d{4}[\s\-]){3}\d{4}\b",
        "[REDACTED_CC]",
        text,
    )

    return text


# ---------------------------------------------------------------------------
# Academic metadata extraction
# ---------------------------------------------------------------------------


def extract_academic_metadata(text: str) -> Dict[str, Any]:
    """
    Heuristically extracts DOI, publication year, and title from the first
    2 000 characters of a document.
    """
    metadata: Dict[str, Any] = {}
    head = text[:2000]

    doi_m = re.search(r"10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+", head)
    if doi_m:
        metadata["doi"] = doi_m.group(0)

    year_m = re.search(r"\b(19|20)\d{2}\b", head)
    if year_m:
        metadata["year"] = year_m.group(0)

    lines = [line.strip() for line in head.split("\n") if line.strip()]
    if lines:
        metadata["extracted_title"] = lines[0][:200]

    return metadata


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------


def _chunk_text_semantically(text: str, max_chars: int = 1500) -> List[str]:
    """
    Splits text at double-newline paragraph boundaries; sub-splits at sentence
    boundaries when a single paragraph exceeds max_chars.
    """
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) < max_chars:
            current += para + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            if len(para) > max_chars:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sub = ""
                for s in sentences:
                    if len(sub) + len(s) < max_chars:
                        sub += s + " "
                    else:
                        chunks.append(sub.strip())
                        sub = s + " "
                current = sub
            else:
                current = para + "\n\n"

    if current:
        chunks.append(current.strip())
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# PDF section detection helpers
# ---------------------------------------------------------------------------


def _extract_sections_from_pdf(doc: "fitz.Document") -> List[DocumentSection]:
    """
    Two-pass font-size and bold-flag based section detection for PDFs.

    Pass 1 — Statistics:
        Collect all font sizes across all text spans to determine the body
        font size (median) and heading threshold (≥112 % of body size, or
        bold flag set at ≥90 % of body size).

    Pass 2 — Classification:
        Iterate blocks; spans meeting the heading criteria start a new section.
        Content between headings is accumulated into the current section.

    Falls back to one section per page if no headings are detected.
    """
    # --- Pass 1: font size statistics ---
    all_sizes: List[float] = []
    page_blocks_cache: List[tuple] = []

    for page_num, page in enumerate(doc, start=1):
        try:
            blocks = page.get_text("dict", flags=~fitz.TEXT_PRESERVE_IMAGES)["blocks"]
        except Exception:
            blocks = []
        page_blocks_cache.append((page_num, blocks))
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sz = span.get("size", 0.0)
                    if sz > 0:
                        all_sizes.append(sz)

    if not all_sizes:
        return []

    body_size = statistics.median(all_sizes)
    heading_threshold = body_size * 1.12

    # --- Pass 2: classify blocks ---
    sections: List[DocumentSection] = []
    current_title: str = "Preamble"
    current_content: List[str] = []
    current_page: int = 1

    def _flush():
        content = " ".join(current_content).strip()
        if content:
            sections.append(
                DocumentSection(
                    title=current_title,
                    content=content,
                    page_number=current_page,
                )
            )

    for page_num, blocks in page_blocks_cache:
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                line_parts: List[str] = []
                is_heading = False
                for span in line.get("spans", []):
                    txt = span.get("text", "").strip()
                    if not txt:
                        continue
                    sz = span.get("size", 0.0)
                    bold = bool(span.get("flags", 0) & 1)
                    if sz >= heading_threshold or (bold and sz >= body_size * 0.9):
                        is_heading = True
                    line_parts.append(txt)

                line_text = " ".join(line_parts).strip()
                if not line_text:
                    continue

                if is_heading and len(line_text) < 150:
                    _flush()
                    current_title = line_text
                    current_content = []
                    current_page = page_num
                else:
                    current_content.append(line_text)

    _flush()

    # Fallback: page-level sections when no headings were detected
    if not sections:
        for page_num, blocks in page_blocks_cache:
            page_text = " ".join(
                span.get("text", "")
                for block in blocks
                if block.get("type") == 0
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            ).strip()
            if page_text:
                sections.append(
                    DocumentSection(
                        title=f"Page {page_num}",
                        content=page_text,
                        page_number=page_num,
                    )
                )

    return sections


# ---------------------------------------------------------------------------
# Ingestion entry point
# ---------------------------------------------------------------------------


def ingest_file(file_path: str, owner_id: str = "default_user") -> IngestionResult:
    """
    Parse a document file and return a fully populated IngestionResult.

    Pipeline:
      1. Detect MIME type and dispatch to the appropriate parser.
      2. Apply PII redaction (Goal 16.2).
      3. Extract academic metadata (DOI, year, title).
      4. Re-generate sections via semantic chunking (overrides parser sections
         for uniformity, except for PDFs which keep their heading-based sections).
      5. Stamp owner_id (Goal 16.1).
    """
    path = Path(file_path)
    if not path.exists():
        return IngestionResult(success=False, error=f"File not found: {file_path}")

    mime_type = detect_mime_type(file_path)

    try:
        if mime_type == "application/pdf":
            result = _ingest_pdf(path)
        elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            result = _ingest_docx(path)
        elif mime_type and (mime_type.startswith("text/") or path.suffix in {".md", ".txt", ".py", ".rst"}):
            result = _ingest_text(path, mime_type)
        else:
            return IngestionResult(
                success=False, error=f"Unsupported file type: {mime_type}"
            )

        if result and result.document:
            doc = result.document

            # 2. PII redaction
            doc.text = redact_text(doc.text)

            # 3. Academic metadata
            doc.metadata.update(extract_academic_metadata(doc.text))

            # 4. For PDFs we already have heading-based sections; for others
            #    overwrite with semantic chunks for consistency.
            if mime_type != "application/pdf" or not doc.sections:
                semantic_chunks = _chunk_text_semantically(doc.text)
                doc.sections = [
                    DocumentSection(
                        title=f"Section {i + 1}", content=chunk, page_number=1
                    )
                    for i, chunk in enumerate(semantic_chunks)
                ]

            # 5. Ownership
            doc.metadata["owner_id"] = owner_id
            doc.metadata["file_path"] = str(path.absolute())

        return result

    except Exception as exc:
        return IngestionResult(success=False, error=str(exc))


# ---------------------------------------------------------------------------
# Format-specific parsers
# ---------------------------------------------------------------------------


def _ingest_pdf(path: Path) -> IngestionResult:
    """
    Parse a PDF using PyMuPDF.

    - Extracts full text from all pages.
    - Detects sections using font-size and bold-flag analysis (two-pass).
    """
    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        return IngestionResult(success=False, error=f"Cannot open PDF: {exc}")

    title = path.stem
    metadata: Dict[str, Any] = {}
    if doc.metadata:
        metadata = dict(doc.metadata)
        if metadata.get("title"):
            title = metadata["title"]

    full_text_parts: List[str] = []
    for page in doc:
        page_text = page.get_text()
        if page_text.strip():
            full_text_parts.append(page_text)

    final_text = "\n\n".join(full_text_parts)
    sections = _extract_sections_from_pdf(doc)
    doc.close()

    return IngestionResult(
        success=True,
        document=IngestedDocument(
            filename=path.name,
            file_path=str(path.absolute()),
            mime_type="application/pdf",
            title=title,
            text=final_text,
            sections=sections,
            metadata=metadata,
        ),
    )


def _ingest_docx(path: Path) -> IngestionResult:
    """
    Parse a Word document using python-docx.

    Heading styles trigger new DocumentSection boundaries.
    """
    try:
        doc = docx.Document(str(path))
    except Exception as exc:
        return IngestionResult(success=False, error=f"Cannot open DOCX: {exc}")

    props = doc.core_properties
    title = props.title or path.stem

    full_text: List[str] = []
    sections: List[DocumentSection] = []
    current_title = "Introduction"
    current_content: List[str] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = para.style.name if para.style else ""
        if style.startswith("Heading"):
            if current_content:
                sections.append(
                    DocumentSection(
                        title=current_title,
                        content="\n".join(current_content),
                        page_number=1,
                    )
                )
            current_title = text
            current_content = []
        else:
            current_content.append(text)
        full_text.append(text)

    if current_content:
        sections.append(
            DocumentSection(
                title=current_title,
                content="\n".join(current_content),
                page_number=1,
            )
        )

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
                "author": props.author or "",
                "created": str(props.created or ""),
                "modified": str(props.modified or ""),
            },
        ),
    )


def _ingest_text(path: Path, mime_type: str) -> IngestionResult:
    """Parse a plain-text or Markdown file; UTF-8 with Latin-1 fallback."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")

    return IngestionResult(
        success=True,
        document=IngestedDocument(
            filename=path.name,
            file_path=str(path.absolute()),
            mime_type=mime_type,
            title=path.stem,
            text=text,
            sections=[DocumentSection(title="Full Text", content=text, page_number=1)],
            metadata={},
        ),
    )
