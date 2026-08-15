"""Deterministic fallback builder for ``ScientificDocumentIR`` from page text.

This is intentionally conservative: it provides reading-order/layout roles when learned
layout/table/formula/figure models are unavailable, while preserving a clear extractor
ID so heuristic structure is never confused with model-verified structure.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tools.document_ir import (
    DocumentBlock,
    DocumentPage,
    FormulaStructure,
    ScientificDocumentIR,
    TableCell,
    TableStructure,
)
from tools.document_linking import enrich_document_links
from tools.multimodal_evidence import NormalizedBBox

_MAX_LINES_PER_PAGE = 200_000
_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*\s+)?[A-Z][^.!?]{0,160}$")
_CAPTION_RE = re.compile(r"^(?:fig(?:ure)?|table|chart)\s*\.?\s*[A-Za-z]?\d+[A-Za-z]?\s*[:.\-]", re.I)
_FORMULA_RE = re.compile(r"(?:=|≤|≥|≈|∑|∫|√|\^|_[A-Za-z0-9])")
_LIST_RE = re.compile(r"^(?:[-*•]|\d+[.)]|[A-Za-z][.)])\s+")
_REFERENCE_RE = re.compile(r"^(?:\[?\d{1,5}\]?\.?\s+|[A-Z][A-Za-z-]+,\s+[A-Z])")
_TABLE_SPLIT_RE = re.compile(r"\s{2,}|\t+|\s*\|\s*")


def _text(value: Any, label: str, maximum: int = 100_000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = value.replace("\x00", " ")
    if (not cleaned.strip() and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


@dataclass(frozen=True)
class PageTextInput:
    page_number: int
    text: str
    width_points: float = 612.0
    height_points: float = 792.0
    rendered_sha256: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.page_number, bool) or not isinstance(self.page_number, int) or not 1 <= self.page_number <= 100_000:
            raise ValueError("page_number is invalid")
        object.__setattr__(self, "text", _text(self.text, "page text", 5_000_000, allow_empty=True))
        for name in ("width_points", "height_points"):
            value = float(getattr(self, name))
            if not 1.0 <= value <= 1_000_000.0:
                raise ValueError(f"{name} is invalid")
            object.__setattr__(self, name, value)
        if self.rendered_sha256:
            digest = self.rendered_sha256.lower().strip()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("rendered_sha256 is invalid")
            object.__setattr__(self, "rendered_sha256", digest)


def _role(line: str, *, references_started: bool) -> str:
    stripped = line.strip()
    if not stripped:
        return "unknown"
    if references_started and _REFERENCE_RE.search(stripped):
        return "reference"
    if _CAPTION_RE.search(stripped):
        return "caption"
    if _LIST_RE.search(stripped):
        return "list_item"
    lower = stripped.casefold()
    if lower in {"references", "bibliography", "works cited"}:
        return "heading"
    if len(stripped) <= 180 and _HEADING_RE.fullmatch(stripped) and stripped[-1:] not in {",", ";"}:
        return "heading"
    table_tokens = _TABLE_SPLIT_RE.split(stripped)
    if len(table_tokens) >= 3 and sum(bool(token.strip()) for token in table_tokens) >= 3:
        numeric = sum(any(ch.isdigit() for ch in token) for token in table_tokens)
        if numeric >= 1:
            return "table"
    if len(stripped) <= 500 and _FORMULA_RE.search(stripped):
        alpha = sum(ch.isalpha() for ch in stripped)
        operators = sum(ch in "=+-*/^_≤≥≈∑∫√()[]{}" for ch in stripped)
        if operators >= 2 and alpha <= max(80, len(stripped) // 2):
            return "formula"
    return "paragraph"


def _line_bbox(index: int, total: int, line: str) -> NormalizedBBox:
    total = max(1, total)
    y0 = min(0.999, index / total)
    y1 = min(1.0, (index + 0.92) / total)
    leading = len(line) - len(line.lstrip())
    x0 = min(0.7, leading / 120.0)
    content = len(line.strip())
    x1 = min(1.0, max(x0 + 0.05, x0 + content / 140.0))
    return NormalizedBBox(x0=x0, y0=y0, x1=x1, y1=y1)


def _table_from_line(line: str, bbox: NormalizedBBox) -> TableStructure | None:
    tokens = [token.strip() for token in _TABLE_SPLIT_RE.split(line.strip()) if token.strip()]
    if len(tokens) < 2 or len(tokens) > 256:
        return None
    width = max(1e-6, bbox.x1 - bbox.x0)
    cells: list[TableCell] = []
    for column, token in enumerate(tokens):
        left = bbox.x0 + width * column / len(tokens)
        right = bbox.x0 + width * (column + 1) / len(tokens)
        cells.append(TableCell(0, column, 1, 1, NormalizedBBox(left, bbox.y0, right, bbox.y1), token, False))
    return TableStructure(tuple(cells))


def build_document_ir_from_pages(
    *,
    owner_id: str,
    doc_id: str,
    source_sha256: str,
    pages: Sequence[PageTextInput],
    metadata: Mapping[str, str] | None = None,
) -> ScientificDocumentIR:
    del metadata  # document-level metadata remains in the authoritative generation store.
    if not pages or len(pages) > 100_000:
        raise ValueError("pages are empty or exceed the page limit")
    if [page.page_number for page in pages] != list(range(1, len(pages) + 1)):
        raise ValueError("page inputs must be contiguous and one-indexed")
    page_records = tuple(DocumentPage(page.page_number, page.width_points, page.height_points, page.rendered_sha256) for page in pages)
    blocks: list[DocumentBlock] = []
    references_started = False
    block_index = 0
    for page in pages:
        lines = page.text.splitlines()
        if len(lines) > _MAX_LINES_PER_PAGE:
            raise ValueError("page contains too many text lines")
        for line_index, raw in enumerate(lines):
            stripped = " ".join(raw.split())
            if not stripped:
                continue
            role = _role(raw, references_started=references_started)
            if role == "heading" and stripped.casefold() in {"references", "bibliography", "works cited"}:
                references_started = True
            bbox = _line_bbox(line_index, max(1, len(lines)), raw)
            content_sha = hashlib.sha256(stripped.encode("utf-8")).hexdigest()
            block_id = hashlib.sha256(f"{source_sha256}:{page.page_number}:{block_index}:{role}:{content_sha}".encode("utf-8")).hexdigest()
            table = _table_from_line(stripped, bbox) if role == "table" else None
            formula = FormulaStructure(normalized_text=stripped) if role == "formula" else None
            blocks.append(
                DocumentBlock(
                    block_id=block_id,
                    page_number=page.page_number,
                    role=role,
                    bbox=bbox,
                    text=stripped,
                    table=table,
                    formula=formula,
                    confidence=0.55 if role in {"table", "formula"} else 0.7,
                    content_sha256=content_sha,
                )
            )
            block_index += 1
    document = ScientificDocumentIR(
        owner_id=owner_id,
        doc_id=doc_id,
        source_sha256=source_sha256,
        pages=page_records,
        blocks=tuple(blocks),
        extractor_id="deterministic-page-text-v1",
        schema_version="1.0.0",
    )
    return enrich_document_links(document)


__all__ = ["PageTextInput", "build_document_ir_from_pages"]
