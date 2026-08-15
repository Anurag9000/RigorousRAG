"""Deterministic cross-modal link construction for ``ScientificDocumentIR``.

The linker derives reading order, caption associations and explicit prose references to
figures/tables/equations/references.  Derived links are structural hypotheses with
bounded confidence; original source text and coordinates remain unchanged.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable

from tools.document_ir import DocumentBlock, DocumentLink, ScientificDocumentIR

_FIGURE_REF_RE = re.compile(r"\b(?:fig(?:ure)?s?\.?)[\s\u00a0]*([A-Za-z]?\d+[A-Za-z]?)\b", re.I)
_TABLE_REF_RE = re.compile(r"\b(?:table?s?\.?)[\s\u00a0]*([A-Za-z]?\d+[A-Za-z]?)\b", re.I)
_EQUATION_REF_RE = re.compile(r"\b(?:eq(?:uation)?s?\.?)[\s\u00a0]*(?:\()?([A-Za-z]?\d+[A-Za-z]?)(?:\))?", re.I)
_REFERENCE_REF_RE = re.compile(r"(?:\[(\d{1,5})\]|\bref(?:erence)?\.?\s+(\d{1,5})\b)", re.I)
_LABEL_RE = re.compile(r"^\s*(?:fig(?:ure)?|table|eq(?:uation)?|reference|ref)\.?\s*([A-Za-z]?\d+[A-Za-z]?)\b", re.I)


def _sort_blocks(blocks: Iterable[DocumentBlock]) -> list[DocumentBlock]:
    return sorted(
        blocks,
        key=lambda block: (
            block.page_number,
            block.bbox.y0,
            block.bbox.x0,
            block.bbox.y1,
            block.bbox.x1,
            block.block_id,
        ),
    )


def _label(block: DocumentBlock) -> str:
    sources = [block.text]
    if block.figure is not None:
        sources.append(block.figure.caption)
    if block.table is not None:
        sources.append(block.table.caption)
    if block.formula is not None:
        sources.extend((block.formula.equation_number, block.formula.normalized_text))
    for source in sources:
        if not source:
            continue
        match = _LABEL_RE.search(source)
        if match:
            return match.group(1).casefold()
    return ""


def _dedupe(links: Iterable[DocumentLink]) -> tuple[DocumentLink, ...]:
    output: list[DocumentLink] = []
    seen: set[tuple[str, str, str]] = set()
    for link in links:
        key = (link.source_block_id, link.target_block_id, link.kind)
        if key in seen:
            continue
        seen.add(key)
        output.append(link)
    return tuple(output)


def derive_reading_order(document: ScientificDocumentIR) -> tuple[DocumentLink, ...]:
    links: list[DocumentLink] = []
    for page in document.pages:
        blocks = _sort_blocks(document.blocks_on_page(page.page_number))
        # Headers/footers should not interrupt the scientific-body reading chain.
        body = [block for block in blocks if block.role not in {"header", "footer", "metadata"}]
        for left, right in zip(body, body[1:]):
            links.append(DocumentLink(left.block_id, right.block_id, "reading_next", 0.85))
    return tuple(links)


def derive_caption_links(document: ScientificDocumentIR) -> tuple[DocumentLink, ...]:
    links: list[DocumentLink] = []
    by_page = {page.page_number: list(document.blocks_on_page(page.page_number)) for page in document.pages}
    for page_number, blocks in by_page.items():
        captions = [block for block in blocks if block.role == "caption"]
        targets = [block for block in blocks if block.role in {"figure", "chart", "table"}]
        for caption in captions:
            caption_label = _label(caption)
            candidates: list[tuple[float, DocumentBlock]] = []
            for target in targets:
                target_label = _label(target)
                if caption_label and target_label and caption_label == target_label:
                    distance = 0.0
                else:
                    vertical_gap = min(abs(caption.bbox.y0 - target.bbox.y1), abs(target.bbox.y0 - caption.bbox.y1))
                    horizontal_gap = abs((caption.bbox.x0 + caption.bbox.x1) / 2 - (target.bbox.x0 + target.bbox.x1) / 2)
                    distance = vertical_gap + 0.25 * horizontal_gap
                candidates.append((distance, target))
            if not candidates:
                continue
            candidates.sort(key=lambda item: (item[0], item[1].block_id))
            distance, target = candidates[0]
            if caption_label and _label(target) == caption_label:
                confidence = 0.98
            elif distance <= 0.08:
                confidence = 0.85
            elif distance <= 0.18:
                confidence = 0.65
            else:
                continue
            links.append(DocumentLink(caption.block_id, target.block_id, "caption_of", confidence))
    return tuple(links)


def derive_reference_links(document: ScientificDocumentIR) -> tuple[DocumentLink, ...]:
    figures: dict[str, DocumentBlock] = {}
    tables: dict[str, DocumentBlock] = {}
    equations: dict[str, DocumentBlock] = {}
    references: dict[str, DocumentBlock] = {}
    for block in document.blocks:
        label = _label(block)
        if not label:
            continue
        if block.role in {"figure", "chart", "caption"} and label not in figures:
            figures[label] = block
        elif block.role == "table" and label not in tables:
            tables[label] = block
        elif block.role == "formula" and label not in equations:
            equations[label] = block
        elif block.role == "reference" and label not in references:
            references[label] = block

    links: list[DocumentLink] = []
    text_roles = {"paragraph", "list_item", "heading", "footnote"}
    for block in document.blocks:
        if block.role not in text_roles or not block.text:
            continue
        for match in _FIGURE_REF_RE.finditer(block.text):
            target = figures.get(match.group(1).casefold())
            if target is not None and target.block_id != block.block_id:
                links.append(DocumentLink(block.block_id, target.block_id, "refers_to", 0.98))
        for match in _TABLE_REF_RE.finditer(block.text):
            target = tables.get(match.group(1).casefold())
            if target is not None and target.block_id != block.block_id:
                links.append(DocumentLink(block.block_id, target.block_id, "refers_to", 0.98))
        for match in _EQUATION_REF_RE.finditer(block.text):
            target = equations.get(match.group(1).casefold())
            if target is not None and target.block_id != block.block_id:
                links.append(DocumentLink(block.block_id, target.block_id, "equation_ref", 0.98))
        for match in _REFERENCE_REF_RE.finditer(block.text):
            label = (match.group(1) or match.group(2) or "").casefold()
            target = references.get(label)
            if target is not None and target.block_id != block.block_id:
                links.append(DocumentLink(block.block_id, target.block_id, "citation_ref", 0.95))
    return tuple(links)


def enrich_document_links(
    document: ScientificDocumentIR,
    *,
    reading_order: bool = True,
    captions: bool = True,
    references: bool = True,
) -> ScientificDocumentIR:
    if not isinstance(document, ScientificDocumentIR):
        raise TypeError("document must be ScientificDocumentIR")
    links = list(document.links)
    if reading_order:
        links.extend(derive_reading_order(document))
    if captions:
        links.extend(derive_caption_links(document))
    if references:
        links.extend(derive_reference_links(document))
    return replace(document, links=_dedupe(links))


__all__ = [
    "derive_caption_links",
    "derive_reading_order",
    "derive_reference_links",
    "enrich_document_links",
]
