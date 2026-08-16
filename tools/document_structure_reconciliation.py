"""Deterministic reconciliation for cross-page tables, formula symbols and figure panels.

The resolver never rewrites extractor output. It emits a separately fingerprinted
structural hypothesis layer and may add bounded ``continues`` / ``defines`` IR links.
Ambiguous table continuations remain unresolved rather than being greedily stitched.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping

from tools.document_ir import DocumentBlock, DocumentLink, ScientificDocumentIR, TableStructure

_MAX_TABLES = 50_000
_MAX_BINDINGS = 100_000
_TOKEN_RE = re.compile(r"[^a-z0-9%±+\-./]+", re.I)
_SYMBOL_DEF_RE = re.compile(
    r"(?:\bwhere\b|\bwith\b|\blet\b|\bdenote(?:s|d)?\b)?\s*"
    r"([A-Za-z][A-Za-z0-9_]{0,31}|[α-ωΑ-Ω])\s+(?:is|denotes|represents|means|=)\s+([^.;]{2,300})",
    re.I,
)
_PANEL_RE = re.compile(r"(?:^|[\s,;])\(?([A-Za-z])\)\s*([^;]{0,300})")
_CONTINUED_RE = re.compile(r"\b(?:continued|cont\.?|continuation)\b", re.I)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _normalize(value: str) -> str:
    return " ".join(_TOKEN_RE.sub(" ", str(value).casefold()).split())


def _table_blocks(document: ScientificDocumentIR) -> tuple[DocumentBlock, ...]:
    values = tuple(block for block in document.blocks if block.role == "table" and block.table is not None)
    if len(values) > _MAX_TABLES:
        raise ValueError("document contains too many table blocks")
    return values


def _column_count(table: TableStructure) -> int:
    return max((cell.column + cell.column_span for cell in table.cells), default=0)


def _header_signature(table: TableStructure) -> tuple[str, ...]:
    if not table.cells:
        return ()
    header_rows = sorted({cell.row for cell in table.cells if cell.is_header})
    selected_row = header_rows[0] if header_rows else min(cell.row for cell in table.cells)
    cells = sorted((cell for cell in table.cells if cell.row == selected_row), key=lambda cell: cell.column)
    return tuple(_normalize(cell.text) for cell in cells if _normalize(cell.text))


def _horizontal_overlap(left: DocumentBlock, right: DocumentBlock) -> float:
    overlap = max(0.0, min(left.bbox.x1, right.bbox.x1) - max(left.bbox.x0, right.bbox.x0))
    denominator = max(left.bbox.x1 - left.bbox.x0, right.bbox.x1 - right.bbox.x0, 1e-9)
    return overlap / denominator


def _caption_signature(block: DocumentBlock) -> str:
    if block.table is None:
        return ""
    return _normalize(block.table.caption or block.text)


def _continuation_score(left: DocumentBlock, right: DocumentBlock) -> tuple[float, tuple[str, ...]]:
    if left.table is None or right.table is None or right.page_number != left.page_number + 1:
        return 0.0, ()
    left_cols, right_cols = _column_count(left.table), _column_count(right.table)
    if left_cols <= 0 or left_cols != right_cols:
        return 0.0, ("column_count_mismatch",)
    overlap = _horizontal_overlap(left, right)
    if overlap < 0.55:
        return 0.0, ("horizontal_alignment_weak",)
    reasons = ["adjacent_pages", "column_count_match", "horizontal_alignment"]
    score = 0.45 + min(0.20, overlap * 0.20)
    left_header, right_header = _header_signature(left.table), _header_signature(right.table)
    if left_header and right_header and left_header == right_header:
        score += 0.25
        reasons.append("repeated_header_match")
    left_caption, right_caption = _caption_signature(left), _caption_signature(right)
    if left_caption and right_caption and left_caption == right_caption:
        score += 0.15
        reasons.append("caption_match")
    elif _CONTINUED_RE.search(right.table.caption or right.text or ""):
        score += 0.15
        reasons.append("explicit_continuation_marker")
    # Strong geometric/column agreement without a header/caption clue remains only a
    # moderate hypothesis and is never auto-selected below the acceptance threshold.
    return min(score, 1.0), tuple(reasons)


@dataclass(frozen=True)
class TableContinuation:
    predecessor_block_id: str
    successor_block_id: str
    confidence: float
    reasons: tuple[str, ...]
    repeated_header: bool


@dataclass(frozen=True)
class CrossPageTable:
    table_id: str
    block_ids: tuple[str, ...]
    pages: tuple[int, ...]
    column_count: int
    matrix: tuple[tuple[str, ...], ...]
    continuation_confidences: tuple[float, ...]


@dataclass(frozen=True)
class FormulaSymbolBinding:
    formula_block_id: str
    symbol: str
    definition: str
    evidence_block_id: str
    confidence: float
    source: str


@dataclass(frozen=True)
class FigurePanelBinding:
    figure_block_id: str
    panel_id: str
    panel_label: str
    description: str
    evidence_block_id: str
    confidence: float


@dataclass(frozen=True)
class DocumentStructureResolution:
    document_fingerprint: str
    table_continuations: tuple[TableContinuation, ...]
    cross_page_tables: tuple[CrossPageTable, ...]
    formula_symbols: tuple[FormulaSymbolBinding, ...]
    figure_panels: tuple[FigurePanelBinding, ...]
    unresolved: tuple[str, ...]
    fingerprint: str


def _merge_tables(chain: tuple[DocumentBlock, ...], confidences: tuple[float, ...]) -> CrossPageTable:
    matrices = [block.table.to_matrix() for block in chain if block.table is not None]
    rows: list[tuple[str, ...]] = []
    previous_header: tuple[str, ...] = ()
    for index, (block, matrix) in enumerate(zip(chain, matrices)):
        if not matrix:
            continue
        header = _header_signature(block.table) if block.table is not None else ()
        start = 0
        if index > 0 and header and previous_header and header == previous_header:
            start = 1
        rows.extend(matrix[start:])
        if header:
            previous_header = header
    payload = {
        "contract": "rigorousrag-cross-page-table-v1",
        "blocks": [block.block_id for block in chain],
        "pages": [block.page_number for block in chain],
        "matrix": rows,
        "confidences": confidences,
    }
    return CrossPageTable(
        table_id=hashlib.sha256(_canonical(payload)).hexdigest(),
        block_ids=tuple(block.block_id for block in chain),
        pages=tuple(block.page_number for block in chain),
        column_count=_column_count(chain[0].table),
        matrix=tuple(rows),
        continuation_confidences=confidences,
    )


def resolve_cross_page_tables(document: ScientificDocumentIR, *, threshold: float = 0.78) -> tuple[tuple[TableContinuation, ...], tuple[CrossPageTable, ...], tuple[str, ...]]:
    if not isinstance(document, ScientificDocumentIR):
        raise TypeError("document must be ScientificDocumentIR")
    if not 0.5 <= float(threshold) <= 1.0:
        raise ValueError("table continuation threshold is invalid")
    tables = _table_blocks(document)
    by_page: dict[int, list[DocumentBlock]] = {}
    for block in tables:
        by_page.setdefault(block.page_number, []).append(block)
    selected: dict[str, TableContinuation] = {}
    unresolved: list[str] = []
    for left in tables:
        candidates: list[tuple[float, DocumentBlock, tuple[str, ...]]] = []
        for right in by_page.get(left.page_number + 1, ()):
            score, reasons = _continuation_score(left, right)
            if score >= threshold:
                candidates.append((score, right, reasons))
        candidates.sort(key=lambda item: (-item[0], item[1].block_id))
        if not candidates:
            continue
        best = candidates[0]
        if len(candidates) > 1 and abs(best[0] - candidates[1][0]) < 1e-9:
            unresolved.append(f"ambiguous_table_continuation:{left.block_id}")
            continue
        continuation = TableContinuation(
            left.block_id,
            best[1].block_id,
            best[0],
            best[2],
            "repeated_header_match" in best[2],
        )
        selected[left.block_id] = continuation

    # A successor may have only one predecessor. Equal-confidence collisions remain unresolved.
    by_successor: dict[str, list[TableContinuation]] = {}
    for item in selected.values():
        by_successor.setdefault(item.successor_block_id, []).append(item)
    accepted: list[TableContinuation] = []
    for successor, items in by_successor.items():
        items.sort(key=lambda item: (-item.confidence, item.predecessor_block_id))
        if len(items) > 1 and abs(items[0].confidence - items[1].confidence) < 1e-9:
            unresolved.append(f"ambiguous_table_predecessor:{successor}")
            continue
        accepted.append(items[0])
    accepted.sort(key=lambda item: (item.predecessor_block_id, item.successor_block_id))

    next_by_left = {item.predecessor_block_id: item for item in accepted}
    predecessor_ids = {item.successor_block_id for item in accepted}
    by_id = {block.block_id: block for block in tables}
    groups: list[CrossPageTable] = []
    for start_id in sorted(set(next_by_left) - predecessor_ids):
        chain = [by_id[start_id]]
        confidences: list[float] = []
        current = start_id
        seen = {current}
        while current in next_by_left:
            edge = next_by_left[current]
            if edge.successor_block_id in seen:
                unresolved.append(f"table_continuation_cycle:{edge.successor_block_id}")
                break
            seen.add(edge.successor_block_id)
            chain.append(by_id[edge.successor_block_id])
            confidences.append(edge.confidence)
            current = edge.successor_block_id
        if len(chain) > 1:
            groups.append(_merge_tables(tuple(chain), tuple(confidences)))
    return tuple(accepted), tuple(groups), tuple(sorted(set(unresolved)))


def _nearby_blocks(document: ScientificDocumentIR, formula: DocumentBlock, window: int = 4) -> tuple[DocumentBlock, ...]:
    ordered = list(document.reading_order())
    positions = {block.block_id: index for index, block in enumerate(ordered)}
    index = positions.get(formula.block_id)
    if index is None:
        return ()
    return tuple(ordered[max(0, index - window): min(len(ordered), index + window + 1)])


def resolve_formula_symbols(document: ScientificDocumentIR) -> tuple[FormulaSymbolBinding, ...]:
    output: list[FormulaSymbolBinding] = []
    seen: set[tuple[str, str, str]] = set()
    for formula in document.blocks:
        if formula.role != "formula" or formula.formula is None:
            continue
        for symbol, definition in formula.formula.symbols.items():
            key = (formula.block_id, symbol, formula.block_id)
            if key not in seen:
                seen.add(key)
                output.append(FormulaSymbolBinding(formula.block_id, symbol, definition, formula.block_id, 1.0, "extractor"))
        formula_text = f"{formula.formula.normalized_text} {formula.formula.latex}"
        candidates = _nearby_blocks(document, formula)
        for block in candidates:
            if block.block_id == formula.block_id or block.role not in {"paragraph", "list_item", "footnote"}:
                continue
            for match in _SYMBOL_DEF_RE.finditer(block.text):
                symbol, definition = match.group(1), " ".join(match.group(2).split())
                if symbol not in formula_text and symbol.casefold() not in formula_text.casefold():
                    continue
                key = (formula.block_id, symbol, block.block_id)
                if key in seen:
                    continue
                seen.add(key)
                output.append(FormulaSymbolBinding(formula.block_id, symbol, definition, block.block_id, 0.82, "nearby_prose"))
                if len(output) > _MAX_BINDINGS:
                    raise ValueError("formula symbol bindings exceed the limit")
    return tuple(output)


def resolve_figure_panels(document: ScientificDocumentIR) -> tuple[FigurePanelBinding, ...]:
    output: list[FigurePanelBinding] = []
    captions_by_page = {
        page.page_number: tuple(block for block in document.blocks_on_page(page.page_number) if block.role == "caption")
        for page in document.pages
    }
    for block in document.blocks:
        if block.role not in {"figure", "chart"} or block.figure is None or not block.figure.panels:
            continue
        caption_sources = [block.figure.caption]
        caption_sources.extend(item.text for item in captions_by_page.get(block.page_number, ()))
        parsed: dict[str, tuple[str, str]] = {}
        for caption in caption_sources:
            for match in _PANEL_RE.finditer(caption or ""):
                label = match.group(1).casefold()
                description = " ".join(match.group(2).split())
                parsed.setdefault(label, (description, caption))
        for panel in block.figure.panels:
            label = panel.label.casefold().strip("() ")
            if not label:
                continue
            description, source_caption = parsed.get(label, ("", ""))
            evidence_id = block.block_id
            if source_caption:
                for candidate in captions_by_page.get(block.page_number, ()):
                    if candidate.text == source_caption:
                        evidence_id = candidate.block_id
                        break
            output.append(
                FigurePanelBinding(
                    block.block_id,
                    panel.panel_id,
                    panel.label,
                    description,
                    evidence_id,
                    0.92 if description else 0.70,
                )
            )
            if len(output) > _MAX_BINDINGS:
                raise ValueError("figure panel bindings exceed the limit")
    return tuple(output)


def resolve_document_structures(document: ScientificDocumentIR) -> DocumentStructureResolution:
    continuations, tables, unresolved = resolve_cross_page_tables(document)
    symbols = resolve_formula_symbols(document)
    panels = resolve_figure_panels(document)
    payload = {
        "contract": "rigorousrag-document-structure-resolution-v1",
        "document_fingerprint": document.fingerprint,
        "table_continuations": [asdict(item) for item in continuations],
        "cross_page_tables": [asdict(item) for item in tables],
        "formula_symbols": [asdict(item) for item in symbols],
        "figure_panels": [asdict(item) for item in panels],
        "unresolved": unresolved,
    }
    return DocumentStructureResolution(
        document.fingerprint,
        continuations,
        tables,
        symbols,
        panels,
        unresolved,
        hashlib.sha256(_canonical(payload)).hexdigest(),
    )


def enrich_with_resolved_structure_links(document: ScientificDocumentIR, resolution: DocumentStructureResolution) -> ScientificDocumentIR:
    if resolution.document_fingerprint != document.fingerprint:
        raise ValueError("resolution does not belong to the supplied document generation")
    links = list(document.links)
    links.extend(DocumentLink(item.predecessor_block_id, item.successor_block_id, "continues", item.confidence) for item in resolution.table_continuations)
    links.extend(DocumentLink(item.evidence_block_id, item.formula_block_id, "defines", item.confidence) for item in resolution.formula_symbols if item.evidence_block_id != item.formula_block_id)
    deduped: list[DocumentLink] = []
    seen: set[tuple[str, str, str]] = set()
    for link in links:
        key = (link.source_block_id, link.target_block_id, link.kind)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(link)
    return replace(document, links=tuple(deduped))


__all__ = [
    "CrossPageTable",
    "DocumentStructureResolution",
    "FigurePanelBinding",
    "FormulaSymbolBinding",
    "TableContinuation",
    "enrich_with_resolved_structure_links",
    "resolve_cross_page_tables",
    "resolve_document_structures",
    "resolve_figure_panels",
    "resolve_formula_symbols",
]
