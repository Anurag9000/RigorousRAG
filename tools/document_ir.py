"""Normalized scientific-document intermediate representation.

The IR is model-neutral and preserves source/page/coordinate lineage for prose, tables,
figures, formulas, captions, references and cross-modal links.  Extractors may be
heuristic or learned, but the representation and deterministic identities are shared.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from tools.multimodal_evidence import NormalizedBBox
from tools.security import normalize_owner_id

_MAX_PAGES = 100_000
_MAX_BLOCKS = 500_000
_MAX_TEXT = 100_000
_MAX_LINKS = 1_000_000
_BLOCK_ROLES = frozenset({
    "title", "heading", "paragraph", "list_item", "table", "figure", "chart", "formula",
    "caption", "footnote", "header", "footer", "reference", "metadata", "unknown",
})
_LINK_KINDS = frozenset({
    "reading_next", "contains", "caption_of", "refers_to", "defines", "continues",
    "table_cell_of", "panel_of", "equation_ref", "citation_ref", "same_entity",
})


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 digest")
    result = value.strip().lower()
    if len(result) != 64 or any(ch not in "0123456789abcdef" for ch in result):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return result


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _finite(value: Any, label: str, minimum: float = 0.0, maximum: float = 1.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{label} is outside its valid range")
    return result


@dataclass(frozen=True)
class TextSpan:
    text: str
    bbox: NormalizedBBox
    style: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _text(self.text, "span text", _MAX_TEXT))
        if not isinstance(self.bbox, NormalizedBBox):
            raise ValueError("bbox must be NormalizedBBox")
        if not isinstance(self.style, Mapping) or len(self.style) > 32:
            raise ValueError("style must be a bounded mapping")
        safe = {str(key)[:100]: str(value)[:500] for key, value in self.style.items()}
        object.__setattr__(self, "style", safe)
        object.__setattr__(self, "confidence", _finite(self.confidence, "confidence"))


@dataclass(frozen=True)
class TableCell:
    row: int
    column: int
    row_span: int
    column_span: int
    bbox: NormalizedBBox
    text: str
    is_header: bool = False

    def __post_init__(self) -> None:
        for name, minimum, maximum in (("row", 0, 100_000), ("column", 0, 10_000), ("row_span", 1, 10_000), ("column_span", 1, 10_000)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{name} is invalid")
        if not isinstance(self.bbox, NormalizedBBox):
            raise ValueError("bbox must be NormalizedBBox")
        object.__setattr__(self, "text", _text(self.text, "cell text", 20_000, allow_empty=True))
        if not isinstance(self.is_header, bool):
            raise ValueError("is_header must be boolean")


@dataclass(frozen=True)
class TableStructure:
    cells: tuple[TableCell, ...]
    caption: str = ""
    footnotes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.cells) > 100_000:
            raise ValueError("table has too many cells")
        occupied: set[tuple[int, int]] = set()
        for cell in self.cells:
            if not isinstance(cell, TableCell):
                raise ValueError("cells must contain TableCell objects")
            for row in range(cell.row, cell.row + cell.row_span):
                for col in range(cell.column, cell.column + cell.column_span):
                    coordinate = (row, col)
                    if coordinate in occupied:
                        raise ValueError("table cells overlap after span expansion")
                    occupied.add(coordinate)
        object.__setattr__(self, "caption", _text(self.caption, "table caption", 10_000, allow_empty=True))
        if len(self.footnotes) > 256:
            raise ValueError("too many table footnotes")
        object.__setattr__(self, "footnotes", tuple(_text(item, "table footnote", 2000) for item in self.footnotes))

    def to_matrix(self, *, fill: str = "") -> tuple[tuple[str, ...], ...]:
        if not self.cells:
            return ()
        max_row = max(cell.row + cell.row_span for cell in self.cells)
        max_col = max(cell.column + cell.column_span for cell in self.cells)
        matrix = [[fill for _ in range(max_col)] for _ in range(max_row)]
        for cell in self.cells:
            matrix[cell.row][cell.column] = cell.text
        return tuple(tuple(row) for row in matrix)


@dataclass(frozen=True)
class FigurePanel:
    panel_id: str
    bbox: NormalizedBBox
    label: str = ""
    content_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "panel_id", _text(self.panel_id, "panel_id", 256))
        if not isinstance(self.bbox, NormalizedBBox):
            raise ValueError("bbox must be NormalizedBBox")
        object.__setattr__(self, "label", _text(self.label, "panel label", 100, allow_empty=True))
        if self.content_sha256:
            object.__setattr__(self, "content_sha256", _digest(self.content_sha256, "content_sha256"))


@dataclass(frozen=True)
class FigureStructure:
    caption: str = ""
    panels: tuple[FigurePanel, ...] = ()
    axis_labels: tuple[str, ...] = ()
    legend_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "caption", _text(self.caption, "figure caption", 10_000, allow_empty=True))
        if len(self.panels) > 256:
            raise ValueError("too many figure panels")
        if any(not isinstance(panel, FigurePanel) for panel in self.panels):
            raise ValueError("panels must contain FigurePanel objects")
        if len({panel.panel_id for panel in self.panels}) != len(self.panels):
            raise ValueError("duplicate figure panel IDs")
        for name in ("axis_labels", "legend_labels"):
            values = getattr(self, name)
            if len(values) > 256:
                raise ValueError(f"{name} exceeds the item limit")
            object.__setattr__(self, name, tuple(_text(item, name, 500) for item in values))


@dataclass(frozen=True)
class FormulaStructure:
    normalized_text: str
    latex: str = ""
    equation_number: str = ""
    symbols: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "normalized_text", _text(self.normalized_text, "normalized formula", 20_000))
        object.__setattr__(self, "latex", _text(self.latex, "latex", 20_000, allow_empty=True))
        object.__setattr__(self, "equation_number", _text(self.equation_number, "equation number", 100, allow_empty=True))
        if not isinstance(self.symbols, Mapping) or len(self.symbols) > 512:
            raise ValueError("symbols must be a bounded mapping")
        object.__setattr__(self, "symbols", {_text(str(key), "symbol", 100): _text(str(value), "definition", 1000) for key, value in self.symbols.items()})


@dataclass(frozen=True)
class DocumentBlock:
    block_id: str
    page_number: int
    role: str
    bbox: NormalizedBBox
    text: str = ""
    spans: tuple[TextSpan, ...] = ()
    table: TableStructure | None = None
    figure: FigureStructure | None = None
    formula: FormulaStructure | None = None
    confidence: float = 1.0
    content_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_id", _text(self.block_id, "block_id", 256))
        if isinstance(self.page_number, bool) or not isinstance(self.page_number, int) or not 1 <= self.page_number <= _MAX_PAGES:
            raise ValueError("page_number is invalid")
        role = _text(self.role, "role", 50).lower()
        if role not in _BLOCK_ROLES:
            raise ValueError("unsupported block role")
        object.__setattr__(self, "role", role)
        if not isinstance(self.bbox, NormalizedBBox):
            raise ValueError("bbox must be NormalizedBBox")
        object.__setattr__(self, "text", _text(self.text, "block text", _MAX_TEXT, allow_empty=True))
        if len(self.spans) > 100_000 or any(not isinstance(span, TextSpan) for span in self.spans):
            raise ValueError("spans are invalid")
        if self.table is not None and role != "table":
            raise ValueError("table structure requires table role")
        if self.figure is not None and role not in {"figure", "chart"}:
            raise ValueError("figure structure requires figure/chart role")
        if self.formula is not None and role != "formula":
            raise ValueError("formula structure requires formula role")
        object.__setattr__(self, "confidence", _finite(self.confidence, "confidence"))
        digest = self.content_sha256
        if not digest:
            payload = self.text.encode("utf-8")
            digest = hashlib.sha256(payload).hexdigest()
        object.__setattr__(self, "content_sha256", _digest(digest, "content_sha256"))


@dataclass(frozen=True)
class DocumentLink:
    source_block_id: str
    target_block_id: str
    kind: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_block_id", _text(self.source_block_id, "source_block_id", 256))
        object.__setattr__(self, "target_block_id", _text(self.target_block_id, "target_block_id", 256))
        if self.source_block_id == self.target_block_id:
            raise ValueError("document links may not self-reference")
        kind = _text(self.kind, "link kind", 64).lower()
        if kind not in _LINK_KINDS:
            raise ValueError("unsupported document link kind")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "confidence", _finite(self.confidence, "confidence"))


@dataclass(frozen=True)
class DocumentPage:
    page_number: int
    width_points: float
    height_points: float
    rendered_sha256: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.page_number, bool) or not isinstance(self.page_number, int) or not 1 <= self.page_number <= _MAX_PAGES:
            raise ValueError("page_number is invalid")
        for name in ("width_points", "height_points"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"{name} is invalid")
            try:
                parsed = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} is invalid") from exc
            if not math.isfinite(parsed) or parsed <= 0 or parsed > 1_000_000:
                raise ValueError(f"{name} is invalid")
            object.__setattr__(self, name, parsed)
        if self.rendered_sha256:
            object.__setattr__(self, "rendered_sha256", _digest(self.rendered_sha256, "rendered_sha256"))


@dataclass(frozen=True)
class ScientificDocumentIR:
    owner_id: str
    doc_id: str
    source_sha256: str
    pages: tuple[DocumentPage, ...]
    blocks: tuple[DocumentBlock, ...]
    links: tuple[DocumentLink, ...] = ()
    extractor_id: str = "deterministic"
    schema_version: str = "1.0.0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "doc_id", _text(self.doc_id, "doc_id", 200))
        object.__setattr__(self, "source_sha256", _digest(self.source_sha256, "source_sha256"))
        if not 1 <= len(self.pages) <= _MAX_PAGES:
            raise ValueError("pages are empty or exceed the limit")
        if len(self.blocks) > _MAX_BLOCKS or len(self.links) > _MAX_LINKS:
            raise ValueError("document IR exceeds its structural limits")
        page_numbers = [page.page_number for page in self.pages]
        if page_numbers != list(range(1, len(self.pages) + 1)):
            raise ValueError("pages must be contiguous and one-indexed")
        block_ids = {block.block_id for block in self.blocks}
        if len(block_ids) != len(self.blocks):
            raise ValueError("duplicate block IDs")
        page_set = set(page_numbers)
        if any(block.page_number not in page_set for block in self.blocks):
            raise ValueError("block references an unknown page")
        for link in self.links:
            if link.source_block_id not in block_ids or link.target_block_id not in block_ids:
                raise ValueError("document link references an unknown block")
        object.__setattr__(self, "extractor_id", _text(self.extractor_id, "extractor_id", 200))
        object.__setattr__(self, "schema_version", _text(self.schema_version, "schema_version", 32))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()

    def blocks_on_page(self, page_number: int) -> tuple[DocumentBlock, ...]:
        return tuple(block for block in self.blocks if block.page_number == page_number)

    def reading_order(self) -> tuple[DocumentBlock, ...]:
        adjacency = {link.source_block_id: link.target_block_id for link in self.links if link.kind == "reading_next"}
        incoming = {target for target in adjacency.values()}
        by_id = {block.block_id: block for block in self.blocks}
        starts = sorted((block.block_id for block in self.blocks if block.block_id not in incoming), key=lambda identifier: (by_id[identifier].page_number, by_id[identifier].bbox.y0, by_id[identifier].bbox.x0))
        output: list[DocumentBlock] = []
        seen: set[str] = set()
        for start in starts:
            current = start
            while current in by_id and current not in seen:
                seen.add(current)
                output.append(by_id[current])
                current = adjacency.get(current, "")
        for block in sorted(self.blocks, key=lambda item: (item.page_number, item.bbox.y0, item.bbox.x0, item.block_id)):
            if block.block_id not in seen:
                output.append(block)
        return tuple(output)


def deterministic_block_id(*, source_sha256: str, page_number: int, role: str, bbox: NormalizedBBox, content_sha256: str) -> str:
    payload = {
        "contract": "rigorousrag-document-block-v1",
        "source_sha256": _digest(source_sha256, "source_sha256"),
        "page_number": page_number,
        "role": _text(role, "role", 50).lower(),
        "bbox": asdict(bbox),
        "content_sha256": _digest(content_sha256, "content_sha256"),
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


__all__ = [
    "DocumentBlock",
    "DocumentLink",
    "DocumentPage",
    "FigurePanel",
    "FigureStructure",
    "FormulaStructure",
    "ScientificDocumentIR",
    "TableCell",
    "TableStructure",
    "TextSpan",
    "deterministic_block_id",
]
