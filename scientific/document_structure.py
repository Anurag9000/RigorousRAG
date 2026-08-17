"""Provenance-aware intermediate representation for scientific document structure.

The IR preserves page geometry, reading order, table topology, formulas and
figure/panel/caption relationships without coupling the repository to a specific OCR or
layout model.  Provider adapters may populate the records; downstream retrieval can use
the validated structure and source anchors.  Importing this module executes no OCR/model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

_MAX_REGIONS = 1_000_000
_MAX_TABLE_CELLS = 1_000_000


def _text(value: Any, label: str, maximum: int = 100_000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if (not allow_empty and not result) or len(result) > maximum or "\x00" in result:
        raise ValueError(f"{label} is empty or too long")
    return result


def _identifier(value: Any, label: str, maximum: int = 1_000) -> str:
    result = _text(value, label, maximum)
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{label} contains control characters")
    return result


def _unit(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be normalized to [0,1]")
    return result


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class RegionKind(str, Enum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    FORMULA = "formula"
    FOOTNOTE = "footnote"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    OTHER = "other"


@dataclass(frozen=True)
class BoundingBox:
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        for name in ("left", "top", "right", "bottom"):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("bounding box must have positive width and height")

    @property
    def area(self) -> float:
        return (self.right - self.left) * (self.bottom - self.top)


@dataclass(frozen=True)
class SourceAnchor:
    document_id: str
    generation_id: str
    page: int
    extraction_artifact_id: str

    def __post_init__(self) -> None:
        for name in ("document_id", "generation_id", "extraction_artifact_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if isinstance(self.page, bool) or not isinstance(self.page, int) or self.page < 1:
            raise ValueError("page must be a positive integer")


@dataclass(frozen=True)
class DocumentRegion:
    region_id: str
    kind: RegionKind
    box: BoundingBox
    anchor: SourceAnchor
    text: str | None = None
    confidence: float | None = None
    parent_region_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_id", _identifier(self.region_id, "region_id"))
        if not isinstance(self.kind, RegionKind):
            object.__setattr__(self, "kind", RegionKind(self.kind))
        if not isinstance(self.box, BoundingBox) or not isinstance(self.anchor, SourceAnchor):
            raise ValueError("box/anchor types are invalid")
        if self.text is not None:
            object.__setattr__(self, "text", _text(self.text, "region text", allow_empty=True))
        if self.confidence is not None:
            value = _unit(self.confidence, "confidence")
            object.__setattr__(self, "confidence", value)
        if self.parent_region_id is not None:
            object.__setattr__(self, "parent_region_id", _identifier(self.parent_region_id, "parent_region_id"))
        if not isinstance(self.metadata, Mapping) or len(self.metadata) > 2_000:
            raise ValueError("metadata must be a bounded mapping")
        object.__setattr__(
            self,
            "metadata",
            {
                _identifier(key, "metadata key", 300): _text(value, "metadata value", 20_000)
                for key, value in self.metadata.items()
            },
        )


@dataclass(frozen=True)
class ReadingOrderEdge:
    before_region_id: str
    after_region_id: str
    reason: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        for name in ("before_region_id", "after_region_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if self.before_region_id == self.after_region_id:
            raise ValueError("reading-order edge may not self-reference")
        object.__setattr__(self, "reason", _text(self.reason, "reading order reason", 10_000))
        if self.confidence is not None:
            object.__setattr__(self, "confidence", _unit(self.confidence, "confidence"))


def topological_reading_order(
    region_ids: Sequence[str],
    edges: Sequence[ReadingOrderEdge],
) -> tuple[str, ...]:
    """Return deterministic reading order or reject cyclic/incomplete references."""

    if len(region_ids) > _MAX_REGIONS or len(edges) > _MAX_REGIONS * 10:
        raise ValueError("reading-order graph exceeds safety bounds")
    ids = tuple(_identifier(value, "region_id") for value in region_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("region_ids must be unique")
    adjacency = {region_id: set() for region_id in ids}
    indegree = {region_id: 0 for region_id in ids}
    for edge in edges:
        if edge.before_region_id not in adjacency or edge.after_region_id not in adjacency:
            raise ValueError("reading-order edge references an unknown region")
        if edge.after_region_id not in adjacency[edge.before_region_id]:
            adjacency[edge.before_region_id].add(edge.after_region_id)
            indegree[edge.after_region_id] += 1
    ready = sorted(region_id for region_id, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for target in sorted(adjacency[current]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    if len(order) != len(ids):
        raise ValueError("reading-order graph contains a cycle")
    return tuple(order)


@dataclass(frozen=True)
class TableCell:
    cell_id: str
    table_region_id: str
    row_start: int
    row_span: int
    column_start: int
    column_span: int
    text: str
    box: BoundingBox | None = None
    is_header: bool = False
    confidence: float | None = None

    def __post_init__(self) -> None:
        for name in ("cell_id", "table_region_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name, minimum in (("row_start", 0), ("column_start", 0), ("row_span", 1), ("column_span", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > 1_000_000:
                raise ValueError(f"{name} is invalid")
        object.__setattr__(self, "text", _text(self.text, "cell text", allow_empty=True))
        if self.box is not None and not isinstance(self.box, BoundingBox):
            raise ValueError("box must be BoundingBox")
        if not isinstance(self.is_header, bool):
            raise ValueError("is_header must be boolean")
        if self.confidence is not None:
            object.__setattr__(self, "confidence", _unit(self.confidence, "confidence"))


@dataclass(frozen=True)
class StructuredTable:
    table_region_id: str
    cells: tuple[TableCell, ...]
    row_count: int
    column_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "table_region_id", _identifier(self.table_region_id, "table_region_id"))
        for name in ("row_count", "column_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1_000_000:
                raise ValueError(f"{name} is invalid")
        if not self.cells or len(self.cells) > _MAX_TABLE_CELLS:
            raise ValueError("cells must be non-empty and bounded")
        cell_ids: set[str] = set()
        occupied: dict[tuple[int, int], str] = {}
        for cell in self.cells:
            if not isinstance(cell, TableCell) or cell.table_region_id != self.table_region_id:
                raise ValueError("table cell has invalid type or table id")
            if cell.cell_id in cell_ids:
                raise ValueError("cell ids must be unique")
            cell_ids.add(cell.cell_id)
            if cell.row_start + cell.row_span > self.row_count or cell.column_start + cell.column_span > self.column_count:
                raise ValueError("cell span exceeds declared table dimensions")
            for row in range(cell.row_start, cell.row_start + cell.row_span):
                for column in range(cell.column_start, cell.column_start + cell.column_span):
                    location = (row, column)
                    if location in occupied:
                        raise ValueError(f"overlapping table cells {occupied[location]} and {cell.cell_id}")
                    occupied[location] = cell.cell_id

    def cell_at(self, row: int, column: int) -> TableCell | None:
        if not 0 <= row < self.row_count or not 0 <= column < self.column_count:
            raise IndexError("table coordinate outside declared dimensions")
        for cell in self.cells:
            if (
                cell.row_start <= row < cell.row_start + cell.row_span
                and cell.column_start <= column < cell.column_start + cell.column_span
            ):
                return cell
        return None


@dataclass(frozen=True)
class FormulaRecord:
    formula_id: str
    region_id: str
    latex: str | None = None
    mathml: str | None = None
    plain_text: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        for name in ("formula_id", "region_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if self.latex is None and self.mathml is None and self.plain_text is None:
            raise ValueError("formula requires at least one representation")
        for name in ("latex", "mathml", "plain_text"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name, 100_000))
        if self.confidence is not None:
            object.__setattr__(self, "confidence", _unit(self.confidence, "confidence"))

    @property
    def lexical_form(self) -> str:
        """Stable whitespace-normalized representation for lexical retrieval/indexing."""

        value = self.latex or self.plain_text or self.mathml or ""
        return " ".join(value.replace("\n", " ").split())


@dataclass(frozen=True)
class FigurePanel:
    panel_id: str
    figure_region_id: str
    box: BoundingBox
    label: str | None = None

    def __post_init__(self) -> None:
        for name in ("panel_id", "figure_region_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if not isinstance(self.box, BoundingBox):
            raise ValueError("panel box must be BoundingBox")
        if self.label is not None:
            object.__setattr__(self, "label", _text(self.label, "panel label", 1_000))


@dataclass(frozen=True)
class FigureStructure:
    figure_region_id: str
    caption_region_ids: tuple[str, ...]
    panels: tuple[FigurePanel, ...] = ()
    cited_from_region_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "figure_region_id", _identifier(self.figure_region_id, "figure_region_id"))
        object.__setattr__(
            self,
            "caption_region_ids",
            tuple(_identifier(value, "caption_region_id") for value in self.caption_region_ids),
        )
        if not self.caption_region_ids:
            raise ValueError("figure structure requires at least one caption region")
        if len(set(self.caption_region_ids)) != len(self.caption_region_ids):
            raise ValueError("caption region ids must be unique")
        if any(not isinstance(panel, FigurePanel) or panel.figure_region_id != self.figure_region_id for panel in self.panels):
            raise ValueError("panels must belong to the figure")
        panel_ids = [panel.panel_id for panel in self.panels]
        if len(set(panel_ids)) != len(panel_ids):
            raise ValueError("panel ids must be unique")
        object.__setattr__(
            self,
            "cited_from_region_ids",
            tuple(_identifier(value, "cited_from_region_id") for value in self.cited_from_region_ids),
        )


@dataclass(frozen=True)
class StructuredDocument:
    document_id: str
    generation_id: str
    regions: tuple[DocumentRegion, ...]
    reading_edges: tuple[ReadingOrderEdge, ...] = ()
    tables: tuple[StructuredTable, ...] = ()
    formulas: tuple[FormulaRecord, ...] = ()
    figures: tuple[FigureStructure, ...] = ()

    def __post_init__(self) -> None:
        for name in ("document_id", "generation_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if not self.regions or len(self.regions) > _MAX_REGIONS:
            raise ValueError("regions must be non-empty and bounded")
        by_id = {region.region_id: region for region in self.regions}
        if len(by_id) != len(self.regions):
            raise ValueError("region ids must be unique")
        for region in self.regions:
            if region.anchor.document_id != self.document_id or region.anchor.generation_id != self.generation_id:
                raise ValueError("region provenance does not match structured document")
            if region.parent_region_id is not None and region.parent_region_id not in by_id:
                raise ValueError("parent_region_id references an unknown region")
        topological_reading_order(tuple(by_id), self.reading_edges)
        for table in self.tables:
            region = by_id.get(table.table_region_id)
            if region is None or region.kind != RegionKind.TABLE:
                raise ValueError("structured table must reference a TABLE region")
        for formula in self.formulas:
            region = by_id.get(formula.region_id)
            if region is None or region.kind != RegionKind.FORMULA:
                raise ValueError("formula must reference a FORMULA region")
        for figure in self.figures:
            region = by_id.get(figure.figure_region_id)
            if region is None or region.kind != RegionKind.FIGURE:
                raise ValueError("figure structure must reference a FIGURE region")
            for caption_id in figure.caption_region_ids:
                caption = by_id.get(caption_id)
                if caption is None or caption.kind != RegionKind.CAPTION:
                    raise ValueError("figure caption must reference a CAPTION region")
            if any(reference not in by_id for reference in figure.cited_from_region_ids):
                raise ValueError("figure citation references an unknown region")

    @property
    def digest(self) -> str:
        return _digest(asdict(self))

    @property
    def reading_order(self) -> tuple[str, ...]:
        return topological_reading_order(tuple(region.region_id for region in self.regions), self.reading_edges)


__all__ = [
    "BoundingBox",
    "DocumentRegion",
    "FigurePanel",
    "FigureStructure",
    "FormulaRecord",
    "ReadingOrderEdge",
    "RegionKind",
    "SourceAnchor",
    "StructuredDocument",
    "StructuredTable",
    "TableCell",
    "topological_reading_order",
]
