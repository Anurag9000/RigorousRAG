"""Table structure and cell-level provenance primitives for multimodal RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Tuple


@dataclass(frozen=True)
class TableCell:
    cell_id: str
    row: int
    column: int
    text: str
    row_span: int = 1
    column_span: int = 1
    header: bool = False
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cell_id.strip():
            raise ValueError("cell_id is required.")
        if self.row < 0 or self.column < 0:
            raise ValueError("row and column must be non-negative.")
        if self.row_span <= 0 or self.column_span <= 0:
            raise ValueError("row_span and column_span must be positive.")

    def occupied_coordinates(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(
            (row, column)
            for row in range(self.row, self.row + self.row_span)
            for column in range(self.column, self.column + self.column_span)
        )


@dataclass(frozen=True)
class TableProvenance:
    table_id: str
    source_id: str
    page: Optional[int]
    cells: Tuple[TableCell, ...]
    caption: str = ""
    section: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.table_id.strip() or not self.source_id.strip():
            raise ValueError("table_id and source_id are required.")
        if self.page is not None and self.page < 0:
            raise ValueError("page must be non-negative when supplied.")
        ids = [cell.cell_id for cell in self.cells]
        if len(ids) != len(set(ids)):
            raise ValueError("cell identifiers must be unique within a table.")
        occupied: dict[tuple[int, int], str] = {}
        for cell in self.cells:
            for coordinate in cell.occupied_coordinates():
                previous = occupied.get(coordinate)
                if previous is not None:
                    raise ValueError(
                        f"cells {previous!r} and {cell.cell_id!r} overlap at {coordinate}."
                    )
                occupied[coordinate] = cell.cell_id

    @property
    def shape(self) -> Tuple[int, int]:
        if not self.cells:
            return 0, 0
        rows = max(cell.row + cell.row_span for cell in self.cells)
        columns = max(cell.column + cell.column_span for cell in self.cells)
        return rows, columns

    def cell(self, cell_id: str) -> TableCell:
        for cell in self.cells:
            if cell.cell_id == cell_id:
                return cell
        raise KeyError(cell_id)

    def citation_key(self, cell_id: Optional[str] = None) -> str:
        page = f":p{self.page}" if self.page is not None else ""
        suffix = f":cell={cell_id}" if cell_id is not None else ""
        if cell_id is not None:
            self.cell(cell_id)
        return f"{self.source_id}{page}:table={self.table_id}{suffix}"

    def row_text(self, row: int) -> str:
        if row < 0:
            raise ValueError("row must be non-negative.")
        cells = [
            cell
            for cell in self.cells
            if cell.row <= row < cell.row + cell.row_span
        ]
        cells.sort(key=lambda cell: (cell.column, cell.cell_id))
        return " | ".join(cell.text for cell in cells)


def table_from_cells(
    *,
    table_id: str,
    source_id: str,
    cells: Iterable[TableCell],
    page: Optional[int] = None,
    caption: str = "",
    section: str = "",
) -> TableProvenance:
    return TableProvenance(
        table_id=table_id,
        source_id=source_id,
        page=page,
        cells=tuple(cells),
        caption=caption,
        section=section,
    )
