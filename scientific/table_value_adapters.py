"""Conservative deterministic extraction of typed numeric table-cell quantities."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass

from evaluation.structured_data_support import Quantity, TableQuantityEvidence, table_quantity_evidence
from scientific.document_structure import StructuredDocument

_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NumericCellParserConfig:
    decimal_separator: str = "."
    thousands_separator: str | None = ","
    allow_accounting_parentheses: bool = True
    percent_as_fraction: bool = False
    allow_unit_suffix: bool = True

    def __post_init__(self) -> None:
        if self.decimal_separator not in {".", ","}:
            raise ValueError("decimal_separator must be . or ,")
        if self.thousands_separator not in {None, ".", ",", " ", "'"}:
            raise ValueError("unsupported thousands_separator")
        if self.thousands_separator == self.decimal_separator:
            raise ValueError("thousands_separator must differ from decimal_separator")
        for name in ("allow_accounting_parentheses", "percent_as_fraction", "allow_unit_suffix"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")

    @property
    def config_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-numeric-cell-parser/v1", **asdict(self)})


@dataclass(frozen=True)
class ParsedNumericCell:
    value: float
    unit: str | None
    parser_sha256: str


def _split_unit(text: str, config: NumericCellParserConfig) -> tuple[str, str | None]:
    selected = text.strip()
    if not selected:
        raise ValueError("table cell is empty")
    if selected.endswith("%"):
        return selected[:-1].strip(), "%"
    if not config.allow_unit_suffix:
        return selected, None
    match = re.fullmatch(r"(.+?)\s+([^\d\s].*)", selected)
    if match is None:
        return selected, None
    number, unit = match.group(1).strip(), match.group(2).strip()
    if not unit or any(ch in unit for ch in "<>≈=±"):
        raise ValueError("table cell unit suffix is ambiguous")
    return number, unit


def _normalize_number(text: str, config: NumericCellParserConfig) -> str:
    selected = text.strip()
    negative = False
    if selected.startswith("(") or selected.endswith(")"):
        if not config.allow_accounting_parentheses or not (selected.startswith("(") and selected.endswith(")")):
            raise ValueError("malformed accounting numeric cell")
        negative = True
        selected = selected[1:-1].strip()
    if any(token in selected for token in ("<", ">", "≤", "≥", "~", "≈", "±", "–", "—")):
        raise ValueError("numeric table cell contains a comparison/range rather than a scalar")
    thousands = config.thousands_separator
    if thousands:
        if thousands == " ":
            selected = selected.replace(" ", "")
        elif thousands in selected:
            integer_part = selected
            exponent = ""
            for marker in ("e", "E"):
                if marker in integer_part:
                    integer_part, exponent = integer_part.split(marker, 1)
                    exponent = marker + exponent
                    break
            sign = ""
            if integer_part[:1] in {"+", "-"}:
                sign, integer_part = integer_part[0], integer_part[1:]
            decimal_parts = integer_part.split(config.decimal_separator)
            if len(decimal_parts) > 2:
                raise ValueError("numeric table cell contains multiple decimal separators")
            groups = decimal_parts[0].split(thousands)
            if len(groups) > 1 and (not 1 <= len(groups[0]) <= 3 or any(len(group) != 3 or not group.isdigit() for group in groups[1:])):
                raise ValueError("numeric table cell has invalid thousands grouping")
            integer_part = sign + "".join(groups)
            if len(decimal_parts) == 2:
                integer_part += config.decimal_separator + decimal_parts[1]
            selected = integer_part + exponent
    if config.decimal_separator == ",":
        selected = selected.replace(",", ".")
    if not _NUMBER.fullmatch(selected):
        raise ValueError("table cell is not one unambiguous numeric scalar")
    if negative:
        if selected.startswith("-"):
            raise ValueError("accounting parentheses may not wrap an already-negative value")
        selected = "-" + selected.lstrip("+")
    return selected


def parse_numeric_cell(text: str, *, config: NumericCellParserConfig = NumericCellParserConfig(), unit_override: str | None = None) -> ParsedNumericCell:
    if not isinstance(text, str) or "\x00" in text or len(text) > 100_000:
        raise ValueError("table cell text is invalid")
    number_text, suffix_unit = _split_unit(text, config)
    if unit_override is not None:
        if not isinstance(unit_override, str) or not unit_override.strip():
            raise ValueError("unit_override must be non-empty when set")
        unit = unit_override.strip()
        if suffix_unit is not None and suffix_unit != unit:
            raise ValueError("unit_override conflicts with cell unit suffix")
    else:
        unit = suffix_unit
    value = float(_normalize_number(number_text, config))
    if not math.isfinite(value):
        raise ValueError("parsed numeric cell is not finite")
    if suffix_unit == "%" and config.percent_as_fraction:
        value /= 100.0
        unit = None if unit_override is None else unit
    return ParsedNumericCell(value, unit, config.config_sha256)


def extract_table_quantity(
    document: StructuredDocument,
    *,
    table_region_id: str,
    cell_id: str,
    config: NumericCellParserConfig = NumericCellParserConfig(),
    unit_override: str | None = None,
) -> TableQuantityEvidence:
    table = next((item for item in document.tables if item.table_region_id == table_region_id), None)
    if table is None:
        raise KeyError(table_region_id)
    cell = next((item for item in table.cells if item.cell_id == cell_id), None)
    if cell is None:
        raise KeyError(cell_id)
    parsed = parse_numeric_cell(cell.text, config=config, unit_override=unit_override)
    return table_quantity_evidence(
        document,
        table_region_id=table_region_id,
        cell_id=cell_id,
        quantity=Quantity(parsed.value, parsed.unit, confidence=cell.confidence),
        value_extraction_sha256=parsed.parser_sha256,
    )


__all__ = ["NumericCellParserConfig", "ParsedNumericCell", "extract_table_quantity", "parse_numeric_cell"]
