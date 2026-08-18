"""Deterministic entailment for provenance-bound table and chart quantities.

The symbolic path is deliberately conservative: it reasons only over typed values with
exact table/chart anchors.  Unit mismatches, overlapping uncertainty intervals, missing
values or mixed trends return neutral/abstention rather than guessed support.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Protocol, Sequence

from evaluation.semantic_support import ModelIdentity, SemanticLabel, SemanticProbabilities
from scientific.chart_structure import ChartPoint, StructuredChart
from scientific.document_structure import StructuredDocument, TableCell


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _identifier(value: Any, label: str, maximum: int = 2000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


def _nonnegative(value: Any, label: str) -> float:
    selected = _finite(value, label)
    if selected < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return selected


def _unit(value: str | None) -> str | None:
    if value is None:
        return None
    selected = _identifier(value, "unit", 500)
    return selected


SYMBOLIC_MODEL = ModelIdentity("rigorousrag", "structured-data-symbolic-entailment", "v1")
_ENTAIL = SemanticProbabilities(1.0, 0.0, 0.0)
_NEUTRAL = SemanticProbabilities(0.0, 1.0, 0.0)
_CONTRADICT = SemanticProbabilities(0.0, 0.0, 1.0)


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str | None = None
    lower: float | None = None
    upper: float | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        value = _finite(self.value, "quantity value")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "unit", _unit(self.unit))
        lower = value if self.lower is None else _finite(self.lower, "quantity lower")
        upper = value if self.upper is None else _finite(self.upper, "quantity upper")
        if lower > value or upper < value or lower > upper:
            raise ValueError("quantity interval must contain its value")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        if self.confidence is not None:
            confidence = _finite(self.confidence, "quantity confidence")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("quantity confidence must be in [0, 1]")
            object.__setattr__(self, "confidence", confidence)


class UnitConverter(Protocol):
    def convert(self, value: float, *, from_unit: str, to_unit: str) -> float | None: ...


def _convert_quantity(quantity: Quantity, target_unit: str | None, converter: UnitConverter | None) -> Quantity | None:
    if quantity.unit == target_unit:
        return quantity
    if quantity.unit is None or target_unit is None or converter is None:
        return None
    values = []
    for raw in (quantity.value, quantity.lower, quantity.upper):
        converted = converter.convert(float(raw), from_unit=quantity.unit, to_unit=target_unit)
        if converted is None:
            return None
        values.append(_finite(converted, "converted quantity"))
    return Quantity(values[0], target_unit, values[1], values[2], quantity.confidence)


@dataclass(frozen=True)
class TableQuantityEvidence:
    document_id: str
    generation_id: str
    table_region_id: str
    cell_id: str
    row_start: int
    column_start: int
    cell_text_sha256: str
    value_extraction_sha256: str
    quantity: Quantity

    def __post_init__(self) -> None:
        for name in ("document_id", "generation_id", "table_region_id", "cell_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in ("row_start", "column_start"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("cell_text_sha256", "value_extraction_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if not isinstance(self.quantity, Quantity):
            raise ValueError("quantity must be Quantity")

    @property
    def evidence_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-table-quantity-evidence/v1", **asdict(self)})


def table_quantity_evidence(
    document: StructuredDocument,
    *,
    table_region_id: str,
    cell_id: str,
    quantity: Quantity,
    value_extraction_sha256: str,
) -> TableQuantityEvidence:
    if not isinstance(document, StructuredDocument):
        raise ValueError("document must be StructuredDocument")
    table_id = _identifier(table_region_id, "table_region_id")
    selected_cell = _identifier(cell_id, "cell_id")
    table = next((item for item in document.tables if item.table_region_id == table_id), None)
    if table is None:
        raise KeyError(table_id)
    cell = next((item for item in table.cells if item.cell_id == selected_cell), None)
    if cell is None:
        raise KeyError(selected_cell)
    region = next((item for item in document.regions if item.region_id == table_id), None)
    if region is None:
        raise RuntimeError("structured table region is missing from document regions")
    return TableQuantityEvidence(
        document.document_id,
        document.generation_id,
        table_id,
        selected_cell,
        cell.row_start,
        cell.column_start,
        hashlib.sha256(cell.text.encode("utf-8")).hexdigest(),
        value_extraction_sha256,
        quantity,
    )


@dataclass(frozen=True)
class ChartQuantityEvidence:
    document_id: str
    generation_id: str
    chart_region_id: str
    chart_sha256: str
    series_id: str
    point_id: str
    point_index: int
    quantity: Quantity

    def __post_init__(self) -> None:
        for name in ("document_id", "generation_id", "chart_region_id", "series_id", "point_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "chart_sha256", _sha(self.chart_sha256, "chart_sha256"))
        if isinstance(self.point_index, bool) or not isinstance(self.point_index, int) or self.point_index < 0:
            raise ValueError("point_index must be non-negative")
        if not isinstance(self.quantity, Quantity):
            raise ValueError("quantity must be Quantity")

    @property
    def evidence_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-chart-quantity-evidence/v1", **asdict(self)})


def chart_point_evidence(chart: StructuredChart, *, series_id: str, point_id: str) -> ChartQuantityEvidence:
    if not isinstance(chart, StructuredChart):
        raise ValueError("chart must be StructuredChart")
    series = chart.series_by_id(series_id)
    selected_point = _identifier(point_id, "point_id")
    point_index = next((index for index, item in enumerate(series.points) if item.point_id == selected_point), None)
    if point_index is None:
        raise KeyError(selected_point)
    point: ChartPoint = series.points[point_index]
    y_axis = chart.axis(series.y_axis_id)
    return ChartQuantityEvidence(
        chart.anchor.document_id,
        chart.anchor.generation_id,
        chart.chart_region_id,
        chart.chart_sha256,
        series.series_id,
        point.point_id,
        point_index,
        Quantity(point.y, y_axis.unit, point.y_lower, point.y_upper, point.confidence),
    )


class NumericOperator(str, Enum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GE = "ge"
    LT = "lt"
    LE = "le"
    BETWEEN = "between"


@dataclass(frozen=True)
class NumericClaim:
    claim_id: str
    claim_sha256: str
    operator: NumericOperator
    unit: str | None
    value: float | None = None
    lower: float | None = None
    upper: float | None = None
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _identifier(self.claim_id, "claim_id"))
        object.__setattr__(self, "claim_sha256", _sha(self.claim_sha256, "claim_sha256"))
        if not isinstance(self.operator, NumericOperator):
            object.__setattr__(self, "operator", NumericOperator(self.operator))
        object.__setattr__(self, "unit", _unit(self.unit))
        if self.operator is NumericOperator.BETWEEN:
            if self.lower is None or self.upper is None or self.value is not None:
                raise ValueError("between claim requires lower/upper and no value")
            low, high = _finite(self.lower, "claim lower"), _finite(self.upper, "claim upper")
            if low > high:
                raise ValueError("claim interval is inverted")
            object.__setattr__(self, "lower", low)
            object.__setattr__(self, "upper", high)
        else:
            if self.value is None or self.lower is not None or self.upper is not None:
                raise ValueError("scalar numeric claim requires value only")
            object.__setattr__(self, "value", _finite(self.value, "claim value"))
        object.__setattr__(self, "absolute_tolerance", _nonnegative(self.absolute_tolerance, "absolute_tolerance"))
        object.__setattr__(self, "relative_tolerance", _nonnegative(self.relative_tolerance, "relative_tolerance"))


@dataclass(frozen=True)
class StructuredSupportScore:
    claim_id: str
    claim_sha256: str
    evidence_sha256: str
    probabilities: SemanticProbabilities
    model: ModelIdentity
    reason_code: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _identifier(self.claim_id, "claim_id"))
        object.__setattr__(self, "claim_sha256", _sha(self.claim_sha256, "claim_sha256"))
        object.__setattr__(self, "evidence_sha256", _sha(self.evidence_sha256, "evidence_sha256"))
        if not isinstance(self.probabilities, SemanticProbabilities) or not isinstance(self.model, ModelIdentity):
            raise ValueError("probabilities/model types are invalid")
        object.__setattr__(self, "reason_code", _identifier(self.reason_code, "reason_code", 500))

    @property
    def label(self) -> SemanticLabel:
        return self.probabilities.predicted_label


def _score(claim: NumericClaim, evidence_sha256: str, probabilities: SemanticProbabilities, reason: str) -> StructuredSupportScore:
    return StructuredSupportScore(claim.claim_id, claim.claim_sha256, evidence_sha256, probabilities, SYMBOLIC_MODEL, reason)


def evaluate_numeric_claim(
    claim: NumericClaim,
    evidence: TableQuantityEvidence | ChartQuantityEvidence,
    *,
    converter: UnitConverter | None = None,
) -> StructuredSupportScore:
    if not isinstance(claim, NumericClaim) or not isinstance(evidence, (TableQuantityEvidence, ChartQuantityEvidence)):
        raise ValueError("numeric claim/evidence types are invalid")
    quantity = _convert_quantity(evidence.quantity, claim.unit, converter)
    if quantity is None:
        return _score(claim, evidence.evidence_sha256, _NEUTRAL, "unit_incomparable")
    low, high, value = float(quantity.lower), float(quantity.upper), quantity.value
    if claim.operator is NumericOperator.BETWEEN:
        lower, upper = float(claim.lower), float(claim.upper)
        if low >= lower and high <= upper:
            return _score(claim, evidence.evidence_sha256, _ENTAIL, "interval_inside_claim_range")
        if high < lower or low > upper:
            return _score(claim, evidence.evidence_sha256, _CONTRADICT, "interval_outside_claim_range")
        return _score(claim, evidence.evidence_sha256, _NEUTRAL, "uncertainty_overlaps_claim_boundary")
    expected = float(claim.value)
    tolerance = max(claim.absolute_tolerance, abs(expected) * claim.relative_tolerance)
    if claim.operator is NumericOperator.EQ:
        allowed_low, allowed_high = expected - tolerance, expected + tolerance
        if low >= allowed_low and high <= allowed_high:
            return _score(claim, evidence.evidence_sha256, _ENTAIL, "quantity_within_equality_tolerance")
        if high < allowed_low or low > allowed_high:
            return _score(claim, evidence.evidence_sha256, _CONTRADICT, "quantity_outside_equality_tolerance")
        return _score(claim, evidence.evidence_sha256, _NEUTRAL, "uncertainty_crosses_equality_tolerance")
    if claim.operator is NumericOperator.NE:
        eq_claim = NumericClaim(claim.claim_id, claim.claim_sha256, NumericOperator.EQ, claim.unit, expected, absolute_tolerance=claim.absolute_tolerance, relative_tolerance=claim.relative_tolerance)
        result = evaluate_numeric_claim(eq_claim, evidence, converter=converter)
        probabilities = _CONTRADICT if result.label is SemanticLabel.ENTAILMENT else _ENTAIL if result.label is SemanticLabel.CONTRADICTION else _NEUTRAL
        return _score(claim, evidence.evidence_sha256, probabilities, "negated_equality")
    if claim.operator is NumericOperator.GT:
        probabilities = _ENTAIL if low > expected else _CONTRADICT if high <= expected else _NEUTRAL
    elif claim.operator is NumericOperator.GE:
        probabilities = _ENTAIL if low >= expected else _CONTRADICT if high < expected else _NEUTRAL
    elif claim.operator is NumericOperator.LT:
        probabilities = _ENTAIL if high < expected else _CONTRADICT if low >= expected else _NEUTRAL
    else:
        probabilities = _ENTAIL if high <= expected else _CONTRADICT if low > expected else _NEUTRAL
    reason = "interval_proves_predicate" if probabilities is _ENTAIL else "interval_refutes_predicate" if probabilities is _CONTRADICT else "uncertainty_crosses_predicate_boundary"
    return _score(claim, evidence.evidence_sha256, probabilities, reason)


class AggregateKind(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    COUNT = "count"


@dataclass(frozen=True)
class DerivedQuantityEvidence:
    evidence_kind: str
    source_evidence_sha256s: tuple[str, ...]
    operation: AggregateKind
    quantity: Quantity
    evidence_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_kind", _identifier(self.evidence_kind, "evidence_kind"))
        sources = tuple(sorted(_sha(value, "source evidence sha256") for value in self.source_evidence_sha256s))
        if not sources:
            raise ValueError("derived evidence requires source evidence")
        object.__setattr__(self, "source_evidence_sha256s", sources)
        if not isinstance(self.operation, AggregateKind):
            object.__setattr__(self, "operation", AggregateKind(self.operation))
        if not isinstance(self.quantity, Quantity):
            raise ValueError("quantity must be Quantity")
        expected = _digest({"schema": "rigorousrag-derived-quantity-evidence/v1", "evidence_kind": self.evidence_kind, "sources": sources, "operation": self.operation.value, "quantity": asdict(self.quantity)})
        provided = _sha(self.evidence_sha256, "evidence_sha256")
        if expected != provided:
            raise ValueError("derived evidence digest mismatch")
        object.__setattr__(self, "evidence_sha256", provided)


def aggregate_quantities(
    evidences: Sequence[TableQuantityEvidence | ChartQuantityEvidence],
    operation: AggregateKind,
    *,
    target_unit: str | None = None,
    converter: UnitConverter | None = None,
) -> DerivedQuantityEvidence:
    values = tuple(evidences)
    if not values or len(values) > 1_000_000:
        raise ValueError("aggregate evidence must be non-empty and bounded")
    selected_operation = operation if isinstance(operation, AggregateKind) else AggregateKind(operation)
    converted: list[Quantity] = []
    if selected_operation is AggregateKind.COUNT:
        quantity = Quantity(float(len(values)), None)
    else:
        desired_unit = target_unit if target_unit is not None else values[0].quantity.unit
        for evidence in values:
            value = _convert_quantity(evidence.quantity, desired_unit, converter)
            if value is None:
                raise ValueError("aggregate quantities use incomparable units")
            converted.append(value)
        point_values = [value.value for value in converted]
        if selected_operation is AggregateKind.SUM:
            quantity = Quantity(sum(point_values), desired_unit, sum(float(value.lower) for value in converted), sum(float(value.upper) for value in converted))
        elif selected_operation is AggregateKind.MEAN:
            count = len(converted)
            quantity = Quantity(sum(point_values) / count, desired_unit, sum(float(value.lower) for value in converted) / count, sum(float(value.upper) for value in converted) / count)
        elif selected_operation is AggregateKind.MIN:
            index = min(range(len(converted)), key=lambda position: converted[position].value)
            quantity = converted[index]
        else:
            index = max(range(len(converted)), key=lambda position: converted[position].value)
            quantity = converted[index]
    sources = tuple(sorted(evidence.evidence_sha256 for evidence in values))
    payload = {"schema": "rigorousrag-derived-quantity-evidence/v1", "evidence_kind": "aggregate", "sources": sources, "operation": selected_operation.value, "quantity": asdict(quantity)}
    return DerivedQuantityEvidence("aggregate", sources, selected_operation, quantity, _digest(payload))


class TrendDirection(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    NONDECREASING = "nondecreasing"
    NONINCREASING = "nonincreasing"
    FLAT = "flat"


@dataclass(frozen=True)
class ChartTrendClaim:
    claim_id: str
    claim_sha256: str
    series_id: str
    direction: TrendDirection
    absolute_tolerance: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _identifier(self.claim_id, "claim_id"))
        object.__setattr__(self, "claim_sha256", _sha(self.claim_sha256, "claim_sha256"))
        object.__setattr__(self, "series_id", _identifier(self.series_id, "series_id"))
        if not isinstance(self.direction, TrendDirection):
            object.__setattr__(self, "direction", TrendDirection(self.direction))
        object.__setattr__(self, "absolute_tolerance", _nonnegative(self.absolute_tolerance, "absolute_tolerance"))


def evaluate_chart_trend(claim: ChartTrendClaim, chart: StructuredChart) -> StructuredSupportScore:
    if not isinstance(claim, ChartTrendClaim) or not isinstance(chart, StructuredChart):
        raise ValueError("chart trend claim/chart types are invalid")
    series = chart.series_by_id(claim.series_id)
    if any(point.x_numeric is None for point in series.points):
        return StructuredSupportScore(claim.claim_id, claim.claim_sha256, chart.chart_sha256, _NEUTRAL, SYMBOLIC_MODEL, "categorical_x_order_not_assumed")
    points = sorted(series.points, key=lambda point: float(point.x_numeric))
    if len({point.x_numeric for point in points}) != len(points):
        return StructuredSupportScore(claim.claim_id, claim.claim_sha256, chart.chart_sha256, _NEUTRAL, SYMBOLIC_MODEL, "duplicate_x_coordinates")
    tolerance = claim.absolute_tolerance
    differences = [points[index + 1].y - points[index].y for index in range(len(points) - 1)]
    if not differences:
        return StructuredSupportScore(claim.claim_id, claim.claim_sha256, chart.chart_sha256, _NEUTRAL, SYMBOLIC_MODEL, "insufficient_points_for_trend")
    if claim.direction is TrendDirection.INCREASING:
        entailed = all(value > tolerance for value in differences)
        contradicted = all(value <= tolerance for value in differences)
    elif claim.direction is TrendDirection.DECREASING:
        entailed = all(value < -tolerance for value in differences)
        contradicted = all(value >= -tolerance for value in differences)
    elif claim.direction is TrendDirection.NONDECREASING:
        entailed = all(value >= -tolerance for value in differences)
        contradicted = any(value < -tolerance for value in differences)
    elif claim.direction is TrendDirection.NONINCREASING:
        entailed = all(value <= tolerance for value in differences)
        contradicted = any(value > tolerance for value in differences)
    else:
        entailed = all(abs(value) <= tolerance for value in differences)
        contradicted = all(abs(value) > tolerance for value in differences)
    probabilities = _ENTAIL if entailed else _CONTRADICT if contradicted else _NEUTRAL
    reason = "series_proves_trend" if entailed else "series_refutes_trend" if contradicted else "series_has_mixed_trend"
    return StructuredSupportScore(claim.claim_id, claim.claim_sha256, chart.chart_sha256, probabilities, SYMBOLIC_MODEL, reason)


__all__ = [
    "AggregateKind",
    "ChartQuantityEvidence",
    "ChartTrendClaim",
    "DerivedQuantityEvidence",
    "NumericClaim",
    "NumericOperator",
    "Quantity",
    "StructuredSupportScore",
    "TableQuantityEvidence",
    "TrendDirection",
    "UnitConverter",
    "aggregate_quantities",
    "chart_point_evidence",
    "evaluate_chart_trend",
    "evaluate_numeric_claim",
    "table_quantity_evidence",
]
