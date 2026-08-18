from __future__ import annotations

import hashlib

import pytest

from evaluation.semantic_support import SemanticLabel
from evaluation.structured_data_support import (
    AggregateKind,
    ChartTrendClaim,
    NumericClaim,
    NumericOperator,
    Quantity,
    TrendDirection,
    aggregate_quantities,
    chart_point_evidence,
    evaluate_chart_trend,
    evaluate_numeric_claim,
    table_quantity_evidence,
)
from scientific.chart_structure import AxisDimension, AxisScale, ChartAxis, ChartPoint, ChartSeries, SeriesKind, StructuredChart
from scientific.document_structure import BoundingBox, DocumentRegion, RegionKind, SourceAnchor, StructuredDocument, StructuredTable, TableCell


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def chart(*, categorical: bool = False) -> StructuredChart:
    anchor = SourceAnchor("doc", "gen", 1, "extract")
    x_axis = ChartAxis("x", AxisDimension.X, "Time", None, AxisScale.CATEGORY if categorical else AxisScale.LINEAR)
    y_axis = ChartAxis("y", AxisDimension.Y, "Flow", "m3/s", AxisScale.LINEAR)
    points = tuple(
        ChartPoint(
            f"p-{index}",
            "series",
            x_category=f"t{index}" if categorical else None,
            x_numeric=None if categorical else float(index),
            y=float(value),
            y_lower=float(value) - 0.1,
            y_upper=float(value) + 0.1,
        )
        for index, value in enumerate((1.0, 2.0, 3.0))
    )
    series = ChartSeries("series", "Observed flow", SeriesKind.LINE, "x", "y", points)
    return StructuredChart("figure-1", anchor, sha("chart-extractor"), (x_axis, y_axis), (series,), ("caption-1",), 0.95)


def table_document() -> StructuredDocument:
    anchor = SourceAnchor("doc", "gen", 1, "layout")
    region = DocumentRegion("table-1", RegionKind.TABLE, BoundingBox(0.1, 0.1, 0.9, 0.5), anchor)
    cells = (
        TableCell("c1", "table-1", 0, 1, 0, 1, "10.0"),
        TableCell("c2", "table-1", 0, 1, 1, 1, "20.0"),
    )
    table = StructuredTable("table-1", cells, 1, 2)
    return StructuredDocument("doc", "gen", (region,), tables=(table,))


def claim(operator, *, value=None, lower=None, upper=None, unit="m3/s", tolerance=0.0):
    return NumericClaim("claim", sha("claim"), operator, unit, value=value, lower=lower, upper=upper, absolute_tolerance=tolerance)


def test_chart_ir_is_content_addressed_and_point_evidence_inherits_axis_unit() -> None:
    value = chart()
    assert len(value.chart_sha256) == 64
    evidence = chart_point_evidence(value, series_id="series", point_id="p-1")
    assert evidence.quantity.value == 2.0
    assert evidence.quantity.unit == "m3/s"
    assert evidence.quantity.lower == pytest.approx(1.9)
    assert len(evidence.evidence_sha256) == 64


def test_interval_aware_numeric_support_entails_refutes_or_abstains_conservatively() -> None:
    evidence = chart_point_evidence(chart(), series_id="series", point_id="p-1")
    assert evaluate_numeric_claim(claim(NumericOperator.GT, value=1.5), evidence).label is SemanticLabel.ENTAILMENT
    assert evaluate_numeric_claim(claim(NumericOperator.LT, value=1.5), evidence).label is SemanticLabel.CONTRADICTION
    assert evaluate_numeric_claim(claim(NumericOperator.GT, value=2.0), evidence).label is SemanticLabel.NEUTRAL


def test_equality_uses_explicit_tolerance_and_uncertainty() -> None:
    evidence = chart_point_evidence(chart(), series_id="series", point_id="p-1")
    assert evaluate_numeric_claim(claim(NumericOperator.EQ, value=2.0, tolerance=0.2), evidence).label is SemanticLabel.ENTAILMENT
    assert evaluate_numeric_claim(claim(NumericOperator.EQ, value=5.0, tolerance=0.2), evidence).label is SemanticLabel.CONTRADICTION


def test_unit_mismatch_is_neutral_without_explicit_converter() -> None:
    evidence = chart_point_evidence(chart(), series_id="series", point_id="p-1")
    result = evaluate_numeric_claim(claim(NumericOperator.GT, value=1.0, unit="ft3/s"), evidence)
    assert result.label is SemanticLabel.NEUTRAL
    assert result.reason_code == "unit_incomparable"


def test_table_quantities_bind_cell_text_and_extractor_identity_and_can_aggregate() -> None:
    document = table_document()
    left = table_quantity_evidence(document, table_region_id="table-1", cell_id="c1", quantity=Quantity(10.0, "m"), value_extraction_sha256=sha("value-parser"))
    right = table_quantity_evidence(document, table_region_id="table-1", cell_id="c2", quantity=Quantity(20.0, "m"), value_extraction_sha256=sha("value-parser"))
    assert left.cell_text_sha256 == sha("10.0")
    total = aggregate_quantities((left, right), AggregateKind.SUM)
    mean = aggregate_quantities((left, right), AggregateKind.MEAN)
    assert total.quantity.value == 30.0
    assert mean.quantity.value == 15.0
    assert set(total.source_evidence_sha256s) == {left.evidence_sha256, right.evidence_sha256}


def test_aggregate_rejects_incomparable_units_without_converter() -> None:
    document = table_document()
    left = table_quantity_evidence(document, table_region_id="table-1", cell_id="c1", quantity=Quantity(10.0, "m"), value_extraction_sha256=sha("parser"))
    right = table_quantity_evidence(document, table_region_id="table-1", cell_id="c2", quantity=Quantity(20.0, "s"), value_extraction_sha256=sha("parser"))
    with pytest.raises(ValueError, match="incomparable units"):
        aggregate_quantities((left, right), AggregateKind.SUM)


def test_numeric_x_chart_can_prove_increasing_trend() -> None:
    result = evaluate_chart_trend(ChartTrendClaim("trend", sha("trend"), "series", TrendDirection.INCREASING, 0.0), chart())
    assert result.label is SemanticLabel.ENTAILMENT
    assert result.reason_code == "series_proves_trend"


def test_categorical_x_chart_does_not_invent_order_semantics() -> None:
    result = evaluate_chart_trend(ChartTrendClaim("trend", sha("trend"), "series", TrendDirection.INCREASING, 0.0), chart(categorical=True))
    assert result.label is SemanticLabel.NEUTRAL
    assert result.reason_code == "categorical_x_order_not_assumed"


def test_chart_point_rejects_inverted_uncertainty() -> None:
    with pytest.raises(ValueError, match="y_lower"):
        ChartPoint("bad", "series", x_numeric=1.0, y=2.0, y_lower=3.0, y_upper=4.0)
