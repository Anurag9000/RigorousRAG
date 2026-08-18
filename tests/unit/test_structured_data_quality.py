from __future__ import annotations

import hashlib

from evaluation.structured_data_support import Quantity, chart_point_evidence, table_quantity_evidence
from scientific.chart_structure import AxisDimension, AxisScale, ChartAxis, ChartPoint, ChartSeries, SeriesKind, StructuredChart
from scientific.document_structure import BoundingBox, DocumentRegion, RegionKind, SourceAnchor, StructuredDocument, StructuredTable, TableCell
from scientific.structured_data_quality import StructuredDataAuthorityPolicy, evaluate_chart_quantity_authority, evaluate_chart_trend_authority, evaluate_table_quantity_authority


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def table_doc(*, confidence=0.95, text="10.0"):
    anchor = SourceAnchor("doc", "gen", 1, "layout")
    region = DocumentRegion("table", RegionKind.TABLE, BoundingBox(0.1, 0.1, 0.9, 0.4), anchor)
    table = StructuredTable("table", (TableCell("cell", "table", 0, 1, 0, 1, text, confidence=confidence),), 1, 1)
    return StructuredDocument("doc", "gen", (region,), tables=(table,))


def chart(*, extraction_confidence=0.95, axis_confidence=0.95, point_confidence=0.95):
    anchor = SourceAnchor("doc", "gen", 1, "chart-extract")
    x = ChartAxis("x", AxisDimension.X, "Time", "s", AxisScale.LINEAR, confidence=axis_confidence)
    y = ChartAxis("y", AxisDimension.Y, "Flow", "m3/s", AxisScale.LINEAR, confidence=axis_confidence)
    points = (
        ChartPoint("p0", "series", x_numeric=0.0, y=1.0, y_lower=0.9, y_upper=1.1, confidence=point_confidence),
        ChartPoint("p1", "series", x_numeric=1.0, y=2.0, y_lower=1.9, y_upper=2.1, confidence=point_confidence),
    )
    series = ChartSeries("series", "Observed", SeriesKind.LINE, "x", "y", points)
    return StructuredChart("figure", anchor, sha("extractor"), (x, y), (series,), extraction_confidence=extraction_confidence)


def policy(**overrides):
    values = dict(min_chart_extraction_confidence=0.8, min_axis_confidence=0.8, min_point_confidence=0.8, min_table_cell_confidence=0.8, max_relative_interval_width=1.0, require_explicit_unit=True, missing_confidence_requires_review=True)
    values.update(overrides)
    return StructuredDataAuthorityPolicy(**values)


def test_high_confidence_table_quantity_is_authoritative() -> None:
    document = table_doc()
    evidence = table_quantity_evidence(document, table_region_id="table", cell_id="cell", quantity=Quantity(10.0, "m"), value_extraction_sha256=sha("parser"))
    decision = evaluate_table_quantity_authority(document, evidence, policy=policy())
    assert decision.action == "authoritative"
    assert decision.reason_codes == ()


def test_low_confidence_table_cell_requires_review() -> None:
    document = table_doc(confidence=0.4)
    evidence = table_quantity_evidence(document, table_region_id="table", cell_id="cell", quantity=Quantity(10.0, "m"), value_extraction_sha256=sha("parser"))
    decision = evaluate_table_quantity_authority(document, evidence, policy=policy())
    assert decision.action == "review_required"
    assert "cell_confidence_below_threshold" in decision.reason_codes


def test_table_lineage_change_blocks_old_evidence() -> None:
    original = table_doc(text="10.0")
    evidence = table_quantity_evidence(original, table_region_id="table", cell_id="cell", quantity=Quantity(10.0, "m"), value_extraction_sha256=sha("parser"))
    changed = table_doc(text="11.0")
    decision = evaluate_table_quantity_authority(changed, evidence, policy=policy())
    assert decision.action == "blocked"
    assert "lineage_mismatch" in decision.reason_codes


def test_high_confidence_chart_quantity_is_authoritative() -> None:
    value = chart()
    evidence = chart_point_evidence(value, series_id="series", point_id="p1")
    decision = evaluate_chart_quantity_authority(value, evidence, policy=policy())
    assert decision.action == "authoritative"


def test_low_chart_or_point_confidence_requires_review() -> None:
    value = chart(extraction_confidence=0.5, point_confidence=0.4)
    evidence = chart_point_evidence(value, series_id="series", point_id="p1")
    decision = evaluate_chart_quantity_authority(value, evidence, policy=policy())
    assert decision.action == "review_required"
    assert "chart_extraction_confidence_below_threshold" in decision.reason_codes
    assert "point_confidence_below_threshold" in decision.reason_codes


def test_wide_numeric_uncertainty_requires_review() -> None:
    document = table_doc()
    evidence = table_quantity_evidence(document, table_region_id="table", cell_id="cell", quantity=Quantity(10.0, "m", 0.0, 20.0), value_extraction_sha256=sha("parser"))
    decision = evaluate_table_quantity_authority(document, evidence, policy=policy(max_relative_interval_width=0.5))
    assert decision.action == "review_required"
    assert "uncertainty_interval_too_wide" in decision.reason_codes


def test_chart_trend_authority_checks_every_series_point() -> None:
    value = chart(point_confidence=0.3)
    decision = evaluate_chart_trend_authority(value, series_id="series", policy=policy())
    assert decision.action == "review_required"
    assert "point_confidence_below_threshold" in decision.reason_codes
