from __future__ import annotations

import hashlib

import pytest

from scientific.chart_structure import AxisScale, SeriesKind
from scientific.document_structure import BoundingBox, DocumentRegion, RegionKind, SourceAnchor, StructuredDocument, StructuredTable, TableCell
from scientific.local_chart_adapters import ChartDecodeConfig, decode_chart_text
from scientific.table_value_adapters import NumericCellParserConfig, extract_table_quantity, parse_numeric_cell


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def anchor() -> SourceAnchor:
    return SourceAnchor("doc", "gen", 1, "layout")


def document(cell_text: str) -> StructuredDocument:
    region = DocumentRegion("table", RegionKind.TABLE, BoundingBox(0.1, 0.1, 0.9, 0.4), anchor())
    table = StructuredTable("table", (TableCell("cell", "table", 0, 1, 0, 1, cell_text, confidence=0.9),), 1, 1)
    return StructuredDocument("doc", "gen", (region,), tables=(table,))


def test_json_chart_decoder_accepts_only_closed_schema_and_builds_chart_ir() -> None:
    payload = """{"axes":[{"axis_id":"x","dimension":"x","label":"Time","unit":"s","scale":"linear"},{"axis_id":"y","dimension":"y","label":"Flow","unit":"m3/s","scale":"linear"}],"series":[{"series_id":"observed","label":"Observed","kind":"line","x_axis_id":"x","y_axis_id":"y","points":[{"point_id":"p1","x_numeric":0,"y":1.0},{"point_id":"p2","x_numeric":1,"y":2.0}]}]}"""
    chart = decode_chart_text(payload, chart_region_id="figure", anchor=anchor(), extraction_artifact_sha256=sha("extractor"), config=ChartDecodeConfig(contract="json_v1"))
    assert chart.axis("y").unit == "m3/s"
    assert chart.series_by_id("observed").kind is SeriesKind.LINE
    assert len(chart.series_by_id("observed").points) == 2


def test_json_chart_decoder_rejects_unknown_fields_instead_of_ignoring_them() -> None:
    payload = """{"axes":[],"series":[],"free_form_answer":"trust me"}"""
    with pytest.raises(ValueError, match="unsupported fields"):
        decode_chart_text(payload, chart_region_id="figure", anchor=anchor(), extraction_artifact_sha256=sha("extractor"), config=ChartDecodeConfig(contract="json_v1"))


def test_tabular_chart_decoder_supports_deplot_style_output() -> None:
    text = "Time\tObserved\tForecast\n0\t1.0\t1.2\n1\t2.0\t2.1\n"
    chart = decode_chart_text(
        text,
        chart_region_id="figure",
        anchor=anchor(),
        extraction_artifact_sha256=sha("extractor"),
        config=ChartDecodeConfig(contract="tabular_v1", x_label="Time", x_unit="s", y_label="Flow", y_unit="m3/s", x_scale=AxisScale.LINEAR, default_series_kind=SeriesKind.LINE),
    )
    assert len(chart.series) == 2
    assert chart.series[0].points[1].x_numeric == 1.0
    assert chart.series[0].points[1].y == 2.0


def test_tabular_chart_decoder_rejects_mixed_numeric_and_categorical_x_values() -> None:
    text = "x\tSeries\n0\t1\nJanuary\t2\n"
    with pytest.raises(ValueError, match="may not mix"):
        decode_chart_text(text, chart_region_id="figure", anchor=anchor(), extraction_artifact_sha256=sha("extractor"), config=ChartDecodeConfig(contract="tabular_v1"))


def test_numeric_cell_parser_validates_locale_grouping_and_units() -> None:
    parsed = parse_numeric_cell("1,234.50 m3/s")
    assert parsed.value == 1234.5
    assert parsed.unit == "m3/s"
    with pytest.raises(ValueError, match="thousands grouping"):
        parse_numeric_cell("12,34.50 m3/s")


def test_numeric_cell_parser_handles_accounting_and_percent_semantics_explicitly() -> None:
    assert parse_numeric_cell("(1,200)").value == -1200.0
    percent = parse_numeric_cell("25%", config=NumericCellParserConfig(percent_as_fraction=True))
    assert percent.value == 0.25
    assert percent.unit is None


def test_numeric_cell_parser_rejects_ranges_comparisons_and_conflicting_units() -> None:
    with pytest.raises(ValueError, match="comparison/range|numeric scalar"):
        parse_numeric_cell("< 5")
    with pytest.raises(ValueError, match="numeric scalar"):
        parse_numeric_cell("10-20")
    with pytest.raises(ValueError, match="conflicts"):
        parse_numeric_cell("10 m", unit_override="s")


def test_table_quantity_extraction_binds_parser_digest_and_cell_confidence() -> None:
    evidence = extract_table_quantity(document("1,234.5 m3/s"), table_region_id="table", cell_id="cell")
    assert evidence.quantity.value == 1234.5
    assert evidence.quantity.unit == "m3/s"
    assert evidence.quantity.confidence == pytest.approx(0.9)
    assert evidence.value_extraction_sha256 == NumericCellParserConfig().config_sha256
    assert evidence.cell_text_sha256 == sha("1,234.5 m3/s")
