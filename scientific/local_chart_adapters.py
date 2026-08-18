"""Verified-local chart-to-structure adapters with closed decoding contracts.

Models may emit either the repository JSON chart schema or a simple chart-to-table TSV
representation.  The adapter performs no network access: model/processor artifacts are
verified before local-only loading and remote code is disabled.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from models.local_hf_adapters import LocalArtifactBinding
from scientific.chart_structure import AxisDimension, AxisScale, ChartAxis, ChartPoint, ChartSeries, SeriesKind, StructuredChart
from scientific.document_structure import SourceAnchor

_MAX_OUTPUT_CHARS = 5_000_000
_MAX_POINTS = 1_000_000


def _text(value: Any, label: str, maximum: int = 20_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or "\x00" in selected:
        raise ValueError(f"{label} is invalid")
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


def _device(torch_module: Any, requested: str) -> Any:
    selected = requested.strip().lower()
    if selected == "auto":
        selected = "cuda" if torch_module.cuda.is_available() else "cpu"
    device = torch_module.device(selected)
    if device.type == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


@dataclass(frozen=True)
class ChartDecodeConfig:
    contract: str = "json_v1"
    max_new_tokens: int = 2048
    x_axis_id: str = "x"
    y_axis_id: str = "y"
    x_label: str | None = None
    x_unit: str | None = None
    y_label: str | None = None
    y_unit: str | None = None
    x_scale: AxisScale = AxisScale.UNKNOWN
    y_scale: AxisScale = AxisScale.UNKNOWN
    default_series_kind: SeriesKind = SeriesKind.OTHER

    def __post_init__(self) -> None:
        if self.contract not in {"json_v1", "tabular_v1"}:
            raise ValueError("chart decoding contract must be json_v1 or tabular_v1")
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int) or not 1 <= self.max_new_tokens <= 65_536:
            raise ValueError("max_new_tokens must be in [1, 65536]")
        for name in ("x_axis_id", "y_axis_id"):
            _text(getattr(self, name), name, 1000)
        if not isinstance(self.x_scale, AxisScale):
            object.__setattr__(self, "x_scale", AxisScale(self.x_scale))
        if not isinstance(self.y_scale, AxisScale):
            object.__setattr__(self, "y_scale", AxisScale(self.y_scale))
        if not isinstance(self.default_series_kind, SeriesKind):
            object.__setattr__(self, "default_series_kind", SeriesKind(self.default_series_kind))


class LocalHFChartToStructureAdapter:
    def __init__(self, binding: LocalArtifactBinding, *, config: ChartDecodeConfig = ChartDecodeConfig(), device: str = "auto") -> None:
        binding.verify()
        self.binding = binding
        self.config = config
        self.device_name = device
        self._processor: Any | None = None
        self._model: Any | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        try:
            import torch
            from transformers import AutoProcessor, VisionEncoderDecoderModel
        except Exception as exc:  # pragma: no cover - optional scientific dependency.
            raise RuntimeError("chart-to-structure execution requires torch + transformers") from exc
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.binding.tokenizer_root, local_files_only=True, trust_remote_code=False)
        if self._model is None:
            self._model = VisionEncoderDecoderModel.from_pretrained(self.binding.model_root, local_files_only=True, trust_remote_code=False)
            self._model.to(_device(torch, self.device_name))
            self._model.eval()
        return torch, self._processor, self._model

    def generate_text(self, image: Any) -> str:
        torch, processor, model = self._load()
        device = _device(torch, self.device_name)
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = getattr(inputs, "pixel_values", None)
        if pixel_values is None and isinstance(inputs, Mapping):
            pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            raise RuntimeError("chart processor did not return pixel_values")
        with torch.inference_mode():
            generated = model.generate(pixel_values.to(device), max_new_tokens=self.config.max_new_tokens)
        if hasattr(processor, "batch_decode"):
            decoded = processor.batch_decode(generated, skip_special_tokens=True)
        elif hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "batch_decode"):
            decoded = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
        else:
            raise RuntimeError("chart processor cannot decode generated token ids")
        if not decoded:
            raise ValueError("chart model produced no decoded sequence")
        text = str(decoded[0]).strip()
        if not text or len(text) > _MAX_OUTPUT_CHARS or "\x00" in text:
            raise ValueError("chart model output is empty or exceeds safety bounds")
        return text

    def extract(self, image: Any, *, chart_region_id: str, anchor: SourceAnchor, extraction_artifact_sha256: str) -> StructuredChart:
        return decode_chart_text(self.generate_text(image), chart_region_id=chart_region_id, anchor=anchor, extraction_artifact_sha256=extraction_artifact_sha256, config=self.config)


def _axis(row: Mapping[str, Any]) -> ChartAxis:
    allowed = {"axis_id", "dimension", "label", "unit", "scale", "confidence"}
    unknown = set(row) - allowed
    if unknown:
        raise ValueError(f"chart axis contains unsupported fields: {sorted(unknown)}")
    return ChartAxis(**row)


def _point(row: Mapping[str, Any], *, series_id: str) -> ChartPoint:
    allowed = {"point_id", "x_numeric", "x_category", "y", "y_lower", "y_upper", "confidence"}
    unknown = set(row) - allowed
    if unknown:
        raise ValueError(f"chart point contains unsupported fields: {sorted(unknown)}")
    return ChartPoint(series_id=series_id, **row)


def _series(row: Mapping[str, Any]) -> ChartSeries:
    allowed = {"series_id", "label", "kind", "x_axis_id", "y_axis_id", "points"}
    unknown = set(row) - allowed
    if unknown:
        raise ValueError(f"chart series contains unsupported fields: {sorted(unknown)}")
    points_raw = row.get("points")
    if not isinstance(points_raw, list) or not points_raw or len(points_raw) > _MAX_POINTS:
        raise ValueError("chart series points must be a bounded non-empty JSON list")
    series_id = _text(row.get("series_id"), "series_id", 2000)
    points = tuple(_point(point, series_id=series_id) for point in points_raw if isinstance(point, Mapping))
    if len(points) != len(points_raw):
        raise ValueError("every chart point must be a JSON object")
    return ChartSeries(series_id, row.get("label"), row.get("kind", "other"), row.get("x_axis_id"), row.get("y_axis_id"), points)


def _decode_json(text: str, *, chart_region_id: str, anchor: SourceAnchor, extraction_artifact_sha256: str) -> StructuredChart:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("json_v1 chart output is not valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("json_v1 chart output must be an object")
    allowed = {"axes", "series", "caption_region_ids", "extraction_confidence"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"json_v1 chart output contains unsupported fields: {sorted(unknown)}")
    axes_raw, series_raw = raw.get("axes"), raw.get("series")
    if not isinstance(axes_raw, list) or not isinstance(series_raw, list):
        raise ValueError("json_v1 chart output requires axes and series lists")
    axes = tuple(_axis(item) for item in axes_raw if isinstance(item, Mapping))
    series = tuple(_series(item) for item in series_raw if isinstance(item, Mapping))
    if len(axes) != len(axes_raw) or len(series) != len(series_raw):
        raise ValueError("axes and series entries must be JSON objects")
    captions = raw.get("caption_region_ids", [])
    if not isinstance(captions, list):
        raise ValueError("caption_region_ids must be a JSON list")
    return StructuredChart(chart_region_id, anchor, extraction_artifact_sha256, axes, series, tuple(captions), raw.get("extraction_confidence"))


def _parse_x(value: str) -> tuple[float | None, str | None]:
    selected = _text(value, "tabular x value")
    try:
        return _finite(selected.replace(",", ""), "tabular x value"), None
    except ValueError:
        return None, selected


def _decode_tabular(text: str, *, chart_region_id: str, anchor: SourceAnchor, extraction_artifact_sha256: str, config: ChartDecodeConfig) -> StructuredChart:
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
    if len(lines) < 2:
        raise ValueError("tabular_v1 chart output requires a header and at least one data row")
    header = [cell.strip() for cell in lines[0].split("\t")]
    if len(header) < 2 or any(not cell for cell in header):
        raise ValueError("tabular_v1 header requires x plus one or more named series")
    rows = [[cell.strip() for cell in line.split("\t")] for line in lines[1:]]
    if any(len(row) != len(header) for row in rows):
        raise ValueError("tabular_v1 rows must have exactly the header column count")
    if (len(header) - 1) * len(rows) > _MAX_POINTS:
        raise ValueError("tabular_v1 chart output exceeds point limit")
    parsed_x = [_parse_x(row[0]) for row in rows]
    x_types = {"numeric" if numeric is not None else "category" for numeric, _ in parsed_x}
    if len(x_types) != 1:
        raise ValueError("tabular_v1 x column may not mix numeric and categorical values")
    x_axis = ChartAxis(config.x_axis_id, AxisDimension.X, config.x_label or header[0], config.x_unit, config.x_scale)
    y_axis = ChartAxis(config.y_axis_id, AxisDimension.Y, config.y_label, config.y_unit, config.y_scale)
    series_values = []
    seen_names: set[str] = set()
    for column in range(1, len(header)):
        name = _text(header[column], "series label")
        if name in seen_names:
            raise ValueError("tabular_v1 series labels must be unique")
        seen_names.add(name)
        series_id = f"series-{column}"
        points = []
        for row_index, row in enumerate(rows):
            y = _finite(row[column].replace(",", ""), "tabular y value")
            x_numeric, x_category = parsed_x[row_index]
            points.append(ChartPoint(f"{series_id}-p{row_index}", series_id, x_numeric=x_numeric, x_category=x_category, y=y))
        series_values.append(ChartSeries(series_id, name, config.default_series_kind, config.x_axis_id, config.y_axis_id, tuple(points)))
    return StructuredChart(chart_region_id, anchor, extraction_artifact_sha256, (x_axis, y_axis), tuple(series_values))


def decode_chart_text(text: str, *, chart_region_id: str, anchor: SourceAnchor, extraction_artifact_sha256: str, config: ChartDecodeConfig = ChartDecodeConfig()) -> StructuredChart:
    selected = _text(text, "chart model output", _MAX_OUTPUT_CHARS)
    if not isinstance(anchor, SourceAnchor):
        raise ValueError("anchor must be SourceAnchor")
    if config.contract == "json_v1":
        return _decode_json(selected, chart_region_id=chart_region_id, anchor=anchor, extraction_artifact_sha256=extraction_artifact_sha256)
    return _decode_tabular(selected, chart_region_id=chart_region_id, anchor=anchor, extraction_artifact_sha256=extraction_artifact_sha256, config=config)


__all__ = ["ChartDecodeConfig", "LocalHFChartToStructureAdapter", "decode_chart_text"]
