"""Provenance-aware structured representation of scientific charts.

Chart extraction may come from OCR, chart-to-table models, vector-PDF parsing or human
review.  This IR does not choose an extractor; it preserves the resulting axes, series,
points, units, uncertainty and source identity so downstream retrieval/entailment can
reason over chart data without treating a screenshot as the only authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Sequence

from scientific.document_structure import BoundingBox, SourceAnchor

_MAX_POINTS = 5_000_000


def _identifier(value: Any, label: str, maximum: int = 2000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _optional_text(value: Any, label: str, maximum: int = 20_000) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string when set")
    selected = value.strip()
    if len(selected) > maximum or "\x00" in selected:
        raise ValueError(f"{label} is invalid")
    return selected or None


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


def _unit_interval(value: Any, label: str) -> float:
    selected = _finite(value, label)
    if not 0.0 <= selected <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return selected


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


class AxisDimension(str, Enum):
    X = "x"
    Y = "y"
    SECONDARY_Y = "secondary_y"


class AxisScale(str, Enum):
    LINEAR = "linear"
    LOG10 = "log10"
    LOG2 = "log2"
    CATEGORY = "category"
    DATETIME = "datetime"
    UNKNOWN = "unknown"


class SeriesKind(str, Enum):
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    AREA = "area"
    BOX = "box"
    ERROR_BAR = "error_bar"
    OTHER = "other"


@dataclass(frozen=True)
class ChartAxis:
    axis_id: str
    dimension: AxisDimension
    label: str | None = None
    unit: str | None = None
    scale: AxisScale = AxisScale.UNKNOWN
    confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis_id", _identifier(self.axis_id, "axis_id"))
        if not isinstance(self.dimension, AxisDimension):
            object.__setattr__(self, "dimension", AxisDimension(self.dimension))
        if not isinstance(self.scale, AxisScale):
            object.__setattr__(self, "scale", AxisScale(self.scale))
        object.__setattr__(self, "label", _optional_text(self.label, "axis label"))
        object.__setattr__(self, "unit", _optional_text(self.unit, "axis unit", 500))
        if self.confidence is not None:
            object.__setattr__(self, "confidence", _unit_interval(self.confidence, "axis confidence"))


@dataclass(frozen=True)
class ChartPoint:
    point_id: str
    series_id: str
    x_numeric: float | None = None
    x_category: str | None = None
    y: float = 0.0
    y_lower: float | None = None
    y_upper: float | None = None
    box: BoundingBox | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_id", _identifier(self.point_id, "point_id"))
        object.__setattr__(self, "series_id", _identifier(self.series_id, "series_id"))
        if (self.x_numeric is None) == (self.x_category is None):
            raise ValueError("chart point requires exactly one of x_numeric or x_category")
        if self.x_numeric is not None:
            object.__setattr__(self, "x_numeric", _finite(self.x_numeric, "x_numeric"))
        if self.x_category is not None:
            object.__setattr__(self, "x_category", _identifier(self.x_category, "x_category", 20_000))
        object.__setattr__(self, "y", _finite(self.y, "y"))
        if self.y_lower is not None:
            object.__setattr__(self, "y_lower", _finite(self.y_lower, "y_lower"))
        if self.y_upper is not None:
            object.__setattr__(self, "y_upper", _finite(self.y_upper, "y_upper"))
        if self.y_lower is not None and self.y_lower > self.y:
            raise ValueError("y_lower may not exceed y")
        if self.y_upper is not None and self.y_upper < self.y:
            raise ValueError("y_upper may not be below y")
        if self.y_lower is not None and self.y_upper is not None and self.y_lower > self.y_upper:
            raise ValueError("chart uncertainty interval is inverted")
        if self.box is not None and not isinstance(self.box, BoundingBox):
            raise ValueError("point box must be BoundingBox")
        if self.confidence is not None:
            object.__setattr__(self, "confidence", _unit_interval(self.confidence, "point confidence"))


@dataclass(frozen=True)
class ChartSeries:
    series_id: str
    label: str | None
    kind: SeriesKind
    x_axis_id: str
    y_axis_id: str
    points: tuple[ChartPoint, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "series_id", _identifier(self.series_id, "series_id"))
        object.__setattr__(self, "label", _optional_text(self.label, "series label"))
        if not isinstance(self.kind, SeriesKind):
            object.__setattr__(self, "kind", SeriesKind(self.kind))
        object.__setattr__(self, "x_axis_id", _identifier(self.x_axis_id, "x_axis_id"))
        object.__setattr__(self, "y_axis_id", _identifier(self.y_axis_id, "y_axis_id"))
        points = tuple(self.points)
        if not points or len(points) > _MAX_POINTS:
            raise ValueError("series points must be non-empty and bounded")
        if any(not isinstance(point, ChartPoint) or point.series_id != self.series_id for point in points):
            raise ValueError("all points must belong to their chart series")
        if len({point.point_id for point in points}) != len(points):
            raise ValueError("point ids must be unique within a series")
        x_kinds = {"numeric" if point.x_numeric is not None else "category" for point in points}
        if len(x_kinds) != 1:
            raise ValueError("a series may not mix numeric and categorical x coordinates")
        object.__setattr__(self, "points", points)


@dataclass(frozen=True)
class StructuredChart:
    chart_region_id: str
    anchor: SourceAnchor
    extraction_artifact_sha256: str
    axes: tuple[ChartAxis, ...]
    series: tuple[ChartSeries, ...]
    caption_region_ids: tuple[str, ...] = ()
    extraction_confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "chart_region_id", _identifier(self.chart_region_id, "chart_region_id"))
        if not isinstance(self.anchor, SourceAnchor):
            raise ValueError("anchor must be SourceAnchor")
        object.__setattr__(self, "extraction_artifact_sha256", _sha(self.extraction_artifact_sha256, "extraction_artifact_sha256"))
        axes = tuple(self.axes)
        if len(axes) < 2 or len(axes) > 100:
            raise ValueError("structured chart requires 2-100 axes")
        if any(not isinstance(axis, ChartAxis) for axis in axes) or len({axis.axis_id for axis in axes}) != len(axes):
            raise ValueError("chart axes must be valid and uniquely identified")
        axis_ids = {axis.axis_id for axis in axes}
        if not any(axis.dimension is AxisDimension.X for axis in axes) or not any(axis.dimension in {AxisDimension.Y, AxisDimension.SECONDARY_Y} for axis in axes):
            raise ValueError("structured chart requires x and y axes")
        series = tuple(self.series)
        if not series or len(series) > 100_000:
            raise ValueError("structured chart requires bounded non-empty series")
        if any(not isinstance(item, ChartSeries) for item in series) or len({item.series_id for item in series}) != len(series):
            raise ValueError("chart series must be valid and uniquely identified")
        if any(item.x_axis_id not in axis_ids or item.y_axis_id not in axis_ids for item in series):
            raise ValueError("chart series references an unknown axis")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "series", series)
        captions = tuple(_identifier(value, "caption_region_id") for value in self.caption_region_ids)
        if len(set(captions)) != len(captions):
            raise ValueError("caption region ids must be unique")
        object.__setattr__(self, "caption_region_ids", captions)
        if self.extraction_confidence is not None:
            object.__setattr__(self, "extraction_confidence", _unit_interval(self.extraction_confidence, "extraction_confidence"))

    @property
    def chart_sha256(self) -> str:
        return _digest({
            "schema": "rigorousrag-structured-chart/v1",
            "chart_region_id": self.chart_region_id,
            "anchor": asdict(self.anchor),
            "extraction_artifact_sha256": self.extraction_artifact_sha256,
            "axes": [asdict(axis) for axis in self.axes],
            "series": [asdict(series) for series in self.series],
            "caption_region_ids": self.caption_region_ids,
            "extraction_confidence": self.extraction_confidence,
        })

    def axis(self, axis_id: str) -> ChartAxis:
        selected = _identifier(axis_id, "axis_id")
        for axis in self.axes:
            if axis.axis_id == selected:
                return axis
        raise KeyError(selected)

    def series_by_id(self, series_id: str) -> ChartSeries:
        selected = _identifier(series_id, "series_id")
        for item in self.series:
            if item.series_id == selected:
                return item
        raise KeyError(selected)


__all__ = [
    "AxisDimension",
    "AxisScale",
    "ChartAxis",
    "ChartPoint",
    "ChartSeries",
    "SeriesKind",
    "StructuredChart",
]
