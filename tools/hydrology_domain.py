"""Hydrology/geospatial evidence contracts for scientific RAG.

These types model provenance-safe rainfall, hydrograph, reservoir and hydraulic-model
results plus adapter interfaces for CHIRPS, HEC-HMS and HEC-RAS.  The module does not
download rainfall data or execute proprietary/external models; operators provide those
adapters explicitly.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from tools.numerical_reasoning import Quantity, UnitRegistry, default_unit_registry

_MAX_SERIES = 10_000_000
_MAX_SCENARIOS = 10_000


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _finite(value: Any, label: str, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum) or (maximum is not None and result > maximum):
        raise ValueError(f"{label} is invalid")
    return result


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 digest")
    digest = value.strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return digest


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def _utc(value: dt.datetime) -> dt.datetime:
    if not isinstance(value, dt.datetime):
        raise ValueError("timestamp must be datetime")
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


@dataclass(frozen=True)
class CRSRef:
    authority: str
    code: str
    axis_order: str = "xy"

    def __post_init__(self) -> None:
        object.__setattr__(self, "authority", _text(self.authority, "CRS authority", 32).upper())
        object.__setattr__(self, "code", _text(self.code, "CRS code", 64))
        axis = _text(self.axis_order, "axis_order", 8).lower()
        if axis not in {"xy", "yx"}:
            raise ValueError("axis_order must be xy or yx")
        object.__setattr__(self, "axis_order", axis)


@dataclass(frozen=True)
class GeoPoint:
    x: float
    y: float
    crs: CRSRef

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite(self.x, "x", -1e12, 1e12))
        object.__setattr__(self, "y", _finite(self.y, "y", -1e12, 1e12))
        if not isinstance(self.crs, CRSRef):
            raise ValueError("crs must be CRSRef")


@dataclass(frozen=True)
class RasterWindowEvidence:
    source_id: str
    source_sha256: str
    crs: CRSRef
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    band: str
    start_time: dt.datetime
    end_time: dt.datetime
    data_sha256: str
    units: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id", 500))
        object.__setattr__(self, "source_sha256", _digest(self.source_sha256, "source_sha256"))
        if not isinstance(self.crs, CRSRef):
            raise ValueError("crs must be CRSRef")
        for name in ("x_min", "y_min", "x_max", "y_max"):
            object.__setattr__(self, name, _finite(getattr(self, name), name, -1e12, 1e12))
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("raster window bounds are invalid")
        object.__setattr__(self, "band", _text(self.band, "band", 128))
        start, end = _utc(self.start_time), _utc(self.end_time)
        if end < start:
            raise ValueError("raster time range is invalid")
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        object.__setattr__(self, "data_sha256", _digest(self.data_sha256, "data_sha256"))
        object.__setattr__(self, "units", _text(self.units, "units", 64))

    @property
    def evidence_id(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True)
class TimeSeriesPoint:
    timestamp: dt.datetime
    value: float
    quality: str = "observed"

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _utc(self.timestamp))
        object.__setattr__(self, "value", _finite(self.value, "value"))
        quality = _text(self.quality, "quality", 32).lower()
        if quality not in {"observed", "simulated", "estimated", "interpolated", "missing"}:
            raise ValueError("unsupported quality flag")
        object.__setattr__(self, "quality", quality)


@dataclass(frozen=True)
class HydroTimeSeries:
    series_id: str
    variable: str
    unit: str
    location_id: str
    points: tuple[TimeSeriesPoint, ...]
    source_id: str
    scenario_id: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "series_id", _text(self.series_id, "series_id", 256))
        object.__setattr__(self, "variable", _text(self.variable, "variable", 128).lower())
        object.__setattr__(self, "unit", _text(self.unit, "unit", 64))
        object.__setattr__(self, "location_id", _text(self.location_id, "location_id", 256))
        if not self.points or len(self.points) > _MAX_SERIES or any(not isinstance(item, TimeSeriesPoint) for item in self.points):
            raise ValueError("time series points are invalid")
        timestamps = [item.timestamp for item in self.points]
        if timestamps != sorted(timestamps) or len(timestamps) != len(set(timestamps)):
            raise ValueError("time series timestamps must be strictly increasing")
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id", 500))
        object.__setattr__(self, "scenario_id", _text(self.scenario_id, "scenario_id", 256, allow_empty=True))
        if not isinstance(self.metadata, Mapping) or len(self.metadata) > 64:
            raise ValueError("metadata must be a bounded mapping")
        object.__setattr__(self, "metadata", {_text(str(k), "metadata key", 100): _text(str(v), "metadata value", 1000) for k, v in self.metadata.items()})

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()

    def peak(self) -> TimeSeriesPoint:
        return max(self.points, key=lambda item: item.value)


@dataclass(frozen=True)
class HydroScenario:
    scenario_id: str
    model_type: str
    model_version: str
    project_sha256: str
    plan_name: str
    start_time: dt.datetime
    end_time: dt.datetime
    parameters_sha256: str
    series: tuple[HydroTimeSeries, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_id", _text(self.scenario_id, "scenario_id", 256))
        model_type = _text(self.model_type, "model_type", 64).lower()
        if model_type not in {"hec-hms", "hec-ras", "observed", "other"}:
            raise ValueError("unsupported hydrology model type")
        object.__setattr__(self, "model_type", model_type)
        object.__setattr__(self, "model_version", _text(self.model_version, "model_version", 100))
        object.__setattr__(self, "project_sha256", _digest(self.project_sha256, "project_sha256"))
        object.__setattr__(self, "plan_name", _text(self.plan_name, "plan_name", 500))
        start, end = _utc(self.start_time), _utc(self.end_time)
        if end <= start:
            raise ValueError("scenario time range is invalid")
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        object.__setattr__(self, "parameters_sha256", _digest(self.parameters_sha256, "parameters_sha256"))
        if len(self.series) > 100_000 or any(not isinstance(item, HydroTimeSeries) for item in self.series):
            raise ValueError("scenario series are invalid")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


class CHIRPSAdapter(Protocol):
    """Operator-supplied adapter; implementation may read local or governed remote CHIRPS data."""
    def rainfall_window(self, *, bbox: tuple[float, float, float, float], crs: CRSRef, start: dt.datetime, end: dt.datetime) -> tuple[RasterWindowEvidence, HydroTimeSeries]: ...


class HECHMSAdapter(Protocol):
    def inspect_project(self, project_path: str) -> Mapping[str, Any]: ...
    def read_scenario(self, project_path: str, run_name: str) -> HydroScenario: ...


class HECRASAdapter(Protocol):
    def inspect_project(self, project_path: str) -> Mapping[str, Any]: ...
    def read_scenario(self, project_path: str, plan_name: str) -> HydroScenario: ...
    def read_profile(self, project_path: str, plan_name: str, profile_name: str) -> Sequence[Mapping[str, Any]]: ...


@dataclass(frozen=True)
class ScenarioMetric:
    variable: str
    location_id: str
    unit: str
    baseline_peak: float
    candidate_peak: float
    peak_delta: float
    baseline_peak_time: dt.datetime
    candidate_peak_time: dt.datetime
    peak_time_delta_seconds: float


def compare_scenarios(
    baseline: HydroScenario,
    candidate: HydroScenario,
    *,
    unit_registry: UnitRegistry | None = None,
) -> tuple[ScenarioMetric, ...]:
    registry = unit_registry or default_unit_registry()
    baseline_index = {(series.variable, series.location_id): series for series in baseline.series}
    candidate_index = {(series.variable, series.location_id): series for series in candidate.series}
    shared = sorted(set(baseline_index) & set(candidate_index))
    metrics: list[ScenarioMetric] = []
    for key in shared:
        left, right = baseline_index[key], candidate_index[key]
        left_peak, right_peak = left.peak(), right.peak()
        try:
            right_value = registry.convert(right_peak.value, right.unit, left.unit)
        except (KeyError, ValueError):
            if right.unit != left.unit:
                continue
            right_value = right_peak.value
        delta = right_value - left_peak.value
        metrics.append(
            ScenarioMetric(
                variable=left.variable,
                location_id=left.location_id,
                unit=left.unit,
                baseline_peak=left_peak.value,
                candidate_peak=right_value,
                peak_delta=delta,
                baseline_peak_time=left_peak.timestamp,
                candidate_peak_time=right_peak.timestamp,
                peak_time_delta_seconds=(right_peak.timestamp - left_peak.timestamp).total_seconds(),
            )
        )
    return tuple(metrics)


def integrate_volume(series: HydroTimeSeries, *, registry: UnitRegistry | None = None) -> Quantity:
    """Trapezoidal integration for discharge series, yielding cubic metres."""
    units = registry or default_unit_registry()
    definition = units.resolve(series.unit)
    discharge_def = units.resolve("m3/s")
    if definition.dimension != discharge_def.dimension:
        raise ValueError("series must be discharge-compatible")
    total_m3 = 0.0
    for left, right in zip(series.points, series.points[1:]):
        dt_seconds = (right.timestamp - left.timestamp).total_seconds()
        if dt_seconds <= 0:
            raise ValueError("series timestamps are not increasing")
        q1 = units.convert(left.value, series.unit, "m3/s")
        q2 = units.convert(right.value, series.unit, "m3/s")
        total_m3 += 0.5 * (q1 + q2) * dt_seconds
    return Quantity(total_m3, "m3", 0.0, (series.source_id,), f"integrated_{series.variable}_volume")


__all__ = [
    "CHIRPSAdapter", "CRSRef", "GeoPoint", "HECHMSAdapter", "HECRASAdapter",
    "HydroScenario", "HydroTimeSeries", "RasterWindowEvidence", "ScenarioMetric",
    "TimeSeriesPoint", "compare_scenarios", "integrate_volume",
]
