"""Typed HEC-RAS/HMS engineering evidence IR for provenance-safe hydrology RAG.

The IR sits between safe local/provider adapters and generic hydrologic topology/reasoning.
It does not execute HEC products or infer missing geometry. Every engineering object binds
its source artifact, units and (where elevation/coordinates matter) explicit CRS/datum.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from tools.hydrology_domain import CRSRef, GeoPoint

_MAX_XS_POINTS = 200_000
_MAX_ELEMENTS = 100_000
_MAX_PROFILES = 1_000_000
_MODEL_TYPES = frozenset({"hec-ras", "hec-hms"})
_ARTIFACT_ROLES = frozenset({
    "project", "plan", "geometry", "steady_flow", "unsteady_flow", "basin",
    "meteorology", "control", "run", "profile_export", "timeseries_export", "other",
})
_UNIT_SYSTEMS = frozenset({"si", "us_customary", "mixed", "unknown"})


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise ValueError(f"{label} contains control characters")
    return cleaned


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    parsed = float(value)
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        raise ValueError(f"{label} is invalid")
    return parsed


def _optional_finite(value: Any, label: str) -> float | None:
    return None if value is None else _finite(value, label)


def _sha(value: Any, label: str) -> str:
    digest = _text(value, label, 64).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _metadata(value: Mapping[str, Any] | None) -> Mapping[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or len(value) > 128:
        raise ValueError("metadata must be a bounded mapping")
    return {
        _text(str(key), "metadata key", 100): _text(str(raw), "metadata value", 2000, allow_empty=True)
        for key, raw in value.items()
    }


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(asdict(value))).hexdigest()


def parse_river_station(value: str) -> float | None:
    """Parse the numeric ordering component of a RAS river station without discarding its label.

    HEC-RAS may append markers such as ``*`` to interpolated stations. The exact original
    label remains authoritative; the parsed number is only an ordering aid.
    """
    raw = _text(value, "river_station", 128)
    cleaned = raw.replace("*", "").replace(",", "").strip()
    try:
        parsed = float(cleaned)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


@dataclass(frozen=True)
class VerticalDatumRef:
    name: str
    epoch: str = ""
    geoid_model: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "vertical datum", 256))
        object.__setattr__(self, "epoch", _text(self.epoch, "datum epoch", 128, allow_empty=True))
        object.__setattr__(self, "geoid_model", _text(self.geoid_model, "geoid model", 256, allow_empty=True))

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


@dataclass(frozen=True)
class EngineeringArtifactRef:
    artifact_id: str
    model_type: str
    role: str
    source_id: str
    content_sha256: str
    relative_path: str = ""
    unit_system: str = "unknown"
    crs: CRSRef | None = None
    vertical_datum: VerticalDatumRef | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _text(self.artifact_id, "artifact_id", 256))
        model = _text(self.model_type, "model_type", 64).lower()
        if model not in _MODEL_TYPES:
            raise ValueError("unsupported HEC model type")
        object.__setattr__(self, "model_type", model)
        role = _text(self.role, "artifact role", 64).lower()
        if role not in _ARTIFACT_ROLES:
            raise ValueError("unsupported engineering artifact role")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id", 1000))
        object.__setattr__(self, "content_sha256", _sha(self.content_sha256, "content_sha256"))
        object.__setattr__(self, "relative_path", _text(self.relative_path, "relative_path", 2000, allow_empty=True))
        units = _text(self.unit_system, "unit_system", 32).lower()
        if units not in _UNIT_SYSTEMS:
            raise ValueError("unsupported engineering unit system")
        object.__setattr__(self, "unit_system", units)
        if self.crs is not None and not isinstance(self.crs, CRSRef):
            raise ValueError("crs must be CRSRef or null")
        if self.vertical_datum is not None and not isinstance(self.vertical_datum, VerticalDatumRef):
            raise ValueError("vertical_datum must be VerticalDatumRef or null")
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


@dataclass(frozen=True)
class StationElevation:
    station: float
    elevation: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "station", _finite(self.station, "station"))
        object.__setattr__(self, "elevation", _finite(self.elevation, "elevation"))


@dataclass(frozen=True)
class ManningSegment:
    station_start: float
    station_end: float
    n_value: float

    def __post_init__(self) -> None:
        start = _finite(self.station_start, "station_start")
        end = _finite(self.station_end, "station_end")
        if end <= start:
            raise ValueError("Manning segment station_end must exceed station_start")
        n = _finite(self.n_value, "n_value", minimum=0.0)
        if n <= 0 or n > 10:
            raise ValueError("Manning n is outside the supported positive range")
        object.__setattr__(self, "station_start", start)
        object.__setattr__(self, "station_end", end)
        object.__setattr__(self, "n_value", n)


@dataclass(frozen=True)
class RASCrossSection:
    cross_section_id: str
    river_name: str
    reach_name: str
    river_station: str
    geometry_artifact_id: str
    station_elevation: tuple[StationElevation, ...]
    bank_left_station: float | None = None
    bank_right_station: float | None = None
    manning_segments: tuple[ManningSegment, ...] = ()
    cutline_start: GeoPoint | None = None
    cutline_end: GeoPoint | None = None
    downstream_reach_length: float | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cross_section_id", _text(self.cross_section_id, "cross_section_id", 256))
        object.__setattr__(self, "river_name", _text(self.river_name, "river_name", 500))
        object.__setattr__(self, "reach_name", _text(self.reach_name, "reach_name", 500))
        object.__setattr__(self, "river_station", _text(self.river_station, "river_station", 128))
        object.__setattr__(self, "geometry_artifact_id", _text(self.geometry_artifact_id, "geometry_artifact_id", 256))
        if not self.station_elevation or len(self.station_elevation) > _MAX_XS_POINTS:
            raise ValueError("cross-section station/elevation points are invalid")
        if any(not isinstance(item, StationElevation) for item in self.station_elevation):
            raise ValueError("station_elevation contains invalid values")
        stations = [item.station for item in self.station_elevation]
        if stations != sorted(stations) or len(stations) != len(set(stations)):
            raise ValueError("cross-section stations must be strictly increasing")
        left = _optional_finite(self.bank_left_station, "bank_left_station")
        right = _optional_finite(self.bank_right_station, "bank_right_station")
        if (left is None) != (right is None):
            raise ValueError("both bank stations must be supplied together")
        if left is not None and (left >= right or left < stations[0] or right > stations[-1]):
            raise ValueError("bank stations are inconsistent with cross-section stationing")
        object.__setattr__(self, "bank_left_station", left)
        object.__setattr__(self, "bank_right_station", right)
        if len(self.manning_segments) > 10_000 or any(not isinstance(item, ManningSegment) for item in self.manning_segments):
            raise ValueError("manning_segments are invalid")
        if self.cutline_start is not None and not isinstance(self.cutline_start, GeoPoint):
            raise ValueError("cutline_start must be GeoPoint")
        if self.cutline_end is not None and not isinstance(self.cutline_end, GeoPoint):
            raise ValueError("cutline_end must be GeoPoint")
        if (self.cutline_start is None) != (self.cutline_end is None):
            raise ValueError("cross-section cutline endpoints must be supplied together")
        if self.cutline_start is not None and self.cutline_start.crs != self.cutline_end.crs:
            raise ValueError("cross-section cutline endpoint CRS mismatch")
        reach_length = _optional_finite(self.downstream_reach_length, "downstream_reach_length")
        if reach_length is not None and reach_length < 0:
            raise ValueError("downstream_reach_length may not be negative")
        object.__setattr__(self, "downstream_reach_length", reach_length)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def numeric_river_station(self) -> float | None:
        return parse_river_station(self.river_station)

    @property
    def minimum_ground_elevation(self) -> float:
        return min(item.elevation for item in self.station_elevation)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


@dataclass(frozen=True)
class RASHydraulicStructure:
    structure_id: str
    structure_type: str
    river_name: str
    reach_name: str
    river_station: str
    geometry_artifact_id: str
    crest_elevation: float | None = None
    invert_elevation: float | None = None
    location: GeoPoint | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "structure_id", _text(self.structure_id, "structure_id", 256))
        kind = _text(self.structure_type, "structure_type", 64).lower()
        if kind not in {"bridge", "culvert", "inline", "lateral", "dam", "weir", "gate", "other"}:
            raise ValueError("unsupported RAS hydraulic structure type")
        object.__setattr__(self, "structure_type", kind)
        object.__setattr__(self, "river_name", _text(self.river_name, "river_name", 500))
        object.__setattr__(self, "reach_name", _text(self.reach_name, "reach_name", 500))
        object.__setattr__(self, "river_station", _text(self.river_station, "river_station", 128))
        object.__setattr__(self, "geometry_artifact_id", _text(self.geometry_artifact_id, "geometry_artifact_id", 256))
        object.__setattr__(self, "crest_elevation", _optional_finite(self.crest_elevation, "crest_elevation"))
        object.__setattr__(self, "invert_elevation", _optional_finite(self.invert_elevation, "invert_elevation"))
        if self.location is not None and not isinstance(self.location, GeoPoint):
            raise ValueError("location must be GeoPoint or null")
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def numeric_river_station(self) -> float | None:
        return parse_river_station(self.river_station)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


@dataclass(frozen=True)
class RASProfilePoint:
    cross_section_id: str
    river_name: str
    reach_name: str
    river_station: str
    profile_name: str
    water_surface_elevation: float | None
    energy_grade_elevation: float | None = None
    velocity: float | None = None
    discharge: float | None = None

    def __post_init__(self) -> None:
        for name, maximum in (("cross_section_id", 256), ("river_name", 500), ("reach_name", 500), ("river_station", 128), ("profile_name", 500)):
            object.__setattr__(self, name, _text(getattr(self, name), name, maximum))
        object.__setattr__(self, "water_surface_elevation", _optional_finite(self.water_surface_elevation, "water_surface_elevation"))
        object.__setattr__(self, "energy_grade_elevation", _optional_finite(self.energy_grade_elevation, "energy_grade_elevation"))
        object.__setattr__(self, "velocity", _optional_finite(self.velocity, "velocity"))
        object.__setattr__(self, "discharge", _optional_finite(self.discharge, "discharge"))


@dataclass(frozen=True)
class RASProfileEvidence:
    profile_id: str
    plan_id: str
    profile_name: str
    source_artifact: EngineeringArtifactRef
    elevation_unit: str
    velocity_unit: str
    discharge_unit: str
    vertical_datum: VerticalDatumRef | None
    points: tuple[RASProfilePoint, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _text(self.profile_id, "profile_id", 256))
        object.__setattr__(self, "plan_id", _text(self.plan_id, "plan_id", 256))
        object.__setattr__(self, "profile_name", _text(self.profile_name, "profile_name", 500))
        if not isinstance(self.source_artifact, EngineeringArtifactRef) or self.source_artifact.model_type != "hec-ras":
            raise ValueError("RAS profile source artifact is invalid")
        for name in ("elevation_unit", "velocity_unit", "discharge_unit"):
            object.__setattr__(self, name, _text(getattr(self, name), name, 64))
        if self.vertical_datum is not None and not isinstance(self.vertical_datum, VerticalDatumRef):
            raise ValueError("vertical_datum must be VerticalDatumRef or null")
        if not self.points or len(self.points) > _MAX_PROFILES or any(not isinstance(item, RASProfilePoint) for item in self.points):
            raise ValueError("RAS profile points are invalid")
        keys = [(item.cross_section_id, item.profile_name) for item in self.points]
        if len(keys) != len(set(keys)):
            raise ValueError("RAS profile contains duplicate cross-section/profile rows")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


@dataclass(frozen=True)
class RASPlanIR:
    plan_id: str
    plan_name: str
    project_artifact: EngineeringArtifactRef
    plan_artifact: EngineeringArtifactRef
    geometry_artifact: EngineeringArtifactRef
    flow_artifact: EngineeringArtifactRef | None
    cross_sections: tuple[RASCrossSection, ...]
    structures: tuple[RASHydraulicStructure, ...] = ()
    profiles: tuple[RASProfileEvidence, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _text(self.plan_id, "plan_id", 256))
        object.__setattr__(self, "plan_name", _text(self.plan_name, "plan_name", 500))
        for name in ("project_artifact", "plan_artifact", "geometry_artifact"):
            artifact = getattr(self, name)
            if not isinstance(artifact, EngineeringArtifactRef) or artifact.model_type != "hec-ras":
                raise ValueError(f"{name} must be a HEC-RAS EngineeringArtifactRef")
        if self.flow_artifact is not None and (not isinstance(self.flow_artifact, EngineeringArtifactRef) or self.flow_artifact.model_type != "hec-ras"):
            raise ValueError("flow_artifact must be a HEC-RAS EngineeringArtifactRef")
        if len(self.cross_sections) > _MAX_ELEMENTS or any(not isinstance(item, RASCrossSection) for item in self.cross_sections):
            raise ValueError("cross_sections are invalid")
        if len({item.cross_section_id for item in self.cross_sections}) != len(self.cross_sections):
            raise ValueError("duplicate RAS cross_section_id")
        if len(self.structures) > _MAX_ELEMENTS or any(not isinstance(item, RASHydraulicStructure) for item in self.structures):
            raise ValueError("structures are invalid")
        if len({item.structure_id for item in self.structures}) != len(self.structures):
            raise ValueError("duplicate RAS structure_id")
        if len(self.profiles) > 10_000 or any(not isinstance(item, RASProfileEvidence) for item in self.profiles):
            raise ValueError("profiles are invalid")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


@dataclass(frozen=True)
class HMSElement:
    element_id: str
    element_type: str
    name: str
    basin_artifact_id: str
    location: GeoPoint | None = None
    area: float | None = None
    area_unit: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "element_id", _text(self.element_id, "element_id", 256))
        kind = _text(self.element_type, "element_type", 64).lower()
        if kind not in {"subbasin", "reach", "junction", "reservoir", "source", "sink", "diversion", "gage", "other"}:
            raise ValueError("unsupported HMS element type")
        object.__setattr__(self, "element_type", kind)
        object.__setattr__(self, "name", _text(self.name, "name", 500))
        object.__setattr__(self, "basin_artifact_id", _text(self.basin_artifact_id, "basin_artifact_id", 256))
        if self.location is not None and not isinstance(self.location, GeoPoint):
            raise ValueError("location must be GeoPoint or null")
        area = _optional_finite(self.area, "area")
        if area is not None and area < 0:
            raise ValueError("area may not be negative")
        object.__setattr__(self, "area", area)
        object.__setattr__(self, "area_unit", _text(self.area_unit, "area_unit", 64, allow_empty=True))
        if area is not None and not self.area_unit:
            raise ValueError("area_unit is required when area is supplied")
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


@dataclass(frozen=True)
class HMSConnection:
    connection_id: str
    upstream_element_id: str
    downstream_element_id: str
    source_artifact_id: str
    length: float | None = None
    length_unit: str = ""

    def __post_init__(self) -> None:
        for name in ("connection_id", "upstream_element_id", "downstream_element_id", "source_artifact_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name, 256))
        if self.upstream_element_id == self.downstream_element_id:
            raise ValueError("HMS connection may not self-reference")
        length = _optional_finite(self.length, "length")
        if length is not None and length < 0:
            raise ValueError("length may not be negative")
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "length_unit", _text(self.length_unit, "length_unit", 64, allow_empty=True))
        if length is not None and not self.length_unit:
            raise ValueError("length_unit is required when length is supplied")


@dataclass(frozen=True)
class HMSBasinIR:
    basin_id: str
    basin_name: str
    basin_artifact: EngineeringArtifactRef
    elements: tuple[HMSElement, ...]
    connections: tuple[HMSConnection, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "basin_id", _text(self.basin_id, "basin_id", 256))
        object.__setattr__(self, "basin_name", _text(self.basin_name, "basin_name", 500))
        if not isinstance(self.basin_artifact, EngineeringArtifactRef) or self.basin_artifact.model_type != "hec-hms":
            raise ValueError("basin_artifact must be a HEC-HMS EngineeringArtifactRef")
        if not self.elements or len(self.elements) > _MAX_ELEMENTS or any(not isinstance(item, HMSElement) for item in self.elements):
            raise ValueError("HMS elements are invalid")
        ids = {item.element_id for item in self.elements}
        if len(ids) != len(self.elements):
            raise ValueError("duplicate HMS element_id")
        if len(self.connections) > _MAX_ELEMENTS * 4 or any(not isinstance(item, HMSConnection) for item in self.connections):
            raise ValueError("HMS connections are invalid")
        for connection in self.connections:
            if connection.upstream_element_id not in ids or connection.downstream_element_id not in ids:
                raise ValueError("HMS connection references an unknown element")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


@dataclass(frozen=True)
class HMSRunIR:
    run_id: str
    run_name: str
    project_artifact: EngineeringArtifactRef
    run_artifact: EngineeringArtifactRef
    basin: HMSBasinIR
    meteorology_artifact: EngineeringArtifactRef | None = None
    control_artifact: EngineeringArtifactRef | None = None
    scenario_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id", 256))
        object.__setattr__(self, "run_name", _text(self.run_name, "run_name", 500))
        for name in ("project_artifact", "run_artifact"):
            artifact = getattr(self, name)
            if not isinstance(artifact, EngineeringArtifactRef) or artifact.model_type != "hec-hms":
                raise ValueError(f"{name} must be a HEC-HMS EngineeringArtifactRef")
        if not isinstance(self.basin, HMSBasinIR):
            raise ValueError("basin must be HMSBasinIR")
        for name in ("meteorology_artifact", "control_artifact"):
            artifact = getattr(self, name)
            if artifact is not None and (not isinstance(artifact, EngineeringArtifactRef) or artifact.model_type != "hec-hms"):
                raise ValueError(f"{name} must be a HEC-HMS EngineeringArtifactRef")
        object.__setattr__(self, "scenario_id", _text(self.scenario_id, "scenario_id", 256, allow_empty=True))

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


__all__ = [
    "EngineeringArtifactRef",
    "HMSBasinIR",
    "HMSConnection",
    "HMSElement",
    "HMSRunIR",
    "ManningSegment",
    "RASCrossSection",
    "RASHydraulicStructure",
    "RASPlanIR",
    "RASProfileEvidence",
    "RASProfilePoint",
    "StationElevation",
    "VerticalDatumRef",
    "parse_river_station",
]
