"""Datum- and unit-safe engineering reasoning over HEC/hydrology evidence.

No missing datum, unit or station relationship is inferred. Functions either return
source-linked deterministic diagnostics or fail closed when the engineering comparison is
not well-defined.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from tools.hydrology_domain import HydroTimeSeries, integrate_volume
from tools.hydrology_engineering_ir import (
    EngineeringArtifactRef,
    RASCrossSection,
    RASHydraulicStructure,
    RASPlanIR,
    RASProfileEvidence,
    RASProfilePoint,
    VerticalDatumRef,
    parse_river_station,
)
from tools.numerical_reasoning import Quantity, UnitRegistry, default_unit_registry


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def _text(value: Any, label: str, maximum: int = 1000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise ValueError(f"{label} is invalid")
    return " ".join(value.replace("\x00", " ").split())


def _elevation_unit(artifact: EngineeringArtifactRef) -> str:
    if artifact.unit_system == "si":
        return "m"
    if artifact.unit_system == "us_customary":
        return "ft"
    raise ValueError("geometry elevation unit is not uniquely determined for mixed/unknown unit_system")


def _require_same_datum(left: VerticalDatumRef | None, right: VerticalDatumRef | None) -> VerticalDatumRef:
    if left is None or right is None:
        raise ValueError("vertical datum is required for elevation comparison")
    if left != right:
        raise ValueError("elevation comparison crosses different vertical datums")
    return left


def _source_ids(profile: RASProfileEvidence, plan: RASPlanIR) -> tuple[str, ...]:
    return tuple(dict.fromkeys((profile.source_artifact.source_id, plan.geometry_artifact.source_id)))


@dataclass(frozen=True)
class ProfileComparisonRow:
    cross_section_id: str
    river_name: str
    reach_name: str
    river_station: str
    unit: str
    baseline_wse: float
    candidate_wse: float
    delta_wse: float
    datum_fingerprint: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class ProfileComparison:
    baseline_profile_id: str
    candidate_profile_id: str
    rows: tuple[ProfileComparisonRow, ...]
    missing_from_baseline: tuple[str, ...]
    missing_from_candidate: tuple[str, ...]
    fingerprint: str


def compare_ras_profiles(
    baseline: RASProfileEvidence,
    candidate: RASProfileEvidence,
    *,
    unit_registry: UnitRegistry | None = None,
) -> ProfileComparison:
    if not isinstance(baseline, RASProfileEvidence) or not isinstance(candidate, RASProfileEvidence):
        raise TypeError("baseline/candidate must be RASProfileEvidence")
    datum = _require_same_datum(baseline.vertical_datum, candidate.vertical_datum)
    registry = unit_registry or default_unit_registry()
    # Resolve before row iteration so unsupported/mismatched dimensions fail even when
    # there happen to be no shared cross sections.
    left_def = registry.resolve(baseline.elevation_unit)
    right_def = registry.resolve(candidate.elevation_unit)
    if left_def.dimension != right_def.dimension:
        raise ValueError("profile elevation units are dimensionally incompatible")
    left = {item.cross_section_id: item for item in baseline.points if item.water_surface_elevation is not None}
    right = {item.cross_section_id: item for item in candidate.points if item.water_surface_elevation is not None}
    shared = sorted(set(left) & set(right))
    rows: list[ProfileComparisonRow] = []
    for cross_section_id in shared:
        a, b = left[cross_section_id], right[cross_section_id]
        if (a.river_name.casefold(), a.reach_name.casefold()) != (b.river_name.casefold(), b.reach_name.casefold()):
            raise ValueError(f"cross-section identity {cross_section_id} changed river/reach between profiles")
        b_value = registry.convert(float(b.water_surface_elevation), candidate.elevation_unit, baseline.elevation_unit)
        a_value = float(a.water_surface_elevation)
        rows.append(
            ProfileComparisonRow(
                cross_section_id=cross_section_id,
                river_name=a.river_name,
                reach_name=a.reach_name,
                river_station=a.river_station,
                unit=baseline.elevation_unit,
                baseline_wse=a_value,
                candidate_wse=b_value,
                delta_wse=b_value - a_value,
                datum_fingerprint=datum.fingerprint,
                source_ids=tuple(dict.fromkeys((baseline.source_artifact.source_id, candidate.source_artifact.source_id))),
            )
        )
    payload = {
        "baseline_profile": baseline.fingerprint,
        "candidate_profile": candidate.fingerprint,
        "rows": [asdict(item) for item in rows],
        "missing_from_baseline": sorted(set(right) - set(left)),
        "missing_from_candidate": sorted(set(left) - set(right)),
    }
    return ProfileComparison(
        baseline.profile_id,
        candidate.profile_id,
        tuple(rows),
        tuple(sorted(set(right) - set(left))),
        tuple(sorted(set(left) - set(right))),
        hashlib.sha256(_canonical(payload)).hexdigest(),
    )


@dataclass(frozen=True)
class CrossSectionDepth:
    cross_section_id: str
    profile_name: str
    river_station: str
    unit: str
    ground_minimum: float
    water_surface_elevation: float
    maximum_section_depth: float
    datum_fingerprint: str
    source_ids: tuple[str, ...]


def cross_section_depths(
    plan: RASPlanIR,
    profile: RASProfileEvidence,
    *,
    unit_registry: UnitRegistry | None = None,
) -> tuple[CrossSectionDepth, ...]:
    if not isinstance(plan, RASPlanIR) or not isinstance(profile, RASProfileEvidence):
        raise TypeError("plan/profile types are invalid")
    geometry_datum = plan.geometry_artifact.vertical_datum
    datum = _require_same_datum(geometry_datum, profile.vertical_datum)
    geometry_unit = _elevation_unit(plan.geometry_artifact)
    registry = unit_registry or default_unit_registry()
    cross_sections = {item.cross_section_id: item for item in plan.cross_sections}
    output: list[CrossSectionDepth] = []
    for point in profile.points:
        if point.water_surface_elevation is None:
            continue
        cross_section = cross_sections.get(point.cross_section_id)
        if cross_section is None:
            continue
        wse = registry.convert(point.water_surface_elevation, profile.elevation_unit, geometry_unit)
        ground = cross_section.minimum_ground_elevation
        output.append(
            CrossSectionDepth(
                cross_section_id=cross_section.cross_section_id,
                profile_name=profile.profile_name,
                river_station=cross_section.river_station,
                unit=geometry_unit,
                ground_minimum=ground,
                water_surface_elevation=wse,
                maximum_section_depth=wse - ground,
                datum_fingerprint=datum.fingerprint,
                source_ids=_source_ids(profile, plan),
            )
        )
    return tuple(output)


def _profile_point_for_structure(
    structure: RASHydraulicStructure,
    profile: RASProfileEvidence,
) -> RASProfilePoint:
    exact = [
        item
        for item in profile.points
        if item.river_name.casefold() == structure.river_name.casefold()
        and item.reach_name.casefold() == structure.reach_name.casefold()
        and item.river_station == structure.river_station
        and item.water_surface_elevation is not None
    ]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError("multiple profile points exactly match the hydraulic structure station")
    target = parse_river_station(structure.river_station)
    if target is None:
        raise ValueError("structure station has no exact profile match and is not numerically comparable")
    numeric = [
        item
        for item in profile.points
        if item.river_name.casefold() == structure.river_name.casefold()
        and item.reach_name.casefold() == structure.reach_name.casefold()
        and item.water_surface_elevation is not None
        and parse_river_station(item.river_station) == target
    ]
    if len(numeric) != 1:
        raise ValueError("structure station cannot be uniquely matched to a profile point")
    return numeric[0]


@dataclass(frozen=True)
class FreeboardAssessment:
    structure_id: str
    profile_id: str
    river_station: str
    unit: str
    crest_elevation: float
    water_surface_elevation: float
    freeboard: float
    overtopping: bool
    datum_fingerprint: str
    source_ids: tuple[str, ...]
    note: str = "Negative freeboard means the profile water surface is above the supplied crest elevation; this does not by itself prove a dam-breach mechanism."


def assess_structure_freeboard(
    plan: RASPlanIR,
    profile: RASProfileEvidence,
    structure: RASHydraulicStructure,
    *,
    unit_registry: UnitRegistry | None = None,
) -> FreeboardAssessment:
    if not isinstance(plan, RASPlanIR) or not isinstance(profile, RASProfileEvidence) or not isinstance(structure, RASHydraulicStructure):
        raise TypeError("plan/profile/structure types are invalid")
    if structure.crest_elevation is None:
        raise ValueError("structure crest elevation is required for freeboard")
    datum = _require_same_datum(plan.geometry_artifact.vertical_datum, profile.vertical_datum)
    geometry_unit = _elevation_unit(plan.geometry_artifact)
    point = _profile_point_for_structure(structure, profile)
    registry = unit_registry or default_unit_registry()
    wse = registry.convert(float(point.water_surface_elevation), profile.elevation_unit, geometry_unit)
    crest = float(structure.crest_elevation)
    freeboard = crest - wse
    return FreeboardAssessment(
        structure_id=structure.structure_id,
        profile_id=profile.profile_id,
        river_station=structure.river_station,
        unit=geometry_unit,
        crest_elevation=crest,
        water_surface_elevation=wse,
        freeboard=freeboard,
        overtopping=freeboard < 0,
        datum_fingerprint=datum.fingerprint,
        source_ids=tuple(dict.fromkeys((plan.geometry_artifact.source_id, profile.source_artifact.source_id))),
    )


@dataclass(frozen=True)
class HydrographMetrics:
    series_id: str
    variable: str
    location_id: str
    scenario_id: str
    unit: str
    peak_value: float
    peak_time: dt.datetime
    integrated_volume_m3: float | None
    source_ids: tuple[str, ...]
    fingerprint: str


def hydrograph_metrics(
    series: HydroTimeSeries,
    *,
    unit_registry: UnitRegistry | None = None,
) -> HydrographMetrics:
    if not isinstance(series, HydroTimeSeries):
        raise TypeError("series must be HydroTimeSeries")
    registry = unit_registry or default_unit_registry()
    peak = series.peak()
    volume: float | None = None
    try:
        volume = integrate_volume(series, registry=registry).value
    except (KeyError, ValueError):
        volume = None
    payload = {
        "series_fingerprint": series.fingerprint,
        "peak_value": peak.value,
        "peak_time": peak.timestamp,
        "integrated_volume_m3": volume,
    }
    return HydrographMetrics(
        series_id=series.series_id,
        variable=series.variable,
        location_id=series.location_id,
        scenario_id=series.scenario_id,
        unit=series.unit,
        peak_value=peak.value,
        peak_time=peak.timestamp,
        integrated_volume_m3=volume,
        source_ids=(series.source_id,),
        fingerprint=hashlib.sha256(_canonical(payload)).hexdigest(),
    )


@dataclass(frozen=True)
class HydrographDelta:
    location_id: str
    variable: str
    unit: str
    baseline_series_id: str
    candidate_series_id: str
    peak_delta: float
    peak_time_delta_seconds: float
    volume_delta_m3: float | None
    source_ids: tuple[str, ...]


def compare_hydrographs(
    baseline: HydroTimeSeries,
    candidate: HydroTimeSeries,
    *,
    unit_registry: UnitRegistry | None = None,
) -> HydrographDelta:
    if not isinstance(baseline, HydroTimeSeries) or not isinstance(candidate, HydroTimeSeries):
        raise TypeError("baseline/candidate must be HydroTimeSeries")
    if baseline.variable != candidate.variable or baseline.location_id != candidate.location_id:
        raise ValueError("hydrograph comparison requires the same variable and location")
    registry = unit_registry or default_unit_registry()
    left_peak, right_peak = baseline.peak(), candidate.peak()
    candidate_peak = registry.convert(right_peak.value, candidate.unit, baseline.unit)
    left_volume = hydrograph_metrics(baseline, unit_registry=registry).integrated_volume_m3
    right_volume = hydrograph_metrics(candidate, unit_registry=registry).integrated_volume_m3
    volume_delta = None if left_volume is None or right_volume is None else right_volume - left_volume
    return HydrographDelta(
        location_id=baseline.location_id,
        variable=baseline.variable,
        unit=baseline.unit,
        baseline_series_id=baseline.series_id,
        candidate_series_id=candidate.series_id,
        peak_delta=candidate_peak - left_peak.value,
        peak_time_delta_seconds=(right_peak.timestamp - left_peak.timestamp).total_seconds(),
        volume_delta_m3=volume_delta,
        source_ids=tuple(dict.fromkeys((baseline.source_id, candidate.source_id))),
    )


@dataclass(frozen=True)
class PeakLagDiagnostic:
    upstream_series_id: str
    downstream_series_id: str
    variable: str
    lag_seconds: float
    upstream_peak_time: dt.datetime
    downstream_peak_time: dt.datetime
    source_ids: tuple[str, ...]
    note: str = "Peak-to-peak lag is an observed/simulated timing diagnostic; it is not automatically equivalent to parcel travel time or wave celerity."


def peak_lag(upstream: HydroTimeSeries, downstream: HydroTimeSeries) -> PeakLagDiagnostic:
    if not isinstance(upstream, HydroTimeSeries) or not isinstance(downstream, HydroTimeSeries):
        raise TypeError("upstream/downstream must be HydroTimeSeries")
    if upstream.variable != downstream.variable:
        raise ValueError("peak lag requires the same variable")
    left, right = upstream.peak(), downstream.peak()
    return PeakLagDiagnostic(
        upstream_series_id=upstream.series_id,
        downstream_series_id=downstream.series_id,
        variable=upstream.variable,
        lag_seconds=(right.timestamp - left.timestamp).total_seconds(),
        upstream_peak_time=left.timestamp,
        downstream_peak_time=right.timestamp,
        source_ids=tuple(dict.fromkeys((upstream.source_id, downstream.source_id))),
    )


__all__ = [
    "CrossSectionDepth",
    "FreeboardAssessment",
    "HydrographDelta",
    "HydrographMetrics",
    "PeakLagDiagnostic",
    "ProfileComparison",
    "ProfileComparisonRow",
    "assess_structure_freeboard",
    "compare_hydrographs",
    "compare_ras_profiles",
    "cross_section_depths",
    "hydrograph_metrics",
    "peak_lag",
]
