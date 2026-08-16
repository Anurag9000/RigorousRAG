"""Strict local-file loaders for HEC engineering IR without executing HEC software.

The loaders consume operator-exported JSON/CSV only through ``LocalArtifactRoot``. They
validate schemas, content hashes, CRS/datum/units and typed geometry/topology fields before
creating the engineering IR used by reconciliation and reasoning.
"""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.hydrology_domain import CRSRef, GeoPoint
from tools.hydrology_engineering_ir import (
    EngineeringArtifactRef,
    HMSBasinIR,
    HMSConnection,
    HMSElement,
    HMSRunIR,
    ManningSegment,
    RASCrossSection,
    RASHydraulicStructure,
    RASPlanIR,
    RASProfileEvidence,
    RASProfilePoint,
    StationElevation,
    VerticalDatumRef,
)
from tools.hydrology_local_adapters import LocalArtifactRoot, _strict_json_load

_MAX_JSON_BYTES = 100_000_000
_MAX_PROFILE_BYTES = 500_000_000
_MAX_ROWS = 10_000_000


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _float(value: Any, label: str, *, allow_none: bool = False) -> float | None:
    if allow_none and value in {None, ""}:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    import math
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: Any, label: str, maximum: int) -> Sequence[Any]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"{label} must be a bounded array")
    return value


def _crs(value: Any) -> CRSRef | None:
    if value in {None, ""}:
        return None
    raw = _mapping(value, "crs")
    return CRSRef(str(raw.get("authority", "")), str(raw.get("code", "")), str(raw.get("axis_order", "xy")))


def _datum(value: Any) -> VerticalDatumRef | None:
    if value in {None, ""}:
        return None
    raw = _mapping(value, "vertical_datum")
    return VerticalDatumRef(str(raw.get("name", "")), str(raw.get("epoch", "")), str(raw.get("geoid_model", "")))


def _point(value: Any, *, inherited_crs: CRSRef | None = None) -> GeoPoint | None:
    if value in {None, ""}:
        return None
    raw = _mapping(value, "point")
    crs = _crs(raw.get("crs")) or inherited_crs
    if crs is None:
        raise ValueError("coordinate point requires an explicit or inherited CRS")
    return GeoPoint(_float(raw.get("x"), "point.x"), _float(raw.get("y"), "point.y"), crs)


def _artifact_from_descriptor(
    root: LocalArtifactRoot,
    value: Any,
    *,
    model_type: str,
    expected_role: str | None = None,
) -> EngineeringArtifactRef:
    raw = _mapping(value, "artifact")
    role = _text(str(raw.get("role", "")), "artifact role", 64).lower()
    if expected_role is not None and role != expected_role:
        raise ValueError(f"artifact role must be {expected_role}")
    relative = _text(str(raw.get("path", "")), "artifact path", 2000)
    payload = root.read_bytes(relative, max_bytes=_MAX_PROFILE_BYTES)
    expected_sha = str(raw.get("sha256", "")).strip().lower()
    actual_sha = _sha_bytes(payload)
    if expected_sha and expected_sha != actual_sha:
        raise ValueError(f"artifact content hash mismatch for {relative}")
    return EngineeringArtifactRef(
        artifact_id=_text(str(raw.get("artifact_id", "")), "artifact_id", 256),
        model_type=model_type,
        role=role,
        source_id=_text(str(raw.get("source_id") or f"local:{relative}:{actual_sha}"), "source_id", 1000),
        content_sha256=actual_sha,
        relative_path=relative,
        unit_system=str(raw.get("unit_system", "unknown")),
        crs=_crs(raw.get("crs")),
        vertical_datum=_datum(raw.get("vertical_datum")),
        metadata={str(k): str(v) for k, v in _mapping(raw.get("metadata", {}), "artifact metadata").items()},
    )


def _station_elevation(value: Any) -> tuple[StationElevation, ...]:
    rows = _list(value, "station_elevation", 200_000)
    return tuple(StationElevation(_float(_mapping(row, "station/elevation row").get("station"), "station"), _float(_mapping(row, "station/elevation row").get("elevation"), "elevation")) for row in rows)


def _manning(value: Any) -> tuple[ManningSegment, ...]:
    rows = _list(value or [], "manning_segments", 10_000)
    output = []
    for item in rows:
        row = _mapping(item, "Manning segment")
        output.append(ManningSegment(_float(row.get("station_start"), "station_start"), _float(row.get("station_end"), "station_end"), _float(row.get("n"), "Manning n")))
    return tuple(output)


def _load_profile_csv(
    root: LocalArtifactRoot,
    descriptor: Mapping[str, Any],
    *,
    plan_id: str,
    inherited_datum: VerticalDatumRef | None,
) -> RASProfileEvidence:
    artifact = _artifact_from_descriptor(root, descriptor.get("artifact"), model_type="hec-ras", expected_role="profile_export")
    payload = root.read_bytes(artifact.relative_path, max_bytes=_MAX_PROFILE_BYTES)
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("RAS profile export must be UTF-8 CSV") from exc
    reader = csv.DictReader(io.StringIO(text))
    required = {"cross_section_id", "river_name", "reach_name", "river_station", "profile_name", "water_surface_elevation"}
    fields = set(reader.fieldnames or ())
    allowed = required | {"energy_grade_elevation", "velocity", "discharge"}
    if not required.issubset(fields) or fields - allowed:
        raise ValueError("RAS profile CSV headers do not match the strict contract")
    points: list[RASProfilePoint] = []
    for index, row in enumerate(reader, start=1):
        if index > _MAX_ROWS:
            raise ValueError("RAS profile CSV exceeds the row limit")
        points.append(
            RASProfilePoint(
                cross_section_id=_text(row["cross_section_id"], "cross_section_id", 256),
                river_name=_text(row["river_name"], "river_name", 500),
                reach_name=_text(row["reach_name"], "reach_name", 500),
                river_station=_text(row["river_station"], "river_station", 128),
                profile_name=_text(row["profile_name"], "profile_name", 500),
                water_surface_elevation=_float(row.get("water_surface_elevation"), "water_surface_elevation", allow_none=True),
                energy_grade_elevation=_float(row.get("energy_grade_elevation"), "energy_grade_elevation", allow_none=True),
                velocity=_float(row.get("velocity"), "velocity", allow_none=True),
                discharge=_float(row.get("discharge"), "discharge", allow_none=True),
            )
        )
    profile_name = _text(str(descriptor.get("profile_name", "")), "profile_name", 500)
    if any(point.profile_name != profile_name for point in points):
        raise ValueError("profile CSV contains rows for a different profile_name")
    return RASProfileEvidence(
        profile_id=_text(str(descriptor.get("profile_id", "")), "profile_id", 256),
        plan_id=plan_id,
        profile_name=profile_name,
        source_artifact=artifact,
        elevation_unit=_text(str(descriptor.get("elevation_unit", "")), "elevation_unit", 64),
        velocity_unit=_text(str(descriptor.get("velocity_unit", "")), "velocity_unit", 64),
        discharge_unit=_text(str(descriptor.get("discharge_unit", "")), "discharge_unit", 64),
        vertical_datum=_datum(descriptor.get("vertical_datum")) or inherited_datum,
        points=tuple(points),
    )


def load_ras_plan_ir(
    root: str | Path | LocalArtifactRoot,
    manifest_path: str | Path,
) -> RASPlanIR:
    artifacts = root if isinstance(root, LocalArtifactRoot) else LocalArtifactRoot(root)
    payload = artifacts.read_bytes(manifest_path, max_bytes=_MAX_JSON_BYTES)
    manifest = _strict_json_load(payload)
    raw = _mapping(manifest, "RAS engineering manifest")
    if raw.get("schema") != "rigorousrag-hec-ras-engineering-v1":
        raise ValueError("RAS engineering manifest schema is invalid")
    plan_id = _text(str(raw.get("plan_id", "")), "plan_id", 256)
    project_artifact = _artifact_from_descriptor(artifacts, raw.get("project_artifact"), model_type="hec-ras", expected_role="project")
    plan_artifact = _artifact_from_descriptor(artifacts, raw.get("plan_artifact"), model_type="hec-ras", expected_role="plan")
    geometry_artifact = _artifact_from_descriptor(artifacts, raw.get("geometry_artifact"), model_type="hec-ras", expected_role="geometry")
    flow_raw = raw.get("flow_artifact")
    flow_artifact = None if flow_raw in {None, ""} else _artifact_from_descriptor(artifacts, flow_raw, model_type="hec-ras")

    cross_sections: list[RASCrossSection] = []
    for item in _list(raw.get("cross_sections"), "cross_sections", 100_000):
        row = _mapping(item, "cross_section")
        start = _point(row.get("cutline_start"), inherited_crs=geometry_artifact.crs)
        end = _point(row.get("cutline_end"), inherited_crs=geometry_artifact.crs)
        cross_sections.append(
            RASCrossSection(
                cross_section_id=str(row.get("cross_section_id", "")),
                river_name=str(row.get("river_name", "")),
                reach_name=str(row.get("reach_name", "")),
                river_station=str(row.get("river_station", "")),
                geometry_artifact_id=geometry_artifact.artifact_id,
                station_elevation=_station_elevation(row.get("station_elevation")),
                bank_left_station=_float(row.get("bank_left_station"), "bank_left_station", allow_none=True),
                bank_right_station=_float(row.get("bank_right_station"), "bank_right_station", allow_none=True),
                manning_segments=_manning(row.get("manning_segments")),
                cutline_start=start,
                cutline_end=end,
                downstream_reach_length=_float(row.get("downstream_reach_length"), "downstream_reach_length", allow_none=True),
                metadata={str(k): str(v) for k, v in _mapping(row.get("metadata", {}), "cross-section metadata").items()},
            )
        )

    structures: list[RASHydraulicStructure] = []
    for item in _list(raw.get("structures", []), "structures", 100_000):
        row = _mapping(item, "hydraulic structure")
        structures.append(
            RASHydraulicStructure(
                structure_id=str(row.get("structure_id", "")),
                structure_type=str(row.get("structure_type", "")),
                river_name=str(row.get("river_name", "")),
                reach_name=str(row.get("reach_name", "")),
                river_station=str(row.get("river_station", "")),
                geometry_artifact_id=geometry_artifact.artifact_id,
                crest_elevation=_float(row.get("crest_elevation"), "crest_elevation", allow_none=True),
                invert_elevation=_float(row.get("invert_elevation"), "invert_elevation", allow_none=True),
                location=_point(row.get("location"), inherited_crs=geometry_artifact.crs),
                metadata={str(k): str(v) for k, v in _mapping(row.get("metadata", {}), "structure metadata").items()},
            )
        )

    profiles = tuple(
        _load_profile_csv(
            artifacts,
            _mapping(item, "profile descriptor"),
            plan_id=plan_id,
            inherited_datum=geometry_artifact.vertical_datum,
        )
        for item in _list(raw.get("profiles", []), "profiles", 10_000)
    )
    return RASPlanIR(
        plan_id=plan_id,
        plan_name=str(raw.get("plan_name", "")),
        project_artifact=project_artifact,
        plan_artifact=plan_artifact,
        geometry_artifact=geometry_artifact,
        flow_artifact=flow_artifact,
        cross_sections=tuple(cross_sections),
        structures=tuple(structures),
        profiles=profiles,
    )


def _hms_element(row: Mapping[str, Any], basin_artifact: EngineeringArtifactRef) -> HMSElement:
    return HMSElement(
        element_id=str(row.get("element_id", "")),
        element_type=str(row.get("element_type", "")),
        name=str(row.get("name", "")),
        basin_artifact_id=basin_artifact.artifact_id,
        location=_point(row.get("location"), inherited_crs=basin_artifact.crs),
        area=_float(row.get("area"), "area", allow_none=True),
        area_unit=str(row.get("area_unit", "")),
        metadata={str(k): str(v) for k, v in _mapping(row.get("metadata", {}), "HMS element metadata").items()},
    )


def load_hms_run_ir(
    root: str | Path | LocalArtifactRoot,
    manifest_path: str | Path,
) -> HMSRunIR:
    artifacts = root if isinstance(root, LocalArtifactRoot) else LocalArtifactRoot(root)
    payload = artifacts.read_bytes(manifest_path, max_bytes=_MAX_JSON_BYTES)
    manifest = _strict_json_load(payload)
    raw = _mapping(manifest, "HMS engineering manifest")
    if raw.get("schema") != "rigorousrag-hec-hms-engineering-v1":
        raise ValueError("HMS engineering manifest schema is invalid")
    project_artifact = _artifact_from_descriptor(artifacts, raw.get("project_artifact"), model_type="hec-hms", expected_role="project")
    run_artifact = _artifact_from_descriptor(artifacts, raw.get("run_artifact"), model_type="hec-hms", expected_role="run")
    basin_artifact = _artifact_from_descriptor(artifacts, raw.get("basin_artifact"), model_type="hec-hms", expected_role="basin")
    elements = tuple(_hms_element(_mapping(item, "HMS element"), basin_artifact) for item in _list(raw.get("elements"), "elements", 100_000))
    connections: list[HMSConnection] = []
    for item in _list(raw.get("connections"), "connections", 400_000):
        row = _mapping(item, "HMS connection")
        connections.append(
            HMSConnection(
                connection_id=str(row.get("connection_id", "")),
                upstream_element_id=str(row.get("upstream_element_id", "")),
                downstream_element_id=str(row.get("downstream_element_id", "")),
                source_artifact_id=basin_artifact.artifact_id,
                length=_float(row.get("length"), "length", allow_none=True),
                length_unit=str(row.get("length_unit", "")),
            )
        )
    basin = HMSBasinIR(
        basin_id=str(raw.get("basin_id", "")),
        basin_name=str(raw.get("basin_name", "")),
        basin_artifact=basin_artifact,
        elements=elements,
        connections=tuple(connections),
    )
    met_raw = raw.get("meteorology_artifact")
    control_raw = raw.get("control_artifact")
    return HMSRunIR(
        run_id=str(raw.get("run_id", "")),
        run_name=str(raw.get("run_name", "")),
        project_artifact=project_artifact,
        run_artifact=run_artifact,
        basin=basin,
        meteorology_artifact=None if met_raw in {None, ""} else _artifact_from_descriptor(artifacts, met_raw, model_type="hec-hms", expected_role="meteorology"),
        control_artifact=None if control_raw in {None, ""} else _artifact_from_descriptor(artifacts, control_raw, model_type="hec-hms", expected_role="control"),
        scenario_id=str(raw.get("scenario_id", "")),
    )


__all__ = ["load_hms_run_ir", "load_ras_plan_ir"]
