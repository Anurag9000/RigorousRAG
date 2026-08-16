"""Compile typed HEC engineering IR into governed hydrology evidence packages.

The compiler deliberately separates *engineering identity* from *retrieval indexability*.
Every source object remains represented in the package, but an object enters the
spatiotemporal index only when it has a real temporal and/or georeferenced spatial scope.
No synthetic coordinates, inferred datums, fuzzy topology matches, or invented timestamps
are introduced to make an object searchable.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from tools.hydro_topology import HydroNetwork
from tools.hydrology_domain import HydroScenario, HydroTimeSeries
from tools.hydrology_engineering_ir import (
    HMSRunIR,
    RASCrossSection,
    RASHydraulicStructure,
    RASPlanIR,
    RASProfileEvidence,
    RASProfilePoint,
)
from tools.hydrology_reconciliation import (
    EngineeringMatch,
    ReconciliationReport,
    build_hms_network,
    reconcile_ras_plan,
    reconcile_series_location,
)
from tools.spatiotemporal_index import (
    SpatialEnvelope,
    SpatiotemporalIndex,
    SpatiotemporalRecord,
    TimeEnvelope,
)

_MAX_RECORDS = 2_000_000
_MAX_DIAGNOSTICS = 200_000


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _point_envelope(x: float, y: float, crs: Any) -> SpatialEnvelope:
    return SpatialEnvelope(crs, x, y, x, y)


def _cross_section_envelope(cross_section: RASCrossSection) -> SpatialEnvelope | None:
    if cross_section.cutline_start is None or cross_section.cutline_end is None:
        return None
    left, right = cross_section.cutline_start, cross_section.cutline_end
    if left.crs != right.crs:
        raise ValueError("cross-section cutline CRS mismatch")
    return SpatialEnvelope(
        left.crs,
        min(left.x, right.x),
        min(left.y, right.y),
        max(left.x, right.x),
        max(left.y, right.y),
    )


def _match_index(report: ReconciliationReport) -> Mapping[tuple[str, str], EngineeringMatch]:
    return {(item.source_kind, item.source_id): item for item in report.matches}


def _match_metadata(match: EngineeringMatch | None) -> dict[str, str]:
    if match is None:
        return {"topology_match_status": "unmatched", "topology_match_reason": "no_reconciliation_result"}
    output = {
        "topology_match_status": match.status,
        "topology_match_reason": match.reason,
    }
    if match.status == "matched":
        if match.target_kind == "node":
            output["hydro_node_id"] = match.target_id
        elif match.target_kind == "reach":
            output["hydro_reach_id"] = match.target_id
    return output


def _time_record(
    series: HydroTimeSeries,
    *,
    scenario: HydroScenario,
    binding: EngineeringMatch,
) -> SpatiotemporalRecord:
    metadata = {
        "scenario_id": scenario.scenario_id,
        "model_type": scenario.model_type,
        "location_id": series.location_id,
        "series_fingerprint": series.fingerprint,
        **_match_metadata(binding),
    }
    return SpatiotemporalRecord(
        record_id=f"hydro:{series.fingerprint}",
        source_id=series.source_id,
        spatial=None,
        temporal=TimeEnvelope(series.points[0].timestamp, series.points[-1].timestamp),
        variable=series.variable,
        modality="timeseries",
        content_sha256=series.fingerprint,
        metadata=metadata,
    )


@dataclass(frozen=True)
class EngineeringEvidenceObject:
    object_id: str
    object_kind: str
    fingerprint: str
    source_id: str
    indexed_record_ids: tuple[str, ...] = ()
    topology_status: str = "not_applicable"
    topology_target_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_id", _text(self.object_id, "object_id", 500))
        object.__setattr__(self, "object_kind", _text(self.object_kind, "object_kind", 64).lower())
        digest = _text(self.fingerprint, "fingerprint", 64).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("fingerprint must be SHA-256")
        object.__setattr__(self, "fingerprint", digest)
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id", 1000))
        if len(self.indexed_record_ids) > 100_000:
            raise ValueError("indexed_record_ids exceed the item limit")
        object.__setattr__(self, "indexed_record_ids", tuple(dict.fromkeys(_text(item, "record_id", 500) for item in self.indexed_record_ids)))
        status = _text(self.topology_status, "topology_status", 32).lower()
        if status not in {"matched", "ambiguous", "unmatched", "not_applicable"}:
            raise ValueError("unsupported topology_status")
        object.__setattr__(self, "topology_status", status)
        object.__setattr__(self, "topology_target_id", _text(self.topology_target_id, "topology_target_id", 256, allow_empty=True))
        if status == "matched" and not self.topology_target_id:
            raise ValueError("matched evidence object requires topology_target_id")
        if status != "matched" and self.topology_target_id:
            raise ValueError("non-matched evidence object may not claim topology_target_id")


@dataclass(frozen=True)
class EngineeringEvidencePackage:
    package_id: str
    model_type: str
    source_fingerprint: str
    topology_fingerprint: str
    scenario_fingerprint: str
    records: tuple[SpatiotemporalRecord, ...]
    objects: tuple[EngineeringEvidenceObject, ...]
    diagnostics: tuple[str, ...]
    reconciliation_fingerprint: str
    fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "package_id", _text(self.package_id, "package_id", 500))
        model = _text(self.model_type, "model_type", 64).lower()
        if model not in {"hec-ras", "hec-hms"}:
            raise ValueError("unsupported engineering evidence model type")
        object.__setattr__(self, "model_type", model)
        for name in ("source_fingerprint", "topology_fingerprint", "fingerprint"):
            digest = _text(getattr(self, name), name, 64).lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError(f"{name} must be SHA-256")
            object.__setattr__(self, name, digest)
        for name in ("scenario_fingerprint", "reconciliation_fingerprint"):
            value = _text(getattr(self, name), name, 64, allow_empty=True).lower()
            if value and (len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)):
                raise ValueError(f"{name} must be empty or SHA-256")
            object.__setattr__(self, name, value)
        if len(self.records) > _MAX_RECORDS or any(not isinstance(item, SpatiotemporalRecord) for item in self.records):
            raise ValueError("records are invalid")
        if len({item.record_id for item in self.records}) != len(self.records):
            raise ValueError("duplicate evidence record_id")
        if len(self.objects) > _MAX_RECORDS:
            raise ValueError("objects exceed the item limit")
        if len(self.diagnostics) > _MAX_DIAGNOSTICS:
            raise ValueError("diagnostics exceed the item limit")
        object.__setattr__(self, "diagnostics", tuple(dict.fromkeys(_text(item, "diagnostic", 2000) for item in self.diagnostics)))

    @property
    def complete(self) -> bool:
        return not any(item.startswith("fatal:") for item in self.diagnostics) and all(
            item.topology_status not in {"ambiguous", "unmatched"}
            for item in self.objects
            if item.topology_status != "not_applicable"
        )

    def populate_index(self, index: SpatiotemporalIndex) -> None:
        if not isinstance(index, SpatiotemporalIndex):
            raise TypeError("index must be SpatiotemporalIndex")
        for record in self.records:
            index.upsert(record)


def _package_fingerprint(
    *,
    package_id: str,
    model_type: str,
    source_fingerprint: str,
    topology_fingerprint: str,
    scenario_fingerprint: str,
    records: Sequence[SpatiotemporalRecord],
    objects: Sequence[EngineeringEvidenceObject],
    diagnostics: Sequence[str],
    reconciliation_fingerprint: str,
) -> str:
    return _sha({
        "package_id": package_id,
        "model_type": model_type,
        "source_fingerprint": source_fingerprint,
        "topology_fingerprint": topology_fingerprint,
        "scenario_fingerprint": scenario_fingerprint,
        "records": [(item.record_id, item.fingerprint) for item in records],
        "objects": [asdict(item) for item in objects],
        "diagnostics": sorted(set(diagnostics)),
        "reconciliation_fingerprint": reconciliation_fingerprint,
    })


def _validate_scenario(scenario: HydroScenario, *, model_type: str, project_sha256: str) -> None:
    if not isinstance(scenario, HydroScenario):
        raise TypeError("scenario must be HydroScenario")
    if scenario.model_type != model_type:
        raise ValueError("scenario model_type does not match engineering IR")
    if scenario.project_sha256 != project_sha256:
        raise ValueError("scenario project hash does not match engineering project artifact")


def compile_hms_evidence(
    run: HMSRunIR,
    *,
    scenario: HydroScenario | None = None,
    series_aliases: Mapping[str, Sequence[str]] | None = None,
) -> tuple[HydroNetwork, EngineeringEvidencePackage]:
    """Compile an HMS run into explicit topology plus indexable engineering evidence."""
    if not isinstance(run, HMSRunIR):
        raise TypeError("run must be HMSRunIR")
    network = build_hms_network(run.basin)
    records: list[SpatiotemporalRecord] = []
    objects: list[EngineeringEvidenceObject] = []
    diagnostics: list[str] = []

    for element in run.basin.elements:
        record_ids: list[str] = []
        if element.location is not None:
            record_id = f"hms-element:{element.fingerprint}"
            records.append(SpatiotemporalRecord(
                record_id=record_id,
                source_id=run.basin.basin_artifact.source_id,
                spatial=_point_envelope(element.location.x, element.location.y, element.location.crs),
                temporal=None,
                variable="",
                modality="engineering",
                content_sha256=element.fingerprint,
                metadata={
                    "model_type": "hec-hms",
                    "hms_element_id": element.element_id,
                    "hms_element_type": element.element_type,
                    "hydro_node_id": element.element_id,
                },
            ))
            record_ids.append(record_id)
        else:
            diagnostics.append(f"unindexed:hms_element_no_spatial_scope:{element.element_id}")
        objects.append(EngineeringEvidenceObject(
            element.element_id,
            "hms_element",
            element.fingerprint,
            run.basin.basin_artifact.source_id,
            tuple(record_ids),
            "matched",
            element.element_id,
        ))

    scenario_fingerprint = ""
    if scenario is not None:
        _validate_scenario(scenario, model_type="hec-hms", project_sha256=run.project_artifact.content_sha256)
        if run.scenario_id and scenario.scenario_id != run.scenario_id:
            raise ValueError("scenario_id does not match HMS run scenario_id")
        scenario_fingerprint = scenario.fingerprint
        for series in scenario.series:
            match = reconcile_series_location(series.location_id, network, aliases=series_aliases)
            record = _time_record(series, scenario=scenario, binding=match)
            records.append(record)
            if match.status != "matched":
                diagnostics.append(f"series_topology_{match.status}:{series.location_id}:{match.reason}")
            objects.append(EngineeringEvidenceObject(
                series.series_id,
                "hydro_timeseries",
                series.fingerprint,
                series.source_id,
                (record.record_id,),
                match.status,
                match.target_id,
            ))

    source_fingerprint = run.fingerprint
    package_id = f"hec-hms:{run.run_id}:{source_fingerprint[:16]}"
    fingerprint = _package_fingerprint(
        package_id=package_id,
        model_type="hec-hms",
        source_fingerprint=source_fingerprint,
        topology_fingerprint=network.fingerprint,
        scenario_fingerprint=scenario_fingerprint,
        records=records,
        objects=objects,
        diagnostics=diagnostics,
        reconciliation_fingerprint="",
    )
    return network, EngineeringEvidencePackage(
        package_id,
        "hec-hms",
        source_fingerprint,
        network.fingerprint,
        scenario_fingerprint,
        tuple(records),
        tuple(objects),
        tuple(diagnostics),
        "",
        fingerprint,
    )


def _ras_series_binding(
    series: HydroTimeSeries,
    *,
    plan: RASPlanIR,
    network: HydroNetwork,
    match_by_source: Mapping[tuple[str, str], EngineeringMatch],
    aliases: Mapping[str, Sequence[str]] | None,
) -> EngineeringMatch:
    # Exact engineering IDs have precedence over topology aliases. This preserves the
    # distinction between a cross-section (reach-bound) and a hydraulic structure
    # (node-bound) when operators export RAS time series using engineering IDs.
    for cross_section in plan.cross_sections:
        if series.location_id == cross_section.cross_section_id:
            match = match_by_source.get(("ras_cross_section", cross_section.cross_section_id))
            if match is not None:
                return match
    for structure in plan.structures:
        if series.location_id == structure.structure_id:
            match = match_by_source.get(("ras_structure", structure.structure_id))
            if match is not None:
                return match
    return reconcile_series_location(series.location_id, network, aliases=aliases)


def _profile_point_fingerprint(profile: RASProfileEvidence, point: RASProfilePoint) -> str:
    return _sha({"profile_fingerprint": profile.fingerprint, "point": asdict(point)})


def compile_ras_evidence(
    plan: RASPlanIR,
    network: HydroNetwork,
    *,
    scenario: HydroScenario | None = None,
    coordinate_tolerance: float | None = None,
    expected_downstream_direction: str = "decreasing",
    series_aliases: Mapping[str, Sequence[str]] | None = None,
) -> EngineeringEvidencePackage:
    """Compile a RAS plan against an existing reviewed/generic hydrologic topology."""
    if not isinstance(plan, RASPlanIR) or not isinstance(network, HydroNetwork):
        raise TypeError("plan/network types are invalid")
    report = reconcile_ras_plan(
        plan,
        network,
        coordinate_tolerance=coordinate_tolerance,
        expected_downstream_direction=expected_downstream_direction,
    )
    matches = _match_index(report)
    records: list[SpatiotemporalRecord] = []
    objects: list[EngineeringEvidenceObject] = []
    diagnostics: list[str] = []

    for issue in report.stationing_issues:
        diagnostics.append(
            f"stationing:{issue.issue}:{issue.river_name}:{issue.reach_name}:"
            f"{issue.upstream_cross_section_id}:{issue.downstream_cross_section_id}"
        )

    cross_sections = {item.cross_section_id: item for item in plan.cross_sections}
    for cross_section in plan.cross_sections:
        match = matches.get(("ras_cross_section", cross_section.cross_section_id))
        record_ids: list[str] = []
        spatial = _cross_section_envelope(cross_section)
        if spatial is not None:
            record_id = f"ras-xs:{cross_section.fingerprint}"
            records.append(SpatiotemporalRecord(
                record_id=record_id,
                source_id=plan.geometry_artifact.source_id,
                spatial=spatial,
                temporal=None,
                variable="cross_section_geometry",
                modality="engineering",
                content_sha256=cross_section.fingerprint,
                metadata={
                    "model_type": "hec-ras",
                    "plan_id": plan.plan_id,
                    "cross_section_id": cross_section.cross_section_id,
                    "river_name": cross_section.river_name,
                    "reach_name": cross_section.reach_name,
                    "river_station": cross_section.river_station,
                    **_match_metadata(match),
                },
            ))
            record_ids.append(record_id)
        else:
            diagnostics.append(f"unindexed:ras_cross_section_no_spatial_scope:{cross_section.cross_section_id}")
        status = match.status if match is not None else "unmatched"
        target = match.target_id if match is not None else ""
        if status != "matched":
            diagnostics.append(f"ras_cross_section_topology_{status}:{cross_section.cross_section_id}:{match.reason if match else 'no_reconciliation_result'}")
        objects.append(EngineeringEvidenceObject(
            cross_section.cross_section_id,
            "ras_cross_section",
            cross_section.fingerprint,
            plan.geometry_artifact.source_id,
            tuple(record_ids),
            status,
            target,
        ))

    for structure in plan.structures:
        match = matches.get(("ras_structure", structure.structure_id))
        record_ids: list[str] = []
        if structure.location is not None:
            record_id = f"ras-structure:{structure.fingerprint}"
            records.append(SpatiotemporalRecord(
                record_id=record_id,
                source_id=plan.geometry_artifact.source_id,
                spatial=_point_envelope(structure.location.x, structure.location.y, structure.location.crs),
                temporal=None,
                variable="hydraulic_structure",
                modality="engineering",
                content_sha256=structure.fingerprint,
                metadata={
                    "model_type": "hec-ras",
                    "plan_id": plan.plan_id,
                    "structure_id": structure.structure_id,
                    "structure_type": structure.structure_type,
                    "river_name": structure.river_name,
                    "reach_name": structure.reach_name,
                    "river_station": structure.river_station,
                    **_match_metadata(match),
                },
            ))
            record_ids.append(record_id)
        else:
            diagnostics.append(f"unindexed:ras_structure_no_spatial_scope:{structure.structure_id}")
        status = match.status if match is not None else "unmatched"
        target = match.target_id if match is not None else ""
        if status != "matched":
            diagnostics.append(f"ras_structure_topology_{status}:{structure.structure_id}:{match.reason if match else 'no_reconciliation_result'}")
        objects.append(EngineeringEvidenceObject(
            structure.structure_id,
            "ras_structure",
            structure.fingerprint,
            plan.geometry_artifact.source_id,
            tuple(record_ids),
            status,
            target,
        ))

    for profile in plan.profiles:
        profile_record_ids: list[str] = []
        for point in profile.points:
            cross_section = cross_sections.get(point.cross_section_id)
            if cross_section is None:
                diagnostics.append(f"fatal:profile_references_unknown_cross_section:{profile.profile_id}:{point.cross_section_id}")
                continue
            spatial = _cross_section_envelope(cross_section)
            if spatial is None:
                diagnostics.append(f"unindexed:ras_profile_point_no_spatial_scope:{profile.profile_id}:{point.cross_section_id}")
                continue
            point_fingerprint = _profile_point_fingerprint(profile, point)
            record_id = f"ras-profile:{point_fingerprint}"
            match = matches.get(("ras_cross_section", point.cross_section_id))
            records.append(SpatiotemporalRecord(
                record_id=record_id,
                source_id=profile.source_artifact.source_id,
                spatial=spatial,
                temporal=None,
                variable="water_surface_elevation",
                modality="profile",
                content_sha256=point_fingerprint,
                metadata={
                    "model_type": "hec-ras",
                    "plan_id": plan.plan_id,
                    "profile_id": profile.profile_id,
                    "profile_name": profile.profile_name,
                    "cross_section_id": point.cross_section_id,
                    "river_name": point.river_name,
                    "reach_name": point.reach_name,
                    "river_station": point.river_station,
                    "elevation_unit": profile.elevation_unit,
                    "vertical_datum": profile.vertical_datum.name if profile.vertical_datum is not None else "",
                    **_match_metadata(match),
                },
            ))
            profile_record_ids.append(record_id)
        objects.append(EngineeringEvidenceObject(
            profile.profile_id,
            "ras_profile",
            profile.fingerprint,
            profile.source_artifact.source_id,
            tuple(profile_record_ids),
            "not_applicable",
            "",
        ))

    scenario_fingerprint = ""
    if scenario is not None:
        _validate_scenario(scenario, model_type="hec-ras", project_sha256=plan.project_artifact.content_sha256)
        if scenario.plan_name != plan.plan_name:
            raise ValueError("scenario plan_name does not exactly match RAS plan_name")
        scenario_fingerprint = scenario.fingerprint
        for series in scenario.series:
            binding = _ras_series_binding(
                series,
                plan=plan,
                network=network,
                match_by_source=matches,
                aliases=series_aliases,
            )
            record = _time_record(series, scenario=scenario, binding=binding)
            records.append(record)
            if binding.status != "matched":
                diagnostics.append(f"series_topology_{binding.status}:{series.location_id}:{binding.reason}")
            objects.append(EngineeringEvidenceObject(
                series.series_id,
                "hydro_timeseries",
                series.fingerprint,
                series.source_id,
                (record.record_id,),
                binding.status,
                binding.target_id,
            ))

    if len(records) > _MAX_RECORDS:
        raise RuntimeError("compiled hydrology evidence exceeds the record limit")
    source_fingerprint = plan.fingerprint
    package_id = f"hec-ras:{plan.plan_id}:{source_fingerprint[:16]}"
    fingerprint = _package_fingerprint(
        package_id=package_id,
        model_type="hec-ras",
        source_fingerprint=source_fingerprint,
        topology_fingerprint=network.fingerprint,
        scenario_fingerprint=scenario_fingerprint,
        records=records,
        objects=objects,
        diagnostics=diagnostics,
        reconciliation_fingerprint=report.fingerprint,
    )
    return EngineeringEvidencePackage(
        package_id,
        "hec-ras",
        source_fingerprint,
        network.fingerprint,
        scenario_fingerprint,
        tuple(records),
        tuple(objects),
        tuple(diagnostics),
        report.fingerprint,
        fingerprint,
    )


__all__ = [
    "EngineeringEvidenceObject",
    "EngineeringEvidencePackage",
    "compile_hms_evidence",
    "compile_ras_evidence",
]
