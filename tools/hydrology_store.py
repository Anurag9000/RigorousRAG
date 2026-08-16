"""Versioned persistence contracts and codecs for hydrology research artifacts.

The persistence boundary is intentionally stricter than a generic JSON document store:
loads reconstruct the typed hydrology objects and recompute their identities.  This keeps
stored package/plan fingerprint columns from becoming an authority by themselves.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol

from tools.hydro_topology import HydroNetwork, HydroNode, HydroReach
from tools.hydrology_domain import CRSRef, GeoPoint
from tools.hydrology_evidence_compiler import EngineeringEvidenceObject, EngineeringEvidencePackage
from tools.hydrology_retrieval import (
    HydrologyQuerySpec,
    HydrologyRetrievalPlan,
    SelectedRecordTrace,
    TopologyTimeWindow,
)
from tools.security import normalize_owner_id
from tools.spatiotemporal_index import SpatialEnvelope, SpatiotemporalRecord, TimeEnvelope

_SCHEMA_VERSION = 1
_KINDS = frozenset({"topology", "engineering_package", "retrieval_plan"})
_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


def _text(value: Any, label: str, maximum: int = 500, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    cleaned = value.strip().lower()
    if len(cleaned) != 64 or any(ch not in "0123456789abcdef" for ch in cleaned):
        raise ValueError(f"{label} must be SHA-256")
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


def strict_json(value: Any) -> str:
    encoded = _canonical(value)
    if len(encoded) > _MAX_PAYLOAD_BYTES:
        raise ValueError("hydrology artifact payload exceeds the size limit")
    return encoded.decode("utf-8")


def _utc_text(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).isoformat()


def _parse_time(value: Any, label: str) -> dt.datetime:
    if not isinstance(value, str) or len(value) > 100:
        raise ValueError(f"{label} is invalid")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _crs_payload(crs: CRSRef) -> Mapping[str, str]:
    return {"authority": crs.authority, "code": crs.code, "axis_order": crs.axis_order}


def _crs_from_payload(value: Any) -> CRSRef:
    if not isinstance(value, Mapping):
        raise ValueError("CRS payload is invalid")
    return CRSRef(str(value.get("authority", "")), str(value.get("code", "")), str(value.get("axis_order", "xy")))


def _point_payload(point: GeoPoint | None) -> Mapping[str, Any] | None:
    if point is None:
        return None
    return {"x": point.x, "y": point.y, "crs": _crs_payload(point.crs)}


def _point_from_payload(value: Any) -> GeoPoint | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("GeoPoint payload is invalid")
    return GeoPoint(float(value["x"]), float(value["y"]), _crs_from_payload(value["crs"]))


def _spatial_payload(value: SpatialEnvelope | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return {
        "crs": _crs_payload(value.crs),
        "min_x": value.min_x,
        "min_y": value.min_y,
        "max_x": value.max_x,
        "max_y": value.max_y,
    }


def _spatial_from_payload(value: Any) -> SpatialEnvelope | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("spatial payload is invalid")
    return SpatialEnvelope(
        _crs_from_payload(value["crs"]),
        float(value["min_x"]),
        float(value["min_y"]),
        float(value["max_x"]),
        float(value["max_y"]),
    )


def _temporal_payload(value: TimeEnvelope | None) -> Mapping[str, str] | None:
    if value is None:
        return None
    return {"start": _utc_text(value.start), "end": _utc_text(value.end)}  # type: ignore[dict-item]


def _temporal_from_payload(value: Any) -> TimeEnvelope | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("temporal payload is invalid")
    return TimeEnvelope(_parse_time(value["start"], "start"), _parse_time(value["end"], "end"))


def _record_payload(record: SpatiotemporalRecord) -> Mapping[str, Any]:
    return {
        "record_id": record.record_id,
        "source_id": record.source_id,
        "spatial": _spatial_payload(record.spatial),
        "temporal": _temporal_payload(record.temporal),
        "variable": record.variable,
        "modality": record.modality,
        "content_sha256": record.content_sha256,
        "metadata": dict(record.metadata),
        "fingerprint": record.fingerprint,
    }


def _record_from_payload(value: Any) -> SpatiotemporalRecord:
    if not isinstance(value, Mapping):
        raise ValueError("spatiotemporal record payload is invalid")
    record = SpatiotemporalRecord(
        record_id=str(value["record_id"]),
        source_id=str(value["source_id"]),
        spatial=_spatial_from_payload(value.get("spatial")),
        temporal=_temporal_from_payload(value.get("temporal")),
        variable=str(value.get("variable", "")),
        modality=str(value.get("modality", "text")),
        content_sha256=str(value.get("content_sha256", "")),
        metadata=dict(value.get("metadata") or {}),
    )
    stored = _digest(value.get("fingerprint"), "record fingerprint")
    if record.fingerprint != stored:
        raise RuntimeError(f"stored hydrology record failed integrity check: {record.record_id}")
    return record


def topology_payload(network: HydroNetwork) -> Mapping[str, Any]:
    if not isinstance(network, HydroNetwork):
        raise TypeError("network must be HydroNetwork")
    return {
        "nodes": [
            {
                "node_id": network.nodes[key].node_id,
                "kind": network.nodes[key].kind,
                "location": _point_payload(network.nodes[key].location),
                "source_id": network.nodes[key].source_id,
            }
            for key in sorted(network.nodes)
        ],
        "reaches": [
            {
                "reach_id": network.reaches[key].reach_id,
                "upstream_node_id": network.reaches[key].upstream_node_id,
                "downstream_node_id": network.reaches[key].downstream_node_id,
                "length_m": network.reaches[key].length_m,
                "source_id": network.reaches[key].source_id,
                "attributes": dict(network.reaches[key].attributes),
            }
            for key in sorted(network.reaches)
        ],
        "fingerprint": network.fingerprint,
    }


def topology_from_payload(value: Any) -> HydroNetwork:
    if not isinstance(value, Mapping):
        raise ValueError("topology payload is invalid")
    nodes_raw, reaches_raw = value.get("nodes"), value.get("reaches")
    if not isinstance(nodes_raw, list) or not isinstance(reaches_raw, list):
        raise ValueError("topology payload collections are invalid")
    nodes = tuple(
        HydroNode(
            node_id=str(item["node_id"]),
            kind=str(item["kind"]),
            location=_point_from_payload(item.get("location")),
            source_id=str(item.get("source_id", "")),
        )
        for item in nodes_raw
        if isinstance(item, Mapping)
    )
    reaches = tuple(
        HydroReach(
            reach_id=str(item["reach_id"]),
            upstream_node_id=str(item["upstream_node_id"]),
            downstream_node_id=str(item["downstream_node_id"]),
            length_m=float(item["length_m"]),
            source_id=str(item.get("source_id", "")),
            attributes=dict(item.get("attributes") or {}),
        )
        for item in reaches_raw
        if isinstance(item, Mapping)
    )
    if len(nodes) != len(nodes_raw) or len(reaches) != len(reaches_raw):
        raise ValueError("topology payload contains non-object rows")
    network = HydroNetwork(nodes, reaches)
    if network.fingerprint != _digest(value.get("fingerprint"), "topology fingerprint"):
        raise RuntimeError("stored hydrology topology failed integrity check")
    return network


def package_payload(package: EngineeringEvidencePackage) -> Mapping[str, Any]:
    if not isinstance(package, EngineeringEvidencePackage):
        raise TypeError("package must be EngineeringEvidencePackage")
    return {
        "package_id": package.package_id,
        "model_type": package.model_type,
        "source_fingerprint": package.source_fingerprint,
        "topology_fingerprint": package.topology_fingerprint,
        "scenario_fingerprint": package.scenario_fingerprint,
        "records": [_record_payload(item) for item in package.records],
        "objects": [asdict(item) for item in package.objects],
        "diagnostics": list(package.diagnostics),
        "reconciliation_fingerprint": package.reconciliation_fingerprint,
        "fingerprint": package.fingerprint,
    }


def _package_identity(package: EngineeringEvidencePackage) -> str:
    value = {
        "package_id": package.package_id,
        "model_type": package.model_type,
        "source_fingerprint": package.source_fingerprint,
        "topology_fingerprint": package.topology_fingerprint,
        "scenario_fingerprint": package.scenario_fingerprint,
        "records": [(item.record_id, item.fingerprint) for item in package.records],
        "objects": [asdict(item) for item in package.objects],
        "diagnostics": sorted(set(package.diagnostics)),
        "reconciliation_fingerprint": package.reconciliation_fingerprint,
    }
    return hashlib.sha256(_canonical(value)).hexdigest()


def package_from_payload(value: Any) -> EngineeringEvidencePackage:
    if not isinstance(value, Mapping):
        raise ValueError("engineering package payload is invalid")
    records_raw, objects_raw = value.get("records"), value.get("objects")
    if not isinstance(records_raw, list) or not isinstance(objects_raw, list):
        raise ValueError("engineering package collections are invalid")
    records = tuple(_record_from_payload(item) for item in records_raw)
    objects = tuple(
        EngineeringEvidenceObject(
            object_id=str(item["object_id"]),
            object_kind=str(item["object_kind"]),
            fingerprint=str(item["fingerprint"]),
            source_id=str(item["source_id"]),
            indexed_record_ids=tuple(item.get("indexed_record_ids") or ()),
            topology_status=str(item.get("topology_status", "not_applicable")),
            topology_target_id=str(item.get("topology_target_id", "")),
        )
        for item in objects_raw
        if isinstance(item, Mapping)
    )
    if len(objects) != len(objects_raw):
        raise ValueError("engineering package contains non-object rows")
    package = EngineeringEvidencePackage(
        package_id=str(value["package_id"]),
        model_type=str(value["model_type"]),
        source_fingerprint=str(value["source_fingerprint"]),
        topology_fingerprint=str(value["topology_fingerprint"]),
        scenario_fingerprint=str(value.get("scenario_fingerprint", "")),
        records=records,
        objects=objects,
        diagnostics=tuple(value.get("diagnostics") or ()),
        reconciliation_fingerprint=str(value.get("reconciliation_fingerprint", "")),
        fingerprint=str(value["fingerprint"]),
    )
    if _package_identity(package) != package.fingerprint:
        raise RuntimeError("stored engineering evidence package failed integrity check")
    return package


def _spec_payload(spec: HydrologyQuerySpec) -> Mapping[str, Any]:
    return {
        "scope": spec.scope,
        "anchor_node_id": spec.anchor_node_id,
        "target_node_id": spec.target_node_id,
        "variable": spec.variable,
        "scenario_ids": list(spec.scenario_ids),
        "start_time": _utc_text(spec.start_time),
        "end_time": _utc_text(spec.end_time),
        "modalities": list(spec.modalities),
        "spatial": _spatial_payload(spec.spatial),
        "max_hops": spec.max_hops,
        "max_paths": spec.max_paths,
        "apply_time_of_travel": spec.apply_time_of_travel,
    }


def _spec_from_payload(value: Any) -> HydrologyQuerySpec:
    if not isinstance(value, Mapping):
        raise ValueError("hydrology query spec payload is invalid")
    start = value.get("start_time")
    end = value.get("end_time")
    return HydrologyQuerySpec(
        scope=str(value["scope"]),
        anchor_node_id=str(value["anchor_node_id"]),
        target_node_id=str(value.get("target_node_id", "")),
        variable=str(value.get("variable", "")),
        scenario_ids=tuple(value.get("scenario_ids") or ()),
        start_time=_parse_time(start, "start_time") if start is not None else None,
        end_time=_parse_time(end, "end_time") if end is not None else None,
        modalities=tuple(value.get("modalities") or ()),
        spatial=_spatial_from_payload(value.get("spatial")),
        max_hops=int(value.get("max_hops", 100)),
        max_paths=int(value.get("max_paths", 1000)),
        apply_time_of_travel=bool(value.get("apply_time_of_travel", False)),
    )


def plan_payload(plan: HydrologyRetrievalPlan) -> Mapping[str, Any]:
    if not isinstance(plan, HydrologyRetrievalPlan):
        raise TypeError("plan must be HydrologyRetrievalPlan")
    return {
        "spec": _spec_payload(plan.spec),
        "node_ids": list(plan.node_ids),
        "reach_ids": list(plan.reach_ids),
        "record_ids": list(plan.record_ids),
        "time_windows": [
            {
                "topology_id": item.topology_id,
                "topology_kind": item.topology_kind,
                "start_time": _utc_text(item.start_time),
                "end_time": _utc_text(item.end_time),
                "travel_seconds_from_anchor": item.travel_seconds_from_anchor,
            }
            for item in plan.time_windows
        ],
        "unresolved": list(plan.unresolved),
        "topology_fingerprint": plan.topology_fingerprint,
        "index_fingerprint": plan.index_fingerprint,
        "fingerprint": plan.fingerprint,
        "selected_records": [asdict(item) for item in plan.selected_records],
        "package_fingerprint": plan.package_fingerprint,
    }


def _plan_identity(plan: HydrologyRetrievalPlan) -> str:
    value = {
        "spec": asdict(plan.spec),
        "node_ids": plan.node_ids,
        "reach_ids": plan.reach_ids,
        "record_ids": list(plan.record_ids),
        "selected_records": [asdict(item) for item in plan.selected_records],
        "time_windows": [asdict(item) for item in plan.time_windows],
        "unresolved": sorted(set(plan.unresolved)),
        "topology_fingerprint": plan.topology_fingerprint,
        "index_fingerprint": plan.index_fingerprint,
        "package_fingerprint": plan.package_fingerprint,
    }
    return hashlib.sha256(_canonical(value)).hexdigest()


def plan_from_payload(value: Any) -> HydrologyRetrievalPlan:
    if not isinstance(value, Mapping):
        raise ValueError("hydrology retrieval-plan payload is invalid")
    windows_raw, traces_raw = value.get("time_windows"), value.get("selected_records")
    if not isinstance(windows_raw, list) or not isinstance(traces_raw, list):
        raise ValueError("hydrology retrieval-plan collections are invalid")
    windows = tuple(
        TopologyTimeWindow(
            topology_id=str(item["topology_id"]),
            topology_kind=str(item["topology_kind"]),
            start_time=_parse_time(item["start_time"], "window start_time"),
            end_time=_parse_time(item["end_time"], "window end_time"),
            travel_seconds_from_anchor=float(item["travel_seconds_from_anchor"]),
        )
        for item in windows_raw
        if isinstance(item, Mapping)
    )
    traces = tuple(
        SelectedRecordTrace(
            record_id=str(item["record_id"]),
            topology_kind=str(item["topology_kind"]),
            topology_id=str(item["topology_id"]),
            scenario_id=str(item.get("scenario_id", "")),
            variable=str(item.get("variable", "")),
            modality=str(item["modality"]),
            reasons=tuple(item.get("reasons") or ()),
            time_window_applied=bool(item.get("time_window_applied", False)),
        )
        for item in traces_raw
        if isinstance(item, Mapping)
    )
    if len(windows) != len(windows_raw) or len(traces) != len(traces_raw):
        raise ValueError("hydrology retrieval plan contains non-object rows")
    plan = HydrologyRetrievalPlan(
        spec=_spec_from_payload(value["spec"]),
        node_ids=tuple(value.get("node_ids") or ()),
        reach_ids=tuple(value.get("reach_ids") or ()),
        record_ids=tuple(value.get("record_ids") or ()),
        time_windows=windows,
        unresolved=tuple(value.get("unresolved") or ()),
        topology_fingerprint=str(value["topology_fingerprint"]),
        index_fingerprint=str(value["index_fingerprint"]),
        fingerprint=str(value["fingerprint"]),
        selected_records=traces,
        package_fingerprint=str(value.get("package_fingerprint", "")),
    )
    if _plan_identity(plan) != plan.fingerprint:
        raise RuntimeError("stored hydrology retrieval plan failed integrity check")
    return plan


@dataclass(frozen=True)
class HydrologyArtifactSummary:
    owner_id: str
    project_id: str
    kind: str
    logical_id: str
    fingerprint: str
    version: int
    created_at: float
    is_current: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "project_id", _text(self.project_id, "project_id", 256))
        kind = _text(self.kind, "kind", 64).lower()
        if kind not in _KINDS:
            raise ValueError("unsupported hydrology artifact kind")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "logical_id", _text(self.logical_id, "logical_id", 500))
        object.__setattr__(self, "fingerprint", _digest(self.fingerprint, "fingerprint"))
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("version is invalid")
        created = float(self.created_at)
        if not math.isfinite(created) or created < 0:
            raise ValueError("created_at is invalid")
        object.__setattr__(self, "created_at", created)
        if not isinstance(self.is_current, bool):
            raise ValueError("is_current must be boolean")


@dataclass(frozen=True)
class HydrologyArtifactEnvelope:
    owner_id: str
    project_id: str
    kind: str
    logical_id: str
    fingerprint: str
    payload: Mapping[str, Any]
    schema_version: int = _SCHEMA_VERSION
    created_at: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "project_id", _text(self.project_id, "project_id", 256))
        kind = _text(self.kind, "kind", 64).lower()
        if kind not in _KINDS:
            raise ValueError("unsupported hydrology artifact kind")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "logical_id", _text(self.logical_id, "logical_id", 500))
        object.__setattr__(self, "fingerprint", _digest(self.fingerprint, "fingerprint"))
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("unsupported hydrology artifact schema version")
        if not isinstance(self.payload, Mapping):
            raise ValueError("payload must be a mapping")
        created = float(self.created_at or time.time())
        if not math.isfinite(created) or created < 0:
            raise ValueError("created_at is invalid")
        object.__setattr__(self, "created_at", created)
        decoded = decode_artifact(self.kind, self.payload)
        if artifact_fingerprint(self.kind, decoded) != self.fingerprint:
            raise RuntimeError("hydrology envelope fingerprint does not match typed payload")


def artifact_fingerprint(kind: str, artifact: Any) -> str:
    if kind == "topology" and isinstance(artifact, HydroNetwork):
        return artifact.fingerprint
    if kind == "engineering_package" and isinstance(artifact, EngineeringEvidencePackage):
        return artifact.fingerprint
    if kind == "retrieval_plan" and isinstance(artifact, HydrologyRetrievalPlan):
        return artifact.fingerprint
    raise TypeError("artifact type does not match hydrology kind")


def encode_artifact(kind: str, artifact: Any) -> Mapping[str, Any]:
    if kind == "topology":
        return topology_payload(artifact)
    if kind == "engineering_package":
        return package_payload(artifact)
    if kind == "retrieval_plan":
        return plan_payload(artifact)
    raise ValueError("unsupported hydrology artifact kind")


def decode_artifact(kind: str, payload: Mapping[str, Any]) -> Any:
    if kind == "topology":
        return topology_from_payload(payload)
    if kind == "engineering_package":
        return package_from_payload(payload)
    if kind == "retrieval_plan":
        return plan_from_payload(payload)
    raise ValueError("unsupported hydrology artifact kind")


def make_envelope(owner_id: str, project_id: str, kind: str, logical_id: str, artifact: Any) -> HydrologyArtifactEnvelope:
    normalized_kind = _text(kind, "kind", 64).lower()
    return HydrologyArtifactEnvelope(
        owner_id=owner_id,
        project_id=project_id,
        kind=normalized_kind,
        logical_id=logical_id,
        fingerprint=artifact_fingerprint(normalized_kind, artifact),
        payload=encode_artifact(normalized_kind, artifact),
    )


class HydrologyArtifactStore(Protocol):
    def put(self, envelope: HydrologyArtifactEnvelope, *, expected_current_fingerprint: str | None = None) -> HydrologyArtifactSummary: ...
    def get(self, owner_id: str, project_id: str, kind: str, logical_id: str, *, fingerprint: str | None = None) -> HydrologyArtifactEnvelope: ...
    def list(self, owner_id: str, project_id: str, *, kind: str | None = None, include_history: bool = False, limit: int = 200) -> tuple[HydrologyArtifactSummary, ...]: ...


__all__ = [
    "HydrologyArtifactEnvelope",
    "HydrologyArtifactStore",
    "HydrologyArtifactSummary",
    "artifact_fingerprint",
    "decode_artifact",
    "encode_artifact",
    "make_envelope",
    "package_from_payload",
    "package_payload",
    "plan_from_payload",
    "plan_payload",
    "strict_json",
    "topology_from_payload",
    "topology_payload",
]
