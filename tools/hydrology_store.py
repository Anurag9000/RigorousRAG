"""Versioned persistence contracts and integrity codecs for hydrology research artifacts.

Every load reconstructs the typed object and recomputes its authoritative identity. Stored
fingerprint columns therefore fence lookup/versioning but never substitute for typed
validation. Unknown schema versions fail closed.
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
from tools.hydrology_projection import HydrologyEvidenceProjection, projection_from_payload, projection_payload
from tools.hydrology_retrieval import HydrologyQuerySpec, HydrologyRetrievalPlan, SelectedRecordTrace, TopologyTimeWindow
from tools.security import normalize_owner_id
from tools.spatiotemporal_index import SpatialEnvelope, SpatiotemporalRecord, TimeEnvelope

_SCHEMA_VERSION = 1
_KINDS = frozenset({"topology", "engineering_package", "retrieval_plan", "evidence_projection"})
_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


def _text(value: Any, label: str, maximum: int = 500, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _digest(value: Any, label: str) -> str:
    cleaned = _text(value, label, 64).lower()
    if len(cleaned) != 64 or any(ch not in "0123456789abcdef" for ch in cleaned):
        raise ValueError(f"{label} must be SHA-256")
    return cleaned


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


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


def _crs_payload(value: CRSRef) -> Mapping[str, str]:
    return {"authority": value.authority, "code": value.code, "axis_order": value.axis_order}


def _crs(value: Any) -> CRSRef:
    if not isinstance(value, Mapping):
        raise ValueError("CRS payload is invalid")
    return CRSRef(str(value.get("authority", "")), str(value.get("code", "")), str(value.get("axis_order", "xy")))


def _point_payload(value: GeoPoint | None) -> Mapping[str, Any] | None:
    return None if value is None else {"x": value.x, "y": value.y, "crs": _crs_payload(value.crs)}


def _point(value: Any) -> GeoPoint | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("GeoPoint payload is invalid")
    return GeoPoint(float(value["x"]), float(value["y"]), _crs(value["crs"]))


def _spatial_payload(value: SpatialEnvelope | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return {"crs": _crs_payload(value.crs), "min_x": value.min_x, "min_y": value.min_y, "max_x": value.max_x, "max_y": value.max_y}


def _spatial(value: Any) -> SpatialEnvelope | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("spatial payload is invalid")
    return SpatialEnvelope(_crs(value["crs"]), float(value["min_x"]), float(value["min_y"]), float(value["max_x"]), float(value["max_y"]))


def _temporal_payload(value: TimeEnvelope | None) -> Mapping[str, str | None] | None:
    return None if value is None else {"start": _utc_text(value.start), "end": _utc_text(value.end)}


def _temporal(value: Any) -> TimeEnvelope | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("temporal payload is invalid")
    return TimeEnvelope(_parse_time(value["start"], "start"), _parse_time(value["end"], "end"))


def _record_payload(value: SpatiotemporalRecord) -> Mapping[str, Any]:
    return {
        "record_id": value.record_id,
        "source_id": value.source_id,
        "spatial": _spatial_payload(value.spatial),
        "temporal": _temporal_payload(value.temporal),
        "variable": value.variable,
        "modality": value.modality,
        "content_sha256": value.content_sha256,
        "metadata": dict(value.metadata),
        "fingerprint": value.fingerprint,
    }


def _record(value: Any) -> SpatiotemporalRecord:
    if not isinstance(value, Mapping):
        raise ValueError("spatiotemporal record payload is invalid")
    result = SpatiotemporalRecord(
        str(value["record_id"]), str(value["source_id"]), _spatial(value.get("spatial")), _temporal(value.get("temporal")),
        str(value.get("variable", "")), str(value.get("modality", "text")), str(value.get("content_sha256", "")), dict(value.get("metadata") or {}),
    )
    if result.fingerprint != _digest(value.get("fingerprint"), "record fingerprint"):
        raise RuntimeError(f"stored hydrology record failed integrity check: {result.record_id}")
    return result


def topology_payload(network: HydroNetwork) -> Mapping[str, Any]:
    if not isinstance(network, HydroNetwork):
        raise TypeError("network must be HydroNetwork")
    return {
        "nodes": [{"node_id": n.node_id, "kind": n.kind, "location": _point_payload(n.location), "source_id": n.source_id} for n in (network.nodes[k] for k in sorted(network.nodes))],
        "reaches": [{"reach_id": r.reach_id, "upstream_node_id": r.upstream_node_id, "downstream_node_id": r.downstream_node_id, "length_m": r.length_m, "source_id": r.source_id, "attributes": dict(r.attributes)} for r in (network.reaches[k] for k in sorted(network.reaches))],
        "fingerprint": network.fingerprint,
    }


def topology_from_payload(value: Any) -> HydroNetwork:
    if not isinstance(value, Mapping) or not isinstance(value.get("nodes"), list) or not isinstance(value.get("reaches"), list):
        raise ValueError("topology payload is invalid")
    node_rows, reach_rows = value["nodes"], value["reaches"]
    nodes = tuple(HydroNode(str(row["node_id"]), str(row["kind"]), _point(row.get("location")), str(row.get("source_id", ""))) for row in node_rows if isinstance(row, Mapping))
    reaches = tuple(HydroReach(str(row["reach_id"]), str(row["upstream_node_id"]), str(row["downstream_node_id"]), float(row["length_m"]), str(row.get("source_id", "")), dict(row.get("attributes") or {})) for row in reach_rows if isinstance(row, Mapping))
    if len(nodes) != len(node_rows) or len(reaches) != len(reach_rows):
        raise ValueError("topology payload contains non-object rows")
    network = HydroNetwork(nodes, reaches)
    if network.fingerprint != _digest(value.get("fingerprint"), "topology fingerprint"):
        raise RuntimeError("stored hydrology topology failed integrity check")
    return network


def package_payload(package: EngineeringEvidencePackage) -> Mapping[str, Any]:
    if not isinstance(package, EngineeringEvidencePackage):
        raise TypeError("package must be EngineeringEvidencePackage")
    return {
        "package_id": package.package_id, "model_type": package.model_type, "source_fingerprint": package.source_fingerprint,
        "topology_fingerprint": package.topology_fingerprint, "scenario_fingerprint": package.scenario_fingerprint,
        "records": [_record_payload(item) for item in package.records], "objects": [asdict(item) for item in package.objects],
        "diagnostics": list(package.diagnostics), "reconciliation_fingerprint": package.reconciliation_fingerprint, "fingerprint": package.fingerprint,
    }


def _package_identity(value: EngineeringEvidencePackage) -> str:
    payload = {
        "package_id": value.package_id, "model_type": value.model_type, "source_fingerprint": value.source_fingerprint,
        "topology_fingerprint": value.topology_fingerprint, "scenario_fingerprint": value.scenario_fingerprint,
        "records": [(r.record_id, r.fingerprint) for r in value.records], "objects": [asdict(item) for item in value.objects],
        "diagnostics": sorted(set(value.diagnostics)), "reconciliation_fingerprint": value.reconciliation_fingerprint,
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


def package_from_payload(value: Any) -> EngineeringEvidencePackage:
    if not isinstance(value, Mapping) or not isinstance(value.get("records"), list) or not isinstance(value.get("objects"), list):
        raise ValueError("engineering package payload is invalid")
    record_rows, object_rows = value["records"], value["objects"]
    records = tuple(_record(row) for row in record_rows)
    objects = tuple(EngineeringEvidenceObject(str(row["object_id"]), str(row["object_kind"]), str(row["fingerprint"]), str(row["source_id"]), tuple(row.get("indexed_record_ids") or ()), str(row.get("topology_status", "not_applicable")), str(row.get("topology_target_id", ""))) for row in object_rows if isinstance(row, Mapping))
    if len(objects) != len(object_rows):
        raise ValueError("engineering package contains non-object rows")
    package = EngineeringEvidencePackage(
        str(value["package_id"]), str(value["model_type"]), str(value["source_fingerprint"]), str(value["topology_fingerprint"]),
        str(value.get("scenario_fingerprint", "")), records, objects, tuple(value.get("diagnostics") or ()), str(value.get("reconciliation_fingerprint", "")), str(value["fingerprint"]),
    )
    if _package_identity(package) != package.fingerprint:
        raise RuntimeError("stored engineering evidence package failed integrity check")
    return package


def query_spec_payload(spec: HydrologyQuerySpec) -> Mapping[str, Any]:
    return {
        "scope": spec.scope, "anchor_node_id": spec.anchor_node_id, "target_node_id": spec.target_node_id, "variable": spec.variable,
        "scenario_ids": list(spec.scenario_ids), "start_time": _utc_text(spec.start_time), "end_time": _utc_text(spec.end_time),
        "modalities": list(spec.modalities), "spatial": _spatial_payload(spec.spatial), "max_hops": spec.max_hops,
        "max_paths": spec.max_paths, "apply_time_of_travel": spec.apply_time_of_travel,
    }


def query_spec_from_payload(value: Any) -> HydrologyQuerySpec:
    if not isinstance(value, Mapping):
        raise ValueError("hydrology query spec payload is invalid")
    start, end = value.get("start_time"), value.get("end_time")
    return HydrologyQuerySpec(
        str(value["scope"]), str(value["anchor_node_id"]), str(value.get("target_node_id", "")), str(value.get("variable", "")),
        tuple(value.get("scenario_ids") or ()), _parse_time(start, "start_time") if start is not None else None,
        _parse_time(end, "end_time") if end is not None else None, tuple(value.get("modalities") or ()), _spatial(value.get("spatial")),
        int(value.get("max_hops", 100)), int(value.get("max_paths", 1000)), bool(value.get("apply_time_of_travel", False)),
    )


_spec_payload = query_spec_payload
_spec_from_payload = query_spec_from_payload


def plan_payload(plan: HydrologyRetrievalPlan) -> Mapping[str, Any]:
    if not isinstance(plan, HydrologyRetrievalPlan):
        raise TypeError("plan must be HydrologyRetrievalPlan")
    return {
        "spec": query_spec_payload(plan.spec), "node_ids": list(plan.node_ids), "reach_ids": list(plan.reach_ids), "record_ids": list(plan.record_ids),
        "time_windows": [{"topology_id": item.topology_id, "topology_kind": item.topology_kind, "start_time": _utc_text(item.start_time), "end_time": _utc_text(item.end_time), "travel_seconds_from_anchor": item.travel_seconds_from_anchor} for item in plan.time_windows],
        "unresolved": list(plan.unresolved), "topology_fingerprint": plan.topology_fingerprint, "index_fingerprint": plan.index_fingerprint,
        "fingerprint": plan.fingerprint, "selected_records": [asdict(item) for item in plan.selected_records], "package_fingerprint": plan.package_fingerprint,
    }


def _plan_identity(plan: HydrologyRetrievalPlan) -> str:
    payload = {
        "spec": asdict(plan.spec), "node_ids": plan.node_ids, "reach_ids": plan.reach_ids, "record_ids": list(plan.record_ids),
        "selected_records": [asdict(item) for item in plan.selected_records], "time_windows": [asdict(item) for item in plan.time_windows],
        "unresolved": sorted(set(plan.unresolved)), "topology_fingerprint": plan.topology_fingerprint,
        "index_fingerprint": plan.index_fingerprint, "package_fingerprint": plan.package_fingerprint,
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


def plan_from_payload(value: Any) -> HydrologyRetrievalPlan:
    if not isinstance(value, Mapping) or not isinstance(value.get("time_windows"), list) or not isinstance(value.get("selected_records"), list):
        raise ValueError("hydrology retrieval-plan payload is invalid")
    window_rows, trace_rows = value["time_windows"], value["selected_records"]
    windows = tuple(TopologyTimeWindow(str(row["topology_id"]), str(row["topology_kind"]), _parse_time(row["start_time"], "window start"), _parse_time(row["end_time"], "window end"), float(row["travel_seconds_from_anchor"])) for row in window_rows if isinstance(row, Mapping))
    traces = tuple(SelectedRecordTrace(str(row["record_id"]), str(row["topology_kind"]), str(row["topology_id"]), str(row.get("scenario_id", "")), str(row.get("variable", "")), str(row["modality"]), tuple(row.get("reasons") or ()), bool(row.get("time_window_applied", False))) for row in trace_rows if isinstance(row, Mapping))
    if len(windows) != len(window_rows) or len(traces) != len(trace_rows):
        raise ValueError("hydrology retrieval plan contains non-object rows")
    plan = HydrologyRetrievalPlan(
        query_spec_from_payload(value["spec"]), tuple(value.get("node_ids") or ()), tuple(value.get("reach_ids") or ()), tuple(value.get("record_ids") or ()),
        windows, tuple(value.get("unresolved") or ()), str(value["topology_fingerprint"]), str(value["index_fingerprint"]), str(value["fingerprint"]), traces, str(value.get("package_fingerprint", "")),
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
        typed = decode_artifact(kind, self.payload)
        if artifact_fingerprint(kind, typed) != self.fingerprint:
            raise RuntimeError("hydrology envelope fingerprint does not match typed payload")


def artifact_fingerprint(kind: str, artifact: Any) -> str:
    if kind == "topology" and isinstance(artifact, HydroNetwork):
        return artifact.fingerprint
    if kind == "engineering_package" and isinstance(artifact, EngineeringEvidencePackage):
        return artifact.fingerprint
    if kind == "retrieval_plan" and isinstance(artifact, HydrologyRetrievalPlan):
        return artifact.fingerprint
    if kind == "evidence_projection" and isinstance(artifact, HydrologyEvidenceProjection):
        return artifact.fingerprint
    raise TypeError("artifact type does not match hydrology kind")


def encode_artifact(kind: str, artifact: Any) -> Mapping[str, Any]:
    if kind == "topology":
        return topology_payload(artifact)
    if kind == "engineering_package":
        return package_payload(artifact)
    if kind == "retrieval_plan":
        return plan_payload(artifact)
    if kind == "evidence_projection":
        return projection_payload(artifact)
    raise ValueError("unsupported hydrology artifact kind")


def decode_artifact(kind: str, payload: Mapping[str, Any]) -> Any:
    if kind == "topology":
        return topology_from_payload(payload)
    if kind == "engineering_package":
        return package_from_payload(payload)
    if kind == "retrieval_plan":
        return plan_from_payload(payload)
    if kind == "evidence_projection":
        return projection_from_payload(payload)
    raise ValueError("unsupported hydrology artifact kind")


def make_envelope(owner_id: str, project_id: str, kind: str, logical_id: str, artifact: Any) -> HydrologyArtifactEnvelope:
    normalized_kind = _text(kind, "kind", 64).lower()
    return HydrologyArtifactEnvelope(owner_id, project_id, normalized_kind, logical_id, artifact_fingerprint(normalized_kind, artifact), encode_artifact(normalized_kind, artifact))


class HydrologyArtifactStore(Protocol):
    def put(self, envelope: HydrologyArtifactEnvelope, *, expected_current_fingerprint: str | None = None) -> HydrologyArtifactSummary: ...
    def get(self, owner_id: str, project_id: str, kind: str, logical_id: str, *, fingerprint: str | None = None) -> HydrologyArtifactEnvelope: ...
    def list(self, owner_id: str, project_id: str, *, kind: str | None = None, include_history: bool = False, limit: int = 200) -> tuple[HydrologyArtifactSummary, ...]: ...


__all__ = [
    "HydrologyArtifactEnvelope", "HydrologyArtifactStore", "HydrologyArtifactSummary", "artifact_fingerprint", "decode_artifact",
    "encode_artifact", "make_envelope", "package_from_payload", "package_payload", "plan_from_payload", "plan_payload",
    "query_spec_from_payload", "query_spec_payload", "strict_json", "topology_from_payload", "topology_payload",
]
