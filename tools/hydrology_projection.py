"""Citation-neutral evidence projections over persisted hydrology packages and plans.

A projection contains only authoritative record identities, scopes, topology bindings and
selection reasons. It never fabricates narrative evidence or copies arbitrary source text.
The public row codec is shared by projections, deterministic reports and future UI/export
surfaces so CRS/time/topology parsing has one authority.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from tools.hydrology_domain import CRSRef
from tools.hydrology_evidence_compiler import EngineeringEvidencePackage
from tools.hydrology_retrieval import HydrologyRetrievalPlan
from tools.spatiotemporal_index import SpatialEnvelope, SpatiotemporalRecord

_MAX_ROWS = 100_000
_MAX_DIAGNOSTICS = 200_000


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _digest(value: Any, label: str, *, allow_empty: bool = False) -> str:
    cleaned = _text(value, label, 64, allow_empty=allow_empty).lower()
    if not cleaned and allow_empty:
        return ""
    if len(cleaned) != 64 or any(ch not in "0123456789abcdef" for ch in cleaned):
        raise ValueError(f"{label} must be SHA-256")
    return cleaned


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def _utc(value: dt.datetime | None) -> dt.datetime | None:
    if value is None:
        return None
    if not isinstance(value, dt.datetime):
        raise ValueError("timestamp must be datetime")
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _parse_time(value: Any) -> dt.datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 100:
        raise ValueError("projection timestamp is invalid")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("projection timestamp is invalid") from exc
    return _utc(parsed)


def _spatial_payload(value: SpatialEnvelope | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return {
        "crs": {"authority": value.crs.authority, "code": value.crs.code, "axis_order": value.crs.axis_order},
        "min_x": value.min_x,
        "min_y": value.min_y,
        "max_x": value.max_x,
        "max_y": value.max_y,
    }


def _spatial_from_payload(value: Any) -> SpatialEnvelope | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not isinstance(value.get("crs"), Mapping):
        raise ValueError("projection spatial scope is invalid")
    crs = value["crs"]
    return SpatialEnvelope(
        CRSRef(str(crs["authority"]), str(crs["code"]), str(crs.get("axis_order", "xy"))),
        float(value["min_x"]),
        float(value["min_y"]),
        float(value["max_x"]),
        float(value["max_y"]),
    )


@dataclass(frozen=True)
class HydrologyEvidenceRow:
    record_id: str
    source_id: str
    content_sha256: str
    variable: str
    modality: str
    scenario_id: str
    topology_kind: str
    topology_id: str
    selection_reasons: tuple[str, ...]
    spatial: SpatialEnvelope | None = None
    start_time: dt.datetime | None = None
    end_time: dt.datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", _text(self.record_id, "record_id", 500))
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id", 1000))
        object.__setattr__(self, "content_sha256", _digest(self.content_sha256, "content_sha256", allow_empty=True))
        object.__setattr__(self, "variable", _text(self.variable, "variable", 256, allow_empty=True))
        object.__setattr__(self, "modality", _text(self.modality, "modality", 64).lower())
        object.__setattr__(self, "scenario_id", _text(self.scenario_id, "scenario_id", 256, allow_empty=True))
        kind = _text(self.topology_kind, "topology_kind", 32).lower()
        if kind not in {"node", "reach"}:
            raise ValueError("topology_kind must be node or reach")
        object.__setattr__(self, "topology_kind", kind)
        object.__setattr__(self, "topology_id", _text(self.topology_id, "topology_id", 256))
        object.__setattr__(self, "selection_reasons", tuple(dict.fromkeys(_text(item, "selection_reason", 500) for item in self.selection_reasons)))
        if self.spatial is not None and not isinstance(self.spatial, SpatialEnvelope):
            raise ValueError("spatial must be SpatialEnvelope or null")
        start, end = _utc(self.start_time), _utc(self.end_time)
        if (start is None) != (end is None):
            raise ValueError("projection time range must provide both endpoints")
        if start is not None and end is not None and end < start:
            raise ValueError("projection time range is invalid")
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)


def evidence_row_payload(row: HydrologyEvidenceRow) -> Mapping[str, Any]:
    if not isinstance(row, HydrologyEvidenceRow):
        raise TypeError("row must be HydrologyEvidenceRow")
    return {
        "record_id": row.record_id,
        "source_id": row.source_id,
        "content_sha256": row.content_sha256,
        "variable": row.variable,
        "modality": row.modality,
        "scenario_id": row.scenario_id,
        "topology_kind": row.topology_kind,
        "topology_id": row.topology_id,
        "selection_reasons": list(row.selection_reasons),
        "spatial": _spatial_payload(row.spatial),
        "start_time": row.start_time.isoformat() if row.start_time is not None else None,
        "end_time": row.end_time.isoformat() if row.end_time is not None else None,
    }


def evidence_row_from_payload(value: Any) -> HydrologyEvidenceRow:
    if not isinstance(value, Mapping):
        raise ValueError("hydrology evidence row payload is invalid")
    return HydrologyEvidenceRow(
        record_id=str(value["record_id"]),
        source_id=str(value["source_id"]),
        content_sha256=str(value.get("content_sha256", "")),
        variable=str(value.get("variable", "")),
        modality=str(value["modality"]),
        scenario_id=str(value.get("scenario_id", "")),
        topology_kind=str(value["topology_kind"]),
        topology_id=str(value["topology_id"]),
        selection_reasons=tuple(value.get("selection_reasons") or ()),
        spatial=_spatial_from_payload(value.get("spatial")),
        start_time=_parse_time(value.get("start_time")),
        end_time=_parse_time(value.get("end_time")),
    )


@dataclass(frozen=True)
class HydrologyEvidenceProjection:
    projection_id: str
    package_fingerprint: str
    topology_fingerprint: str
    plan_fingerprint: str
    index_fingerprint: str
    rows: tuple[HydrologyEvidenceRow, ...]
    package_diagnostics: tuple[str, ...] = ()
    plan_unresolved: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "projection_id", _text(self.projection_id, "projection_id", 500))
        for name in ("package_fingerprint", "topology_fingerprint", "plan_fingerprint", "index_fingerprint"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if len(self.rows) > _MAX_ROWS or any(not isinstance(item, HydrologyEvidenceRow) for item in self.rows):
            raise ValueError("projection rows are invalid")
        if len({item.record_id for item in self.rows}) != len(self.rows):
            raise ValueError("projection contains duplicate record IDs")
        if len(self.package_diagnostics) > _MAX_DIAGNOSTICS or len(self.plan_unresolved) > _MAX_DIAGNOSTICS:
            raise ValueError("projection diagnostics exceed the item limit")
        object.__setattr__(self, "package_diagnostics", tuple(dict.fromkeys(_text(item, "package_diagnostic", 2000) for item in self.package_diagnostics)))
        object.__setattr__(self, "plan_unresolved", tuple(dict.fromkeys(_text(item, "plan_unresolved", 2000) for item in self.plan_unresolved)))

    @property
    def fingerprint(self) -> str:
        payload = {
            "projection_id": self.projection_id,
            "package_fingerprint": self.package_fingerprint,
            "topology_fingerprint": self.topology_fingerprint,
            "plan_fingerprint": self.plan_fingerprint,
            "index_fingerprint": self.index_fingerprint,
            "rows": [asdict(item) for item in self.rows],
            "package_diagnostics": list(self.package_diagnostics),
            "plan_unresolved": list(self.plan_unresolved),
        }
        return hashlib.sha256(_canonical(payload)).hexdigest()

    @property
    def complete(self) -> bool:
        return not any(item.startswith("fatal:") for item in (*self.package_diagnostics, *self.plan_unresolved))


def build_hydrology_projection(
    package: EngineeringEvidencePackage,
    plan: HydrologyRetrievalPlan,
    *,
    projection_id: str,
) -> HydrologyEvidenceProjection:
    if not isinstance(package, EngineeringEvidencePackage) or not isinstance(plan, HydrologyRetrievalPlan):
        raise TypeError("package and plan types are invalid")
    if plan.package_fingerprint != package.fingerprint:
        raise ValueError("retrieval plan is not bound to the selected engineering package")
    if plan.topology_fingerprint != package.topology_fingerprint:
        raise ValueError("retrieval plan topology does not match the engineering package")
    records = {item.record_id: item for item in package.records}
    traces = {item.record_id: item for item in plan.selected_records}
    if set(plan.record_ids) != set(traces):
        raise ValueError("retrieval plan record IDs and selection traces are inconsistent")
    rows: list[HydrologyEvidenceRow] = []
    for record_id in plan.record_ids:
        record: SpatiotemporalRecord | None = records.get(record_id)
        trace = traces.get(record_id)
        if record is None or trace is None:
            raise ValueError(f"retrieval plan references a record absent from the engineering package: {record_id}")
        rows.append(HydrologyEvidenceRow(
            record_id=record.record_id,
            source_id=record.source_id,
            content_sha256=record.content_sha256,
            variable=record.variable,
            modality=record.modality,
            scenario_id=str(record.metadata.get("scenario_id", "")).strip(),
            topology_kind=trace.topology_kind,
            topology_id=trace.topology_id,
            selection_reasons=trace.reasons,
            spatial=record.spatial,
            start_time=record.temporal.start if record.temporal is not None else None,
            end_time=record.temporal.end if record.temporal is not None else None,
        ))
    return HydrologyEvidenceProjection(
        projection_id=projection_id,
        package_fingerprint=package.fingerprint,
        topology_fingerprint=package.topology_fingerprint,
        plan_fingerprint=plan.fingerprint,
        index_fingerprint=plan.index_fingerprint,
        rows=tuple(rows),
        package_diagnostics=package.diagnostics,
        plan_unresolved=plan.unresolved,
    )


def projection_payload(projection: HydrologyEvidenceProjection) -> Mapping[str, Any]:
    if not isinstance(projection, HydrologyEvidenceProjection):
        raise TypeError("projection must be HydrologyEvidenceProjection")
    return {
        "projection_id": projection.projection_id,
        "package_fingerprint": projection.package_fingerprint,
        "topology_fingerprint": projection.topology_fingerprint,
        "plan_fingerprint": projection.plan_fingerprint,
        "index_fingerprint": projection.index_fingerprint,
        "rows": [evidence_row_payload(item) for item in projection.rows],
        "package_diagnostics": list(projection.package_diagnostics),
        "plan_unresolved": list(projection.plan_unresolved),
        "fingerprint": projection.fingerprint,
    }


def projection_from_payload(value: Any) -> HydrologyEvidenceProjection:
    if not isinstance(value, Mapping) or not isinstance(value.get("rows"), list):
        raise ValueError("hydrology projection payload is invalid")
    rows = tuple(evidence_row_from_payload(item) for item in value["rows"])
    projection = HydrologyEvidenceProjection(
        projection_id=str(value["projection_id"]),
        package_fingerprint=str(value["package_fingerprint"]),
        topology_fingerprint=str(value["topology_fingerprint"]),
        plan_fingerprint=str(value["plan_fingerprint"]),
        index_fingerprint=str(value["index_fingerprint"]),
        rows=rows,
        package_diagnostics=tuple(value.get("package_diagnostics") or ()),
        plan_unresolved=tuple(value.get("plan_unresolved") or ()),
    )
    stored = _digest(value.get("fingerprint"), "projection fingerprint")
    if projection.fingerprint != stored:
        raise RuntimeError("stored hydrology evidence projection failed integrity check")
    return projection


__all__ = [
    "HydrologyEvidenceProjection",
    "HydrologyEvidenceRow",
    "build_hydrology_projection",
    "evidence_row_from_payload",
    "evidence_row_payload",
    "projection_from_payload",
    "projection_payload",
]
