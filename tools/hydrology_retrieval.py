"""Topology-, scenario- and time-aware retrieval planning for hydrology evidence.

The planner is deterministic and bounded. It can optionally fence execution to a compiled
engineering-evidence package and/or an expected index fingerprint, preventing a topology,
package and index from being mixed across generations without an explicit fatal diagnostic.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from tools.hydro_topology import HydroNetwork
from tools.hydrology_domain import HydroScenario
from tools.hydrology_evidence_compiler import EngineeringEvidencePackage
from tools.hydrology_reconciliation import EngineeringMatch, reconcile_series_location
from tools.spatiotemporal_index import (
    SpatialEnvelope,
    SpatiotemporalIndex,
    SpatiotemporalQuery,
    SpatiotemporalRecord,
    TimeEnvelope,
)

_SCOPES = frozenset({"local", "upstream", "downstream", "between"})
_MAX_RECORDS = 10_000


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _digest(value: str, label: str, *, allow_empty: bool = True) -> str:
    cleaned = _text(value, label, 64, allow_empty=allow_empty).lower()
    if cleaned and (len(cleaned) != 64 or any(ch not in "0123456789abcdef" for ch in cleaned)):
        raise ValueError(f"{label} must be SHA-256")
    return cleaned


def _utc(value: dt.datetime) -> dt.datetime:
    if not isinstance(value, dt.datetime):
        raise ValueError("timestamp must be datetime")
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be non-negative")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{label} must be non-negative")
    return parsed


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def _record_scenario(record: SpatiotemporalRecord) -> str:
    return str(record.metadata.get("scenario_id", "")).strip()


def _record_node(record: SpatiotemporalRecord) -> str:
    return str(record.metadata.get("hydro_node_id") or record.metadata.get("location_id") or "").strip()


def _record_reach(record: SpatiotemporalRecord) -> str:
    return str(record.metadata.get("hydro_reach_id") or "").strip()


@dataclass(frozen=True)
class HydrologyQuerySpec:
    scope: str
    anchor_node_id: str
    target_node_id: str = ""
    variable: str = ""
    scenario_ids: tuple[str, ...] = ()
    start_time: dt.datetime | None = None
    end_time: dt.datetime | None = None
    modalities: tuple[str, ...] = ()
    spatial: SpatialEnvelope | None = None
    max_hops: int = 100
    max_paths: int = 1000
    apply_time_of_travel: bool = False

    def __post_init__(self) -> None:
        scope = _text(self.scope, "scope", 32).lower()
        if scope not in _SCOPES:
            raise ValueError("unsupported hydrology topology scope")
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "anchor_node_id", _text(self.anchor_node_id, "anchor_node_id", 256))
        object.__setattr__(self, "target_node_id", _text(self.target_node_id, "target_node_id", 256, allow_empty=True))
        if scope == "between" and not self.target_node_id:
            raise ValueError("between scope requires target_node_id")
        object.__setattr__(self, "variable", _text(self.variable, "variable", 256, allow_empty=True).lower())
        if len(self.scenario_ids) > 1000:
            raise ValueError("scenario_ids exceed the item limit")
        object.__setattr__(self, "scenario_ids", tuple(dict.fromkeys(_text(item, "scenario_id", 256) for item in self.scenario_ids)))
        if (self.start_time is None) != (self.end_time is None):
            raise ValueError("start_time and end_time must be supplied together")
        if self.start_time is not None:
            start, end = _utc(self.start_time), _utc(self.end_time)
            if end < start:
                raise ValueError("hydrology query time range is invalid")
            object.__setattr__(self, "start_time", start)
            object.__setattr__(self, "end_time", end)
        object.__setattr__(self, "modalities", tuple(dict.fromkeys(_text(item, "modality", 64).lower() for item in self.modalities)))
        if self.spatial is not None and not isinstance(self.spatial, SpatialEnvelope):
            raise ValueError("spatial must be SpatialEnvelope or null")
        if isinstance(self.max_hops, bool) or not isinstance(self.max_hops, int) or not 1 <= self.max_hops <= 10_000:
            raise ValueError("max_hops is invalid")
        if isinstance(self.max_paths, bool) or not isinstance(self.max_paths, int) or not 1 <= self.max_paths <= 10_000:
            raise ValueError("max_paths is invalid")
        if not isinstance(self.apply_time_of_travel, bool):
            raise ValueError("apply_time_of_travel must be boolean")


@dataclass(frozen=True)
class TopologyTimeWindow:
    topology_id: str
    topology_kind: str
    start_time: dt.datetime
    end_time: dt.datetime
    travel_seconds_from_anchor: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "topology_id", _text(self.topology_id, "topology_id", 256))
        kind = _text(self.topology_kind, "topology_kind", 32).lower()
        if kind not in {"node", "reach"}:
            raise ValueError("topology_kind must be node or reach")
        object.__setattr__(self, "topology_kind", kind)
        start, end = _utc(self.start_time), _utc(self.end_time)
        if end < start:
            raise ValueError("topology time window is invalid")
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        object.__setattr__(self, "travel_seconds_from_anchor", _nonnegative(self.travel_seconds_from_anchor, "travel_seconds_from_anchor"))


@dataclass(frozen=True)
class SelectedRecordTrace:
    record_id: str
    topology_kind: str
    topology_id: str
    scenario_id: str
    variable: str
    modality: str
    reasons: tuple[str, ...]
    time_window_applied: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", _text(self.record_id, "record_id", 500))
        kind = _text(self.topology_kind, "topology_kind", 32).lower()
        if kind not in {"node", "reach"}:
            raise ValueError("topology_kind must be node or reach")
        object.__setattr__(self, "topology_kind", kind)
        object.__setattr__(self, "topology_id", _text(self.topology_id, "topology_id", 256))
        object.__setattr__(self, "scenario_id", _text(self.scenario_id, "scenario_id", 256, allow_empty=True))
        object.__setattr__(self, "variable", _text(self.variable, "variable", 256, allow_empty=True))
        object.__setattr__(self, "modality", _text(self.modality, "modality", 64))
        object.__setattr__(self, "reasons", tuple(dict.fromkeys(_text(item, "reason", 500) for item in self.reasons)))
        if not isinstance(self.time_window_applied, bool):
            raise ValueError("time_window_applied must be boolean")


@dataclass(frozen=True)
class HydrologyRetrievalPlan:
    spec: HydrologyQuerySpec
    node_ids: tuple[str, ...]
    reach_ids: tuple[str, ...]
    record_ids: tuple[str, ...]
    time_windows: tuple[TopologyTimeWindow, ...]
    unresolved: tuple[str, ...]
    topology_fingerprint: str
    index_fingerprint: str
    fingerprint: str
    selected_records: tuple[SelectedRecordTrace, ...] = ()
    package_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "topology_fingerprint", _digest(self.topology_fingerprint, "topology_fingerprint", allow_empty=False))
        object.__setattr__(self, "index_fingerprint", _digest(self.index_fingerprint, "index_fingerprint", allow_empty=False))
        object.__setattr__(self, "fingerprint", _digest(self.fingerprint, "fingerprint", allow_empty=False))
        object.__setattr__(self, "package_fingerprint", _digest(self.package_fingerprint, "package_fingerprint"))

    @property
    def executable(self) -> bool:
        return bool(self.node_ids or self.reach_ids) and not any(item.startswith("fatal:") for item in self.unresolved)


def _scope_members(network: HydroNetwork, spec: HydrologyQuerySpec) -> tuple[tuple[str, ...], tuple[str, ...], list[str]]:
    if spec.anchor_node_id not in network.nodes:
        return (), (), [f"fatal:anchor_node_not_found:{spec.anchor_node_id}"]
    unresolved: list[str] = []
    if spec.scope == "local":
        nodes = (spec.anchor_node_id,)
        reaches = tuple(sorted({row.reach_id for row in (*network.up.get(spec.anchor_node_id, ()), *network.down.get(spec.anchor_node_id, ()))}))
        return nodes, reaches, unresolved
    if spec.scope == "upstream":
        nodes = (spec.anchor_node_id, *network.upstream_nodes(spec.anchor_node_id, max_hops=spec.max_hops))
    elif spec.scope == "downstream":
        nodes = (spec.anchor_node_id, *network.downstream_nodes(spec.anchor_node_id, max_hops=spec.max_hops))
    else:
        if spec.target_node_id not in network.nodes:
            return (), (), [f"fatal:target_node_not_found:{spec.target_node_id}"]
        paths = network.paths(spec.anchor_node_id, spec.target_node_id, max_paths=spec.max_paths, max_hops=spec.max_hops)
        if not paths:
            return (), (), [f"fatal:no_directed_path:{spec.anchor_node_id}:{spec.target_node_id}"]
        if len(paths) > 1:
            unresolved.append(f"multiple_topology_paths:{len(paths)}")
        node_set = {node for path in paths for node in path.node_ids}
        reach_set = {reach for path in paths for reach in path.reach_ids}
        return tuple(sorted(node_set)), tuple(sorted(reach_set)), unresolved
    node_set = set(nodes)
    reach_set = {
        reach.reach_id
        for reach in network.reaches.values()
        if reach.upstream_node_id in node_set and reach.downstream_node_id in node_set
    }
    return tuple(dict.fromkeys(nodes)), tuple(sorted(reach_set)), unresolved


def _travel_offsets(
    network: HydroNetwork,
    spec: HydrologyQuerySpec,
    node_ids: Sequence[str],
    reach_travel_seconds: Mapping[str, float],
) -> tuple[Mapping[str, float], tuple[str, ...]]:
    if not spec.apply_time_of_travel:
        return {node: 0.0 for node in node_ids}, ()
    missing: set[str] = set()
    offsets: dict[str, float] = {spec.anchor_node_id: 0.0}
    for node in node_ids:
        if node == spec.anchor_node_id:
            continue
        paths = (
            network.paths(node, spec.anchor_node_id, max_paths=spec.max_paths, max_hops=spec.max_hops)
            if spec.scope == "upstream"
            else network.paths(spec.anchor_node_id, node, max_paths=spec.max_paths, max_hops=spec.max_hops)
        )
        totals: list[float] = []
        for path in paths:
            total = 0.0
            valid = True
            for reach_id in path.reach_ids:
                raw = reach_travel_seconds.get(reach_id)
                if raw is None:
                    missing.add(reach_id)
                    valid = False
                    break
                total += _nonnegative(raw, f"travel_seconds:{reach_id}")
            if valid:
                totals.append(total)
        offsets[node] = min(totals) if totals else 0.0
    return offsets, tuple(f"travel_time_missing:{reach_id}" for reach_id in sorted(missing))


def _window_for_node(spec: HydrologyQuerySpec, node_id: str, offset: float) -> TopologyTimeWindow | None:
    if spec.start_time is None or spec.end_time is None:
        return None
    delta = dt.timedelta(seconds=offset)
    if spec.scope == "upstream":
        start, end = spec.start_time - delta, spec.end_time - delta
    else:
        start, end = spec.start_time + delta, spec.end_time + delta
    return TopologyTimeWindow(node_id, "node", start, end, offset)


def _record_matches_window(record: SpatiotemporalRecord, windows: Mapping[str, TopologyTimeWindow]) -> bool:
    node_id = _record_node(record)
    if not node_id or node_id not in windows:
        return True
    window = windows[node_id]
    if record.temporal is None:
        return False
    return record.temporal.intersects(TimeEnvelope(window.start_time, window.end_time))


def _package_records(package: EngineeringEvidencePackage | None) -> Mapping[str, SpatiotemporalRecord]:
    if package is None:
        return {}
    return {item.record_id: item for item in package.records}


def plan_hydrology_retrieval(
    network: HydroNetwork,
    index: SpatiotemporalIndex,
    spec: HydrologyQuerySpec,
    *,
    reach_travel_seconds: Mapping[str, float] | None = None,
    limit: int = 1000,
    package: EngineeringEvidencePackage | None = None,
    expected_index_fingerprint: str = "",
) -> HydrologyRetrievalPlan:
    if not isinstance(network, HydroNetwork) or not isinstance(index, SpatiotemporalIndex) or not isinstance(spec, HydrologyQuerySpec):
        raise TypeError("network/index/spec types are invalid")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_RECORDS:
        raise ValueError(f"limit must be between 1 and {_MAX_RECORDS} for the reference planner")
    if package is not None and not isinstance(package, EngineeringEvidencePackage):
        raise TypeError("package must be EngineeringEvidencePackage or null")
    expected_index = _digest(expected_index_fingerprint, "expected_index_fingerprint")
    current_index = index.fingerprint
    unresolved: list[str] = []
    if expected_index and current_index != expected_index:
        unresolved.append(f"fatal:index_fingerprint_mismatch:{expected_index}:{current_index}")
    if package is not None and package.topology_fingerprint != network.fingerprint:
        unresolved.append(
            f"fatal:package_topology_fingerprint_mismatch:{package.topology_fingerprint}:{network.fingerprint}"
        )

    node_ids, reach_ids, scope_unresolved = _scope_members(network, spec)
    unresolved.extend(scope_unresolved)
    offsets, travel_unresolved = _travel_offsets(network, spec, node_ids, reach_travel_seconds or {})
    unresolved.extend(travel_unresolved)
    windows = tuple(
        row for node in node_ids if (row := _window_for_node(spec, node, offsets.get(node, 0.0))) is not None
    )
    window_index = {item.topology_id: item for item in windows}

    temporal = None
    if spec.start_time is not None and spec.end_time is not None:
        temporal = (
            TimeEnvelope(min(item.start_time for item in windows), max(item.end_time for item in windows))
            if windows
            else TimeEnvelope(spec.start_time, spec.end_time)
        )
    if spec.spatial is None and temporal is None and not spec.variable and not spec.modalities:
        candidates = index.records(limit=limit)
    else:
        query = SpatiotemporalQuery(
            spatial=spec.spatial,
            temporal=temporal,
            variable=spec.variable,
            modalities=spec.modalities,
        )
        candidates = index.search(query, limit=limit)

    package_records = _package_records(package)
    selected: list[str] = []
    traces: list[SelectedRecordTrace] = []
    node_set, reach_set = set(node_ids), set(reach_ids)
    scenario_set = set(spec.scenario_ids)
    for record in candidates:
        if package is not None:
            authoritative = package_records.get(record.record_id)
            if authoritative is None:
                continue
            if authoritative.fingerprint != record.fingerprint:
                unresolved.append(f"fatal:package_index_record_mismatch:{record.record_id}")
                continue
        if scenario_set and _record_scenario(record) not in scenario_set:
            continue
        node_id, reach_id = _record_node(record), _record_reach(record)
        if node_id:
            if node_id not in node_set:
                continue
            topology_kind, topology_id = "node", node_id
        elif reach_id:
            if reach_id not in reach_set:
                continue
            topology_kind, topology_id = "reach", reach_id
        else:
            unresolved.append(f"record_topology_binding_missing:{record.record_id}")
            continue
        if not _record_matches_window(record, window_index):
            continue
        reasons = [f"topology:{spec.scope}:{topology_kind}:{topology_id}"]
        scenario_id = _record_scenario(record)
        if scenario_set:
            reasons.append(f"scenario:{scenario_id}")
        if spec.variable:
            reasons.append(f"variable:{spec.variable}")
        if spec.modalities:
            reasons.append(f"modality:{record.modality}")
        time_applied = bool(node_id and node_id in window_index and record.temporal is not None)
        if time_applied:
            reasons.append("time_window:topology_shifted" if spec.apply_time_of_travel else "time_window:requested")
        if spec.spatial is not None:
            reasons.append("spatial:intersection")
        selected.append(record.record_id)
        traces.append(SelectedRecordTrace(
            record.record_id,
            topology_kind,
            topology_id,
            scenario_id,
            record.variable,
            record.modality,
            tuple(reasons),
            time_applied,
        ))

    payload = {
        "spec": asdict(spec),
        "node_ids": node_ids,
        "reach_ids": reach_ids,
        "record_ids": selected,
        "selected_records": [asdict(item) for item in traces],
        "time_windows": [asdict(item) for item in windows],
        "unresolved": sorted(set(unresolved)),
        "topology_fingerprint": network.fingerprint,
        "index_fingerprint": current_index,
        "package_fingerprint": package.fingerprint if package is not None else "",
    }
    return HydrologyRetrievalPlan(
        spec=spec,
        node_ids=tuple(node_ids),
        reach_ids=tuple(reach_ids),
        record_ids=tuple(selected),
        time_windows=windows,
        unresolved=tuple(sorted(set(unresolved))),
        topology_fingerprint=network.fingerprint,
        index_fingerprint=current_index,
        fingerprint=hashlib.sha256(_canonical(payload)).hexdigest(),
        selected_records=tuple(traces),
        package_fingerprint=package.fingerprint if package is not None else "",
    )


def scenario_records(
    scenario: HydroScenario,
    network: HydroNetwork,
    *,
    aliases: Mapping[str, Sequence[str]] | None = None,
    modality: str = "timeseries",
) -> tuple[SpatiotemporalRecord, ...]:
    """Compile a scenario's time-series identities into topology-bound index records."""
    if not isinstance(scenario, HydroScenario) or not isinstance(network, HydroNetwork):
        raise TypeError("scenario/network types are invalid")
    output: list[SpatiotemporalRecord] = []
    for series in scenario.series:
        match: EngineeringMatch = reconcile_series_location(series.location_id, network, aliases=aliases)
        metadata: dict[str, str] = {
            "scenario_id": scenario.scenario_id,
            "model_type": scenario.model_type,
            "location_id": series.location_id,
            "series_fingerprint": series.fingerprint,
            "topology_match_status": match.status,
        }
        if match.status == "matched":
            metadata["hydro_node_id"] = match.target_id
        else:
            metadata["topology_match_reason"] = match.reason
        output.append(SpatiotemporalRecord(
            record_id=f"hydro:{series.fingerprint}",
            source_id=series.source_id,
            spatial=None,
            temporal=TimeEnvelope(series.points[0].timestamp, series.points[-1].timestamp),
            variable=series.variable,
            modality=_text(modality, "modality", 64).lower(),
            content_sha256=series.fingerprint,
            metadata=metadata,
        ))
    return tuple(output)


__all__ = [
    "HydrologyQuerySpec",
    "HydrologyRetrievalPlan",
    "SelectedRecordTrace",
    "TopologyTimeWindow",
    "plan_hydrology_retrieval",
    "scenario_records",
]
