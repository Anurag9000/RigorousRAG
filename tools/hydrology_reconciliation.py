"""Ambiguity-safe reconciliation between HEC engineering IR and hydrologic topology."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from tools.hydro_topology import HydroNetwork, HydroNode, HydroReach
from tools.hydrology_domain import GeoPoint
from tools.hydrology_engineering_ir import (
    HMSBasinIR,
    HMSElement,
    RASCrossSection,
    RASHydraulicStructure,
    RASPlanIR,
)
from tools.numerical_reasoning import UnitRegistry, default_unit_registry

_STATUSES = frozenset({"matched", "ambiguous", "unmatched"})


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def _normalize(value: str) -> str:
    return "".join(ch for ch in value.casefold() if ch.isalnum())


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be non-negative")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{label} must be non-negative")
    return parsed


def _distance(left: GeoPoint, right: GeoPoint) -> float:
    if left.crs != right.crs:
        raise ValueError("coordinate snapping requires identical CRS")
    return math.hypot(left.x - right.x, left.y - right.y)


def _reach_names(reach: HydroReach) -> tuple[str, str]:
    attrs = {str(key).casefold(): str(value) for key, value in (reach.attributes or {}).items()}
    river = attrs.get("river_name") or attrs.get("river") or attrs.get("stream") or ""
    name = attrs.get("reach_name") or attrs.get("reach") or attrs.get("name") or reach.reach_id
    return _normalize(river), _normalize(name)


@dataclass(frozen=True)
class ReconciliationCandidate:
    target_id: str
    score: int
    basis: str
    coordinate_distance: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _text(self.target_id, "target_id", 256))
        if isinstance(self.score, bool) or not isinstance(self.score, int) or not 0 <= self.score <= 10_000:
            raise ValueError("candidate score is invalid")
        object.__setattr__(self, "basis", _text(self.basis, "basis", 256))
        if self.coordinate_distance is not None:
            object.__setattr__(self, "coordinate_distance", _finite_nonnegative(self.coordinate_distance, "coordinate_distance"))


@dataclass(frozen=True)
class EngineeringMatch:
    source_kind: str
    source_id: str
    status: str
    target_kind: str
    target_id: str = ""
    candidates: tuple[ReconciliationCandidate, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_kind", _text(self.source_kind, "source_kind", 64).lower())
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id", 256))
        status = _text(self.status, "status", 32).lower()
        if status not in _STATUSES:
            raise ValueError("unsupported reconciliation status")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "target_kind", _text(self.target_kind, "target_kind", 64).lower())
        object.__setattr__(self, "target_id", _text(self.target_id, "target_id", 256, allow_empty=True))
        if len(self.candidates) > 1000 or any(not isinstance(item, ReconciliationCandidate) for item in self.candidates):
            raise ValueError("reconciliation candidates are invalid")
        object.__setattr__(self, "reason", _text(self.reason, "reason", 2000, allow_empty=True))
        if status == "matched" and not self.target_id:
            raise ValueError("matched reconciliation requires target_id")
        if status != "matched" and self.target_id:
            raise ValueError("ambiguous/unmatched reconciliation may not claim target_id")


@dataclass(frozen=True)
class StationingIssue:
    river_name: str
    reach_name: str
    upstream_cross_section_id: str
    downstream_cross_section_id: str
    upstream_station: float | None
    downstream_station: float | None
    issue: str


@dataclass(frozen=True)
class ReconciliationReport:
    source_fingerprint: str
    topology_fingerprint: str
    matches: tuple[EngineeringMatch, ...]
    stationing_issues: tuple[StationingIssue, ...]
    fingerprint: str

    @property
    def complete(self) -> bool:
        return all(item.status == "matched" for item in self.matches) and not self.stationing_issues


def _finish_match(
    *,
    source_kind: str,
    source_id: str,
    target_kind: str,
    candidates: Sequence[ReconciliationCandidate],
    unmatched_reason: str,
) -> EngineeringMatch:
    ordered = tuple(sorted(candidates, key=lambda item: (-item.score, item.target_id, item.basis)))
    if not ordered:
        return EngineeringMatch(source_kind, source_id, "unmatched", target_kind, candidates=(), reason=unmatched_reason)
    best = ordered[0].score
    best_rows = tuple(item for item in ordered if item.score == best)
    if len(best_rows) != 1:
        return EngineeringMatch(
            source_kind,
            source_id,
            "ambiguous",
            target_kind,
            candidates=ordered,
            reason="multiple topology candidates have the same strongest reconciliation evidence",
        )
    return EngineeringMatch(source_kind, source_id, "matched", target_kind, best_rows[0].target_id, ordered, best_rows[0].basis)


def reconcile_ras_cross_section(
    cross_section: RASCrossSection,
    network: HydroNetwork,
) -> EngineeringMatch:
    if not isinstance(cross_section, RASCrossSection) or not isinstance(network, HydroNetwork):
        raise TypeError("cross_section/network types are invalid")
    explicit = str(cross_section.metadata.get("hydro_reach_id", "")).strip()
    if explicit:
        if explicit in network.reaches:
            return EngineeringMatch(
                "ras_cross_section",
                cross_section.cross_section_id,
                "matched",
                "reach",
                explicit,
                (ReconciliationCandidate(explicit, 10_000, "explicit_hydro_reach_id"),),
                "explicit reviewed topology binding",
            )
        return EngineeringMatch(
            "ras_cross_section",
            cross_section.cross_section_id,
            "unmatched",
            "reach",
            reason="explicit hydro_reach_id does not exist in the topology",
        )
    river_key = _normalize(cross_section.river_name)
    reach_key = _normalize(cross_section.reach_name)
    candidates: list[ReconciliationCandidate] = []
    for reach in network.reaches.values():
        river, name = _reach_names(reach)
        score = 0
        basis: list[str] = []
        if reach_key and _normalize(reach.reach_id) == reach_key:
            score += 500
            basis.append("reach_id")
        if reach_key and name == reach_key:
            score += 1000
            basis.append("reach_name")
        if river_key and river == river_key:
            score += 1000
            basis.append("river_name")
        if score:
            candidates.append(ReconciliationCandidate(reach.reach_id, score, "+".join(basis)))
    return _finish_match(
        source_kind="ras_cross_section",
        source_id=cross_section.cross_section_id,
        target_kind="reach",
        candidates=candidates,
        unmatched_reason="no explicit or normalized river/reach topology match",
    )


def reconcile_ras_structure(
    structure: RASHydraulicStructure,
    network: HydroNetwork,
    *,
    coordinate_tolerance: float | None = None,
) -> EngineeringMatch:
    if not isinstance(structure, RASHydraulicStructure) or not isinstance(network, HydroNetwork):
        raise TypeError("structure/network types are invalid")
    explicit = str(structure.metadata.get("hydro_node_id", "")).strip()
    if explicit:
        if explicit in network.nodes:
            return EngineeringMatch(
                "ras_structure",
                structure.structure_id,
                "matched",
                "node",
                explicit,
                (ReconciliationCandidate(explicit, 10_000, "explicit_hydro_node_id"),),
                "explicit reviewed topology binding",
            )
        return EngineeringMatch("ras_structure", structure.structure_id, "unmatched", "node", reason="explicit hydro_node_id does not exist in topology")
    tolerance = None if coordinate_tolerance is None else _finite_nonnegative(coordinate_tolerance, "coordinate_tolerance")
    candidates: list[ReconciliationCandidate] = []
    if structure.location is not None and tolerance is not None:
        for node in network.nodes.values():
            if node.location is None or node.location.crs != structure.location.crs:
                continue
            distance = _distance(structure.location, node.location)
            if distance <= tolerance:
                # Same-coordinate evidence is strongest. Distances are deliberately not
                # called metres because CRS coordinate units are not encoded by CRSRef.
                score = 2000 if distance == 0 else max(1001, 2000 - int(999 * distance / max(tolerance, 1e-15)))
                candidates.append(ReconciliationCandidate(node.node_id, score, "coordinate_snap_same_crs", distance))
    normalized_name = _normalize(str(structure.metadata.get("name", structure.structure_id)))
    if normalized_name:
        for node in network.nodes.values():
            attrs_name = _normalize(node.node_id)
            if attrs_name == normalized_name:
                candidates.append(ReconciliationCandidate(node.node_id, 1000, "normalized_node_id"))
    return _finish_match(
        source_kind="ras_structure",
        source_id=structure.structure_id,
        target_kind="node",
        candidates=candidates,
        unmatched_reason="no explicit, same-CRS coordinate, or normalized topology-node match",
    )


def validate_ras_stationing(
    plan: RASPlanIR,
    *,
    expected_downstream_direction: str = "decreasing",
) -> tuple[StationingIssue, ...]:
    if not isinstance(plan, RASPlanIR):
        raise TypeError("plan must be RASPlanIR")
    direction = _text(expected_downstream_direction, "expected_downstream_direction", 32).lower()
    if direction not in {"decreasing", "increasing"}:
        raise ValueError("expected_downstream_direction must be decreasing or increasing")
    grouped: dict[tuple[str, str], list[RASCrossSection]] = {}
    for cross_section in plan.cross_sections:
        grouped.setdefault((_normalize(cross_section.river_name), _normalize(cross_section.reach_name)), []).append(cross_section)
    issues: list[StationingIssue] = []
    for rows in grouped.values():
        for upstream, downstream in zip(rows, rows[1:]):
            left = upstream.numeric_river_station
            right = downstream.numeric_river_station
            if left is None or right is None:
                issues.append(StationingIssue(upstream.river_name, upstream.reach_name, upstream.cross_section_id, downstream.cross_section_id, left, right, "non_numeric_river_station_prevents_order_validation"))
                continue
            valid = left > right if direction == "decreasing" else left < right
            if not valid:
                issues.append(StationingIssue(upstream.river_name, upstream.reach_name, upstream.cross_section_id, downstream.cross_section_id, left, right, f"river_station_not_{direction}_in_source_order"))
    return tuple(issues)


def reconcile_ras_plan(
    plan: RASPlanIR,
    network: HydroNetwork,
    *,
    coordinate_tolerance: float | None = None,
    expected_downstream_direction: str = "decreasing",
) -> ReconciliationReport:
    if not isinstance(plan, RASPlanIR) or not isinstance(network, HydroNetwork):
        raise TypeError("plan/network types are invalid")
    matches = tuple(
        [reconcile_ras_cross_section(item, network) for item in plan.cross_sections]
        + [reconcile_ras_structure(item, network, coordinate_tolerance=coordinate_tolerance) for item in plan.structures]
    )
    issues = validate_ras_stationing(plan, expected_downstream_direction=expected_downstream_direction)
    payload = {
        "source_fingerprint": plan.fingerprint,
        "topology_fingerprint": network.fingerprint,
        "matches": [asdict(item) for item in matches],
        "stationing_issues": [asdict(item) for item in issues],
    }
    return ReconciliationReport(
        plan.fingerprint,
        network.fingerprint,
        matches,
        issues,
        hashlib.sha256(_canonical(payload)).hexdigest(),
    )


def build_hms_network(
    basin: HMSBasinIR,
    *,
    unit_registry: UnitRegistry | None = None,
) -> HydroNetwork:
    """Convert explicit HMS element connectivity into the generic directed hydro network."""
    if not isinstance(basin, HMSBasinIR):
        raise TypeError("basin must be HMSBasinIR")
    registry = unit_registry or default_unit_registry()
    kind_map = {
        "junction": "junction",
        "reservoir": "reservoir",
        "gage": "gauge",
        "source": "source",
        "sink": "sink",
    }
    nodes = tuple(
        HydroNode(
            item.element_id,
            kind_map.get(item.element_type, "other"),
            item.location,
            basin.basin_artifact.source_id,
        )
        for item in basin.elements
    )
    reaches: list[HydroReach] = []
    for connection in basin.connections:
        length_m = 0.0
        attributes: dict[str, str] = {"hms_connection_id": connection.connection_id}
        if connection.length is not None:
            try:
                length_m = registry.convert(connection.length, connection.length_unit, "m")
            except (KeyError, ValueError) as exc:
                raise ValueError(f"HMS connection {connection.connection_id} length unit is not convertible to metres") from exc
            attributes["length_source_unit"] = connection.length_unit
        else:
            attributes["length_unknown"] = "true"
        reaches.append(
            HydroReach(
                connection.connection_id,
                connection.upstream_element_id,
                connection.downstream_element_id,
                length_m,
                basin.basin_artifact.source_id,
                attributes,
            )
        )
    return HydroNetwork(nodes, tuple(reaches))


def reconcile_series_location(
    location_id: str,
    network: HydroNetwork,
    *,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> EngineeringMatch:
    """Resolve time-series location IDs to topology nodes without fuzzy guessing."""
    location = _text(location_id, "location_id", 256)
    normalized = _normalize(location)
    candidates: list[ReconciliationCandidate] = []
    for node in network.nodes.values():
        if node.node_id == location:
            candidates.append(ReconciliationCandidate(node.node_id, 10_000, "exact_node_id"))
        elif _normalize(node.node_id) == normalized:
            candidates.append(ReconciliationCandidate(node.node_id, 1000, "normalized_node_id"))
    for node_id, values in (aliases or {}).items():
        if node_id not in network.nodes:
            raise ValueError(f"series alias references unknown topology node {node_id}")
        normalized_aliases = {_normalize(str(item)) for item in values}
        if normalized in normalized_aliases:
            candidates.append(ReconciliationCandidate(node_id, 5000, "reviewed_alias"))
    return _finish_match(
        source_kind="hydro_series_location",
        source_id=location,
        target_kind="node",
        candidates=candidates,
        unmatched_reason="series location has no exact, reviewed-alias, or normalized node identity",
    )


__all__ = [
    "EngineeringMatch",
    "ReconciliationCandidate",
    "ReconciliationReport",
    "StationingIssue",
    "build_hms_network",
    "reconcile_ras_cross_section",
    "reconcile_ras_plan",
    "reconcile_ras_structure",
    "reconcile_series_location",
    "validate_ras_stationing",
]
