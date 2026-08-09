"""Derived temporal and hypergraph semantics for scientific evidence graphs.

This module never mutates the authoritative evidence graph. Temporal validity is
read only from explicit node metadata, retraction propagation is labelled as a
conservative derived risk signal, and scientific hyperedges are deterministic
secondary artifacts that can be projected for GNN-style processing.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from tools.evidence_graph_types import EvidenceEdge, EvidenceNode

_MAX_MEMBERS = 256
_MAX_HOPS = 12
_ALLOWED_ROLES = frozenset({"claim", "evidence", "method", "dataset", "result"})
_ALLOWED_RELATIONS = frozenset(
    {
        "experiment",
        "joint_support",
        "joint_contradiction",
        "method_result",
        "dataset_result",
    }
)
_DEPENDENCY_EDGES = frozenset({"derived_from", "cites", "uses_method", "uses_dataset"})


def _identifier(value: Any, label: str, maximum: int = 200) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    result = value.strip()
    if (
        not result
        or len(result) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in result)
    ):
        raise ValueError(f"{label} is invalid.")
    return result


def _digest(value: Any, label: str) -> str:
    result = _identifier(value, label, 64).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return result


def _timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return result


def _optional_timestamp(value: Any, label: str) -> float | None:
    return None if value is None else _timestamp(value, label)


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        result = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return result


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class TemporalEvidenceStatus:
    node_id: str
    as_of: float
    status: str
    valid_from: float | None = None
    valid_to: float | None = None
    retracted_at: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _digest(self.node_id, "node_id"))
        object.__setattr__(self, "as_of", _timestamp(self.as_of, "as_of"))
        if self.status not in {"active", "not_yet_valid", "expired", "retracted"}:
            raise ValueError("temporal status is unsupported.")
        start = _optional_timestamp(self.valid_from, "valid_from")
        end = _optional_timestamp(self.valid_to, "valid_to")
        retracted = _optional_timestamp(self.retracted_at, "retracted_at")
        if start is not None and end is not None and end < start:
            raise ValueError("valid_to may not precede valid_from.")
        object.__setattr__(self, "valid_from", start)
        object.__setattr__(self, "valid_to", end)
        object.__setattr__(self, "retracted_at", retracted)

    @property
    def status_digest(self) -> str:
        return _sha256(asdict(self))


def temporal_evidence_status(
    node: EvidenceNode,
    *,
    as_of: float,
) -> TemporalEvidenceStatus:
    """Interpret explicit temporal metadata without inferring missing timestamps."""

    if not isinstance(node, EvidenceNode):
        raise ValueError("node must be EvidenceNode.")
    current = _timestamp(as_of, "as_of")
    metadata = node.metadata
    start = _optional_timestamp(metadata.get("valid_from"), "valid_from")
    end = _optional_timestamp(metadata.get("valid_to"), "valid_to")
    retracted = _optional_timestamp(metadata.get("retracted_at"), "retracted_at")
    if start is not None and end is not None and end < start:
        raise ValueError("valid_to may not precede valid_from.")
    if retracted is not None and retracted <= current:
        status = "retracted"
    elif start is not None and current < start:
        status = "not_yet_valid"
    elif end is not None and current >= end:
        status = "expired"
    else:
        status = "active"
    return TemporalEvidenceStatus(
        node_id=node.node_id,
        as_of=current,
        status=status,
        valid_from=start,
        valid_to=end,
        retracted_at=retracted,
    )


@dataclass(frozen=True)
class RetractionRisk:
    node_id: str
    risk: float
    distance: int
    retracted_source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _digest(self.node_id, "node_id"))
        if isinstance(self.risk, bool):
            raise ValueError("risk must be between 0 and 1.")
        try:
            selected = float(self.risk)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("risk must be between 0 and 1.") from exc
        if not math.isfinite(selected) or not 0.0 <= selected <= 1.0:
            raise ValueError("risk must be between 0 and 1.")
        object.__setattr__(self, "risk", selected)
        object.__setattr__(self, "distance", _integer(self.distance, "distance", 0, _MAX_HOPS))
        sources = tuple(sorted({_digest(value, "retracted_source_id") for value in self.retracted_source_ids}))
        if not sources:
            raise ValueError("retraction risk requires at least one source.")
        object.__setattr__(self, "retracted_source_ids", sources)


def propagate_retraction_risk(
    nodes: Sequence[EvidenceNode],
    edges: Sequence[EvidenceEdge],
    *,
    as_of: float,
    max_hops: int = 4,
    decay: float = 0.70,
) -> dict[str, RetractionRisk]:
    """Propagate conservative dependency risk backwards from explicitly retracted nodes."""

    if isinstance(nodes, (str, bytes, bytearray)) or isinstance(edges, (str, bytes, bytearray)):
        raise ValueError("nodes and edges must be sequences.")
    if len(nodes) > 100_000 or len(edges) > 500_000:
        raise ValueError("graph exceeds the derived temporal processing limit.")
    hop_limit = _integer(max_hops, "max_hops", 1, _MAX_HOPS)
    if isinstance(decay, bool):
        raise ValueError("decay must be between 0 and 1.")
    try:
        selected_decay = float(decay)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("decay must be between 0 and 1.") from exc
    if not math.isfinite(selected_decay) or not 0.0 < selected_decay <= 1.0:
        raise ValueError("decay must be between 0 and 1.")
    node_map = {node.node_id: node for node in nodes if isinstance(node, EvidenceNode)}
    if len(node_map) != len(nodes):
        raise ValueError("nodes must contain unique EvidenceNode values.")
    reverse_dependencies: dict[str, set[str]] = {node_id: set() for node_id in node_map}
    for edge in edges:
        if not isinstance(edge, EvidenceEdge):
            raise ValueError("edges must contain EvidenceEdge values.")
        if edge.source_node_id not in node_map or edge.target_node_id not in node_map:
            raise ValueError("edge endpoints must exist in nodes.")
        if edge.edge_type in _DEPENDENCY_EDGES:
            reverse_dependencies[edge.target_node_id].add(edge.source_node_id)
    retracted = {
        node_id
        for node_id, node in node_map.items()
        if temporal_evidence_status(node, as_of=as_of).status == "retracted"
    }
    if not retracted:
        return {}
    best_distance: dict[str, int] = {node_id: 0 for node_id in retracted}
    sources: dict[str, set[str]] = {node_id: {node_id} for node_id in retracted}
    queue = deque((node_id, 0, node_id) for node_id in sorted(retracted))
    while queue:
        current, distance, origin = queue.popleft()
        if distance >= hop_limit:
            continue
        for dependent in sorted(reverse_dependencies[current]):
            next_distance = distance + 1
            previous = best_distance.get(dependent)
            if previous is None or next_distance < previous:
                best_distance[dependent] = next_distance
                sources[dependent] = {origin}
                queue.append((dependent, next_distance, origin))
            elif next_distance == previous and origin not in sources[dependent]:
                sources[dependent].add(origin)
                queue.append((dependent, next_distance, origin))
    return {
        node_id: RetractionRisk(
            node_id=node_id,
            risk=1.0 if distance == 0 else selected_decay**distance,
            distance=distance,
            retracted_source_ids=tuple(sorted(sources[node_id])),
        )
        for node_id, distance in sorted(best_distance.items())
    }


@dataclass(frozen=True)
class ScientificHyperedge:
    hyperedge_id: str
    relation_type: str
    roles: Mapping[str, tuple[str, ...]]
    weight: float = 1.0

    def __post_init__(self) -> None:
        relation = _identifier(self.relation_type, "relation_type", 100)
        if relation not in _ALLOWED_RELATIONS:
            raise ValueError("scientific hyperedge relation is unsupported.")
        object.__setattr__(self, "relation_type", relation)
        if not isinstance(self.roles, Mapping) or not self.roles:
            raise ValueError("scientific hyperedge roles must be a mapping.")
        normalized: dict[str, tuple[str, ...]] = {}
        total = 0
        for role, raw_ids in self.roles.items():
            selected_role = _identifier(role, "role", 50)
            if selected_role not in _ALLOWED_ROLES:
                raise ValueError("scientific hyperedge role is unsupported.")
            if not isinstance(raw_ids, tuple) or not raw_ids:
                raise ValueError("scientific hyperedge role members must be a non-empty tuple.")
            values = tuple(sorted({_digest(value, "member_node_id") for value in raw_ids}))
            total += len(values)
            normalized[selected_role] = values
        if total < 2 or total > _MAX_MEMBERS:
            raise ValueError("scientific hyperedge member count is invalid.")
        object.__setattr__(self, "roles", normalized)
        if isinstance(self.weight, bool):
            raise ValueError("hyperedge weight must be between 0 and 1.")
        try:
            selected_weight = float(self.weight)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("hyperedge weight must be between 0 and 1.") from exc
        if not math.isfinite(selected_weight) or not 0.0 <= selected_weight <= 1.0:
            raise ValueError("hyperedge weight must be between 0 and 1.")
        object.__setattr__(self, "weight", selected_weight)
        expected = _sha256(
            {
                "contract": "rigorousrag-scientific-hyperedge-v1",
                "relation_type": relation,
                "roles": normalized,
            }
        )
        if _digest(self.hyperedge_id, "hyperedge_id") != expected:
            raise ValueError("hyperedge_id does not match deterministic identity.")
        object.__setattr__(self, "hyperedge_id", expected)

    @property
    def members(self) -> tuple[str, ...]:
        return tuple(sorted({value for values in self.roles.values() for value in values}))

    @property
    def provenance_digest(self) -> str:
        return _sha256(asdict(self))


def build_scientific_hyperedge(
    *,
    relation_type: str,
    roles: Mapping[str, Sequence[str]],
    weight: float = 1.0,
) -> ScientificHyperedge:
    if not isinstance(roles, Mapping):
        raise ValueError("roles must be a mapping.")
    normalized = {
        _identifier(role, "role", 50): tuple(values)
        for role, values in roles.items()
    }
    relation = _identifier(relation_type, "relation_type", 100)
    hyperedge_id = _sha256(
        {
            "contract": "rigorousrag-scientific-hyperedge-v1",
            "relation_type": relation,
            "roles": {
                role: tuple(sorted({_digest(value, "member_node_id") for value in values}))
                for role, values in normalized.items()
            },
        }
    )
    return ScientificHyperedge(
        hyperedge_id=hyperedge_id,
        relation_type=relation,
        roles={role: tuple(values) for role, values in normalized.items()},
        weight=weight,
    )


@dataclass(frozen=True)
class HypergraphProjectionEdge:
    hyperedge_id: str
    left_node_id: str
    right_node_id: str
    weight: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "hyperedge_id", _digest(self.hyperedge_id, "hyperedge_id"))
        left = _digest(self.left_node_id, "left_node_id")
        right = _digest(self.right_node_id, "right_node_id")
        if left == right:
            raise ValueError("projected hypergraph edges may not self-loop.")
        if right < left:
            left, right = right, left
        object.__setattr__(self, "left_node_id", left)
        object.__setattr__(self, "right_node_id", right)
        if isinstance(self.weight, bool):
            raise ValueError("projection weight must be between 0 and 1.")
        selected = float(self.weight)
        if not math.isfinite(selected) or not 0.0 <= selected <= 1.0:
            raise ValueError("projection weight must be between 0 and 1.")
        object.__setattr__(self, "weight", selected)


def project_hyperedges_for_gnn(
    hyperedges: Sequence[ScientificHyperedge],
) -> tuple[HypergraphProjectionEdge, ...]:
    """Clique-project hyperedges with degree-normalized weights for derived GNN input."""

    if isinstance(hyperedges, (str, bytes, bytearray)) or len(hyperedges) > 100_000:
        raise ValueError("hyperedges must be a bounded sequence.")
    projected: list[HypergraphProjectionEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for hyperedge in hyperedges:
        if not isinstance(hyperedge, ScientificHyperedge):
            raise ValueError("every hyperedge must be ScientificHyperedge.")
        members = hyperedge.members
        degree_weight = hyperedge.weight / max(len(members) - 1, 1)
        for index, left in enumerate(members):
            for right in members[index + 1 :]:
                key = (hyperedge.hyperedge_id, left, right)
                if key in seen:
                    continue
                seen.add(key)
                projected.append(
                    HypergraphProjectionEdge(
                        hyperedge_id=hyperedge.hyperedge_id,
                        left_node_id=left,
                        right_node_id=right,
                        weight=degree_weight,
                    )
                )
    return tuple(projected)


__all__ = [
    "HypergraphProjectionEdge",
    "RetractionRisk",
    "ScientificHyperedge",
    "TemporalEvidenceStatus",
    "build_scientific_hyperedge",
    "project_hyperedges_for_gnn",
    "propagate_retraction_risk",
    "temporal_evidence_status",
]
