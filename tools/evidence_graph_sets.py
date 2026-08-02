"""Provenance-preserving cross-document evidence graph sets.

Graph sets reference exact authoritative document graph generations. They never
merge source identity and never infer cross-document relations from text.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from tools.security import normalize_owner_id

_CROSS_EDGE_TYPES = frozenset(
    {"cites", "same_as", "supports", "contradicts", "derived_from", "mentions"}
)
_MAX_MEMBERS = 1_000
_MAX_RELATIONS = 100_000
_MAX_METADATA_ITEMS = 64
_MAX_METADATA_BYTES = 50_000


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > maximum or any(
        ord(character) < 32 or ord(character) == 127 for character in cleaned
    ):
        raise ValueError(f"{label} is invalid.")
    return cleaned


def _digest(value: Any, label: str) -> str:
    cleaned = _identifier(value, label, 64).lower()
    if len(cleaned) != 64 or any(character not in "0123456789abcdef" for character in cleaned):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return cleaned


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return result


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a mapping.")
    result: dict[str, Any] = {}
    for index, (key, item) in enumerate(value.items()):
        if index >= _MAX_METADATA_ITEMS:
            raise ValueError("metadata contains too many fields.")
        selected = _identifier(key, "metadata key", 200)
        if item is None or isinstance(item, (bool, int)):
            result[selected] = item
        elif isinstance(item, float) and math.isfinite(item):
            result[selected] = item
        elif isinstance(item, str) and len(item) <= 10_000 and "\x00" not in item:
            result[selected] = item
        else:
            raise ValueError("metadata contains an unsupported value.")
    encoded = json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode("utf-8")) > _MAX_METADATA_BYTES:
        raise ValueError("metadata exceeds the byte limit.")
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
class GraphGenerationReference:
    owner_id: str
    doc_id: str
    generation: int
    content_sha256: str
    profile_fingerprint: str
    graph_digest: str
    authority_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "doc_id", _identifier(self.doc_id, "doc_id", 200))
        object.__setattr__(
            self, "generation", _integer(self.generation, "generation", 1, 2**63 - 1)
        )
        for name in (
            "content_sha256",
            "profile_fingerprint",
            "graph_digest",
            "authority_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))

    @property
    def member_digest(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class CrossDocumentNodeReference:
    doc_id: str
    generation: int
    graph_digest: str
    node_id: str
    node_type: str
    provenance_digest: str
    label: str
    page_number: int | None = None
    section: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "doc_id", _identifier(self.doc_id, "doc_id", 200))
        object.__setattr__(
            self, "generation", _integer(self.generation, "generation", 1, 2**63 - 1)
        )
        for name in ("graph_digest", "node_id", "provenance_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "node_type", _identifier(self.node_type, "node_type", 50))
        object.__setattr__(self, "label", _identifier(self.label, "label", 2_000))
        if self.page_number is not None:
            object.__setattr__(
                self,
                "page_number",
                _integer(self.page_number, "page_number", 1, 1_000_000),
            )
        if self.section is not None:
            object.__setattr__(self, "section", _identifier(self.section, "section", 2_000))

    @property
    def reference_digest(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class ExplicitCrossDocumentRelation:
    relation_key: str
    source_doc_id: str
    source_node_id: str
    target_doc_id: str
    target_node_id: str
    edge_type: str
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation_key", _identifier(self.relation_key, "relation_key", 2_000))
        object.__setattr__(self, "source_doc_id", _identifier(self.source_doc_id, "source_doc_id", 200))
        object.__setattr__(self, "target_doc_id", _identifier(self.target_doc_id, "target_doc_id", 200))
        if self.source_doc_id == self.target_doc_id:
            raise ValueError("cross-document relations must connect different documents.")
        object.__setattr__(self, "source_node_id", _digest(self.source_node_id, "source_node_id"))
        object.__setattr__(self, "target_node_id", _digest(self.target_node_id, "target_node_id"))
        selected = _identifier(self.edge_type, "edge_type", 50)
        if selected not in _CROSS_EDGE_TYPES:
            raise ValueError("cross-document edge_type is unsupported.")
        object.__setattr__(self, "edge_type", selected)
        if isinstance(self.weight, bool):
            raise ValueError("weight must be finite and between 0 and 1.")
        try:
            weight = float(self.weight)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("weight must be finite and between 0 and 1.") from exc
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("weight must be finite and between 0 and 1.")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True)
class CrossDocumentEdge:
    edge_id: str
    owner_id: str
    graph_set_id: str
    source: CrossDocumentNodeReference
    target: CrossDocumentNodeReference
    edge_type: str
    relation_key: str
    weight: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        set_id = _digest(self.graph_set_id, "graph_set_id")
        if not isinstance(self.source, CrossDocumentNodeReference) or not isinstance(
            self.target, CrossDocumentNodeReference
        ):
            raise ValueError("cross-document edge endpoints must be node references.")
        if self.source.doc_id == self.target.doc_id:
            raise ValueError("cross-document edges must connect different documents.")
        selected = _identifier(self.edge_type, "edge_type", 50)
        if selected not in _CROSS_EDGE_TYPES:
            raise ValueError("cross-document edge_type is unsupported.")
        relation_key = _identifier(self.relation_key, "relation_key", 2_000)
        if isinstance(self.weight, bool):
            raise ValueError("weight must be finite and between 0 and 1.")
        weight = float(self.weight)
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("weight must be finite and between 0 and 1.")
        metadata = _metadata(self.metadata)
        expected = _sha256(
            {
                "scope": "rigorousrag-cross-document-edge-v1",
                "owner_id": owner,
                "graph_set_id": set_id,
                "source": self.source.reference_digest,
                "target": self.target.reference_digest,
                "edge_type": selected,
                "relation_key": relation_key,
            }
        )
        if _digest(self.edge_id, "edge_id") != expected:
            raise ValueError("edge_id does not match deterministic cross-document identity.")
        object.__setattr__(self, "edge_id", expected)
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "graph_set_id", set_id)
        object.__setattr__(self, "edge_type", selected)
        object.__setattr__(self, "relation_key", relation_key)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "metadata", metadata)

    @property
    def provenance_digest(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class EvidenceGraphSet:
    graph_set_id: str
    owner_id: str
    graph_set_key: str
    members: tuple[GraphGenerationReference, ...]
    edges: tuple[CrossDocumentEdge, ...]
    created_at: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        key = _identifier(self.graph_set_key, "graph_set_key", 500)
        if not isinstance(self.members, tuple) or not 2 <= len(self.members) <= _MAX_MEMBERS:
            raise ValueError("graph sets require a bounded tuple of at least two members.")
        members = tuple(sorted(self.members, key=lambda item: item.doc_id))
        if any(not isinstance(item, GraphGenerationReference) for item in members):
            raise ValueError("every member must be GraphGenerationReference.")
        if any(item.owner_id != owner for item in members):
            raise ValueError("graph set member escaped owner scope.")
        if len({item.doc_id for item in members}) != len(members):
            raise ValueError("graph set members must use unique document IDs.")
        expected_set_id = _sha256(
            {
                "scope": "rigorousrag-evidence-graph-set-v1",
                "owner_id": owner,
                "graph_set_key": key,
                "members": [item.member_digest for item in members],
            }
        )
        if _digest(self.graph_set_id, "graph_set_id") != expected_set_id:
            raise ValueError("graph_set_id does not match member generation identity.")
        if not isinstance(self.edges, tuple) or len(self.edges) > _MAX_RELATIONS:
            raise ValueError("edges must be a bounded tuple.")
        member_map = {item.doc_id: item for item in members}
        edge_ids: set[str] = set()
        for edge in self.edges:
            if not isinstance(edge, CrossDocumentEdge):
                raise ValueError("every edge must be CrossDocumentEdge.")
            if edge.owner_id != owner or edge.graph_set_id != expected_set_id:
                raise ValueError("cross-document edge escaped graph set scope.")
            if edge.edge_id in edge_ids:
                raise ValueError("graph set contains duplicate edge IDs.")
            edge_ids.add(edge.edge_id)
            for endpoint in (edge.source, edge.target):
                member = member_map.get(endpoint.doc_id)
                if member is None or (
                    endpoint.generation != member.generation
                    or endpoint.graph_digest != member.graph_digest
                ):
                    raise ValueError("cross-document edge endpoint differs from member identity.")
        object.__setattr__(self, "graph_set_id", expected_set_id)
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "graph_set_key", key)
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "created_at"))
        if self.schema_version != 1:
            raise ValueError("graph set schema is unsupported.")

    @property
    def graph_set_digest(self) -> str:
        value = asdict(self)
        value.pop("created_at", None)
        return _sha256(value)


@dataclass(frozen=True)
class CrossDocumentPath:
    nodes: tuple[CrossDocumentNodeReference, ...]
    edges: tuple[CrossDocumentEdge, ...]

    def __post_init__(self) -> None:
        if len(self.nodes) != len(self.edges) + 1 or not self.edges:
            raise ValueError("cross-document path shape is invalid.")
        for index, edge in enumerate(self.edges):
            if edge.source != self.nodes[index] or edge.target != self.nodes[index + 1]:
                raise ValueError("cross-document path adjacency is invalid.")
        identities = [(node.doc_id, node.node_id) for node in self.nodes]
        if len(set(identities)) != len(identities):
            raise ValueError("cross-document paths must be simple.")

    @property
    def path_digest(self) -> str:
        return _sha256(
            {
                "nodes": [node.reference_digest for node in self.nodes],
                "edges": [edge.edge_id for edge in self.edges],
            }
        )


def _bounded_tuple(values: Iterable[Any], maximum: int, label: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be an iterable.")
    result: list[Any] = []
    for value in values:
        if len(result) >= maximum:
            raise ValueError(f"{label} exceeds the item limit.")
        result.append(value)
    return tuple(result)


def build_evidence_graph_set(
    *,
    owner_id: str,
    graph_set_key: str,
    authority_views: Iterable[Any],
    relations: Iterable[ExplicitCrossDocumentRelation],
    now: float | None = None,
) -> EvidenceGraphSet:
    """Build an immutable graph set from current authoritative graph views only."""

    owner = normalize_owner_id(owner_id)
    key = _identifier(graph_set_key, "graph_set_key", 500)
    views = _bounded_tuple(authority_views, _MAX_MEMBERS, "authority_views")
    if len(views) < 2:
        raise ValueError("at least two authoritative graph views are required.")
    members: list[GraphGenerationReference] = []
    node_lookup: dict[tuple[str, str], CrossDocumentNodeReference] = {}
    for view in views:
        if getattr(view, "authoritative_current", None) is not True:
            raise ValueError("every graph-set member must be authoritative current.")
        batch = getattr(view, "batch", None)
        if batch is None or getattr(batch, "owner_id", None) != owner:
            raise ValueError("graph-set member escaped owner scope.")
        member = GraphGenerationReference(
            owner_id=owner,
            doc_id=batch.doc_id,
            generation=batch.generation,
            content_sha256=batch.content_sha256,
            profile_fingerprint=batch.profile_fingerprint,
            graph_digest=batch.graph_digest,
            authority_digest=view.authority_digest,
        )
        members.append(member)
        for node in batch.nodes:
            reference = CrossDocumentNodeReference(
                doc_id=batch.doc_id,
                generation=batch.generation,
                graph_digest=batch.graph_digest,
                node_id=node.node_id,
                node_type=node.node_type,
                provenance_digest=(
                    getattr(node, "provenance_digest", None)
                    or _sha256(asdict(node))
                ),
                label=node.label,
                page_number=node.page_number,
                section=node.section,
            )
            identity = (batch.doc_id, node.node_id)
            if identity in node_lookup:
                raise ValueError("member graph contains duplicate node identity.")
            node_lookup[identity] = reference
    members_tuple = tuple(sorted(members, key=lambda item: item.doc_id))
    if len({item.doc_id for item in members_tuple}) != len(members_tuple):
        raise ValueError("authority_views contain duplicate documents.")
    set_id = _sha256(
        {
            "scope": "rigorousrag-evidence-graph-set-v1",
            "owner_id": owner,
            "graph_set_key": key,
            "members": [item.member_digest for item in members_tuple],
        }
    )
    relation_values = _bounded_tuple(relations, _MAX_RELATIONS, "relations")
    edges: list[CrossDocumentEdge] = []
    relation_keys: set[str] = set()
    for relation in relation_values:
        if not isinstance(relation, ExplicitCrossDocumentRelation):
            raise ValueError("every relation must be ExplicitCrossDocumentRelation.")
        if relation.relation_key in relation_keys:
            raise ValueError("relation keys must be unique.")
        relation_keys.add(relation.relation_key)
        source = node_lookup.get((relation.source_doc_id, relation.source_node_id))
        target = node_lookup.get((relation.target_doc_id, relation.target_node_id))
        if source is None or target is None:
            raise ValueError("cross-document relation references an unknown node.")
        edge_id = _sha256(
            {
                "scope": "rigorousrag-cross-document-edge-v1",
                "owner_id": owner,
                "graph_set_id": set_id,
                "source": source.reference_digest,
                "target": target.reference_digest,
                "edge_type": relation.edge_type,
                "relation_key": relation.relation_key,
            }
        )
        edges.append(
            CrossDocumentEdge(
                edge_id=edge_id,
                owner_id=owner,
                graph_set_id=set_id,
                source=source,
                target=target,
                edge_type=relation.edge_type,
                relation_key=relation.relation_key,
                weight=relation.weight,
                metadata={"explicit_cross_document_relation": True, **dict(relation.metadata)},
            )
        )
    return EvidenceGraphSet(
        graph_set_id=set_id,
        owner_id=owner,
        graph_set_key=key,
        members=members_tuple,
        edges=tuple(sorted(edges, key=lambda item: item.edge_id)),
        created_at=time.time() if now is None else now,
    )


def cross_document_neighbors(
    graph_set: EvidenceGraphSet,
    *,
    doc_id: str,
    node_id: str,
    edge_types: Iterable[str] | None = None,
) -> tuple[tuple[CrossDocumentEdge, CrossDocumentNodeReference], ...]:
    document = _identifier(doc_id, "doc_id", 200)
    node = _digest(node_id, "node_id")
    allowed = None
    if edge_types is not None:
        allowed = frozenset(_identifier(value, "edge_type", 50) for value in edge_types)
        if not allowed <= _CROSS_EDGE_TYPES:
            raise ValueError("edge_types contain unsupported values.")
    values = [
        (edge, edge.target)
        for edge in graph_set.edges
        if edge.source.doc_id == document
        and edge.source.node_id == node
        and (allowed is None or edge.edge_type in allowed)
    ]
    return tuple(sorted(values, key=lambda item: item[0].edge_id))


def find_cross_document_paths(
    graph_set: EvidenceGraphSet,
    *,
    source_doc_id: str,
    source_node_id: str,
    target_doc_id: str,
    target_node_id: str,
    edge_types: Iterable[str] | None = None,
    max_depth: int = 6,
    max_paths: int = 20,
) -> tuple[CrossDocumentPath, ...]:
    if not isinstance(graph_set, EvidenceGraphSet):
        raise ValueError("graph_set must be EvidenceGraphSet.")
    depth = _integer(max_depth, "max_depth", 1, 20)
    limit = _integer(max_paths, "max_paths", 1, 1_000)
    source_key = (_identifier(source_doc_id, "source_doc_id", 200), _digest(source_node_id, "source_node_id"))
    target_key = (_identifier(target_doc_id, "target_doc_id", 200), _digest(target_node_id, "target_node_id"))
    node_index: dict[tuple[str, str], CrossDocumentNodeReference] = {}
    adjacency: dict[tuple[str, str], list[CrossDocumentEdge]] = {}
    allowed = None
    if edge_types is not None:
        allowed = frozenset(_identifier(value, "edge_type", 50) for value in edge_types)
        if not allowed <= _CROSS_EDGE_TYPES:
            raise ValueError("edge_types contain unsupported values.")
    for edge in graph_set.edges:
        node_index[(edge.source.doc_id, edge.source.node_id)] = edge.source
        node_index[(edge.target.doc_id, edge.target.node_id)] = edge.target
        if allowed is None or edge.edge_type in allowed:
            adjacency.setdefault((edge.source.doc_id, edge.source.node_id), []).append(edge)
    if source_key not in node_index or target_key not in node_index:
        raise ValueError("path endpoint is not referenced by a cross-document edge.")
    queue = deque([(source_key, (node_index[source_key],), ())])
    results: list[CrossDocumentPath] = []
    while queue and len(results) < limit:
        current, nodes, edges = queue.popleft()
        if len(edges) >= depth:
            continue
        for edge in sorted(adjacency.get(current, ()), key=lambda item: item.edge_id):
            next_key = (edge.target.doc_id, edge.target.node_id)
            if next_key in {(item.doc_id, item.node_id) for item in nodes}:
                continue
            next_nodes = nodes + (edge.target,)
            next_edges = edges + (edge,)
            if next_key == target_key:
                results.append(CrossDocumentPath(next_nodes, next_edges))
                if len(results) >= limit:
                    break
            else:
                queue.append((next_key, next_nodes, next_edges))
    return tuple(results)


__all__ = [
    "CrossDocumentEdge",
    "CrossDocumentNodeReference",
    "CrossDocumentPath",
    "EvidenceGraphSet",
    "ExplicitCrossDocumentRelation",
    "GraphGenerationReference",
    "build_evidence_graph_set",
    "cross_document_neighbors",
    "find_cross_document_paths",
]
