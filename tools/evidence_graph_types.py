"""Validated provenance-preserving evidence-graph value types.

The graph is generation scoped and deterministic. These types describe only
explicitly supplied nodes and relations; they do not infer semantic support,
contradiction, entity equivalence, or citation intent.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from tools.security import normalize_owner_id

NODE_TYPES = frozenset(
    {"document", "section", "claim", "entity", "method", "dataset", "citation"}
)
EDGE_TYPES = frozenset(
    {
        "contains",
        "mentions",
        "supports",
        "contradicts",
        "cites",
        "uses_method",
        "uses_dataset",
        "derived_from",
        "same_as",
    }
)

_MAX_IDENTIFIER = 500
_MAX_LABEL = 2_000
_MAX_TEXT = 5_000_000
_MAX_METADATA_ITEMS = 128
_MAX_METADATA_BYTES = 100_000
_MAX_NODES = 100_000
_MAX_EDGES = 500_000
_MAX_PATH_NODES = 1_000
_MAX_DEPTH = 12
_MAX_JSON_ITEMS = 100_000
_SCHEMA_VERSION = 1


def _identifier(value: Any, label: str, maximum: int = _MAX_IDENTIFIER) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip()
    if (
        not cleaned
        or len(cleaned) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in cleaned)
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


def _optional_page(value: Any) -> int | None:
    if value is None:
        return None
    return _integer(value, "page_number", 1, 1_000_000)


def _timestamp(value: Any, label: str = "timestamp") -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return result


def _canonical(value: Any, *, depth: int, counter: list[int]) -> Any:
    if depth > _MAX_DEPTH:
        raise ValueError("graph metadata exceeds the nesting limit.")
    counter[0] += 1
    if counter[0] > _MAX_JSON_ITEMS:
        raise ValueError("graph metadata exceeds the item limit.")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("graph metadata may not contain non-finite numbers.")
        return value
    if isinstance(value, str):
        if len(value) > 10_000 or "\x00" in value:
            raise ValueError("graph metadata text is invalid or too long.")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        try:
            items = value.items()
        except Exception as exc:
            raise ValueError("graph metadata mapping is unreadable.") from exc
        for index, (raw_key, item) in enumerate(items):
            if index >= _MAX_METADATA_ITEMS:
                raise ValueError("graph metadata contains too many fields.")
            key = _identifier(raw_key, "metadata key", 200)
            if key in result:
                raise ValueError("graph metadata contains a duplicate key.")
            result[key] = _canonical(item, depth=depth + 1, counter=counter)
        return result
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError("graph metadata contains unsupported bytes.")
    if not isinstance(value, Sequence):
        raise ValueError("graph metadata contains an unsupported value.")
    return [
        _canonical(item, depth=depth + 1, counter=counter)
        for item in value
    ]


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a mapping.")
    result = _canonical(value, depth=0, counter=[0])
    if not isinstance(result, dict):
        raise ValueError("metadata must normalize to an object.")
    payload = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(payload) > _MAX_METADATA_BYTES:
        raise ValueError("metadata exceeds the serialized byte limit.")
    return result


def _bounded_text(value: Any, label: str, maximum: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text.")
    cleaned = value.strip()
    if len(cleaned) > maximum or "\x00" in cleaned:
        raise ValueError(f"{label} is invalid or too long.")
    if not allow_empty and not cleaned:
        raise ValueError(f"{label} is required.")
    return cleaned


def _sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_node_id(
    *,
    owner_id: str,
    doc_id: str,
    generation: int,
    node_type: str,
    natural_key: str,
) -> str:
    owner = normalize_owner_id(owner_id)
    document = _identifier(doc_id, "doc_id", 200)
    sequence = _integer(generation, "generation", 1, 2**63 - 1)
    kind = _identifier(node_type, "node_type", 50)
    if kind not in NODE_TYPES:
        raise ValueError("node_type is unsupported.")
    key = _identifier(natural_key, "natural_key", 2_000)
    return _sha256(
        {
            "scope": "rigorousrag-evidence-node-v1",
            "owner_id": owner,
            "doc_id": document,
            "generation": sequence,
            "node_type": kind,
            "natural_key": key,
        }
    )


def deterministic_edge_id(
    *,
    owner_id: str,
    doc_id: str,
    generation: int,
    source_node_id: str,
    target_node_id: str,
    edge_type: str,
    relation_key: str,
) -> str:
    owner = normalize_owner_id(owner_id)
    document = _identifier(doc_id, "doc_id", 200)
    sequence = _integer(generation, "generation", 1, 2**63 - 1)
    source = _digest(source_node_id, "source_node_id")
    target = _digest(target_node_id, "target_node_id")
    kind = _identifier(edge_type, "edge_type", 50)
    if kind not in EDGE_TYPES:
        raise ValueError("edge_type is unsupported.")
    key = _identifier(relation_key, "relation_key", 2_000)
    return _sha256(
        {
            "scope": "rigorousrag-evidence-edge-v1",
            "owner_id": owner,
            "doc_id": document,
            "generation": sequence,
            "source_node_id": source,
            "target_node_id": target,
            "edge_type": kind,
            "relation_key": key,
        }
    )


@dataclass(frozen=True)
class EvidenceNode:
    node_id: str
    owner_id: str
    doc_id: str
    generation: int
    node_type: str
    natural_key: str
    label: str
    text: str = ""
    page_number: int | None = None
    section: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "doc_id", _identifier(self.doc_id, "doc_id", 200))
        object.__setattr__(
            self,
            "generation",
            _integer(self.generation, "generation", 1, 2**63 - 1),
        )
        kind = _identifier(self.node_type, "node_type", 50)
        if kind not in NODE_TYPES:
            raise ValueError("node_type is unsupported.")
        object.__setattr__(self, "node_type", kind)
        object.__setattr__(
            self,
            "natural_key",
            _identifier(self.natural_key, "natural_key", 2_000),
        )
        object.__setattr__(
            self,
            "label",
            _bounded_text(self.label, "label", _MAX_LABEL),
        )
        object.__setattr__(
            self,
            "text",
            _bounded_text(self.text, "text", _MAX_TEXT, allow_empty=True),
        )
        object.__setattr__(self, "page_number", _optional_page(self.page_number))
        if self.section is not None:
            object.__setattr__(
                self,
                "section",
                _bounded_text(self.section, "section", _MAX_LABEL),
            )
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        expected = deterministic_node_id(
            owner_id=self.owner_id,
            doc_id=self.doc_id,
            generation=self.generation,
            node_type=self.node_type,
            natural_key=self.natural_key,
        )
        if _digest(self.node_id, "node_id") != expected:
            raise ValueError("node_id does not match deterministic node identity.")
        object.__setattr__(self, "node_id", expected)

    @property
    def provenance_digest(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class EvidenceEdge:
    edge_id: str
    owner_id: str
    doc_id: str
    generation: int
    source_node_id: str
    target_node_id: str
    edge_type: str
    relation_key: str
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "doc_id", _identifier(self.doc_id, "doc_id", 200))
        object.__setattr__(
            self,
            "generation",
            _integer(self.generation, "generation", 1, 2**63 - 1),
        )
        source = _digest(self.source_node_id, "source_node_id")
        target = _digest(self.target_node_id, "target_node_id")
        if source == target:
            raise ValueError("evidence edges may not be self loops.")
        object.__setattr__(self, "source_node_id", source)
        object.__setattr__(self, "target_node_id", target)
        kind = _identifier(self.edge_type, "edge_type", 50)
        if kind not in EDGE_TYPES:
            raise ValueError("edge_type is unsupported.")
        object.__setattr__(self, "edge_type", kind)
        object.__setattr__(
            self,
            "relation_key",
            _identifier(self.relation_key, "relation_key", 2_000),
        )
        if isinstance(self.weight, bool):
            raise ValueError("weight must be finite and between 0 and 1.")
        try:
            numeric = float(self.weight)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("weight must be finite and between 0 and 1.") from exc
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError("weight must be finite and between 0 and 1.")
        object.__setattr__(self, "weight", numeric)
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        expected = deterministic_edge_id(
            owner_id=self.owner_id,
            doc_id=self.doc_id,
            generation=self.generation,
            source_node_id=source,
            target_node_id=target,
            edge_type=kind,
            relation_key=self.relation_key,
        )
        if _digest(self.edge_id, "edge_id") != expected:
            raise ValueError("edge_id does not match deterministic edge identity.")
        object.__setattr__(self, "edge_id", expected)

    @property
    def provenance_digest(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class EvidenceGraphBatch:
    owner_id: str
    doc_id: str
    generation: int
    content_sha256: str
    profile_fingerprint: str
    nodes: tuple[EvidenceNode, ...]
    edges: tuple[EvidenceEdge, ...]
    created_at: float
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        document = _identifier(self.doc_id, "doc_id", 200)
        sequence = _integer(self.generation, "generation", 1, 2**63 - 1)
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "doc_id", document)
        object.__setattr__(self, "generation", sequence)
        object.__setattr__(self, "content_sha256", _digest(self.content_sha256, "content_sha256"))
        object.__setattr__(
            self,
            "profile_fingerprint",
            _digest(self.profile_fingerprint, "profile_fingerprint"),
        )
        if not isinstance(self.nodes, tuple) or not 1 <= len(self.nodes) <= _MAX_NODES:
            raise ValueError("nodes must be a non-empty bounded tuple.")
        if not isinstance(self.edges, tuple) or len(self.edges) > _MAX_EDGES:
            raise ValueError("edges must be a bounded tuple.")
        node_ids: set[str] = set()
        document_nodes = 0
        for node in self.nodes:
            if not isinstance(node, EvidenceNode):
                raise ValueError("every node must be EvidenceNode.")
            if (node.owner_id, node.doc_id, node.generation) != (owner, document, sequence):
                raise ValueError("node scope differs from graph scope.")
            if node.node_id in node_ids:
                raise ValueError("graph contains duplicate node IDs.")
            node_ids.add(node.node_id)
            document_nodes += int(node.node_type == "document")
        if document_nodes != 1:
            raise ValueError("graph must contain exactly one document node.")
        edge_ids: set[str] = set()
        for edge in self.edges:
            if not isinstance(edge, EvidenceEdge):
                raise ValueError("every edge must be EvidenceEdge.")
            if (edge.owner_id, edge.doc_id, edge.generation) != (owner, document, sequence):
                raise ValueError("edge scope differs from graph scope.")
            if edge.edge_id in edge_ids:
                raise ValueError("graph contains duplicate edge IDs.")
            edge_ids.add(edge.edge_id)
            if edge.source_node_id not in node_ids or edge.target_node_id not in node_ids:
                raise ValueError("graph edge endpoint is missing.")
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "created_at"))
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("evidence graph schema is unsupported.")

    @property
    def graph_digest(self) -> str:
        return _sha256(
            {
                "schema_version": self.schema_version,
                "owner_id": self.owner_id,
                "doc_id": self.doc_id,
                "generation": self.generation,
                "content_sha256": self.content_sha256,
                "profile_fingerprint": self.profile_fingerprint,
                "nodes": [node.provenance_digest for node in self.nodes],
                "edges": [edge.provenance_digest for edge in self.edges],
            }
        )


@dataclass(frozen=True)
class EvidencePath:
    nodes: tuple[EvidenceNode, ...]
    edges: tuple[EvidenceEdge, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.nodes, tuple) or not 1 <= len(self.nodes) <= _MAX_PATH_NODES:
            raise ValueError("path nodes must be a non-empty bounded tuple.")
        if not isinstance(self.edges, tuple) or len(self.edges) != len(self.nodes) - 1:
            raise ValueError("path must contain exactly one edge between adjacent nodes.")
        first = self.nodes[0]
        if not isinstance(first, EvidenceNode):
            raise ValueError("path nodes must be EvidenceNode.")
        scope = (first.owner_id, first.doc_id, first.generation)
        for node in self.nodes:
            if not isinstance(node, EvidenceNode) or (
                node.owner_id,
                node.doc_id,
                node.generation,
            ) != scope:
                raise ValueError("path node scope is inconsistent.")
        for index, edge in enumerate(self.edges):
            if not isinstance(edge, EvidenceEdge) or (
                edge.owner_id,
                edge.doc_id,
                edge.generation,
            ) != scope:
                raise ValueError("path edge scope is inconsistent.")
            if (
                edge.source_node_id != self.nodes[index].node_id
                or edge.target_node_id != self.nodes[index + 1].node_id
            ):
                raise ValueError("path edge does not connect adjacent nodes.")

    @property
    def path_digest(self) -> str:
        return _sha256(
            {
                "nodes": [node.node_id for node in self.nodes],
                "edges": [edge.edge_id for edge in self.edges],
            }
        )


__all__ = [
    "EDGE_TYPES",
    "NODE_TYPES",
    "EvidenceEdge",
    "EvidenceGraphBatch",
    "EvidenceNode",
    "EvidencePath",
    "deterministic_edge_id",
    "deterministic_node_id",
]
