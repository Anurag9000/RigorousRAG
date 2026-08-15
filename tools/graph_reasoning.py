"""Bounded provenance-preserving evidence-graph reasoning primitives.

Existing evidence graph stores can adapt their nodes/edges into this model to perform
hybrid seed selection, bounded path expansion, support/contradiction clustering and
optional learned graph reranking without making graph summaries citation authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence

_MAX_NODES = 100_000
_MAX_EDGES = 1_000_000
_NODE_KINDS = frozenset({"document", "section", "claim", "entity", "method", "dataset", "result", "figure", "table", "formula", "study", "other"})
_EDGE_KINDS = frozenset({"contains", "mentions", "supports", "contradicts", "cites", "uses_method", "uses_dataset", "reports", "same_entity", "derived_from", "references"})


def _text(value: Any, label: str, maximum: int = 500, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _prob(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1")
    return parsed


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    kind: str
    source_id: str
    content_sha256: str
    label: str = ""
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _text(self.node_id, "node_id", 256))
        kind = _text(self.kind, "kind", 64).lower()
        if kind not in _NODE_KINDS:
            raise ValueError("unsupported graph node kind")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id", 1000))
        digest = _text(self.content_sha256, "content_sha256", 64).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("content_sha256 is invalid")
        object.__setattr__(self, "content_sha256", digest)
        object.__setattr__(self, "label", _text(self.label, "label", 1000, allow_empty=True))
        if not isinstance(self.attributes, Mapping) or len(self.attributes) > 64:
            raise ValueError("attributes must be a bounded mapping")
        object.__setattr__(self, "attributes", {_text(str(k), "attribute key", 100): _text(str(v), "attribute value", 1000) for k, v in self.attributes.items()})


@dataclass(frozen=True)
class GraphEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    kind: str
    confidence: float = 1.0
    evidence_source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "edge_id", _text(self.edge_id, "edge_id", 256))
        object.__setattr__(self, "source_node_id", _text(self.source_node_id, "source_node_id", 256))
        object.__setattr__(self, "target_node_id", _text(self.target_node_id, "target_node_id", 256))
        if self.source_node_id == self.target_node_id:
            raise ValueError("graph edges may not self-reference")
        kind = _text(self.kind, "kind", 64).lower()
        if kind not in _EDGE_KINDS:
            raise ValueError("unsupported graph edge kind")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "confidence", _prob(self.confidence, "confidence"))
        if len(self.evidence_source_ids) > 100:
            raise ValueError("evidence_source_ids exceed the item limit")
        object.__setattr__(self, "evidence_source_ids", tuple(dict.fromkeys(_text(item, "evidence source", 1000) for item in self.evidence_source_ids)))


@dataclass(frozen=True)
class EvidenceGraph:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]

    def __post_init__(self) -> None:
        if len(self.nodes) > _MAX_NODES or len(self.edges) > _MAX_EDGES:
            raise ValueError("evidence graph exceeds its size limits")
        node_ids = {node.node_id for node in self.nodes}
        edge_ids = {edge.edge_id for edge in self.edges}
        if len(node_ids) != len(self.nodes) or len(edge_ids) != len(self.edges):
            raise ValueError("graph IDs must be unique")
        if any(edge.source_node_id not in node_ids or edge.target_node_id not in node_ids for edge in self.edges):
            raise ValueError("graph edge references an unknown node")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical({"nodes": [asdict(node) for node in self.nodes], "edges": [asdict(edge) for edge in self.edges]})).hexdigest()


@dataclass(frozen=True)
class GraphSeed:
    node_id: str
    retrieval_score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _text(self.node_id, "node_id", 256))
        object.__setattr__(self, "retrieval_score", _prob(self.retrieval_score, "retrieval_score"))


@dataclass(frozen=True)
class GraphPath:
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    score: float
    evidence_source_ids: tuple[str, ...]


class GraphPathReranker(Protocol):
    @property
    def version(self) -> str: ...
    def score(self, query: str, path: GraphPath, nodes: Sequence[GraphNode], edges: Sequence[GraphEdge]) -> float: ...


def retrieve_paths(
    graph: EvidenceGraph,
    seeds: Sequence[GraphSeed],
    *,
    max_hops: int = 3,
    max_paths: int = 100,
    edge_kinds: Sequence[str] = tuple(_EDGE_KINDS),
) -> tuple[GraphPath, ...]:
    if not 1 <= max_hops <= 8 or not 1 <= max_paths <= 5000:
        raise ValueError("graph retrieval limits are invalid")
    node_ids = {node.node_id for node in graph.nodes}
    allowed = frozenset(_text(item, "edge kind", 64).lower() for item in edge_kinds)
    if any(item not in _EDGE_KINDS for item in allowed):
        raise ValueError("unsupported edge kind filter")
    adjacency: dict[str, list[GraphEdge]] = defaultdict(list)
    for edge in graph.edges:
        if edge.kind in allowed:
            adjacency[edge.source_node_id].append(edge)
            adjacency[edge.target_node_id].append(edge)
    paths: list[GraphPath] = []
    for seed in seeds:
        if seed.node_id not in node_ids:
            continue
        queue = deque([(seed.node_id, (seed.node_id,), (), seed.retrieval_score, tuple())])
        while queue and len(paths) < max_paths * 4:
            current, nodes, edges, score, sources = queue.popleft()
            if edges:
                paths.append(GraphPath(nodes, edges, score, tuple(dict.fromkeys(sources))))
            if len(edges) >= max_hops:
                continue
            for edge in sorted(adjacency.get(current, ()), key=lambda item: (-item.confidence, item.edge_id)):
                neighbor = edge.target_node_id if edge.source_node_id == current else edge.source_node_id
                if neighbor in nodes:
                    continue
                next_score = score * edge.confidence * (1.0 / (1.0 + 0.15 * len(edges)))
                queue.append((neighbor, (*nodes, neighbor), (*edges, edge.edge_id), next_score, (*sources, *edge.evidence_source_ids)))
    paths.sort(key=lambda item: (-item.score, item.node_ids, item.edge_ids))
    return tuple(paths[:max_paths])


def rerank_paths(query: str, graph: EvidenceGraph, paths: Sequence[GraphPath], reranker: GraphPathReranker, *, top_k: int = 20) -> tuple[GraphPath, ...]:
    if not 1 <= top_k <= 1000:
        raise ValueError("top_k is invalid")
    nodes = {node.node_id: node for node in graph.nodes}
    edges = {edge.edge_id: edge for edge in graph.edges}
    scored: list[tuple[float, GraphPath]] = []
    for path in paths[:5000]:
        try:
            score = _prob(reranker.score(query, path, [nodes[item] for item in path.node_ids], [edges[item] for item in path.edge_ids]), "graph reranker score")
        except Exception:
            score = path.score
        scored.append((score, path))
    scored.sort(key=lambda item: (-item[0], item[1].node_ids))
    return tuple(GraphPath(path.node_ids, path.edge_ids, score, path.evidence_source_ids) for score, path in scored[:top_k])


def support_contradiction_clusters(graph: EvidenceGraph) -> Mapping[str, Mapping[str, tuple[str, ...]]]:
    result: dict[str, dict[str, list[str]]] = {}
    for edge in graph.edges:
        if edge.kind not in {"supports", "contradicts"}:
            continue
        bucket = result.setdefault(edge.target_node_id, {"supports": [], "contradicts": []})
        bucket[edge.kind].append(edge.source_node_id)
    return {target: {kind: tuple(sorted(set(values))) for kind, values in groups.items()} for target, groups in sorted(result.items())}


__all__ = [
    "EvidenceGraph", "GraphEdge", "GraphNode", "GraphPath", "GraphPathReranker", "GraphSeed",
    "rerank_paths", "retrieve_paths", "support_contradiction_clusters",
]
