"""Content-addressed provenance queries and downstream impact analysis.

The query engine is deliberately model-free. It traverses explicit derivation edges to
answer where an artifact came from, what depends on it, which policies/models were used,
and what must be reconsidered when an input is retracted or replaced.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

_MAX_NODES = 1_000_000
_MAX_EDGES = 5_000_000


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _sha(value: Any, label: str) -> str:
    digest = _text(value, label, 64).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class ProvenanceNode:
    node_id: str
    kind: str
    content_sha256: str
    owner_scope_sha256: str
    label: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _text(self.node_id, "node_id", 256))
        object.__setattr__(self, "kind", _text(self.kind, "kind", 64).lower())
        object.__setattr__(self, "content_sha256", _sha(self.content_sha256, "content_sha256"))
        object.__setattr__(self, "owner_scope_sha256", _sha(self.owner_scope_sha256, "owner_scope_sha256"))
        object.__setattr__(self, "label", _text(self.label, "label", 1000, allow_empty=True))
        if not isinstance(self.metadata, Mapping) or len(self.metadata) > 64:
            raise ValueError("metadata must be a bounded mapping")
        safe: dict[str, str] = {}
        for key, value in self.metadata.items():
            name = _text(str(key), "metadata key", 100).lower()
            if any(token in name for token in ("secret", "password", "token", "credential", "api_key")):
                raise ValueError("provenance metadata may not contain secret-like fields")
            safe[name] = _text(str(value), "metadata value", 1000, allow_empty=True)
        object.__setattr__(self, "metadata", safe)


@dataclass(frozen=True)
class ProvenanceEdge:
    edge_id: str
    parent_id: str
    child_id: str
    relation: str
    operation_sha256: str

    def __post_init__(self) -> None:
        for name in ("edge_id", "parent_id", "child_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name, 256))
        if self.parent_id == self.child_id:
            raise ValueError("provenance edges may not self-reference")
        object.__setattr__(self, "relation", _text(self.relation, "relation", 64).lower())
        object.__setattr__(self, "operation_sha256", _sha(self.operation_sha256, "operation_sha256"))


@dataclass(frozen=True)
class ProvenancePath:
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]


class ProvenanceIndex:
    def __init__(self, nodes: Sequence[ProvenanceNode] = (), edges: Sequence[ProvenanceEdge] = ()) -> None:
        if len(nodes) > _MAX_NODES or len(edges) > _MAX_EDGES:
            raise ValueError("provenance index exceeds its size limits")
        self.nodes = {node.node_id: node for node in nodes}
        if len(self.nodes) != len(nodes):
            raise ValueError("duplicate provenance node IDs")
        self.edges = {edge.edge_id: edge for edge in edges}
        if len(self.edges) != len(edges):
            raise ValueError("duplicate provenance edge IDs")
        self.parents: dict[str, list[ProvenanceEdge]] = defaultdict(list)
        self.children: dict[str, list[ProvenanceEdge]] = defaultdict(list)
        for edge in edges:
            if edge.parent_id not in self.nodes or edge.child_id not in self.nodes:
                raise ValueError("provenance edge references an unknown node")
            if self.nodes[edge.parent_id].owner_scope_sha256 != self.nodes[edge.child_id].owner_scope_sha256:
                raise ValueError("cross-owner provenance edge is forbidden")
            self.parents[edge.child_id].append(edge)
            self.children[edge.parent_id].append(edge)
        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        indegree = {node_id: len(self.parents.get(node_id, ())) for node_id in self.nodes}
        queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
        visited = 0
        while queue:
            node_id = queue.popleft()
            visited += 1
            for edge in self.children.get(node_id, ()):
                indegree[edge.child_id] -= 1
                if indegree[edge.child_id] == 0:
                    queue.append(edge.child_id)
        if visited != len(self.nodes):
            raise ValueError("provenance graph contains a cycle")

    def lineage(self, node_id: str, *, max_depth: int = 16, max_paths: int = 1000) -> tuple[ProvenancePath, ...]:
        target = _text(node_id, "node_id", 256)
        if target not in self.nodes or not 1 <= max_depth <= 64 or not 1 <= max_paths <= 100_000:
            raise ValueError("lineage query is invalid")
        output: list[ProvenancePath] = []
        queue = deque([(target, (target,), ())])
        while queue and len(output) < max_paths:
            current, nodes, edges = queue.popleft()
            parents = sorted(self.parents.get(current, ()), key=lambda edge: edge.edge_id)
            if not parents:
                output.append(ProvenancePath(tuple(reversed(nodes)), tuple(reversed(edges))))
                continue
            if len(edges) >= max_depth:
                output.append(ProvenancePath(tuple(reversed(nodes)), tuple(reversed(edges))))
                continue
            for edge in parents:
                queue.append((edge.parent_id, (*nodes, edge.parent_id), (*edges, edge.edge_id)))
        return tuple(output)

    def impact(self, node_id: str, *, max_depth: int = 32, kinds: Sequence[str] = ()) -> tuple[ProvenanceNode, ...]:
        source = _text(node_id, "node_id", 256)
        if source not in self.nodes or not 1 <= max_depth <= 128:
            raise ValueError("impact query is invalid")
        allowed = frozenset(_text(item, "kind", 64).lower() for item in kinds)
        queue = deque([(source, 0)])
        seen = {source}
        output: list[ProvenanceNode] = []
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for edge in sorted(self.children.get(current, ()), key=lambda item: item.edge_id):
                child = edge.child_id
                if child in seen:
                    continue
                seen.add(child)
                node = self.nodes[child]
                if not allowed or node.kind in allowed:
                    output.append(node)
                queue.append((child, depth + 1))
        output.sort(key=lambda node: (node.kind, node.node_id))
        return tuple(output)

    def explain(self, node_id: str) -> Mapping[str, Any]:
        node = self.nodes[_text(node_id, "node_id", 256)]
        direct_parents = [self.nodes[edge.parent_id] for edge in sorted(self.parents.get(node.node_id, ()), key=lambda item: item.edge_id)]
        direct_children = [self.nodes[edge.child_id] for edge in sorted(self.children.get(node.node_id, ()), key=lambda item: item.edge_id)]
        return {
            "node": asdict(node),
            "direct_parents": [asdict(item) for item in direct_parents],
            "direct_children": [asdict(item) for item in direct_children],
            "root_lineages": len(self.lineage(node.node_id, max_depth=16, max_paths=1000)),
            "downstream_impact_count": len(self.impact(node.node_id, max_depth=32)),
        }

    @property
    def fingerprint(self) -> str:
        payload = {"nodes": [asdict(self.nodes[key]) for key in sorted(self.nodes)], "edges": [asdict(self.edges[key]) for key in sorted(self.edges)]}
        return hashlib.sha256(_canonical(payload)).hexdigest()


__all__ = ["ProvenanceEdge", "ProvenanceIndex", "ProvenanceNode", "ProvenancePath"]
