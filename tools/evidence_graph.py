"""Evidence graph primitives for provenance-aware multi-hop support analysis."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class EvidenceNode:
    node_id: str
    kind: str
    text: str = ""
    source_id: Optional[str] = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SupportPath:
    nodes: Tuple[str, ...]
    relations: Tuple[str, ...]
    score: float


class EvidenceGraph:
    """Directed evidence graph with bounded traversal and support-path extraction."""

    SUPPORT_RELATIONS = frozenset(
        {"supports", "cites", "derived_from", "mentions", "entails", "contains"}
    )
    CONTRADICTION_RELATIONS = frozenset({"contradicts", "refutes", "disagrees_with"})

    def __init__(self) -> None:
        self._nodes: Dict[str, EvidenceNode] = {}
        self._out: Dict[str, List[EvidenceEdge]] = {}
        self._in: Dict[str, List[EvidenceEdge]] = {}

    def add_node(self, node: EvidenceNode) -> None:
        if not node.node_id or not node.kind:
            raise ValueError("node_id and kind are required.")
        existing = self._nodes.get(node.node_id)
        if existing is not None and existing != node:
            raise ValueError(f"node {node.node_id!r} already exists with different content.")
        self._nodes[node.node_id] = node

    def add_edge(self, edge: EvidenceEdge) -> None:
        if edge.source not in self._nodes or edge.target not in self._nodes:
            raise KeyError("both edge endpoints must exist before an edge is added.")
        if edge.weight < 0:
            raise ValueError("edge weights must be non-negative.")
        if edge not in self._out.setdefault(edge.source, []):
            self._out[edge.source].append(edge)
            self._in.setdefault(edge.target, []).append(edge)

    def node(self, node_id: str) -> EvidenceNode:
        return self._nodes[node_id]

    def outgoing(
        self,
        node_id: str,
        *,
        relations: Optional[Iterable[str]] = None,
    ) -> Tuple[EvidenceEdge, ...]:
        allowed = set(relations) if relations is not None else None
        return tuple(
            edge
            for edge in self._out.get(node_id, ())
            if allowed is None or edge.relation in allowed
        )

    def incoming(
        self,
        node_id: str,
        *,
        relations: Optional[Iterable[str]] = None,
    ) -> Tuple[EvidenceEdge, ...]:
        allowed = set(relations) if relations is not None else None
        return tuple(
            edge
            for edge in self._in.get(node_id, ())
            if allowed is None or edge.relation in allowed
        )

    def reachable(
        self,
        start: str,
        *,
        max_hops: int = 3,
        relations: Optional[Iterable[str]] = None,
        reverse: bool = False,
    ) -> Set[str]:
        if start not in self._nodes:
            raise KeyError(start)
        if max_hops < 0:
            raise ValueError("max_hops must be non-negative.")
        allowed = set(relations) if relations is not None else None
        seen = {start}
        frontier = deque([(start, 0)])
        while frontier:
            current, depth = frontier.popleft()
            if depth >= max_hops:
                continue
            edges = self._in.get(current, ()) if reverse else self._out.get(current, ())
            for edge in edges:
                if allowed is not None and edge.relation not in allowed:
                    continue
                neighbor = edge.source if reverse else edge.target
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                frontier.append((neighbor, depth + 1))
        seen.remove(start)
        return seen

    def support_paths(
        self,
        claim_id: str,
        *,
        max_hops: int = 4,
        source_kinds: Sequence[str] = ("source", "document", "chunk"),
        limit: int = 20,
    ) -> List[SupportPath]:
        """Return strongest acyclic support paths from a claim to source-like nodes."""

        if claim_id not in self._nodes:
            raise KeyError(claim_id)
        if max_hops <= 0 or limit <= 0:
            return []
        source_kind_set = set(source_kinds)
        results: List[SupportPath] = []
        stack: List[Tuple[str, Tuple[str, ...], Tuple[str, ...], float]] = [
            (claim_id, (claim_id,), (), 1.0)
        ]
        while stack:
            current, nodes, relations, score = stack.pop()
            if len(nodes) - 1 >= max_hops:
                continue
            edges = list(self._out.get(current, ())) + list(self._in.get(current, ()))
            for edge in edges:
                if edge.relation not in self.SUPPORT_RELATIONS:
                    continue
                neighbor = edge.target if edge.source == current else edge.source
                if neighbor in nodes:
                    continue
                next_nodes = nodes + (neighbor,)
                next_relations = relations + (edge.relation,)
                next_score = score * edge.weight
                if self._nodes[neighbor].kind in source_kind_set:
                    results.append(SupportPath(next_nodes, next_relations, next_score))
                stack.append((neighbor, next_nodes, next_relations, next_score))
        results.sort(key=lambda path: (-path.score, len(path.nodes), path.nodes))
        return results[:limit]

    def contradictions(self, node_id: str, *, max_hops: int = 2) -> Set[str]:
        return self.reachable(
            node_id,
            max_hops=max_hops,
            relations=self.CONTRADICTION_RELATIONS,
        ) | self.reachable(
            node_id,
            max_hops=max_hops,
            relations=self.CONTRADICTION_RELATIONS,
            reverse=True,
        )

    def evidence_coverage(self, claim_ids: Iterable[str]) -> float:
        claims = list(dict.fromkeys(claim_ids))
        if not claims:
            return 1.0
        supported = sum(bool(self.support_paths(claim_id)) for claim_id in claims)
        return supported / len(claims)

    def subgraph(self, node_ids: Iterable[str]) -> "EvidenceGraph":
        selected = set(node_ids)
        graph = EvidenceGraph()
        for node_id in selected:
            if node_id in self._nodes:
                graph.add_node(self._nodes[node_id])
        for source in selected:
            for edge in self._out.get(source, ()):
                if edge.target in selected:
                    graph.add_edge(edge)
        return graph
