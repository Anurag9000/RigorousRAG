"""Bounded path-aware retrieval over provenance evidence graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from tools.evidence_graph import EvidenceGraph, SupportPath


@dataclass(frozen=True)
class GraphEvidence:
    node_id: str
    source_id: str
    score: float
    support_path: Tuple[str, ...]
    relations: Tuple[str, ...]
    text: str
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class GraphRetrievalResult:
    claim_id: str
    evidence: Tuple[GraphEvidence, ...]
    complete_paths: int
    unique_sources: int
    truncated: bool


def retrieve_supporting_evidence(
    graph: EvidenceGraph,
    claim_id: str,
    *,
    max_hops: int = 4,
    limit: int = 8,
    per_source_cap: int = 2,
    source_kinds: Sequence[str] = ("source", "document", "chunk"),
) -> GraphRetrievalResult:
    """Select strongest support paths while preserving source diversity and lineage."""

    if max_hops <= 0:
        raise ValueError("max_hops must be positive.")
    if limit <= 0 or per_source_cap <= 0:
        raise ValueError("limit and per_source_cap must be positive.")
    paths = graph.support_paths(
        claim_id,
        max_hops=max_hops,
        source_kinds=source_kinds,
        limit=max(limit * max(per_source_cap, 2) * 4, limit),
    )
    selected: List[GraphEvidence] = []
    source_counts: dict[str, int] = {}
    seen_nodes = set()
    for path in paths:
        terminal_id = path.nodes[-1]
        if terminal_id in seen_nodes:
            continue
        node = graph.node(terminal_id)
        source_id = node.source_id or node.node_id
        if source_counts.get(source_id, 0) >= per_source_cap:
            continue
        selected.append(
            GraphEvidence(
                node_id=node.node_id,
                source_id=source_id,
                score=path.score,
                support_path=path.nodes,
                relations=path.relations,
                text=node.text,
                metadata=dict(node.metadata),
            )
        )
        seen_nodes.add(terminal_id)
        source_counts[source_id] = source_counts.get(source_id, 0) + 1
        if len(selected) >= limit:
            break
    return GraphRetrievalResult(
        claim_id=claim_id,
        evidence=tuple(selected),
        complete_paths=len(paths),
        unique_sources=len(source_counts),
        truncated=len(selected) < len(paths) and len(selected) >= limit,
    )


def path_completeness(
    expected_paths: Iterable[Iterable[str]],
    retrieved: GraphRetrievalResult,
) -> float:
    """Fraction of expected node paths represented by retrieved lineage paths."""

    expected = {tuple(path) for path in expected_paths}
    if not expected:
        return 1.0
    observed = {item.support_path for item in retrieved.evidence}
    return len(expected & observed) / len(expected)


def evidence_source_diversity(result: GraphRetrievalResult) -> float:
    if not result.evidence:
        return 0.0
    return result.unique_sources / len(result.evidence)
