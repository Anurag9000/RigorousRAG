"""Explicit-edge-only support and contradiction analysis for evidence graphs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from tools.evidence_graph_types import EvidenceEdge, EvidenceGraphBatch, EvidenceNode


def _sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ClaimEvidenceCluster:
    claim: EvidenceNode
    supporting_nodes: tuple[EvidenceNode, ...]
    contradicting_nodes: tuple[EvidenceNode, ...]
    support_edges: tuple[EvidenceEdge, ...]
    contradiction_edges: tuple[EvidenceEdge, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.claim, EvidenceNode) or self.claim.node_type != "claim":
            raise ValueError("claim must be a claim EvidenceNode.")
        scope = (self.claim.owner_id, self.claim.doc_id, self.claim.generation)
        if len(self.supporting_nodes) != len(self.support_edges):
            raise ValueError("support nodes and edges must align one-to-one.")
        if len(self.contradicting_nodes) != len(self.contradiction_edges):
            raise ValueError("contradiction nodes and edges must align one-to-one.")
        for node, edge in zip(self.supporting_nodes, self.support_edges, strict=True):
            if (
                not isinstance(node, EvidenceNode)
                or not isinstance(edge, EvidenceEdge)
                or edge.edge_type != "supports"
                or edge.source_node_id != node.node_id
                or edge.target_node_id != self.claim.node_id
                or (node.owner_id, node.doc_id, node.generation) != scope
            ):
                raise ValueError("support cluster provenance is invalid.")
        for node, edge in zip(
            self.contradicting_nodes,
            self.contradiction_edges,
            strict=True,
        ):
            if (
                not isinstance(node, EvidenceNode)
                or not isinstance(edge, EvidenceEdge)
                or edge.edge_type != "contradicts"
                or edge.source_node_id != node.node_id
                or edge.target_node_id != self.claim.node_id
                or (node.owner_id, node.doc_id, node.generation) != scope
            ):
                raise ValueError("contradiction cluster provenance is invalid.")

    @property
    def has_conflict(self) -> bool:
        return bool(self.support_edges and self.contradiction_edges)

    @property
    def cluster_digest(self) -> str:
        return _sha256(
            {
                "claim_node_id": self.claim.node_id,
                "support_edge_ids": [edge.edge_id for edge in self.support_edges],
                "contradiction_edge_ids": [
                    edge.edge_id for edge in self.contradiction_edges
                ],
            }
        )


@dataclass(frozen=True)
class EvidenceGraphAnalysis:
    graph_digest: str
    node_counts: dict[str, int]
    edge_counts: dict[str, int]
    claim_clusters: tuple[ClaimEvidenceCluster, ...]

    @property
    def analysis_digest(self) -> str:
        return _sha256(
            {
                "graph_digest": self.graph_digest,
                "node_counts": self.node_counts,
                "edge_counts": self.edge_counts,
                "claim_cluster_digests": [
                    cluster.cluster_digest for cluster in self.claim_clusters
                ],
            }
        )


def analyze_evidence_graph(batch: EvidenceGraphBatch) -> EvidenceGraphAnalysis:
    """Summarize stored structure; never infer unstored semantic relations."""

    if not isinstance(batch, EvidenceGraphBatch):
        raise ValueError("batch must be EvidenceGraphBatch.")
    nodes = {node.node_id: node for node in batch.nodes}
    node_counts: dict[str, int] = {}
    edge_counts: dict[str, int] = {}
    for node in batch.nodes:
        node_counts[node.node_type] = node_counts.get(node.node_type, 0) + 1
    for edge in batch.edges:
        edge_counts[edge.edge_type] = edge_counts.get(edge.edge_type, 0) + 1

    incoming_support: dict[str, list[tuple[EvidenceEdge, EvidenceNode]]] = {}
    incoming_contradiction: dict[str, list[tuple[EvidenceEdge, EvidenceNode]]] = {}
    for edge in batch.edges:
        if edge.edge_type not in {"supports", "contradicts"}:
            continue
        target = nodes[edge.target_node_id]
        if target.node_type != "claim":
            continue
        source = nodes[edge.source_node_id]
        destination = (
            incoming_support if edge.edge_type == "supports" else incoming_contradiction
        )
        destination.setdefault(target.node_id, []).append((edge, source))

    clusters: list[ClaimEvidenceCluster] = []
    for claim in sorted(
        (node for node in batch.nodes if node.node_type == "claim"),
        key=lambda node: node.node_id,
    ):
        supports = sorted(
            incoming_support.get(claim.node_id, ()),
            key=lambda pair: pair[0].edge_id,
        )
        contradictions = sorted(
            incoming_contradiction.get(claim.node_id, ()),
            key=lambda pair: pair[0].edge_id,
        )
        if not supports and not contradictions:
            continue
        clusters.append(
            ClaimEvidenceCluster(
                claim=claim,
                supporting_nodes=tuple(source for _edge, source in supports),
                contradicting_nodes=tuple(source for _edge, source in contradictions),
                support_edges=tuple(edge for edge, _source in supports),
                contradiction_edges=tuple(edge for edge, _source in contradictions),
            )
        )
    return EvidenceGraphAnalysis(
        graph_digest=batch.graph_digest,
        node_counts=dict(sorted(node_counts.items())),
        edge_counts=dict(sorted(edge_counts.items())),
        claim_clusters=tuple(clusters),
    )


__all__ = [
    "ClaimEvidenceCluster",
    "EvidenceGraphAnalysis",
    "analyze_evidence_graph",
]
