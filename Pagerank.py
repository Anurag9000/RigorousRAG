"""Simplified PageRank implementation for the crawled link graph."""

from __future__ import annotations

from typing import Dict, Iterable, Set


def compute_pagerank(
    graph: Dict[str, Set[str]],
    damping: float = 0.85,
    iterations: int = 20,
) -> Dict[str, float]:
    """Compute PageRank scores for the given link graph."""
    nodes: Set[str] = set(graph.keys())
    for targets in graph.values():
        nodes.update(targets)

    if not nodes:
        return {}

    total_nodes = len(nodes)
    rank = {node: 1.0 / total_nodes for node in nodes}

    # Ensure every node has an outgoing set for simplified logic
    adjacency = {node: set(graph.get(node, set())) for node in nodes}

    for _ in range(iterations):
        new_rank = {node: (1.0 - damping) / total_nodes for node in nodes}
        sink_share = sum(rank[node] for node, edges in adjacency.items() if not edges)
        sink_distribution = damping * sink_share / total_nodes

        for node, edges in adjacency.items():
            if not edges:
                continue
            share = damping * rank[node] / len(edges)
            for target in edges:
                new_rank[target] += share

        for node in nodes:
            new_rank[node] += sink_distribution

        rank = new_rank

    return rank
