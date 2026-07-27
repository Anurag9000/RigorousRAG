"""Validated PageRank for the fetched-page link graph."""

from __future__ import annotations

from typing import Dict, Iterable, Set


def compute_pagerank(
    graph: Dict[str, Iterable[str]],
    damping: float = 0.85,
    iterations: int = 100,
    tolerance: float = 1e-10,
) -> Dict[str, float]:
    if not 0.0 <= damping < 1.0:
        raise ValueError("damping must be in the interval [0, 1).")
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if tolerance < 0:
        raise ValueError("tolerance cannot be negative.")

    nodes: Set[str] = set(graph)
    for targets in graph.values():
        nodes.update(str(target) for target in targets)
    if not nodes:
        return {}

    adjacency: Dict[str, Set[str]] = {
        node: {str(target) for target in graph.get(node, ()) if str(target) in nodes}
        for node in nodes
    }
    total_nodes = len(nodes)
    rank = {node: 1.0 / total_nodes for node in nodes}
    teleport = (1.0 - damping) / total_nodes

    for _ in range(iterations):
        sink_mass = sum(rank[node] for node, targets in adjacency.items() if not targets)
        sink_share = damping * sink_mass / total_nodes
        new_rank = {node: teleport + sink_share for node in nodes}
        for source, targets in adjacency.items():
            if not targets:
                continue
            share = damping * rank[source] / len(targets)
            for target in targets:
                new_rank[target] += share
        delta = sum(abs(new_rank[node] - rank[node]) for node in nodes)
        rank = new_rank
        if delta <= tolerance:
            break

    total = sum(rank.values())
    if total <= 0:
        return {node: 1.0 / total_nodes for node in nodes}
    return {node: value / total for node, value in rank.items()}
