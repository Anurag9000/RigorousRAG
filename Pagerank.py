"""Validated and bounded PageRank for the fetched-page link graph."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from typing import Dict, Iterable, Set

_MAX_NODES = 100_000
_MAX_EDGES = 5_000_000
_MAX_EDGES_PER_NODE = 100_000
_MAX_NODE_CHARS = 4096
_MAX_ITERATIONS = 10_000


def _finite_float(value: object, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite.")
    return numeric


def _positive_integer(value: object, label: str, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer.")
    if not 1 <= numeric <= maximum:
        raise ValueError(f"{label} must be between 1 and {maximum}.")
    return numeric


def _node(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("PageRank node identifiers must be strings.")
    node = value.strip()
    if not node or len(node) > _MAX_NODE_CHARS:
        raise ValueError(
            f"PageRank node identifiers must contain 1-{_MAX_NODE_CHARS} characters."
        )
    return node


def compute_pagerank(
    graph: Dict[str, Iterable[str]],
    damping: float = 0.85,
    iterations: int = 100,
    tolerance: float = 1e-10,
) -> Dict[str, float]:
    """Compute normalized PageRank under explicit graph and iteration budgets."""

    damping_value = _finite_float(damping, "damping")
    if not 0.0 <= damping_value < 1.0:
        raise ValueError("damping must be in the interval [0, 1).")
    iteration_count = _positive_integer(iterations, "iterations", _MAX_ITERATIONS)
    tolerance_value = _finite_float(tolerance, "tolerance")
    if tolerance_value < 0:
        raise ValueError("tolerance cannot be negative.")
    if not isinstance(graph, Mapping):
        raise ValueError("graph must be a node-to-targets mapping.")
    if len(graph) > _MAX_NODES:
        raise ValueError(f"PageRank supports at most {_MAX_NODES} nodes.")

    adjacency: Dict[str, Set[str]] = {}
    nodes: Set[str] = set()
    total_edges = 0
    for raw_source, raw_targets in graph.items():
        source = _node(raw_source)
        nodes.add(source)
        if len(nodes) > _MAX_NODES:
            raise ValueError(f"PageRank supports at most {_MAX_NODES} nodes.")
        if isinstance(raw_targets, (str, bytes)):
            raise ValueError("PageRank target collections may not be strings.")
        try:
            target_iterator = iter(raw_targets)
        except TypeError as exc:
            raise ValueError("Every PageRank target collection must be iterable.") from exc

        targets: Set[str] = set()
        exhausted = True
        for raw_target in itertools.islice(
            target_iterator,
            _MAX_EDGES_PER_NODE + 1,
        ):
            if len(targets) >= _MAX_EDGES_PER_NODE:
                exhausted = False
                break
            target = _node(raw_target)
            if target in targets:
                continue
            targets.add(target)
            nodes.add(target)
            total_edges += 1
            if len(nodes) > _MAX_NODES:
                raise ValueError(f"PageRank supports at most {_MAX_NODES} nodes.")
            if total_edges > _MAX_EDGES:
                raise ValueError(f"PageRank supports at most {_MAX_EDGES} edges.")
        if not exhausted:
            raise ValueError(
                f"A PageRank node may have at most {_MAX_EDGES_PER_NODE} targets."
            )
        adjacency[source] = targets

    if not nodes:
        return {}
    for node in nodes:
        adjacency.setdefault(node, set())

    ordered_nodes = sorted(nodes)
    total_nodes = len(ordered_nodes)
    rank = {node: 1.0 / total_nodes for node in ordered_nodes}
    teleport = (1.0 - damping_value) / total_nodes

    for _ in range(iteration_count):
        sink_mass = sum(
            rank[node]
            for node, targets in adjacency.items()
            if not targets
        )
        sink_share = damping_value * sink_mass / total_nodes
        new_rank = {node: teleport + sink_share for node in ordered_nodes}
        for source in ordered_nodes:
            targets = adjacency[source]
            if not targets:
                continue
            share = damping_value * rank[source] / len(targets)
            for target in targets:
                new_rank[target] += share
        delta = sum(abs(new_rank[node] - rank[node]) for node in ordered_nodes)
        rank = new_rank
        if delta <= tolerance_value:
            break

    total = sum(rank.values())
    if not math.isfinite(total) or total <= 0:
        return {node: 1.0 / total_nodes for node in ordered_nodes}
    normalized = {node: value / total for node, value in rank.items()}
    if any(not math.isfinite(value) or value < 0 for value in normalized.values()):
        raise ValueError("PageRank computation produced an invalid score.")
    return normalized
