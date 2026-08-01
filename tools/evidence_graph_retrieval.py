"""Bounded deterministic retrieval and path explanations over evidence graphs."""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from tools.evidence_graph_types import (
    EDGE_TYPES,
    NODE_TYPES,
    EvidenceEdge,
    EvidenceGraphBatch,
    EvidenceNode,
    EvidencePath,
)

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_MAX_QUERY_CHARS = 20_000
_MAX_QUERY_TERMS = 256
_MAX_RESULTS = 1_000
_MAX_DEPTH = 20
_MAX_PATHS = 1_000
_MAX_VISITED = 100_000


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _tokens(value: str) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise ValueError("query must be text.")
    query = value.strip()
    if not query or len(query) > _MAX_QUERY_CHARS or "\x00" in query:
        raise ValueError("query is empty, invalid or too long.")
    result: list[str] = []
    for match in _TOKEN_RE.finditer(query.casefold()):
        token = match.group(0)
        if token not in result:
            result.append(token)
        if len(result) >= _MAX_QUERY_TERMS:
            break
    if not result:
        raise ValueError("query contains no searchable terms.")
    return tuple(result)


def _type_filter(
    values: Iterable[str] | None,
    *,
    allowed: frozenset[str],
    label: str,
) -> frozenset[str] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be an iterable of type names.")
    try:
        result = frozenset(values)
    except Exception as exc:
        raise ValueError(f"{label} is not safely iterable.") from exc
    if not result or any(not isinstance(item, str) or item not in allowed for item in result):
        raise ValueError(f"{label} contains unsupported values.")
    return result


def _text_terms(node: EvidenceNode) -> tuple[str, ...]:
    payload = " ".join(
        part
        for part in (node.label, node.text, node.section or "")
        if isinstance(part, str) and part
    ).casefold()
    return tuple(_TOKEN_RE.findall(payload))


@dataclass(frozen=True)
class NodeSearchResult:
    node: EvidenceNode
    score: float
    matched_terms: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.node, EvidenceNode):
            raise ValueError("node must be EvidenceNode.")
        if isinstance(self.score, bool):
            raise ValueError("score must be finite and non-negative.")
        try:
            score = float(self.score)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("score must be finite and non-negative.") from exc
        if not math.isfinite(score) or score < 0:
            raise ValueError("score must be finite and non-negative.")
        object.__setattr__(self, "score", score)
        if (
            not isinstance(self.matched_terms, tuple)
            or not self.matched_terms
            or tuple(sorted(set(self.matched_terms))) != self.matched_terms
        ):
            raise ValueError("matched_terms must be a non-empty sorted unique tuple.")


def search_nodes(
    batch: EvidenceGraphBatch,
    query: str,
    *,
    node_types: Iterable[str] | None = None,
    limit: int = 20,
) -> tuple[NodeSearchResult, ...]:
    """Rank graph nodes lexically without synthesizing semantic relations."""

    if not isinstance(batch, EvidenceGraphBatch):
        raise ValueError("batch must be EvidenceGraphBatch.")
    terms = _tokens(query)
    selected_types = _type_filter(node_types, allowed=NODE_TYPES, label="node_types")
    count = _integer(limit, "limit", 1, _MAX_RESULTS)
    candidates: list[NodeSearchResult] = []
    for node in batch.nodes:
        if selected_types is not None and node.node_type not in selected_types:
            continue
        tokens = _text_terms(node)
        if not tokens:
            continue
        frequencies = {term: tokens.count(term) for term in terms}
        matched = tuple(sorted(term for term, frequency in frequencies.items() if frequency))
        if not matched:
            continue
        label_tokens = set(_TOKEN_RE.findall(node.label.casefold()))
        exact_label_bonus = 2.0 if " ".join(terms) == node.label.casefold() else 0.0
        label_bonus = sum(1.5 for term in matched if term in label_tokens)
        frequency_score = sum(1.0 + math.log1p(frequencies[term]) for term in matched)
        coverage = len(matched) / len(terms)
        type_bonus = 0.25 if node.node_type in {"claim", "method", "dataset", "citation"} else 0.0
        score = exact_label_bonus + label_bonus + frequency_score + coverage + type_bonus
        candidates.append(NodeSearchResult(node, score, matched))
    candidates.sort(
        key=lambda item: (
            -item.score,
            item.node.node_type,
            item.node.label.casefold(),
            item.node.node_id,
        )
    )
    return tuple(candidates[:count])


def find_paths(
    batch: EvidenceGraphBatch,
    *,
    source_node_id: str,
    target_node_id: str,
    max_depth: int = 6,
    max_paths: int = 20,
    edge_types: Iterable[str] | None = None,
    intermediate_node_types: Iterable[str] | None = None,
) -> tuple[EvidencePath, ...]:
    """Find bounded simple directed paths using only stored explicit edges."""

    if not isinstance(batch, EvidenceGraphBatch):
        raise ValueError("batch must be EvidenceGraphBatch.")
    if not isinstance(source_node_id, str) or not isinstance(target_node_id, str):
        raise ValueError("source_node_id and target_node_id must be strings.")
    depth_limit = _integer(max_depth, "max_depth", 1, _MAX_DEPTH)
    path_limit = _integer(max_paths, "max_paths", 1, _MAX_PATHS)
    selected_edges = _type_filter(edge_types, allowed=EDGE_TYPES, label="edge_types")
    selected_intermediate = _type_filter(
        intermediate_node_types,
        allowed=NODE_TYPES,
        label="intermediate_node_types",
    )
    nodes = {node.node_id: node for node in batch.nodes}
    if source_node_id not in nodes or target_node_id not in nodes:
        raise KeyError("source or target node is unavailable.")
    if source_node_id == target_node_id:
        return (EvidencePath((nodes[source_node_id],), ()),)

    adjacency: dict[str, list[EvidenceEdge]] = {}
    for edge in batch.edges:
        if selected_edges is not None and edge.edge_type not in selected_edges:
            continue
        adjacency.setdefault(edge.source_node_id, []).append(edge)
    for values in adjacency.values():
        values.sort(key=lambda edge: (edge.edge_type, edge.target_node_id, edge.edge_id))

    queue: deque[tuple[tuple[str, ...], tuple[EvidenceEdge, ...]]] = deque(
        [((source_node_id,), ())]
    )
    results: list[EvidencePath] = []
    inspected = 0
    while queue and len(results) < path_limit:
        node_ids, edges = queue.popleft()
        inspected += 1
        if inspected > _MAX_VISITED:
            raise RuntimeError("path traversal exceeded the visited-state limit.")
        if len(edges) >= depth_limit:
            continue
        for edge in adjacency.get(node_ids[-1], ()):
            next_id = edge.target_node_id
            if next_id in node_ids:
                continue
            next_node = nodes[next_id]
            is_target = next_id == target_node_id
            if (
                not is_target
                and selected_intermediate is not None
                and next_node.node_type not in selected_intermediate
            ):
                continue
            next_ids = node_ids + (next_id,)
            next_edges = edges + (edge,)
            if is_target:
                results.append(
                    EvidencePath(
                        tuple(nodes[node_id] for node_id in next_ids),
                        next_edges,
                    )
                )
                if len(results) >= path_limit:
                    break
            else:
                queue.append((next_ids, next_edges))
    results.sort(
        key=lambda path: (
            len(path.edges),
            tuple(edge.edge_type for edge in path.edges),
            path.path_digest,
        )
    )
    return tuple(results)


def outgoing_neighbors(
    batch: EvidenceGraphBatch,
    node_id: str,
    *,
    edge_types: Iterable[str] | None = None,
    limit: int = 100,
) -> tuple[tuple[EvidenceEdge, EvidenceNode], ...]:
    if not isinstance(batch, EvidenceGraphBatch):
        raise ValueError("batch must be EvidenceGraphBatch.")
    if not isinstance(node_id, str):
        raise ValueError("node_id must be a string.")
    selected_edges = _type_filter(edge_types, allowed=EDGE_TYPES, label="edge_types")
    count = _integer(limit, "limit", 1, _MAX_RESULTS)
    nodes = {node.node_id: node for node in batch.nodes}
    if node_id not in nodes:
        raise KeyError(node_id)
    values = [
        (edge, nodes[edge.target_node_id])
        for edge in batch.edges
        if edge.source_node_id == node_id
        and (selected_edges is None or edge.edge_type in selected_edges)
    ]
    values.sort(key=lambda item: (item[0].edge_type, item[1].node_type, item[1].node_id))
    return tuple(values[:count])


__all__ = [
    "NodeSearchResult",
    "find_paths",
    "outgoing_neighbors",
    "search_nodes",
]
