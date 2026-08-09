"""Bounded deterministic retrieval and path explanations over evidence graphs."""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from tools.evidence_graph_temporal import (
    propagate_retraction_risk,
    temporal_evidence_status,
)
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
_TEMPORAL_STATUSES = frozenset({"active", "not_yet_valid", "expired", "retracted"})


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _unit(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be between 0 and 1.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be between 0 and 1.") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return result


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


@dataclass(frozen=True)
class GovernedNodeSearchResult:
    """One lexical result annotated with derived temporal/retraction governance."""

    lexical_result: NodeSearchResult
    adjusted_score: float
    temporal_status: str
    retraction_risk: float
    retracted_source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.lexical_result, NodeSearchResult):
            raise ValueError("lexical_result must be NodeSearchResult.")
        if isinstance(self.adjusted_score, bool):
            raise ValueError("adjusted_score must be finite and non-negative.")
        try:
            score = float(self.adjusted_score)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("adjusted_score must be finite and non-negative.") from exc
        if not math.isfinite(score) or score < 0.0:
            raise ValueError("adjusted_score must be finite and non-negative.")
        object.__setattr__(self, "adjusted_score", score)
        if self.temporal_status not in _TEMPORAL_STATUSES:
            raise ValueError("temporal_status is unsupported.")
        object.__setattr__(
            self,
            "retraction_risk",
            _unit(self.retraction_risk, "retraction_risk"),
        )
        if (
            not isinstance(self.retracted_source_ids, tuple)
            or tuple(sorted(set(self.retracted_source_ids))) != self.retracted_source_ids
        ):
            raise ValueError("retracted_source_ids must be a sorted unique tuple.")

    @property
    def node(self) -> EvidenceNode:
        return self.lexical_result.node

    @property
    def matched_terms(self) -> tuple[str, ...]:
        return self.lexical_result.matched_terms

    @property
    def lexical_score(self) -> float:
        return self.lexical_result.score


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


def search_nodes_governed(
    batch: EvidenceGraphBatch,
    query: str,
    *,
    as_of: float,
    node_types: Iterable[str] | None = None,
    limit: int = 20,
    retraction_policy: str = "exclude",
    max_retraction_risk: float = 0.0,
    risk_penalty: float = 1.0,
) -> tuple[GovernedNodeSearchResult, ...]:
    """Apply explicit temporal validity and conservative retraction risk to search.

    Non-active nodes are always excluded. Under ``exclude``, active nodes above
    ``max_retraction_risk`` are excluded. Under ``penalize``, active nodes remain
    eligible and their lexical score is multiplied by ``1 - risk_penalty * risk``.
    The authoritative graph is never mutated and ``as_of`` is mandatory.
    """

    if not isinstance(batch, EvidenceGraphBatch):
        raise ValueError("batch must be EvidenceGraphBatch.")
    if retraction_policy not in {"exclude", "penalize"}:
        raise ValueError("retraction_policy must be exclude or penalize.")
    count = _integer(limit, "limit", 1, _MAX_RESULTS)
    threshold = _unit(max_retraction_risk, "max_retraction_risk")
    penalty = _unit(risk_penalty, "risk_penalty")

    # The derived temporal layer validates the complete graph and explicit as_of.
    statuses = {
        node.node_id: temporal_evidence_status(node, as_of=as_of)
        for node in batch.nodes
    }
    risks = propagate_retraction_risk(
        batch.nodes,
        batch.edges,
        as_of=as_of,
    )
    lexical = search_nodes(
        batch,
        query,
        node_types=node_types,
        limit=_MAX_RESULTS,
    )
    candidates: list[GovernedNodeSearchResult] = []
    for item in lexical:
        status = statuses[item.node.node_id]
        if status.status != "active":
            continue
        risk_record = risks.get(item.node.node_id)
        risk = 0.0 if risk_record is None else risk_record.risk
        sources = () if risk_record is None else risk_record.retracted_source_ids
        if retraction_policy == "exclude" and risk > threshold:
            continue
        adjusted = item.score
        if retraction_policy == "penalize":
            adjusted *= max(0.0, 1.0 - penalty * risk)
        candidates.append(
            GovernedNodeSearchResult(
                lexical_result=item,
                adjusted_score=adjusted,
                temporal_status=status.status,
                retraction_risk=risk,
                retracted_source_ids=sources,
            )
        )
    candidates.sort(
        key=lambda item: (
            -item.adjusted_score,
            -item.lexical_score,
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
    "GovernedNodeSearchResult",
    "NodeSearchResult",
    "find_paths",
    "outgoing_neighbors",
    "search_nodes",
    "search_nodes_governed",
]
