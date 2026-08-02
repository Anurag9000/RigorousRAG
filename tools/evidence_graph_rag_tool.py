"""Authoritative GraphRAG retrieval tool returning canonical Citation objects."""

from __future__ import annotations

import operator
from collections.abc import Iterable
from typing import Any

from tools.evidence_graph_citations import graph_evidence_to_citations
from tools.evidence_graph_rag import select_current_graph_set_evidence
from tools.evidence_graph_runtime import get_evidence_graph_store
from tools.evidence_graph_set_runtime import get_evidence_graph_set_store
from tools.evidence_graph_sets import _CROSS_EDGE_TYPES
from tools.evidence_graph_types import EDGE_TYPES, NODE_TYPES
from tools.models import Citation
from tools.security import normalize_owner_id
from tools.sparse_runtime import get_generation_store

_MAX_CITATIONS = 50
_ORIGINS = frozenset({"lexical", "within_document", "cross_document"})

GRAPH_RAG_SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_evidence_graph",
        "description": (
            "Search one current reviewed evidence-graph set. Returns only "
            "server-owned citations from generation-validated graph evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 20_000,
                },
                "graph_set_key": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 500,
                },
                "node_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": sorted(NODE_TYPES)},
                    "minItems": 1,
                    "maxItems": len(NODE_TYPES),
                    "uniqueItems": True,
                },
                "within_edge_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": sorted(EDGE_TYPES)},
                    "minItems": 1,
                    "maxItems": len(EDGE_TYPES),
                    "uniqueItems": True,
                },
                "cross_edge_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": sorted(_CROSS_EDGE_TYPES),
                    },
                    "minItems": 1,
                    "maxItems": len(_CROSS_EDGE_TYPES),
                    "uniqueItems": True,
                },
                "allowed_origins": {
                    "type": "array",
                    "items": {"type": "string", "enum": sorted(_ORIGINS)},
                    "minItems": 1,
                    "maxItems": len(_ORIGINS),
                    "uniqueItems": True,
                },
                "per_document_hits": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                },
                "max_lexical_seeds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 2_000,
                    "default": 100,
                },
                "max_within_per_seed": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "default": 3,
                },
                "max_cross_depth": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 6,
                    "default": 2,
                },
                "max_cross_per_seed": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 5_000,
                    "default": 20,
                },
                "max_citations": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": _MAX_CITATIONS,
                    "default": 20,
                },
            },
            "required": ["query", "graph_set_key"],
            "additionalProperties": False,
        },
    },
}


def _text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text.")
    cleaned = value.strip()
    if (
        not cleaned
        or len(cleaned) > maximum
        or any(
            ord(character) < 32 or ord(character) == 127
            for character in cleaned
        )
    ):
        raise ValueError(f"{label} is invalid or too long.")
    return cleaned


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        result = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return result


def _choices(
    values: Iterable[str] | None,
    *,
    label: str,
    allowed: frozenset[str],
) -> tuple[str, ...] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be an iterable.")
    try:
        result = tuple(sorted(set(values)))
    except Exception as exc:
        raise ValueError(f"{label} is not safely iterable.") from exc
    if not result or any(
        not isinstance(value, str) or value not in allowed for value in result
    ):
        raise ValueError(f"{label} contains unsupported values.")
    return result


def search_evidence_graph(
    query: str,
    *,
    owner_id: str = "default_user",
    graph_set_key: str,
    node_types: Iterable[str] | None = None,
    within_edge_types: Iterable[str] | None = None,
    cross_edge_types: Iterable[str] | None = None,
    allowed_origins: Iterable[str] | None = None,
    per_document_hits: int = 10,
    max_lexical_seeds: int = 100,
    max_within_per_seed: int = 3,
    max_cross_depth: int = 2,
    max_cross_per_seed: int = 20,
    max_citations: int = 20,
    set_store: Any | None = None,
    generations: Any | None = None,
    graphs: Any | None = None,
) -> list[Citation]:
    """Select current graph evidence and convert it through canonical citations."""

    retrieval_query = _text(query, "query", 20_000)
    owner = normalize_owner_id(owner_id)
    key = _text(graph_set_key, "graph_set_key", 500)
    selected_node_types = _choices(
        node_types,
        label="node_types",
        allowed=NODE_TYPES,
    )
    selected_within_edges = _choices(
        within_edge_types,
        label="within_edge_types",
        allowed=EDGE_TYPES,
    )
    selected_cross_edges = _choices(
        cross_edge_types,
        label="cross_edge_types",
        allowed=_CROSS_EDGE_TYPES,
    )
    selected_origins = _choices(
        allowed_origins,
        label="allowed_origins",
        allowed=_ORIGINS,
    )
    per_doc = _integer(per_document_hits, "per_document_hits", 1, 100)
    seeds = _integer(max_lexical_seeds, "max_lexical_seeds", 1, 2_000)
    within = _integer(max_within_per_seed, "max_within_per_seed", 0, 100)
    depth = _integer(max_cross_depth, "max_cross_depth", 0, 6)
    cross = _integer(max_cross_per_seed, "max_cross_per_seed", 0, 5_000)
    citation_limit = _integer(
        max_citations,
        "max_citations",
        1,
        _MAX_CITATIONS,
    )

    selected_set_store = (
        get_evidence_graph_set_store() if set_store is None else set_store
    )
    selected_generations = (
        get_generation_store() if generations is None else generations
    )
    selected_graphs = get_evidence_graph_store() if graphs is None else graphs
    selection = select_current_graph_set_evidence(
        owner_id=owner,
        graph_set_key=key,
        query=retrieval_query,
        set_store=selected_set_store,
        generations=selected_generations,
        graphs=selected_graphs,
        node_types=selected_node_types,
        within_edge_types=selected_within_edges,
        cross_edge_types=selected_cross_edges,
        per_document_hits=per_doc,
        max_lexical_seeds=seeds,
        max_within_per_seed=within,
        max_cross_depth=depth,
        max_cross_per_seed=cross,
        max_total_items=citation_limit,
    )
    return graph_evidence_to_citations(
        selection,
        max_citations=citation_limit,
        allowed_origins=selected_origins,
    )


__all__ = ["GRAPH_RAG_SEARCH_TOOL_DEF", "search_evidence_graph"]
