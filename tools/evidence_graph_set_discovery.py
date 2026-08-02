"""Discover owner-scoped current reviewed evidence-graph sets without graph text."""

from __future__ import annotations

import math
import operator
from typing import Any

from tools.evidence_graph_runtime import get_evidence_graph_store
from tools.evidence_graph_set_runtime import get_evidence_graph_set_store
from tools.evidence_graph_set_store import assess_graph_set_authority
from tools.security import normalize_owner_id
from tools.sparse_runtime import get_generation_store

_MAX_SETS = 50

LIST_EVIDENCE_GRAPH_SETS_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "list_evidence_graph_sets",
        "description": (
            "List the authenticated user's current reviewed evidence-graph sets "
            "before selecting one for graph retrieval. Returns identifiers and "
            "authority/count metadata only, never graph text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": _MAX_SETS,
                    "default": 20,
                },
                "include_unavailable": {
                    "type": "boolean",
                    "default": False,
                },
            },
            "additionalProperties": False,
        },
    },
}


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


def _finite_timestamp(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("graph-set creation time must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("graph-set creation time must be finite.") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError("graph-set creation time must be finite.")
    return result


def _current_values(
    store: Any,
    *,
    owner_id: str,
    limit: int,
) -> tuple[Any, ...]:
    public = getattr(store, "list_current", None)
    if callable(public):
        values = public(owner_id=owner_id, limit=limit)
        if isinstance(values, (str, bytes, bytearray)):
            raise RuntimeError(
                "graph-set store returned an invalid current collection."
            )
        try:
            result = tuple(values)
        except Exception as exc:
            raise RuntimeError(
                "graph-set store returned an unreadable current collection."
            ) from exc
        if len(result) > limit:
            raise RuntimeError(
                "graph-set store exceeded the requested current-set limit."
            )
        return result

    lock = getattr(store, "_lock", None)
    connect = getattr(store, "_connect", None)
    decode = getattr(store, "_value", None)
    if lock is None or not callable(connect) or not callable(decode):
        raise ValueError(
            "graph-set store lacks a bounded current-set read boundary."
        )
    with lock, connect() as connection:
        rows = connection.execute(
            """
            SELECT sets.*,
                   current.graph_set_id AS pointer_graph_set_id,
                   current.graph_set_digest AS pointer_graph_set_digest,
                   current.schema_version AS pointer_schema_version
            FROM evidence_graph_set_current AS current
            JOIN evidence_graph_sets AS sets
              ON sets.owner_id = current.owner_id
             AND sets.graph_set_id = current.graph_set_id
            WHERE current.owner_id = ?
            ORDER BY current.graph_set_key, current.graph_set_id
            LIMIT ?
            """,
            (owner_id, limit),
        ).fetchall()
    result: list[Any] = []
    for row in rows:
        try:
            pointer_id = row["pointer_graph_set_id"]
            pointer_digest = row["pointer_graph_set_digest"]
            pointer_schema = int(row["pointer_schema_version"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                "graph-set current pointer row is corrupt."
            ) from exc
        value = decode(row)
        if (
            pointer_schema != 1
            or getattr(value, "owner_id", None) != owner_id
            or getattr(value, "graph_set_id", None) != pointer_id
            or getattr(value, "graph_set_digest", None) != pointer_digest
        ):
            raise RuntimeError(
                "graph-set current pointer identity is corrupt."
            )
        result.append(value)
    return tuple(result)


def list_evidence_graph_sets(
    *,
    owner_id: str = "default_user",
    limit: int = 20,
    include_unavailable: bool = False,
    set_store: Any | None = None,
    generations: Any | None = None,
    graphs: Any | None = None,
) -> list[dict[str, Any]]:
    """Return bounded text-free summaries of current owner-scoped graph sets."""

    owner = normalize_owner_id(owner_id)
    count = _integer(limit, "limit", 1, _MAX_SETS)
    if not isinstance(include_unavailable, bool):
        raise ValueError("include_unavailable must be boolean.")
    selected_store = (
        get_evidence_graph_set_store() if set_store is None else set_store
    )
    selected_generations = (
        get_generation_store() if generations is None else generations
    )
    selected_graphs = (
        get_evidence_graph_store() if graphs is None else graphs
    )
    values = _current_values(selected_store, owner_id=owner, limit=count)
    summaries: list[dict[str, Any]] = []
    for value in values:
        if getattr(value, "owner_id", None) != owner:
            raise RuntimeError("graph-set discovery escaped owner scope.")
        report = assess_graph_set_authority(
            value,
            generations=selected_generations,
            graphs=selected_graphs,
        )
        authoritative = bool(
            getattr(report, "authoritative_current", False)
        )
        if not authoritative and not include_unavailable:
            continue
        members = getattr(value, "members", None)
        edges = getattr(value, "edges", None)
        if not isinstance(members, tuple) or not isinstance(edges, tuple):
            raise RuntimeError(
                "graph-set discovery encountered invalid bounded arrays."
            )
        stale = getattr(report, "stale_member_doc_ids", ())
        missing = getattr(report, "missing_member_doc_ids", ())
        if not isinstance(stale, tuple) or not isinstance(missing, tuple):
            raise RuntimeError("graph-set authority report is invalid.")
        summaries.append(
            {
                "graph_set_key": str(getattr(value, "graph_set_key")),
                "graph_set_id": str(getattr(value, "graph_set_id")),
                "graph_set_digest": str(
                    getattr(value, "graph_set_digest")
                ),
                "member_count": len(members),
                "edge_count": len(edges),
                "created_at": _finite_timestamp(
                    getattr(value, "created_at")
                ),
                "authoritative_current": authoritative,
                "authority_digest": str(
                    getattr(report, "authority_digest")
                ),
                "stale_member_count": len(stale),
                "missing_member_count": len(missing),
            }
        )
    summaries.sort(
        key=lambda item: (item["graph_set_key"], item["graph_set_id"])
    )
    return summaries


__all__ = [
    "LIST_EVIDENCE_GRAPH_SETS_TOOL_DEF",
    "list_evidence_graph_sets",
]
