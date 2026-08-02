"""Exact compare-and-swap pointer compensation for graph-set publication."""

from __future__ import annotations

from tools.evidence_graph_set_store import EvidenceGraphSetStore
from tools.evidence_graph_sets import _digest, _identifier
from tools.security import normalize_owner_id


def clear_current_graph_set_pointer(
    store: EvidenceGraphSetStore,
    *,
    owner_id: str,
    graph_set_key: str,
    expected_current_set_id: str,
) -> bool:
    """Clear one current pointer only when it still equals the exact expected set."""

    if not isinstance(store, EvidenceGraphSetStore):
        raise ValueError("store must be EvidenceGraphSetStore.")
    owner = normalize_owner_id(owner_id)
    key = _identifier(graph_set_key, "graph_set_key", 500)
    expected = _digest(expected_current_set_id, "expected_current_set_id")
    with store._lock, store._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT graph_set_id FROM evidence_graph_set_current "
                "WHERE owner_id=? AND graph_set_key=?",
                (owner, key),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return False
            if row["graph_set_id"] != expected:
                raise RuntimeError("graph set current pointer changed concurrently.")
            cursor = connection.execute(
                "DELETE FROM evidence_graph_set_current "
                "WHERE owner_id=? AND graph_set_key=? AND graph_set_id=?",
                (owner, key, expected),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("graph set current pointer could not be cleared.")
            connection.execute("COMMIT")
            return True
        except Exception:
            connection.execute("ROLLBACK")
            raise


__all__ = ["clear_current_graph_set_pointer"]
