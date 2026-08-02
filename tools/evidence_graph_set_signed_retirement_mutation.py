"""Exact journal mutation used by signed publication retirement recovery."""

from __future__ import annotations

import time
from typing import Any

from tools.evidence_graph_set_publish_attempts import (
    EvidenceGraphSetPublicationAttempt,
    EvidenceGraphSetPublicationJournal,
)
from tools.evidence_graph_set_publish_contracts import _digest, _optional_digest, _timestamp
from tools.evidence_graph_sets import _identifier
from tools.security import normalize_owner_id


def retire_expired_authorization_publication_attempt(
    journal: EvidenceGraphSetPublicationJournal,
    *,
    operation_id: str,
    owner_id: str,
    graph_set_key: str,
    expected_candidate_set_id: str | None,
    now: float | None = None,
) -> tuple[EvidenceGraphSetPublicationAttempt, bool]:
    """Cancel one expired running attempt after its pointer is known safe.

    This is intentionally narrower than the ordinary journal cancel operation. It is
    used only after the signed-retirement saga has durably established pointer safety.
    It never changes a live lease, completed publication, or mismatched candidate.
    """

    if not isinstance(journal, EvidenceGraphSetPublicationJournal):
        raise ValueError("journal must be EvidenceGraphSetPublicationJournal.")
    operation = _digest(operation_id, "operation_id")
    owner = normalize_owner_id(owner_id)
    key = _identifier(graph_set_key, "graph_set_key", 500)
    candidate = _optional_digest(
        expected_candidate_set_id, "expected_candidate_set_id"
    )
    timestamp = _timestamp(time.time() if now is None else now, "now")
    with journal._lock, journal._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT * FROM evidence_graph_set_publications WHERE operation_id=?",
                (operation,),
            ).fetchone()
            if row is None:
                raise KeyError(operation)
            current = journal._attempt(row)
            if (
                current.owner_id != owner
                or current.graph_set_key != key
                or current.candidate_graph_set_id != candidate
            ):
                raise RuntimeError(
                    "authorization-only publication escaped retirement scope."
                )
            if current.state == "cancelled":
                connection.execute("COMMIT")
                return current, False
            if current.state != "running":
                raise RuntimeError(
                    "authorization-only publication is not expired running work."
                )
            if (
                current.lease_expires_at is None
                or current.lease_expires_at > timestamp
            ):
                raise RuntimeError(
                    "authorization-only publication lease is still active."
                )
            connection.execute(
                """
                UPDATE evidence_graph_set_publications
                SET state='cancelled', lease_owner=NULL, lease_expires_at=NULL,
                    failure_type=NULL, compensation_errors_json='[]',
                    updated_at=?, completed_at=? WHERE operation_id=?
                """,
                (timestamp, timestamp, operation),
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    return journal.get(operation), True


__all__ = ["retire_expired_authorization_publication_attempt"]
