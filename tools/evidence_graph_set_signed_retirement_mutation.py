"""Exact publication-journal mutations used by signed retirement recovery."""

from __future__ import annotations

import time

from tools.evidence_graph_set_publish_attempts import (
    EvidenceGraphSetPublicationAttempt,
    EvidenceGraphSetPublicationJournal,
)
from tools.evidence_graph_set_publish_contracts import (
    _digest,
    _integer,
    _optional_digest,
    _timestamp,
)
from tools.evidence_graph_sets import _identifier
from tools.security import normalize_owner_id


def signed_retirement_lease_owner(retirement_id: str) -> str:
    return f"signed-retirement:{_digest(retirement_id, 'retirement_id')}"


def _scope(
    current: EvidenceGraphSetPublicationAttempt,
    *,
    owner_id: str,
    graph_set_key: str,
    expected_candidate_set_id: str | None,
) -> None:
    if (
        current.owner_id != owner_id
        or current.graph_set_key != graph_set_key
        or current.candidate_graph_set_id != expected_candidate_set_id
    ):
        raise RuntimeError(
            "authorization-only publication escaped retirement scope."
        )


def claim_or_renew_authorization_publication_retirement_lease(
    journal: EvidenceGraphSetPublicationJournal,
    *,
    operation_id: str,
    retirement_id: str,
    owner_id: str,
    graph_set_key: str,
    expected_candidate_set_id: str | None,
    lease_seconds: int,
    now: float | None = None,
) -> EvidenceGraphSetPublicationAttempt:
    """Take over only an expired weaker lease, or renew this saga's exact lease."""

    if not isinstance(journal, EvidenceGraphSetPublicationJournal):
        raise ValueError("journal must be EvidenceGraphSetPublicationJournal.")
    operation = _digest(operation_id, "operation_id")
    lease_owner = signed_retirement_lease_owner(retirement_id)
    owner = normalize_owner_id(owner_id)
    key = _identifier(graph_set_key, "graph_set_key", 500)
    candidate = _optional_digest(
        expected_candidate_set_id, "expected_candidate_set_id"
    )
    duration = _integer(lease_seconds, "lease_seconds", 1, 86_400)
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
            _scope(
                current,
                owner_id=owner,
                graph_set_key=key,
                expected_candidate_set_id=candidate,
            )
            if current.state == "cancelled":
                connection.execute("COMMIT")
                return current
            if current.state != "running":
                raise RuntimeError(
                    "authorization-only publication is not running retirement work."
                )
            held_by_this_saga = current.lease_owner == lease_owner
            expired = bool(
                current.lease_expires_at is not None
                and current.lease_expires_at <= timestamp
            )
            if not held_by_this_saga and not expired:
                raise RuntimeError(
                    "authorization-only publication lease is held by another worker."
                )
            connection.execute(
                """
                UPDATE evidence_graph_set_publications
                SET lease_owner=?, lease_expires_at=?, updated_at=?
                WHERE operation_id=?
                """,
                (lease_owner, timestamp + duration, timestamp, operation),
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    return journal.get(operation)


def retire_claimed_authorization_publication_attempt(
    journal: EvidenceGraphSetPublicationJournal,
    *,
    operation_id: str,
    retirement_id: str,
    owner_id: str,
    graph_set_key: str,
    expected_candidate_set_id: str | None,
    now: float | None = None,
) -> tuple[EvidenceGraphSetPublicationAttempt, bool]:
    """Cancel one weaker attempt only while this saga owns its live lease."""

    if not isinstance(journal, EvidenceGraphSetPublicationJournal):
        raise ValueError("journal must be EvidenceGraphSetPublicationJournal.")
    operation = _digest(operation_id, "operation_id")
    lease_owner = signed_retirement_lease_owner(retirement_id)
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
            _scope(
                current,
                owner_id=owner,
                graph_set_key=key,
                expected_candidate_set_id=candidate,
            )
            if current.state == "cancelled":
                connection.execute("COMMIT")
                return current, False
            if current.state != "running" or current.lease_owner != lease_owner:
                raise RuntimeError(
                    "authorization-only publication is not leased by this retirement."
                )
            if (
                current.lease_expires_at is None
                or current.lease_expires_at <= timestamp
            ):
                raise RuntimeError("signed retirement lease expired.")
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


__all__ = [
    "claim_or_renew_authorization_publication_retirement_lease",
    "retire_claimed_authorization_publication_attempt",
    "signed_retirement_lease_owner",
]
