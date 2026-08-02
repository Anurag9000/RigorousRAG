"""Lease-guarded state transitions for graph-set publication attempts."""

from __future__ import annotations

import time
from typing import Iterable

from tools.evidence_graph_set_publish_contracts import (
    EvidenceGraphSetPublicationAttempt,
    _STATES,
    _digest,
    _generic_errors,
    _identifier,
    _integer,
    _optional_digest,
    _timestamp,
)
from tools.security import normalize_owner_id


class _PublicationJournalTransitions:
    def record_candidate(
        self,
        operation_id: str,
        *,
        worker_id: str,
        previous_graph_set_id: str | None,
        previous_graph_set_digest: str | None,
        candidate_graph_set_id: str,
        candidate_graph_set_digest: str,
        member_count: int,
        edge_count: int,
        now: float | None = None,
    ) -> EvidenceGraphSetPublicationAttempt:
        selected = _digest(operation_id, "operation_id")
        worker = _identifier(worker_id, "worker_id", 200)
        previous_id = _optional_digest(previous_graph_set_id, "previous_graph_set_id")
        previous_digest = _optional_digest(
            previous_graph_set_digest, "previous_graph_set_digest"
        )
        if (previous_id is None) != (previous_digest is None):
            raise ValueError("previous graph-set identity must be complete or absent.")
        candidate_id = _digest(candidate_graph_set_id, "candidate_graph_set_id")
        candidate_digest = _digest(
            candidate_graph_set_digest, "candidate_graph_set_digest"
        )
        members = _integer(member_count, "member_count", 2, 100_000)
        edges = _integer(edge_count, "edge_count", 1, 500_000)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._require_running(
                    connection, operation_id=selected, worker_id=worker, now=timestamp
                )
                current = self._attempt(row)
                if current.phase not in {"planned", "candidate_stored"}:
                    raise RuntimeError(
                        "candidate identity cannot be recorded in this phase."
                    )
                if current.phase == "candidate_stored" and (
                    current.previous_graph_set_id != previous_id
                    or current.previous_graph_set_digest != previous_digest
                    or current.candidate_graph_set_id != candidate_id
                    or current.candidate_graph_set_digest != candidate_digest
                    or current.member_count != members
                    or current.edge_count != edges
                ):
                    raise RuntimeError(
                        "publication candidate identity changed during replay."
                    )
                connection.execute(
                    """
                    UPDATE evidence_graph_set_publications
                    SET phase='candidate_stored', previous_graph_set_id=?,
                        previous_graph_set_digest=?, candidate_graph_set_id=?,
                        candidate_graph_set_digest=?, member_count=?, edge_count=?,
                        updated_at=? WHERE operation_id=?
                    """,
                    (
                        previous_id,
                        previous_digest,
                        candidate_id,
                        candidate_digest,
                        members,
                        edges,
                        timestamp,
                        selected,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def record_pointer_activated(
        self,
        operation_id: str,
        *,
        worker_id: str,
        now: float | None = None,
    ) -> EvidenceGraphSetPublicationAttempt:
        return self._phase_update(
            operation_id,
            worker_id=worker_id,
            allowed_phases={"candidate_stored", "pointer_activated"},
            phase="pointer_activated",
            now=now,
        )

    def _phase_update(
        self,
        operation_id: str,
        *,
        worker_id: str,
        allowed_phases: set[str],
        phase: str,
        now: float | None,
    ) -> EvidenceGraphSetPublicationAttempt:
        selected = _digest(operation_id, "operation_id")
        worker = _identifier(worker_id, "worker_id", 200)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._require_running(
                    connection, operation_id=selected, worker_id=worker, now=timestamp
                )
                if row["phase"] not in allowed_phases:
                    raise RuntimeError("publication phase transition is invalid.")
                connection.execute(
                    "UPDATE evidence_graph_set_publications SET phase=?, updated_at=? "
                    "WHERE operation_id=?",
                    (phase, timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def complete(
        self,
        operation_id: str,
        *,
        worker_id: str,
        verification_digest: str,
        now: float | None = None,
    ) -> EvidenceGraphSetPublicationAttempt:
        return self._terminal(
            operation_id,
            worker_id=worker_id,
            state="completed",
            phase="verified",
            verification_digest=verification_digest,
            failure_type=None,
            compensation_errors=(),
            now=now,
        )

    def mark_compensated(
        self,
        operation_id: str,
        *,
        worker_id: str,
        verification_digest: str,
        failure_type: str,
        now: float | None = None,
    ) -> EvidenceGraphSetPublicationAttempt:
        return self._terminal(
            operation_id,
            worker_id=worker_id,
            state="compensated",
            phase="compensated",
            verification_digest=verification_digest,
            failure_type=failure_type,
            compensation_errors=(),
            now=now,
        )

    def _terminal(
        self,
        operation_id: str,
        *,
        worker_id: str,
        state: str,
        phase: str,
        verification_digest: str,
        failure_type: str | None,
        compensation_errors: tuple[str, ...],
        now: float | None,
    ) -> EvidenceGraphSetPublicationAttempt:
        selected = _digest(operation_id, "operation_id")
        worker = _identifier(worker_id, "worker_id", 200)
        authority = _digest(verification_digest, "verification_digest")
        failure = None if failure_type is None else _identifier(
            failure_type, "failure_type", 200
        )
        errors = _generic_errors(compensation_errors)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._require_running(
                    connection, operation_id=selected, worker_id=worker, now=timestamp
                )
                if row["phase"] not in {"pointer_activated", phase}:
                    raise RuntimeError(
                        "publication cannot enter terminal state from this phase."
                    )
                connection.execute(
                    """
                    UPDATE evidence_graph_set_publications
                    SET state=?, phase=?, lease_owner=NULL, lease_expires_at=NULL,
                        verification_digest=?, failure_type=?, compensation_errors_json=?,
                        updated_at=?, completed_at=? WHERE operation_id=?
                    """,
                    (
                        state,
                        phase,
                        authority,
                        failure,
                        self._errors_json(errors),
                        timestamp,
                        timestamp,
                        selected,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def fail(
        self,
        operation_id: str,
        *,
        worker_id: str,
        failure_type: str,
        compensation_errors: Iterable[str] = (),
        now: float | None = None,
    ) -> EvidenceGraphSetPublicationAttempt:
        selected = _digest(operation_id, "operation_id")
        worker = _identifier(worker_id, "worker_id", 200)
        failure = _identifier(failure_type, "failure_type", 200)
        errors = _generic_errors(compensation_errors)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_running(
                    connection, operation_id=selected, worker_id=worker, now=timestamp
                )
                connection.execute(
                    """
                    UPDATE evidence_graph_set_publications
                    SET state='failed', lease_owner=NULL, lease_expires_at=NULL,
                        failure_type=?, compensation_errors_json=?, updated_at=?
                    WHERE operation_id=?
                    """,
                    (failure, self._errors_json(errors), timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def retry(
        self,
        operation_id: str,
        *,
        owner_id: str,
        confirm_operation_id: str,
        now: float | None = None,
    ) -> EvidenceGraphSetPublicationAttempt:
        selected = _digest(operation_id, "operation_id")
        confirmation = _digest(confirm_operation_id, "confirm_operation_id")
        if selected != confirmation:
            raise ValueError("operation confirmation differs.")
        owner = normalize_owner_id(owner_id)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_set_publications WHERE operation_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise KeyError(selected)
                current = self._attempt(row)
                if current.owner_id != owner:
                    raise RuntimeError("publication attempt escaped owner scope.")
                if current.state not in {"failed", "compensated"}:
                    raise RuntimeError(
                        "only failed or compensated attempts may be retried."
                    )
                if current.attempt_count >= current.max_attempts:
                    raise RuntimeError(
                        "publication attempt exhausted its attempt ceiling."
                    )
                next_phase = (
                    "candidate_stored"
                    if current.state == "compensated"
                    else current.phase
                )
                connection.execute(
                    """
                    UPDATE evidence_graph_set_publications
                    SET state='planned', phase=?, lease_owner=NULL,
                        lease_expires_at=NULL, verification_digest=NULL,
                        failure_type=NULL, compensation_errors_json='[]',
                        completed_at=NULL, updated_at=? WHERE operation_id=?
                    """,
                    (next_phase, timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def cancel(
        self,
        operation_id: str,
        *,
        owner_id: str,
        confirm_operation_id: str,
        now: float | None = None,
    ) -> EvidenceGraphSetPublicationAttempt:
        selected = _digest(operation_id, "operation_id")
        confirmation = _digest(confirm_operation_id, "confirm_operation_id")
        if selected != confirmation:
            raise ValueError("operation confirmation differs.")
        owner = normalize_owner_id(owner_id)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_set_publications WHERE operation_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise KeyError(selected)
                current = self._attempt(row)
                if current.owner_id != owner:
                    raise RuntimeError("publication attempt escaped owner scope.")
                if current.state not in {"planned", "failed"}:
                    raise RuntimeError(
                        "only planned or failed attempts may be cancelled."
                    )
                if current.phase == "pointer_activated":
                    raise RuntimeError(
                        "activated publication attempts may not be cancelled."
                    )
                connection.execute(
                    """
                    UPDATE evidence_graph_set_publications
                    SET state='cancelled', lease_owner=NULL, lease_expires_at=NULL,
                        failure_type=NULL, compensation_errors_json='[]',
                        updated_at=?, completed_at=? WHERE operation_id=?
                    """,
                    (timestamp, timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def next_claimable_id(
        self,
        *,
        owner_id: str,
        now: float | None = None,
    ) -> str | None:
        owner = normalize_owner_id(owner_id)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT operation_id FROM evidence_graph_set_publications
                WHERE owner_id=? AND attempt_count < max_attempts AND (
                    state='planned' OR
                    (state='running' AND lease_expires_at IS NOT NULL AND lease_expires_at<=?)
                )
                ORDER BY updated_at, operation_id LIMIT 1
                """,
                (owner, timestamp),
            ).fetchone()
        return None if row is None else str(row["operation_id"])


__all__ = ["_PublicationJournalTransitions"]
