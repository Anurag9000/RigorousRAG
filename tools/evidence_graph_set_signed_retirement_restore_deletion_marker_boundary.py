"""Recover exact aborted restore-deletion markers without weakening scope checks."""

from __future__ import annotations

from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_contracts import _timestamp
from tools.evidence_graph_set_signed_retirement_restore_deletion_mutation import (
    _marker_digest,
    _marker_row,
    canonical_restore_record_digest,
    ensure_active_deletion_marker as _ensure_active,
)


def ensure_active_deletion_marker(
    restore_journal: Any,
    attempt: Any,
    *,
    now: float,
) -> tuple[str, str, str | None]:
    """Reactivate only the same aborted deletion over the unchanged restore row."""

    timestamp = _timestamp(now, "now")
    with restore_journal._lock, restore_journal._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = _marker_row(connection, attempt.restore_id)
            if row is None or row["state"] != "aborted":
                connection.execute("COMMIT")
            else:
                if (
                    row["deletion_id"] != attempt.deletion_id
                    or row["authorization_id"] != attempt.authorization_id
                    or row["owner_id"] != attempt.owner_id
                    or row["restore_record_digest"]
                    != attempt.restore_record_digest
                ):
                    raise RuntimeError(
                        "aborted deletion marker escaped immutable scope."
                    )
                restore_row = connection.execute(
                    "SELECT * FROM "
                    "evidence_graph_set_signed_retirement_restores "
                    "WHERE restore_id=?",
                    (attempt.restore_id,),
                ).fetchone()
                if (
                    restore_row is None
                    or canonical_restore_record_digest(
                        restore_journal._attempt(restore_row)
                    )
                    != attempt.restore_record_digest
                ):
                    raise RuntimeError(
                        "restore record changed after marker abort."
                    )
                marker = _marker_digest(
                    deletion_id=attempt.deletion_id,
                    authorization_id=attempt.authorization_id,
                    owner_id=attempt.owner_id,
                    restore_id=attempt.restore_id,
                    restore_record_digest=attempt.restore_record_digest,
                    state="active",
                    created_at=row["created_at"],
                    updated_at=timestamp,
                    deleted_at=None,
                    tombstone_digest=None,
                )
                connection.execute(
                    "UPDATE signed_retirement_restore_deletion_markers "
                    "SET state='active', marker_digest=?, "
                    "tombstone_digest=NULL, updated_at=?, deleted_at=NULL "
                    "WHERE deletion_id=? AND state='aborted'",
                    (marker, timestamp, attempt.deletion_id),
                )
                connection.execute("COMMIT")
                return marker, "active", None
        except Exception:
            connection.execute("ROLLBACK")
            raise
    return _ensure_active(restore_journal, attempt, now=timestamp)


__all__ = ["ensure_active_deletion_marker"]
