"""Atomic deletion markers and tombstones inside the restore-intent journal."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _timestamp,
)


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def canonical_restore_record_digest(value: Any) -> str:
    return _canonical_digest(asdict(value))


def _ensure_table(connection: Any) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS signed_retirement_restore_deletion_markers (
            deletion_id TEXT PRIMARY KEY,
            authorization_id TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            restore_id TEXT UNIQUE NOT NULL,
            restore_record_digest TEXT NOT NULL,
            state TEXT NOT NULL,
            marker_digest TEXT NOT NULL,
            tombstone_digest TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            deleted_at REAL
        )
        """
    )


def _marker_digest(
    *,
    deletion_id: str,
    authorization_id: str,
    owner_id: str,
    restore_id: str,
    restore_record_digest: str,
    state: str,
    created_at: float,
    updated_at: float,
    deleted_at: float | None,
    tombstone_digest: str | None,
) -> str:
    return _canonical_digest(
        {
            "scope": "rigorousrag-restore-deletion-marker-v1",
            "deletion_id": deletion_id,
            "authorization_id": authorization_id,
            "owner_id": owner_id,
            "restore_id": restore_id,
            "restore_record_digest": restore_record_digest,
            "state": state,
            "created_at": created_at,
            "updated_at": updated_at,
            "deleted_at": deleted_at,
            "tombstone_digest": tombstone_digest,
        }
    )


def _marker_row(connection: Any, restore_id: str) -> Any:
    _ensure_table(connection)
    return connection.execute(
        "SELECT * FROM signed_retirement_restore_deletion_markers "
        "WHERE restore_id=?",
        (_digest(restore_id, "restore_id"),),
    ).fetchone()


def assert_restore_not_under_deletion(
    restore_journal: Any,
    restore_id: str,
) -> None:
    """Refuse new holds/authorizations while an active marker owns the restore."""

    with restore_journal._lock, restore_journal._connect() as connection:
        row = _marker_row(connection, restore_id)
        if row is not None and row["state"] in {"active", "deleted"}:
            raise RuntimeError("restore intent is under deletion control.")


def ensure_active_deletion_marker(
    restore_journal: Any,
    attempt: Any,
    *,
    now: float,
) -> tuple[str, str, str | None]:
    timestamp = _timestamp(now, "now")
    with restore_journal._lock, restore_journal._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = _marker_row(connection, attempt.restore_id)
            if row is not None:
                if (
                    row["deletion_id"] != attempt.deletion_id
                    or row["authorization_id"] != attempt.authorization_id
                    or row["restore_record_digest"]
                    != attempt.restore_record_digest
                ):
                    raise RuntimeError(
                        "restore deletion marker collision detected."
                    )
                if row["state"] == "aborted":
                    raise RuntimeError("restore deletion marker was aborted.")
                connection.execute("COMMIT")
                return (
                    row["marker_digest"],
                    row["state"],
                    row["tombstone_digest"],
                )
            restore_row = connection.execute(
                "SELECT * FROM "
                "evidence_graph_set_signed_retirement_restores "
                "WHERE restore_id=?",
                (attempt.restore_id,),
            ).fetchone()
            if restore_row is None:
                raise KeyError(attempt.restore_id)
            restore = restore_journal._attempt(restore_row)
            if (
                canonical_restore_record_digest(restore)
                != attempt.restore_record_digest
            ):
                raise RuntimeError(
                    "restore record differs from deletion scope."
                )
            marker = _marker_digest(
                deletion_id=attempt.deletion_id,
                authorization_id=attempt.authorization_id,
                owner_id=attempt.owner_id,
                restore_id=attempt.restore_id,
                restore_record_digest=attempt.restore_record_digest,
                state="active",
                created_at=timestamp,
                updated_at=timestamp,
                deleted_at=None,
                tombstone_digest=None,
            )
            connection.execute(
                "INSERT INTO signed_retirement_restore_deletion_markers "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    attempt.deletion_id,
                    attempt.authorization_id,
                    attempt.owner_id,
                    attempt.restore_id,
                    attempt.restore_record_digest,
                    "active",
                    marker,
                    None,
                    timestamp,
                    timestamp,
                    None,
                ),
            )
            connection.execute("COMMIT")
            return marker, "active", None
        except Exception:
            connection.execute("ROLLBACK")
            raise


def abort_deletion_marker(
    restore_journal: Any,
    attempt: Any,
    *,
    now: float,
) -> bool:
    timestamp = _timestamp(now, "now")
    with restore_journal._lock, restore_journal._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = _marker_row(connection, attempt.restore_id)
            if row is None:
                connection.execute("COMMIT")
                return False
            if (
                row["deletion_id"] != attempt.deletion_id
                or row["state"] != "active"
            ):
                raise RuntimeError(
                    "restore deletion marker cannot be aborted."
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
                    "restore record changed before marker abort."
                )
            marker = _marker_digest(
                deletion_id=attempt.deletion_id,
                authorization_id=attempt.authorization_id,
                owner_id=attempt.owner_id,
                restore_id=attempt.restore_id,
                restore_record_digest=attempt.restore_record_digest,
                state="aborted",
                created_at=row["created_at"],
                updated_at=timestamp,
                deleted_at=None,
                tombstone_digest=None,
            )
            connection.execute(
                "UPDATE signed_retirement_restore_deletion_markers "
                "SET state='aborted', marker_digest=?, updated_at=? "
                "WHERE deletion_id=?",
                (marker, timestamp, attempt.deletion_id),
            )
            connection.execute("COMMIT")
            return True
        except Exception:
            connection.execute("ROLLBACK")
            raise


def delete_restore_with_tombstone(
    restore_journal: Any,
    attempt: Any,
    *,
    authorization_consumption_digest: str,
    now: float,
) -> tuple[str, str, bool]:
    timestamp = _timestamp(now, "now")
    consumption = _digest(
        authorization_consumption_digest,
        "authorization_consumption_digest",
    )
    with restore_journal._lock, restore_journal._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = _marker_row(connection, attempt.restore_id)
            if row is None or row["deletion_id"] != attempt.deletion_id:
                raise RuntimeError("active deletion marker is missing.")
            if row["state"] == "deleted":
                connection.execute("COMMIT")
                return (
                    row["marker_digest"],
                    row["tombstone_digest"],
                    False,
                )
            if row["state"] != "active":
                raise RuntimeError(
                    "restore deletion marker is not active."
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
                    "restore record differs before deletion."
                )
            tombstone = _canonical_digest(
                {
                    "scope": "rigorousrag-restore-deletion-tombstone-v1",
                    "deletion_id": attempt.deletion_id,
                    "authorization_id": attempt.authorization_id,
                    "authorization_digest": attempt.authorization_digest,
                    "authorization_consumption_digest": consumption,
                    "owner_id": attempt.owner_id,
                    "restore_id": attempt.restore_id,
                    "snapshot_digest": attempt.snapshot_digest,
                    "target_path_digest": attempt.target_path_digest,
                    "restore_record_digest": attempt.restore_record_digest,
                    "custody_id": attempt.custody_id,
                    "custody_manifest_digest": (
                        attempt.custody_manifest_digest
                    ),
                    "deleted_at": timestamp,
                }
            )
            marker = _marker_digest(
                deletion_id=attempt.deletion_id,
                authorization_id=attempt.authorization_id,
                owner_id=attempt.owner_id,
                restore_id=attempt.restore_id,
                restore_record_digest=attempt.restore_record_digest,
                state="deleted",
                created_at=row["created_at"],
                updated_at=timestamp,
                deleted_at=timestamp,
                tombstone_digest=tombstone,
            )
            connection.execute(
                "DELETE FROM evidence_graph_set_signed_retirement_restores "
                "WHERE restore_id=?",
                (attempt.restore_id,),
            )
            connection.execute(
                "UPDATE signed_retirement_restore_deletion_markers "
                "SET state='deleted', marker_digest=?, tombstone_digest=?, "
                "updated_at=?, deleted_at=? WHERE deletion_id=?",
                (
                    marker,
                    tombstone,
                    timestamp,
                    timestamp,
                    attempt.deletion_id,
                ),
            )
            connection.execute("COMMIT")
            return marker, tombstone, True
        except Exception:
            connection.execute("ROLLBACK")
            raise


def verify_deleted_tombstone(
    restore_journal: Any,
    attempt: Any,
) -> tuple[str, str]:
    with restore_journal._lock, restore_journal._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = _marker_row(connection, attempt.restore_id)
            restore_row = connection.execute(
                "SELECT 1 FROM "
                "evidence_graph_set_signed_retirement_restores "
                "WHERE restore_id=?",
                (attempt.restore_id,),
            ).fetchone()
            if (
                restore_row is not None
                or row is None
                or row["deletion_id"] != attempt.deletion_id
                or row["state"] != "deleted"
                or row["tombstone_digest"] is None
            ):
                raise RuntimeError(
                    "restore deletion tombstone is not exact."
                )
            result = row["marker_digest"], row["tombstone_digest"]
            connection.execute("COMMIT")
            return result
        except Exception:
            connection.execute("ROLLBACK")
            raise


__all__ = [
    "abort_deletion_marker",
    "assert_restore_not_under_deletion",
    "canonical_restore_record_digest",
    "delete_restore_with_tombstone",
    "ensure_active_deletion_marker",
    "verify_deleted_tombstone",
]
