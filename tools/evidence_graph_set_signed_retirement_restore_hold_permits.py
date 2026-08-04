"""Restore-database permits serializing legal-hold placement and deletion."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _timestamp,
)
from tools.security import normalize_owner_id


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


def _ensure_table(connection: Any) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS
            signed_retirement_restore_hold_placement_permits (
                hold_id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                restore_id TEXT NOT NULL,
                state TEXT NOT NULL,
                permit_digest TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                released_at REAL
            )
        """
    )


def _permit_digest(
    *,
    hold_id: str,
    owner_id: str,
    restore_id: str,
    state: str,
    created_at: float,
    updated_at: float,
    released_at: float | None,
) -> str:
    return _canonical_digest(
        {
            "scope": "rigorousrag-restore-hold-placement-permit-v1",
            "hold_id": hold_id,
            "owner_id": owner_id,
            "restore_id": restore_id,
            "state": state,
            "created_at": created_at,
            "updated_at": updated_at,
            "released_at": released_at,
        }
    )


def active_hold_placement_permit(connection: Any, restore_id: str) -> Any:
    _ensure_table(connection)
    return connection.execute(
        "SELECT * FROM signed_retirement_restore_hold_placement_permits "
        "WHERE restore_id=? AND state='active'",
        (_digest(restore_id, "restore_id"),),
    ).fetchone()


def acquire_hold_placement_permit(
    restore_journal: Any,
    *,
    owner_id: str,
    restore_id: str,
    hold_id: str,
    now: float,
) -> str:
    owner = normalize_owner_id(owner_id)
    restore = _digest(restore_id, "restore_id")
    hold = _digest(hold_id, "hold_id")
    timestamp = _timestamp(now, "now")
    with restore_journal._lock, restore_journal._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            from tools.evidence_graph_set_signed_retirement_restore_deletion_mutation import (
                _marker_row,
            )

            marker = _marker_row(connection, restore)
            if marker is not None and marker["state"] in {
                "active",
                "deleted",
            }:
                raise RuntimeError(
                    "restore intent is under deletion control."
                )
            current = active_hold_placement_permit(connection, restore)
            if current is not None:
                if (
                    current["hold_id"] != hold
                    or current["owner_id"] != owner
                ):
                    raise RuntimeError(
                        "another hold placement owns the restore permit."
                    )
                connection.execute("COMMIT")
                return current["permit_digest"]
            digest = _permit_digest(
                hold_id=hold,
                owner_id=owner,
                restore_id=restore,
                state="active",
                created_at=timestamp,
                updated_at=timestamp,
                released_at=None,
            )
            connection.execute(
                "INSERT INTO signed_retirement_restore_hold_placement_permits "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    hold,
                    owner,
                    restore,
                    "active",
                    digest,
                    timestamp,
                    timestamp,
                    None,
                ),
            )
            connection.execute("COMMIT")
            return digest
        except Exception:
            connection.execute("ROLLBACK")
            raise


def release_hold_placement_permit(
    restore_journal: Any,
    *,
    owner_id: str,
    restore_id: str,
    hold_id: str,
    now: float,
) -> bool:
    owner = normalize_owner_id(owner_id)
    restore = _digest(restore_id, "restore_id")
    hold = _digest(hold_id, "hold_id")
    timestamp = _timestamp(now, "now")
    with restore_journal._lock, restore_journal._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            _ensure_table(connection)
            row = connection.execute(
                "SELECT * FROM "
                "signed_retirement_restore_hold_placement_permits "
                "WHERE hold_id=?",
                (hold,),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return False
            if row["owner_id"] != owner or row["restore_id"] != restore:
                raise RuntimeError(
                    "hold placement permit escaped scope."
                )
            if row["state"] == "released":
                connection.execute("COMMIT")
                return False
            digest = _permit_digest(
                hold_id=hold,
                owner_id=owner,
                restore_id=restore,
                state="released",
                created_at=row["created_at"],
                updated_at=timestamp,
                released_at=timestamp,
            )
            connection.execute(
                "UPDATE signed_retirement_restore_hold_placement_permits "
                "SET state='released', permit_digest=?, updated_at=?, "
                "released_at=? WHERE hold_id=?",
                (digest, timestamp, timestamp, hold),
            )
            connection.execute("COMMIT")
            return True
        except Exception:
            connection.execute("ROLLBACK")
            raise


def assert_no_active_hold_placement_permits(
    connection: Any,
    restore_id: str,
) -> None:
    if active_hold_placement_permit(connection, restore_id) is not None:
        raise RuntimeError(
            "active hold placement permit blocks deletion."
        )


__all__ = [
    "acquire_hold_placement_permit",
    "active_hold_placement_permit",
    "assert_no_active_hold_placement_permits",
    "release_hold_placement_permit",
]
