"""Canonical integrity-backed store for signed-retirement restore legal holds."""

from __future__ import annotations

import time
from typing import Any

from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _integer,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_restore_holds import (
    SignedRetirementRestoreHold,
    SignedRetirementRestoreHoldStore,
)
from tools.security import normalize_owner_id

_MAX_LIMIT = 10_000


class IntegritySignedRetirementRestoreHoldStore(
    SignedRetirementRestoreHoldStore
):
    """Hold store whose complete mutable row is digest-committed atomically."""

    def __init__(self, path: Any) -> None:
        super().__init__(path)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_graph_set_signed_restore_hold_integrity (
                    hold_id TEXT PRIMARY KEY,
                    hold_digest TEXT NOT NULL,
                    FOREIGN KEY(hold_id)
                        REFERENCES evidence_graph_set_signed_restore_holds(hold_id)
                        ON DELETE RESTRICT
                )
                """
            )

    @staticmethod
    def _verified_value(connection: Any, row: Any) -> SignedRetirementRestoreHold:
        value = SignedRetirementRestoreHoldStore._value(row)
        integrity = connection.execute(
            "SELECT hold_digest "
            "FROM evidence_graph_set_signed_restore_hold_integrity "
            "WHERE hold_id=?",
            (value.hold_id,),
        ).fetchone()
        if integrity is None:
            raise RuntimeError("restore hold integrity record is missing.")
        stored_digest = _digest(integrity["hold_digest"], "hold_digest")
        if stored_digest != value.hold_digest:
            raise RuntimeError("stored restore hold integrity differs.")
        return value

    def place(
        self,
        *,
        owner_id: str,
        restore_id: str,
        hold_key: str,
        reason_code: str,
        actor: ReviewActorBinding,
        restore_journal: Any,
        now: float | None = None,
    ) -> SignedRetirementRestoreHold:
        owner = normalize_owner_id(owner_id)
        restore = _digest(restore_id, "restore_id")
        if not callable(getattr(restore_journal, "get", None)):
            raise ValueError("restore_journal lacks the required read boundary.")
        restore_value = restore_journal.get(restore)
        if restore_value.owner_id != owner:
            raise RuntimeError("restore escaped legal-hold owner scope.")
        value = SignedRetirementRestoreHold.create(
            owner_id=owner,
            restore_id=restore,
            hold_key=hold_key,
            reason_code=reason_code,
            actor=actor,
            now=now,
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_set_signed_restore_holds "
                    "WHERE hold_id=?",
                    (value.hold_id,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO evidence_graph_set_signed_restore_holds "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
                        (
                            value.hold_id,
                            value.owner_id,
                            value.restore_id,
                            value.hold_key,
                            value.reason_code,
                            value.status,
                            value.created_actor_id,
                            value.created_binding_method,
                            value.created_binding_digest,
                            value.created_at,
                            None,
                            None,
                            None,
                            None,
                        ),
                    )
                    connection.execute(
                        "INSERT INTO evidence_graph_set_signed_restore_hold_integrity "
                        "VALUES (?,?)",
                        (value.hold_id, value.hold_digest),
                    )
                    connection.execute("COMMIT")
                    return value
                stored = self._verified_value(connection, row)
                if (
                    stored.owner_id != value.owner_id
                    or stored.restore_id != value.restore_id
                    or stored.hold_key != value.hold_key
                    or stored.reason_code != value.reason_code
                    or stored.created_actor_id != value.created_actor_id
                    or stored.created_binding_method != value.created_binding_method
                    or stored.created_binding_digest != value.created_binding_digest
                ):
                    raise RuntimeError("restore hold identity collision detected.")
                connection.execute("COMMIT")
                return stored
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def get(self, hold_id: str) -> SignedRetirementRestoreHold:
        selected = _digest(hold_id, "hold_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM evidence_graph_set_signed_restore_holds "
                "WHERE hold_id=?",
                (selected,),
            ).fetchone()
            if row is None:
                raise KeyError(selected)
            return self._verified_value(connection, row)

    def release(
        self,
        hold_id: str,
        *,
        owner_id: str,
        confirm_hold_id: str,
        actor: ReviewActorBinding,
        now: float | None = None,
    ) -> SignedRetirementRestoreHold:
        selected = _digest(hold_id, "hold_id")
        if selected != _digest(confirm_hold_id, "confirm_hold_id"):
            raise ValueError("hold confirmation differs.")
        owner = normalize_owner_id(owner_id)
        if not isinstance(actor, ReviewActorBinding):
            raise ValueError("actor must be ReviewActorBinding.")
        timestamp = _timestamp(time.time() if now is None else now, "now")
        if actor.expires_at is not None and actor.expires_at < timestamp:
            raise PermissionError("review actor binding expired before hold release.")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_set_signed_restore_holds "
                    "WHERE hold_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise KeyError(selected)
                current = self._verified_value(connection, row)
                if current.owner_id != owner:
                    raise RuntimeError("restore hold escaped owner scope.")
                if current.status == "released":
                    connection.execute("COMMIT")
                    return current
                released = SignedRetirementRestoreHold(
                    hold_id=current.hold_id,
                    owner_id=current.owner_id,
                    restore_id=current.restore_id,
                    hold_key=current.hold_key,
                    reason_code=current.reason_code,
                    status="released",
                    created_actor_id=current.created_actor_id,
                    created_binding_method=current.created_binding_method,
                    created_binding_digest=current.created_binding_digest,
                    created_at=current.created_at,
                    released_actor_id=actor.actor_id,
                    released_binding_method=actor.binding_method,
                    released_binding_digest=actor.binding_digest,
                    released_at=timestamp,
                )
                connection.execute(
                    "UPDATE evidence_graph_set_signed_restore_holds "
                    "SET status='released', released_actor_id=?, "
                    "released_binding_method=?, released_binding_digest=?, "
                    "released_at=? WHERE hold_id=? AND status='active'",
                    (
                        released.released_actor_id,
                        released.released_binding_method,
                        released.released_binding_digest,
                        released.released_at,
                        selected,
                    ),
                )
                connection.execute(
                    "UPDATE evidence_graph_set_signed_restore_hold_integrity "
                    "SET hold_digest=? WHERE hold_id=?",
                    (released.hold_digest, selected),
                )
                connection.execute("COMMIT")
                return released
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def list(
        self,
        *,
        owner_id: str,
        restore_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> tuple[SignedRetirementRestoreHold, ...]:
        owner = normalize_owner_id(owner_id)
        restore = None if restore_id is None else _digest(restore_id, "restore_id")
        selected_status = (
            None if status is None else _identifier(status, "status", 20)
        )
        if selected_status is not None and selected_status not in {
            "active",
            "released",
        }:
            raise ValueError("restore hold status is unsupported.")
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        query = (
            "SELECT * FROM evidence_graph_set_signed_restore_holds "
            "WHERE owner_id=?"
        )
        params: list[Any] = [owner]
        if restore is not None:
            query += " AND restore_id=?"
            params.append(restore)
        if selected_status is not None:
            query += " AND status=?"
            params.append(selected_status)
        query += " ORDER BY created_at DESC, hold_id DESC LIMIT ?"
        params.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
            return tuple(
                self._verified_value(connection, row) for row in rows
            )

    def active_restore_ids(
        self,
        *,
        owner_id: str,
        limit: int = _MAX_LIMIT,
    ) -> frozenset[str]:
        values = self.list(
            owner_id=owner_id,
            status="active",
            limit=limit,
        )
        if len(values) >= limit:
            raise RuntimeError("active restore hold list reached the bounded limit.")
        return frozenset(value.restore_id for value in values)


__all__ = ["IntegritySignedRetirementRestoreHoldStore"]
