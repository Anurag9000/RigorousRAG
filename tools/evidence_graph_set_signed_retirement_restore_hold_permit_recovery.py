"""Governed recovery of abandoned restore hold-placement permits."""

from __future__ import annotations

import time
from typing import Any

from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _integer,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_mutation import (
    _marker_row,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_permit_recovery_contracts import (
    RestoreHoldPermitRecoveryReceipt,
    ensure_recovery_table,
    receipt_from_row,
    recovery_table_exists,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_permits import (
    _ensure_table,
    _permit_digest,
)
from tools.evidence_graph_set_signed_retirement_restore_holds import (
    SignedRetirementRestoreHold,
)
from tools.security import normalize_owner_id

_MAX_LIMIT = 10_000


def _verified_hold(hold_store: Any, connection: Any, hold_id: str) -> Any:
    row = connection.execute(
        "SELECT * FROM evidence_graph_set_signed_restore_holds "
        "WHERE hold_id=?",
        (hold_id,),
    ).fetchone()
    if row is None:
        return None
    verifier = getattr(hold_store, "_verified_value", None)
    if not callable(verifier):
        raise ValueError("hold store lacks the integrity verification boundary.")
    return verifier(connection, row)


def _quarantine_hold(
    *,
    hold_store: Any,
    connection: Any,
    owner_id: str,
    restore_id: str,
    original_hold_id: str,
    actor: ReviewActorBinding,
    now: float,
) -> SignedRetirementRestoreHold:
    value = SignedRetirementRestoreHold.create(
        owner_id=owner_id,
        restore_id=restore_id,
        hold_key=f"permit-recovery-{original_hold_id}",
        reason_code="abandoned_hold_placement_permit",
        actor=actor,
        now=now,
    )
    row = connection.execute(
        "SELECT * FROM evidence_graph_set_signed_restore_holds "
        "WHERE hold_id=?",
        (value.hold_id,),
    ).fetchone()
    if row is not None:
        stored = hold_store._verified_value(connection, row)
        if (
            stored.owner_id != value.owner_id
            or stored.restore_id != value.restore_id
            or stored.hold_key != value.hold_key
            or stored.reason_code != value.reason_code
            or stored.status != "active"
        ):
            raise RuntimeError("quarantine hold differs from recovery scope.")
        return stored
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
    return value


def _permit_row(connection: Any, hold_id: str) -> Any:
    _ensure_table(connection)
    return connection.execute(
        "SELECT * FROM signed_retirement_restore_hold_placement_permits "
        "WHERE hold_id=?",
        (hold_id,),
    ).fetchone()


def recover_abandoned_hold_placement_permit(
    *,
    restore_journal: Any,
    hold_store: Any,
    owner_id: str,
    hold_id: str,
    confirm_hold_id: str,
    confirm_permit_digest: str,
    actor: ReviewActorBinding,
    minimum_age_seconds: int = 3600,
    now: float | None = None,
) -> tuple[RestoreHoldPermitRecoveryReceipt, bool]:
    owner = normalize_owner_id(owner_id)
    selected_hold = _digest(hold_id, "hold_id")
    if selected_hold != _digest(confirm_hold_id, "confirm_hold_id"):
        raise ValueError("hold confirmation differs.")
    confirmed_permit = _digest(
        confirm_permit_digest,
        "confirm_permit_digest",
    )
    if not isinstance(actor, ReviewActorBinding):
        raise ValueError("actor must be ReviewActorBinding.")
    timestamp = _timestamp(time.time() if now is None else now, "now")
    if actor.expires_at is not None and actor.expires_at < timestamp:
        raise PermissionError("review actor binding expired before recovery.")
    age_floor = _integer(
        minimum_age_seconds,
        "minimum_age_seconds",
        60,
        365 * 24 * 60 * 60,
    )
    if (
        not hasattr(restore_journal, "_lock")
        or not callable(getattr(restore_journal, "_connect", None))
        or not hasattr(hold_store, "_lock")
        or not callable(getattr(hold_store, "_connect", None))
    ):
        raise ValueError("stores lack coordinated recovery boundaries.")

    with restore_journal._lock, hold_store._lock:
        restore_connection = restore_journal._connect()
        hold_connection = hold_store._connect()
        restore_tx = hold_tx = False
        try:
            restore_connection.execute("BEGIN IMMEDIATE")
            restore_tx = True
            hold_connection.execute("BEGIN IMMEDIATE")
            hold_tx = True
            ensure_recovery_table(restore_connection)
            permit = _permit_row(restore_connection, selected_hold)
            if permit is None:
                raise KeyError(selected_hold)
            restore_id = _digest(permit["restore_id"], "restore_id")
            if permit["owner_id"] != owner:
                raise RuntimeError("hold placement permit escaped owner scope.")
            state = permit["state"]
            if state not in {"active", "released"}:
                raise RuntimeError("hold placement permit state is corrupt.")
            created_at = _timestamp(permit["created_at"], "created_at")
            updated_at = _timestamp(permit["updated_at"], "updated_at")
            released_at = (
                None
                if permit["released_at"] is None
                else _timestamp(permit["released_at"], "released_at")
            )
            expected = _permit_digest(
                hold_id=selected_hold,
                owner_id=owner,
                restore_id=restore_id,
                state=state,
                created_at=created_at,
                updated_at=updated_at,
                released_at=released_at,
            )
            stored_digest = _digest(
                permit["permit_digest"],
                "permit_digest",
            )
            if stored_digest != expected:
                raise RuntimeError("hold placement permit integrity differs.")

            existing = restore_connection.execute(
                "SELECT * FROM "
                "signed_retirement_restore_hold_permit_recoveries "
                "WHERE hold_id=?",
                (selected_hold,),
            ).fetchone()
            if existing is not None:
                receipt = receipt_from_row(existing)
                if confirmed_permit != receipt.original_permit_digest:
                    raise ValueError("permit digest confirmation differs.")
                if (
                    state != "released"
                    or stored_digest != receipt.released_permit_digest
                    or receipt.owner_id != owner
                    or receipt.restore_id != restore_id
                ):
                    raise RuntimeError(
                        "permit recovery receipt differs from live permit."
                    )
                restore_connection.execute("COMMIT")
                restore_tx = False
                hold_connection.execute("ROLLBACK")
                hold_tx = False
                return receipt, False

            if state != "active":
                raise RuntimeError(
                    "released permit lacks a governed recovery receipt."
                )
            if stored_digest != confirmed_permit:
                raise ValueError("permit digest confirmation differs.")
            if timestamp - updated_at < age_floor:
                raise RuntimeError(
                    "hold placement permit has not reached recovery age."
                )
            marker = _marker_row(restore_connection, restore_id)
            if marker is not None and marker["state"] in {"active", "deleted"}:
                raise RuntimeError("restore intent is under deletion control.")
            restore_row = restore_connection.execute(
                "SELECT * FROM "
                "evidence_graph_set_signed_retirement_restores "
                "WHERE restore_id=?",
                (restore_id,),
            ).fetchone()
            if restore_row is None:
                raise KeyError(restore_id)
            if restore_journal._attempt(restore_row).owner_id != owner:
                raise RuntimeError("restore escaped recovery owner scope.")

            original_hold = _verified_hold(
                hold_store,
                hold_connection,
                selected_hold,
            )
            quarantine = None
            if original_hold is None:
                classification = "abandoned_without_hold_quarantined"
                quarantine = _quarantine_hold(
                    hold_store=hold_store,
                    connection=hold_connection,
                    owner_id=owner,
                    restore_id=restore_id,
                    original_hold_id=selected_hold,
                    actor=actor,
                    now=timestamp,
                )
                hold_connection.execute("COMMIT")
                hold_tx = False
            elif (
                original_hold.owner_id != owner
                or original_hold.restore_id != restore_id
            ):
                raise RuntimeError("hold placement permit escaped hold scope.")
            elif original_hold.status == "active":
                raise RuntimeError(
                    "active hold requires exact hold replay, not recovery."
                )
            elif original_hold.status == "released":
                classification = "released_hold_cleanup"
                hold_connection.execute("ROLLBACK")
                hold_tx = False
            else:
                raise RuntimeError("stored restore hold status is unsupported.")

            released_digest = _permit_digest(
                hold_id=selected_hold,
                owner_id=owner,
                restore_id=restore_id,
                state="released",
                created_at=created_at,
                updated_at=timestamp,
                released_at=timestamp,
            )
            receipt = RestoreHoldPermitRecoveryReceipt.create(
                owner_id=owner,
                restore_id=restore_id,
                hold_id=selected_hold,
                original_permit_digest=stored_digest,
                released_permit_digest=released_digest,
                classification=classification,
                quarantine_hold_id=(
                    None if quarantine is None else quarantine.hold_id
                ),
                quarantine_hold_digest=(
                    None if quarantine is None else quarantine.hold_digest
                ),
                actor_id=actor.actor_id,
                actor_binding_method=actor.binding_method,
                actor_binding_digest=actor.binding_digest,
                recovered_at=timestamp,
            )
            cursor = restore_connection.execute(
                "UPDATE signed_retirement_restore_hold_placement_permits "
                "SET state='released', permit_digest=?, updated_at=?, "
                "released_at=? WHERE hold_id=? AND state='active'",
                (
                    released_digest,
                    timestamp,
                    timestamp,
                    selected_hold,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("hold placement permit changed during recovery.")
            restore_connection.execute(
                "INSERT INTO "
                "signed_retirement_restore_hold_permit_recoveries "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
                (
                    receipt.recovery_id,
                    receipt.owner_id,
                    receipt.restore_id,
                    receipt.hold_id,
                    receipt.original_permit_digest,
                    receipt.released_permit_digest,
                    receipt.classification,
                    receipt.quarantine_hold_id,
                    receipt.quarantine_hold_digest,
                    receipt.actor_id,
                    receipt.actor_binding_method,
                    receipt.actor_binding_digest,
                    receipt.recovered_at,
                    receipt.receipt_digest,
                ),
            )
            restore_connection.execute("COMMIT")
            restore_tx = False
            return receipt, True
        except Exception:
            if restore_tx:
                try:
                    restore_connection.execute("ROLLBACK")
                except Exception:
                    pass
            if hold_tx:
                try:
                    hold_connection.execute("ROLLBACK")
                except Exception:
                    pass
            raise
        finally:
            restore_connection.close()
            hold_connection.close()


def get_hold_permit_recovery(
    restore_journal: Any,
    recovery_id: str,
) -> RestoreHoldPermitRecoveryReceipt:
    selected = _digest(recovery_id, "recovery_id")
    with restore_journal._lock, restore_journal._connect() as connection:
        if not recovery_table_exists(connection):
            raise KeyError(selected)
        row = connection.execute(
            "SELECT * FROM "
            "signed_retirement_restore_hold_permit_recoveries "
            "WHERE recovery_id=?",
            (selected,),
        ).fetchone()
    if row is None:
        raise KeyError(selected)
    return receipt_from_row(row)


def list_hold_permit_recoveries(
    restore_journal: Any,
    *,
    owner_id: str,
    limit: int = 100,
) -> tuple[RestoreHoldPermitRecoveryReceipt, ...]:
    owner = normalize_owner_id(owner_id)
    count = _integer(limit, "limit", 1, _MAX_LIMIT)
    with restore_journal._lock, restore_journal._connect() as connection:
        if not recovery_table_exists(connection):
            return ()
        rows = connection.execute(
            "SELECT * FROM "
            "signed_retirement_restore_hold_permit_recoveries "
            "WHERE owner_id=? "
            "ORDER BY recovered_at DESC, recovery_id DESC LIMIT ?",
            (owner, count),
        ).fetchall()
    if len(rows) >= count:
        raise RuntimeError("permit recovery list reached the bounded limit.")
    return tuple(receipt_from_row(row) for row in rows)


__all__ = [
    "get_hold_permit_recovery",
    "list_hold_permit_recoveries",
    "recover_abandoned_hold_placement_permit",
]
