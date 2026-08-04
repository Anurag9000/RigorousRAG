"""Authorization reservation and terminal consumption for restore deletion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class RestoreDeletionAuthorizationConsumption:
    authorization_id: str
    deletion_id: str
    state: str
    reserved_at: float
    consumed_at: float | None
    consumption_digest: str

    def __post_init__(self) -> None:
        authorization = _digest(self.authorization_id, "authorization_id")
        deletion = _digest(self.deletion_id, "deletion_id")
        if self.state not in {"reserved", "consumed"}:
            raise ValueError("consumption state is unsupported.")
        reserved = _timestamp(self.reserved_at, "reserved_at")
        consumed = (
            None
            if self.consumed_at is None
            else _timestamp(self.consumed_at, "consumed_at")
        )
        if (self.state == "reserved") != (consumed is None):
            raise ValueError("consumption timestamps are inconsistent.")
        if consumed is not None and consumed < reserved:
            raise ValueError("consumption predates reservation.")
        expected = _canonical_digest(
            {
                "scope": "rigorousrag-restore-deletion-authorization-consumption-v1",
                "authorization_id": authorization,
                "deletion_id": deletion,
                "state": self.state,
                "reserved_at": reserved,
                "consumed_at": consumed,
            }
        )
        digest = _digest(self.consumption_digest, "consumption_digest")
        if digest != expected:
            raise ValueError("consumption_digest differs.")
        object.__setattr__(self, "authorization_id", authorization)
        object.__setattr__(self, "deletion_id", deletion)
        object.__setattr__(self, "reserved_at", reserved)
        object.__setattr__(self, "consumed_at", consumed)
        object.__setattr__(self, "consumption_digest", digest)


def _ensure_table(connection: Any) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS
            signed_retirement_restore_deletion_authorization_consumptions (
                authorization_id TEXT PRIMARY KEY,
                deletion_id TEXT NOT NULL,
                state TEXT NOT NULL,
                reserved_at REAL NOT NULL,
                consumed_at REAL,
                consumption_digest TEXT NOT NULL
            )
        """
    )


def _create(
    *,
    authorization_id: str,
    deletion_id: str,
    state: str,
    reserved_at: float,
    consumed_at: float | None,
) -> RestoreDeletionAuthorizationConsumption:
    stable = {
        "scope": "rigorousrag-restore-deletion-authorization-consumption-v1",
        "authorization_id": _digest(authorization_id, "authorization_id"),
        "deletion_id": _digest(deletion_id, "deletion_id"),
        "state": state,
        "reserved_at": _timestamp(reserved_at, "reserved_at"),
        "consumed_at": (
            None
            if consumed_at is None
            else _timestamp(consumed_at, "consumed_at")
        ),
    }
    return RestoreDeletionAuthorizationConsumption(
        authorization_id=stable["authorization_id"],
        deletion_id=stable["deletion_id"],
        state=state,
        reserved_at=stable["reserved_at"],
        consumed_at=stable["consumed_at"],
        consumption_digest=_canonical_digest(stable),
    )


def _value(row: Any) -> RestoreDeletionAuthorizationConsumption:
    try:
        return RestoreDeletionAuthorizationConsumption(
            authorization_id=row["authorization_id"],
            deletion_id=row["deletion_id"],
            state=row["state"],
            reserved_at=row["reserved_at"],
            consumed_at=row["consumed_at"],
            consumption_digest=row["consumption_digest"],
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("stored authorization consumption is corrupt.") from exc


def get_authorization_consumption(
    store: Any,
    authorization_id: str,
) -> RestoreDeletionAuthorizationConsumption | None:
    authorization = _digest(authorization_id, "authorization_id")
    with store._lock, store._connect() as connection:
        _ensure_table(connection)
        row = connection.execute(
            "SELECT * FROM "
            "signed_retirement_restore_deletion_authorization_consumptions "
            "WHERE authorization_id=?",
            (authorization,),
        ).fetchone()
    return None if row is None else _value(row)


def reserve_authorization_for_deletion(
    store: Any,
    *,
    authorization_id: str,
    deletion_id: str,
    now: float,
) -> RestoreDeletionAuthorizationConsumption:
    authorization = _digest(authorization_id, "authorization_id")
    deletion = _digest(deletion_id, "deletion_id")
    timestamp = _timestamp(now, "now")
    with store._lock, store._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            _ensure_table(connection)
            authorization_row = connection.execute(
                "SELECT * FROM "
                "signed_retirement_restore_deletion_authorizations "
                "WHERE authorization_id=?",
                (authorization,),
            ).fetchone()
            if authorization_row is None:
                raise KeyError(authorization)
            authorization_value = store._verified_value(
                connection, authorization_row
            )
            if (
                authorization_value.status != "authorized"
                or authorization_value.expires_at <= timestamp
            ):
                raise RuntimeError("deletion authorization is not active.")
            row = connection.execute(
                "SELECT * FROM "
                "signed_retirement_restore_deletion_authorization_consumptions "
                "WHERE authorization_id=?",
                (authorization,),
            ).fetchone()
            if row is None:
                value = _create(
                    authorization_id=authorization,
                    deletion_id=deletion,
                    state="reserved",
                    reserved_at=timestamp,
                    consumed_at=None,
                )
                connection.execute(
                    "INSERT INTO "
                    "signed_retirement_restore_deletion_authorization_consumptions "
                    "VALUES (?,?,?,?,?,?)",
                    (
                        value.authorization_id,
                        value.deletion_id,
                        value.state,
                        value.reserved_at,
                        value.consumed_at,
                        value.consumption_digest,
                    ),
                )
                connection.execute("COMMIT")
                return value
            value = _value(row)
            if value.deletion_id != deletion:
                raise RuntimeError(
                    "authorization is bound to another deletion."
                )
            connection.execute("COMMIT")
            return value
        except Exception:
            connection.execute("ROLLBACK")
            raise


def release_authorization_reservation(
    store: Any,
    *,
    authorization_id: str,
    deletion_id: str,
) -> bool:
    authorization = _digest(authorization_id, "authorization_id")
    deletion = _digest(deletion_id, "deletion_id")
    with store._lock, store._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            _ensure_table(connection)
            row = connection.execute(
                "SELECT * FROM "
                "signed_retirement_restore_deletion_authorization_consumptions "
                "WHERE authorization_id=?",
                (authorization,),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return False
            value = _value(row)
            if value.deletion_id != deletion or value.state != "reserved":
                raise RuntimeError(
                    "authorization reservation cannot be released."
                )
            connection.execute(
                "DELETE FROM "
                "signed_retirement_restore_deletion_authorization_consumptions "
                "WHERE authorization_id=?",
                (authorization,),
            )
            connection.execute("COMMIT")
            return True
        except Exception:
            connection.execute("ROLLBACK")
            raise


def mark_authorization_consumed(
    store: Any,
    *,
    authorization_id: str,
    deletion_id: str,
    now: float,
) -> RestoreDeletionAuthorizationConsumption:
    authorization = _digest(authorization_id, "authorization_id")
    deletion = _digest(deletion_id, "deletion_id")
    timestamp = _timestamp(now, "now")
    with store._lock, store._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            _ensure_table(connection)
            row = connection.execute(
                "SELECT * FROM "
                "signed_retirement_restore_deletion_authorization_consumptions "
                "WHERE authorization_id=?",
                (authorization,),
            ).fetchone()
            if row is None:
                raise RuntimeError("authorization was not reserved.")
            current = _value(row)
            if current.deletion_id != deletion:
                raise RuntimeError(
                    "authorization is bound to another deletion."
                )
            if current.state == "consumed":
                connection.execute("COMMIT")
                return current
            value = _create(
                authorization_id=authorization,
                deletion_id=deletion,
                state="consumed",
                reserved_at=current.reserved_at,
                consumed_at=max(timestamp, current.reserved_at),
            )
            connection.execute(
                "UPDATE "
                "signed_retirement_restore_deletion_authorization_consumptions "
                "SET state='consumed', consumed_at=?, consumption_digest=? "
                "WHERE authorization_id=?",
                (
                    value.consumed_at,
                    value.consumption_digest,
                    authorization,
                ),
            )
            connection.execute("COMMIT")
            return value
        except Exception:
            connection.execute("ROLLBACK")
            raise


def require_authorization_unconsumed(store: Any, authorization_id: str) -> None:
    if get_authorization_consumption(store, authorization_id) is not None:
        raise RuntimeError(
            "deletion authorization is reserved or consumed."
        )


__all__ = [
    "RestoreDeletionAuthorizationConsumption",
    "get_authorization_consumption",
    "mark_authorization_consumed",
    "release_authorization_reservation",
    "require_authorization_unconsumed",
    "reserve_authorization_for_deletion",
]
