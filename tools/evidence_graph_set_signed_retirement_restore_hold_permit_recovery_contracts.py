"""Contracts for governed stale hold-placement permit recovery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _timestamp,
)
from tools.security import normalize_owner_id

_SCHEMA_VERSION = 1
_ALLOWED_METHODS = frozenset(
    {"process_environment", "descriptor_file", "hmac_assertion"}
)
_CLASSIFICATIONS = frozenset(
    {"abandoned_without_hold_quarantined", "released_hold_cleanup"}
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


def deterministic_hold_permit_recovery_id(
    *,
    owner_id: str,
    restore_id: str,
    hold_id: str,
    original_permit_digest: str,
) -> str:
    return _canonical_digest(
        {
            "scope": "rigorousrag-restore-hold-permit-recovery-v1",
            "owner_id": normalize_owner_id(owner_id),
            "restore_id": _digest(restore_id, "restore_id"),
            "hold_id": _digest(hold_id, "hold_id"),
            "original_permit_digest": _digest(
                original_permit_digest,
                "original_permit_digest",
            ),
        }
    )


@dataclass(frozen=True)
class RestoreHoldPermitRecoveryReceipt:
    recovery_id: str
    owner_id: str
    restore_id: str
    hold_id: str
    original_permit_digest: str
    released_permit_digest: str
    classification: str
    quarantine_hold_id: str | None
    quarantine_hold_digest: str | None
    actor_id: str
    actor_binding_method: str
    actor_binding_digest: str
    recovered_at: float
    receipt_digest: str
    schema_version: int = _SCHEMA_VERSION

    def stable_payload(self) -> dict[str, Any]:
        return {
            "scope": "rigorousrag-restore-hold-permit-recovery-receipt-v1",
            "recovery_id": self.recovery_id,
            "owner_id": self.owner_id,
            "restore_id": self.restore_id,
            "hold_id": self.hold_id,
            "original_permit_digest": self.original_permit_digest,
            "released_permit_digest": self.released_permit_digest,
            "classification": self.classification,
            "quarantine_hold_id": self.quarantine_hold_id,
            "quarantine_hold_digest": self.quarantine_hold_digest,
            "actor_id": self.actor_id,
            "actor_binding_method": self.actor_binding_method,
            "actor_binding_digest": self.actor_binding_digest,
            "recovered_at": self.recovered_at,
            "schema_version": self.schema_version,
        }

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        restore = _digest(self.restore_id, "restore_id")
        hold = _digest(self.hold_id, "hold_id")
        original = _digest(
            self.original_permit_digest,
            "original_permit_digest",
        )
        released = _digest(
            self.released_permit_digest,
            "released_permit_digest",
        )
        recovery = _digest(self.recovery_id, "recovery_id")
        if recovery != deterministic_hold_permit_recovery_id(
            owner_id=owner,
            restore_id=restore,
            hold_id=hold,
            original_permit_digest=original,
        ):
            raise ValueError("recovery_id differs from immutable permit scope.")
        classification = _identifier(
            self.classification,
            "classification",
            100,
        )
        if classification not in _CLASSIFICATIONS:
            raise ValueError("permit recovery classification is unsupported.")
        quarantine_id = (
            None
            if self.quarantine_hold_id is None
            else _digest(self.quarantine_hold_id, "quarantine_hold_id")
        )
        quarantine_digest = (
            None
            if self.quarantine_hold_digest is None
            else _digest(
                self.quarantine_hold_digest,
                "quarantine_hold_digest",
            )
        )
        if classification == "abandoned_without_hold_quarantined":
            if quarantine_id is None or quarantine_digest is None:
                raise ValueError(
                    "abandoned recovery requires quarantine hold evidence."
                )
        elif quarantine_id is not None or quarantine_digest is not None:
            raise ValueError(
                "released-hold cleanup may not contain quarantine evidence."
            )
        actor_id = _identifier(self.actor_id, "actor_id", 200)
        method = _identifier(
            self.actor_binding_method,
            "actor_binding_method",
            50,
        )
        if method not in _ALLOWED_METHODS:
            raise ValueError("permit recovery actor method is unsupported.")
        actor_digest = _digest(
            self.actor_binding_digest,
            "actor_binding_digest",
        )
        recovered = _timestamp(self.recovered_at, "recovered_at")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("permit recovery receipt schema is unsupported.")
        values = {
            "recovery_id": recovery,
            "owner_id": owner,
            "restore_id": restore,
            "hold_id": hold,
            "original_permit_digest": original,
            "released_permit_digest": released,
            "classification": classification,
            "quarantine_hold_id": quarantine_id,
            "quarantine_hold_digest": quarantine_digest,
            "actor_id": actor_id,
            "actor_binding_method": method,
            "actor_binding_digest": actor_digest,
            "recovered_at": recovered,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        receipt = _digest(self.receipt_digest, "receipt_digest")
        if receipt != _canonical_digest(self.stable_payload()):
            raise ValueError("receipt_digest differs from recovery receipt.")
        object.__setattr__(self, "receipt_digest", receipt)

    @classmethod
    def create(
        cls,
        *,
        owner_id: str,
        restore_id: str,
        hold_id: str,
        original_permit_digest: str,
        released_permit_digest: str,
        classification: str,
        quarantine_hold_id: str | None,
        quarantine_hold_digest: str | None,
        actor_id: str,
        actor_binding_method: str,
        actor_binding_digest: str,
        recovered_at: float,
    ) -> "RestoreHoldPermitRecoveryReceipt":
        recovery_id = deterministic_hold_permit_recovery_id(
            owner_id=owner_id,
            restore_id=restore_id,
            hold_id=hold_id,
            original_permit_digest=original_permit_digest,
        )
        payload = {
            "scope": "rigorousrag-restore-hold-permit-recovery-receipt-v1",
            "recovery_id": recovery_id,
            "owner_id": normalize_owner_id(owner_id),
            "restore_id": _digest(restore_id, "restore_id"),
            "hold_id": _digest(hold_id, "hold_id"),
            "original_permit_digest": _digest(
                original_permit_digest,
                "original_permit_digest",
            ),
            "released_permit_digest": _digest(
                released_permit_digest,
                "released_permit_digest",
            ),
            "classification": _identifier(
                classification,
                "classification",
                100,
            ),
            "quarantine_hold_id": (
                None
                if quarantine_hold_id is None
                else _digest(quarantine_hold_id, "quarantine_hold_id")
            ),
            "quarantine_hold_digest": (
                None
                if quarantine_hold_digest is None
                else _digest(
                    quarantine_hold_digest,
                    "quarantine_hold_digest",
                )
            ),
            "actor_id": _identifier(actor_id, "actor_id", 200),
            "actor_binding_method": _identifier(
                actor_binding_method,
                "actor_binding_method",
                50,
            ),
            "actor_binding_digest": _digest(
                actor_binding_digest,
                "actor_binding_digest",
            ),
            "recovered_at": _timestamp(recovered_at, "recovered_at"),
            "schema_version": _SCHEMA_VERSION,
        }
        return cls(
            **{key: value for key, value in payload.items() if key != "scope"},
            receipt_digest=_canonical_digest(payload),
        )


def ensure_recovery_table(connection: Any) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS
            signed_retirement_restore_hold_permit_recoveries (
                recovery_id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                restore_id TEXT NOT NULL,
                hold_id TEXT UNIQUE NOT NULL,
                original_permit_digest TEXT NOT NULL,
                released_permit_digest TEXT NOT NULL,
                classification TEXT NOT NULL,
                quarantine_hold_id TEXT,
                quarantine_hold_digest TEXT,
                actor_id TEXT NOT NULL,
                actor_binding_method TEXT NOT NULL,
                actor_binding_digest TEXT NOT NULL,
                recovered_at REAL NOT NULL,
                receipt_digest TEXT NOT NULL,
                schema_version INTEGER NOT NULL
            )
        """
    )


def recovery_table_exists(connection: Any) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='signed_retirement_restore_hold_permit_recoveries'"
        ).fetchone()
        is not None
    )


def receipt_from_row(row: Any) -> RestoreHoldPermitRecoveryReceipt:
    try:
        return RestoreHoldPermitRecoveryReceipt(
            recovery_id=row["recovery_id"],
            owner_id=row["owner_id"],
            restore_id=row["restore_id"],
            hold_id=row["hold_id"],
            original_permit_digest=row["original_permit_digest"],
            released_permit_digest=row["released_permit_digest"],
            classification=row["classification"],
            quarantine_hold_id=row["quarantine_hold_id"],
            quarantine_hold_digest=row["quarantine_hold_digest"],
            actor_id=row["actor_id"],
            actor_binding_method=row["actor_binding_method"],
            actor_binding_digest=row["actor_binding_digest"],
            recovered_at=row["recovered_at"],
            receipt_digest=row["receipt_digest"],
            schema_version=int(row["schema_version"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("stored permit recovery receipt is corrupt.") from exc


__all__ = [
    "RestoreHoldPermitRecoveryReceipt",
    "deterministic_hold_permit_recovery_id",
    "ensure_recovery_table",
    "receipt_from_row",
    "recovery_table_exists",
]
