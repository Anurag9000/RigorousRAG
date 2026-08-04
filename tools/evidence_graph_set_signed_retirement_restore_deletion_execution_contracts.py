"""Durable contracts for crash-recoverable restore-intent deletion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _integer,
    _timestamp,
)
from tools.security import normalize_owner_id

_SCHEMA_VERSION = 1
_STATES = frozenset({"planned", "running", "completed", "failed", "cancelled"})
_PHASES = frozenset({"planned", "marker_active", "restore_deleted", "verified"})


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


def deterministic_restore_deletion_id(
    *,
    authorization_id: str,
    authorization_digest: str,
    owner_id: str,
    restore_id: str,
    snapshot_digest: str,
    target_path_digest: str,
    restore_record_digest: str,
    custody_manifest_digest: str | None,
) -> str:
    return _canonical_digest(
        {
            "scope": "rigorousrag-restore-intent-deletion-v1",
            "authorization_id": _digest(
                authorization_id, "authorization_id"
            ),
            "authorization_digest": _digest(
                authorization_digest, "authorization_digest"
            ),
            "owner_id": normalize_owner_id(owner_id),
            "restore_id": _digest(restore_id, "restore_id"),
            "snapshot_digest": _digest(snapshot_digest, "snapshot_digest"),
            "target_path_digest": _digest(
                target_path_digest, "target_path_digest"
            ),
            "restore_record_digest": _digest(
                restore_record_digest, "restore_record_digest"
            ),
            "custody_manifest_digest": (
                None
                if custody_manifest_digest is None
                else _digest(
                    custody_manifest_digest, "custody_manifest_digest"
                )
            ),
        }
    )


@dataclass(frozen=True)
class SignedRetirementRestoreDeletionAttempt:
    deletion_id: str
    authorization_id: str
    authorization_digest: str
    owner_id: str
    restore_id: str
    snapshot_digest: str
    target_path_digest: str
    restore_state: str
    restore_phase: str
    restore_record_digest: str
    custody_id: str | None
    custody_manifest_digest: str | None
    state: str
    phase: str
    attempt_count: int
    max_attempts: int
    lease_owner: str | None
    lease_expires_at: float | None
    marker_digest: str | None
    tombstone_digest: str | None
    failure_type: str | None
    created_at: float
    updated_at: float
    completed_at: float | None
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        authorization = _digest(self.authorization_id, "authorization_id")
        authorization_digest = _digest(
            self.authorization_digest, "authorization_digest"
        )
        restore = _digest(self.restore_id, "restore_id")
        snapshot = _digest(self.snapshot_digest, "snapshot_digest")
        target = _digest(self.target_path_digest, "target_path_digest")
        record = _digest(self.restore_record_digest, "restore_record_digest")
        restore_state = _identifier(self.restore_state, "restore_state", 30)
        restore_phase = _identifier(self.restore_phase, "restore_phase", 30)
        if restore_state not in {"completed", "cancelled"}:
            raise ValueError("only terminal restore records can be deleted.")
        if restore_state == "completed" and restore_phase != "verified":
            raise ValueError("completed restore deletion requires verified phase.")
        if restore_state == "cancelled" and restore_phase != "planned":
            raise ValueError("cancelled restore deletion requires planned phase.")
        custody_id = (
            None
            if self.custody_id is None
            else _digest(self.custody_id, "custody_id")
        )
        custody_digest = (
            None
            if self.custody_manifest_digest is None
            else _digest(
                self.custody_manifest_digest, "custody_manifest_digest"
            )
        )
        if (custody_id is None) != (custody_digest is None):
            raise ValueError(
                "custody identity and digest must be supplied together."
            )
        if restore_state == "completed" and custody_id is None:
            raise ValueError("completed restore deletion requires custody evidence.")
        deletion = _digest(self.deletion_id, "deletion_id")
        expected = deterministic_restore_deletion_id(
            authorization_id=authorization,
            authorization_digest=authorization_digest,
            owner_id=owner,
            restore_id=restore,
            snapshot_digest=snapshot,
            target_path_digest=target,
            restore_record_digest=record,
            custody_manifest_digest=custody_digest,
        )
        if deletion != expected:
            raise ValueError(
                "deletion_id differs from immutable deletion scope."
            )
        state = _identifier(self.state, "state", 20)
        phase = _identifier(self.phase, "phase", 30)
        if state not in _STATES or phase not in _PHASES:
            raise ValueError("deletion state or phase is unsupported.")
        attempts = _integer(
            self.attempt_count, "attempt_count", 0, 1_000_000
        )
        maximum = _integer(self.max_attempts, "max_attempts", 1, 1_000_000)
        if attempts > maximum:
            raise ValueError("attempt_count exceeds max_attempts.")
        lease_owner = (
            None
            if self.lease_owner is None
            else _identifier(self.lease_owner, "lease_owner", 200)
        )
        lease_expires = (
            None
            if self.lease_expires_at is None
            else _timestamp(self.lease_expires_at, "lease_expires_at")
        )
        marker = (
            None
            if self.marker_digest is None
            else _digest(self.marker_digest, "marker_digest")
        )
        tombstone = (
            None
            if self.tombstone_digest is None
            else _digest(self.tombstone_digest, "tombstone_digest")
        )
        failure = (
            None
            if self.failure_type is None
            else _identifier(self.failure_type, "failure_type", 200)
        )
        created = _timestamp(self.created_at, "created_at")
        updated = _timestamp(self.updated_at, "updated_at")
        completed = (
            None
            if self.completed_at is None
            else _timestamp(self.completed_at, "completed_at")
        )
        if updated < created or (
            completed is not None and completed < created
        ):
            raise ValueError("deletion timestamps are not monotonic.")
        if phase in {"marker_active", "restore_deleted", "verified"} and marker is None:
            raise ValueError("deletion phase requires marker digest.")
        if phase in {"restore_deleted", "verified"} and tombstone is None:
            raise ValueError("deleted phase requires tombstone digest.")
        if state == "completed" and (
            phase != "verified"
            or completed is None
            or lease_owner is not None
            or lease_expires is not None
        ):
            raise ValueError("completed deletion fields are inconsistent.")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("deletion schema is unsupported.")
        for name, value in {
            "deletion_id": deletion,
            "authorization_id": authorization,
            "authorization_digest": authorization_digest,
            "owner_id": owner,
            "restore_id": restore,
            "snapshot_digest": snapshot,
            "target_path_digest": target,
            "restore_state": restore_state,
            "restore_phase": restore_phase,
            "restore_record_digest": record,
            "custody_id": custody_id,
            "custody_manifest_digest": custody_digest,
            "state": state,
            "phase": phase,
            "attempt_count": attempts,
            "max_attempts": maximum,
            "lease_owner": lease_owner,
            "lease_expires_at": lease_expires,
            "marker_digest": marker,
            "tombstone_digest": tombstone,
            "failure_type": failure,
            "created_at": created,
            "updated_at": updated,
            "completed_at": completed,
        }.items():
            object.__setattr__(self, name, value)

    @classmethod
    def create(
        cls,
        *,
        authorization_id: str,
        authorization_digest: str,
        owner_id: str,
        restore_id: str,
        snapshot_digest: str,
        target_path_digest: str,
        restore_state: str,
        restore_phase: str,
        restore_record_digest: str,
        custody_id: str | None,
        custody_manifest_digest: str | None,
        max_attempts: int = 3,
        now: float,
    ) -> "SignedRetirementRestoreDeletionAttempt":
        deletion_id = deterministic_restore_deletion_id(
            authorization_id=authorization_id,
            authorization_digest=authorization_digest,
            owner_id=owner_id,
            restore_id=restore_id,
            snapshot_digest=snapshot_digest,
            target_path_digest=target_path_digest,
            restore_record_digest=restore_record_digest,
            custody_manifest_digest=custody_manifest_digest,
        )
        return cls(
            deletion_id=deletion_id,
            authorization_id=authorization_id,
            authorization_digest=authorization_digest,
            owner_id=owner_id,
            restore_id=restore_id,
            snapshot_digest=snapshot_digest,
            target_path_digest=target_path_digest,
            restore_state=restore_state,
            restore_phase=restore_phase,
            restore_record_digest=restore_record_digest,
            custody_id=custody_id,
            custody_manifest_digest=custody_manifest_digest,
            state="planned",
            phase="planned",
            attempt_count=0,
            max_attempts=max_attempts,
            lease_owner=None,
            lease_expires_at=None,
            marker_digest=None,
            tombstone_digest=None,
            failure_type=None,
            created_at=now,
            updated_at=now,
            completed_at=None,
        )

    @property
    def immutable_digest(self) -> str:
        return _canonical_digest(
            {
                "scope": "rigorousrag-restore-intent-deletion-attempt-immutable-v1",
                "deletion_id": self.deletion_id,
                "authorization_id": self.authorization_id,
                "authorization_digest": self.authorization_digest,
                "owner_id": self.owner_id,
                "restore_id": self.restore_id,
                "snapshot_digest": self.snapshot_digest,
                "target_path_digest": self.target_path_digest,
                "restore_state": self.restore_state,
                "restore_phase": self.restore_phase,
                "restore_record_digest": self.restore_record_digest,
                "custody_id": self.custody_id,
                "custody_manifest_digest": self.custody_manifest_digest,
                "max_attempts": self.max_attempts,
                "schema_version": self.schema_version,
            }
        )


__all__ = [
    "SignedRetirementRestoreDeletionAttempt",
    "deterministic_restore_deletion_id",
]
