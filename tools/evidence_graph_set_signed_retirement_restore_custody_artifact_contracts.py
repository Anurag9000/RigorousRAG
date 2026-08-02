"""Durable contracts for pre-restore custody artifact publication attempts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _integer,
    _timestamp,
)
from tools.security import normalize_owner_id

_SCHEMA_VERSION = 1
_STATES = frozenset(
    {"planned", "running", "completed", "orphaned", "failed", "cancelled"}
)
_PHASES = frozenset({"planned", "publication_intent", "observed", "verified"})
_ORPHAN_DISPOSITIONS = frozenset(
    {"backup_without_receipt", "receipt_without_backup", "artifact_collision"}
)


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def deterministic_custody_artifact_id(
    *,
    owner_id: str,
    snapshot_digest: str,
    target_path_digest: str,
    backup_path_digest: str,
    receipt_path_digest: str,
) -> str:
    return _canonical_digest(
        {
            "scope": "rigorousrag-restore-custody-artifact-attempt-v1",
            "owner_id": normalize_owner_id(owner_id),
            "snapshot_digest": _digest(snapshot_digest, "snapshot_digest"),
            "target_path_digest": _digest(
                target_path_digest, "target_path_digest"
            ),
            "backup_path_digest": _digest(
                backup_path_digest, "backup_path_digest"
            ),
            "receipt_path_digest": _digest(
                receipt_path_digest, "receipt_path_digest"
            ),
        }
    )


@dataclass(frozen=True)
class RestoreCustodyArtifactAttempt:
    artifact_id: str
    owner_id: str
    snapshot_digest: str
    target_path_digest: str
    backup_path_digest: str
    receipt_path_digest: str
    state: str
    phase: str
    attempt_count: int
    max_attempts: int
    lease_owner: str | None
    lease_expires_at: float | None
    backup_sha256: str | None
    backup_size_bytes: int | None
    receipt_digest: str | None
    receipt_actor_id: str | None
    receipt_binding_method: str | None
    receipt_binding_digest: str | None
    disposition: str | None
    failure_type: str | None
    created_at: float
    updated_at: float
    completed_at: float | None
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        snapshot = _digest(self.snapshot_digest, "snapshot_digest")
        target = _digest(self.target_path_digest, "target_path_digest")
        backup_path = _digest(self.backup_path_digest, "backup_path_digest")
        receipt_path = _digest(self.receipt_path_digest, "receipt_path_digest")
        artifact = _digest(self.artifact_id, "artifact_id")
        if artifact != deterministic_custody_artifact_id(
            owner_id=owner,
            snapshot_digest=snapshot,
            target_path_digest=target,
            backup_path_digest=backup_path,
            receipt_path_digest=receipt_path,
        ):
            raise ValueError("artifact_id differs from immutable artifact scope.")
        state = _identifier(self.state, "state", 30)
        phase = _identifier(self.phase, "phase", 40)
        if state not in _STATES or phase not in _PHASES:
            raise ValueError("artifact attempt state or phase is unsupported.")
        attempts = _integer(self.attempt_count, "attempt_count", 0, 1_000_000)
        maximum = _integer(self.max_attempts, "max_attempts", 1, 1_000_000)
        if attempts > maximum:
            raise ValueError("artifact attempt count exceeds its ceiling.")
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
        if state == "running":
            if lease_owner is None or lease_expires is None:
                raise ValueError("running artifact attempt requires a lease.")
        elif lease_owner is not None or lease_expires is not None:
            raise ValueError("non-running artifact attempt may not retain a lease.")
        backup_sha = (
            None
            if self.backup_sha256 is None
            else _digest(self.backup_sha256, "backup_sha256")
        )
        backup_size = (
            None
            if self.backup_size_bytes is None
            else _integer(
                self.backup_size_bytes,
                "backup_size_bytes",
                1,
                1024 * 1024 * 1024 * 1024,
            )
        )
        if (backup_sha is None) != (backup_size is None):
            raise ValueError("backup artifact digest and size must be paired.")
        receipt = (
            None
            if self.receipt_digest is None
            else _digest(self.receipt_digest, "receipt_digest")
        )
        actor = (
            None
            if self.receipt_actor_id is None
            else _identifier(self.receipt_actor_id, "receipt_actor_id", 200)
        )
        method = (
            None
            if self.receipt_binding_method is None
            else _identifier(
                self.receipt_binding_method,
                "receipt_binding_method",
                50,
            )
        )
        binding = (
            None
            if self.receipt_binding_digest is None
            else _digest(
                self.receipt_binding_digest,
                "receipt_binding_digest",
            )
        )
        if any(value is None for value in (receipt, actor, method, binding)) != all(
            value is None for value in (receipt, actor, method, binding)
        ):
            raise ValueError("receipt provenance fields must be all present or absent.")
        disposition = (
            None
            if self.disposition is None
            else _identifier(self.disposition, "disposition", 80)
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
        if updated < created or (completed is not None and completed < created):
            raise ValueError("artifact attempt timestamps are not monotonic.")
        if state == "completed":
            if (
                phase != "verified"
                or backup_sha is None
                or receipt is None
                or disposition != "paired"
                or completed is None
                or failure is not None
            ):
                raise ValueError("completed artifact attempt is incomplete.")
        elif state == "orphaned":
            if (
                phase != "observed"
                or disposition not in _ORPHAN_DISPOSITIONS
                or completed is None
                or failure is not None
            ):
                raise ValueError("orphaned artifact attempt is invalid.")
        elif state == "cancelled":
            if phase != "planned" or completed is None or failure is not None:
                raise ValueError("cancelled artifact attempt is invalid.")
        elif state == "failed":
            if failure is None or completed is not None:
                raise ValueError("failed artifact attempt requires a failure type.")
        else:
            if completed is not None or disposition is not None or failure is not None:
                raise ValueError("nonterminal artifact attempt has terminal fields.")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("artifact attempt schema is unsupported.")
        object.__setattr__(self, "artifact_id", artifact)
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "snapshot_digest", snapshot)
        object.__setattr__(self, "target_path_digest", target)
        object.__setattr__(self, "backup_path_digest", backup_path)
        object.__setattr__(self, "receipt_path_digest", receipt_path)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "attempt_count", attempts)
        object.__setattr__(self, "max_attempts", maximum)
        object.__setattr__(self, "lease_owner", lease_owner)
        object.__setattr__(self, "lease_expires_at", lease_expires)
        object.__setattr__(self, "backup_sha256", backup_sha)
        object.__setattr__(self, "backup_size_bytes", backup_size)
        object.__setattr__(self, "receipt_digest", receipt)
        object.__setattr__(self, "receipt_actor_id", actor)
        object.__setattr__(self, "receipt_binding_method", method)
        object.__setattr__(self, "receipt_binding_digest", binding)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "failure_type", failure)
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)
        object.__setattr__(self, "completed_at", completed)

    @property
    def immutable_digest(self) -> str:
        return _canonical_digest(
            {
                "scope": "rigorousrag-restore-custody-artifact-immutable-v1",
                "artifact_id": self.artifact_id,
                "owner_id": self.owner_id,
                "snapshot_digest": self.snapshot_digest,
                "target_path_digest": self.target_path_digest,
                "backup_path_digest": self.backup_path_digest,
                "receipt_path_digest": self.receipt_path_digest,
                "max_attempts": self.max_attempts,
                "schema_version": self.schema_version,
            }
        )

    @classmethod
    def create(
        cls,
        *,
        owner_id: str,
        snapshot_digest: str,
        target_path_digest: str,
        backup_path_digest: str,
        receipt_path_digest: str,
        max_attempts: int = 3,
        now: float,
    ) -> "RestoreCustodyArtifactAttempt":
        timestamp = _timestamp(now, "now")
        return cls(
            artifact_id=deterministic_custody_artifact_id(
                owner_id=owner_id,
                snapshot_digest=snapshot_digest,
                target_path_digest=target_path_digest,
                backup_path_digest=backup_path_digest,
                receipt_path_digest=receipt_path_digest,
            ),
            owner_id=owner_id,
            snapshot_digest=snapshot_digest,
            target_path_digest=target_path_digest,
            backup_path_digest=backup_path_digest,
            receipt_path_digest=receipt_path_digest,
            state="planned",
            phase="planned",
            attempt_count=0,
            max_attempts=max_attempts,
            lease_owner=None,
            lease_expires_at=None,
            backup_sha256=None,
            backup_size_bytes=None,
            receipt_digest=None,
            receipt_actor_id=None,
            receipt_binding_method=None,
            receipt_binding_digest=None,
            disposition=None,
            failure_type=None,
            created_at=timestamp,
            updated_at=timestamp,
            completed_at=None,
        )


__all__ = [
    "RestoreCustodyArtifactAttempt",
    "deterministic_custody_artifact_id",
]
