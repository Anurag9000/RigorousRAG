"""Durable contracts for empty-target signed retirement snapshot restores."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from tools.security import normalize_owner_id

_SCHEMA_VERSION = 1
_STATES = frozenset({"planned", "running", "completed", "failed", "cancelled"})
_PHASES = frozenset({"planned", "target_committed", "verified"})
_MAX_LIMIT = 10_000


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    selected = value.strip()
    if (
        not selected
        or len(selected) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in selected)
    ):
        raise ValueError(f"{label} is invalid.")
    return selected


def _digest(value: Any, label: str) -> str:
    selected = _identifier(value, label, 64).lower()
    if len(selected) != 64 or any(
        character not in "0123456789abcdef" for character in selected
    ):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return selected


def _optional_digest(value: Any, label: str) -> str | None:
    return None if value is None else _digest(value, label)


def _timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(selected) or selected < 0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return selected


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


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


def deterministic_signed_retirement_restore_id(
    *, owner_id: str, snapshot_digest: str, target_path_digest: str
) -> str:
    return _canonical_digest(
        {
            "scope": "rigorousrag-signed-retirement-empty-target-restore-v1",
            "owner_id": normalize_owner_id(owner_id),
            "snapshot_digest": _digest(snapshot_digest, "snapshot_digest"),
            "target_path_digest": _digest(
                target_path_digest, "target_path_digest"
            ),
        }
    )


@dataclass(frozen=True)
class SignedRetirementRestoreAttempt:
    restore_id: str
    owner_id: str
    snapshot_digest: str
    target_path_digest: str
    snapshot_record_count: int
    state: str
    phase: str
    attempt_count: int
    max_attempts: int
    lease_owner: str | None
    lease_expires_at: float | None
    target_verification_digest: str | None
    failure_type: str | None
    created_at: float
    updated_at: float
    completed_at: float | None
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        snapshot = _digest(self.snapshot_digest, "snapshot_digest")
        target = _digest(self.target_path_digest, "target_path_digest")
        expected = deterministic_signed_retirement_restore_id(
            owner_id=owner,
            snapshot_digest=snapshot,
            target_path_digest=target,
        )
        restore = _digest(self.restore_id, "restore_id")
        if restore != expected:
            raise ValueError("restore_id differs from immutable restore scope.")
        records = _integer(
            self.snapshot_record_count,
            "snapshot_record_count",
            1,
            _MAX_LIMIT,
        )
        state = _identifier(self.state, "state", 30)
        phase = _identifier(self.phase, "phase", 30)
        if state not in _STATES or phase not in _PHASES:
            raise ValueError("restore state or phase is unsupported.")
        attempts = _integer(self.attempt_count, "attempt_count", 0, 1_000_000)
        ceiling = _integer(self.max_attempts, "max_attempts", 1, 100)
        if attempts > ceiling:
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
        verification = _optional_digest(
            self.target_verification_digest,
            "target_verification_digest",
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
        if updated < created or (completed is not None and completed < updated):
            raise ValueError("restore timestamps are not monotonic.")
        if state == "running":
            if lease_owner is None or lease_expires is None or completed is not None:
                raise ValueError("running restore lease is incomplete.")
        elif lease_owner is not None or lease_expires is not None:
            raise ValueError("non-running restore may not retain a lease.")
        if phase == "planned" and verification is not None:
            raise ValueError("planned restore may not have target verification.")
        if phase in {"target_committed", "verified"} and verification is None:
            raise ValueError(
                "committed restore phase requires target verification."
            )
        if state == "completed":
            if phase != "verified" or completed is None:
                raise ValueError("completed restore must be verified.")
        elif state == "cancelled":
            if phase != "planned" or completed is None:
                raise ValueError("cancelled restore must remain unstarted.")
        elif completed is not None:
            raise ValueError("nonterminal restore may not have completed_at.")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("restore schema version is unsupported.")
        object.__setattr__(self, "restore_id", restore)
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "snapshot_digest", snapshot)
        object.__setattr__(self, "target_path_digest", target)
        object.__setattr__(self, "snapshot_record_count", records)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "attempt_count", attempts)
        object.__setattr__(self, "max_attempts", ceiling)
        object.__setattr__(self, "lease_owner", lease_owner)
        object.__setattr__(self, "lease_expires_at", lease_expires)
        object.__setattr__(
            self, "target_verification_digest", verification
        )
        object.__setattr__(self, "failure_type", failure)
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)
        object.__setattr__(self, "completed_at", completed)

    @property
    def immutable_digest(self) -> str:
        return _canonical_digest(
            {
                "scope": "rigorousrag-signed-retirement-restore-immutable-v1",
                "restore_id": self.restore_id,
                "owner_id": self.owner_id,
                "snapshot_digest": self.snapshot_digest,
                "target_path_digest": self.target_path_digest,
                "snapshot_record_count": self.snapshot_record_count,
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
        snapshot_record_count: int,
        max_attempts: int = 3,
        now: float,
    ) -> "SignedRetirementRestoreAttempt":
        owner = normalize_owner_id(owner_id)
        snapshot = _digest(snapshot_digest, "snapshot_digest")
        target = _digest(target_path_digest, "target_path_digest")
        timestamp = _timestamp(now, "now")
        return cls(
            restore_id=deterministic_signed_retirement_restore_id(
                owner_id=owner,
                snapshot_digest=snapshot,
                target_path_digest=target,
            ),
            owner_id=owner,
            snapshot_digest=snapshot,
            target_path_digest=target,
            snapshot_record_count=snapshot_record_count,
            state="planned",
            phase="planned",
            attempt_count=0,
            max_attempts=max_attempts,
            lease_owner=None,
            lease_expires_at=None,
            target_verification_digest=None,
            failure_type=None,
            created_at=timestamp,
            updated_at=timestamp,
            completed_at=None,
        )


__all__ = [
    "SignedRetirementRestoreAttempt",
    "deterministic_signed_retirement_restore_id",
]
