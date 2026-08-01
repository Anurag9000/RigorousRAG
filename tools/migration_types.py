"""Validated profile-migration planning and journal value types."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from tools.security import normalize_owner_id

_STATES = {"planned", "running", "validated", "committed", "failed", "cancelled"}


def identifier(value: Any, label: str, maximum: int = 200) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip()
    if (
        not cleaned
        or len(cleaned) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in cleaned)
    ):
        raise ValueError(f"{label} is invalid.")
    return cleaned


def digest(value: Any, label: str) -> str:
    cleaned = identifier(value, label, 64).lower()
    if len(cleaned) != 64 or any(character not in "0123456789abcdef" for character in cleaned):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return cleaned


def exact_integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def timestamp(value: Any, label: str = "timestamp") -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite.") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return parsed


@dataclass(frozen=True)
class MigrationCandidate:
    owner_id: str
    doc_id: str
    source_sequence: int
    source_profile_fingerprint: str
    target_profile_name: str
    target_profile_fingerprint: str
    retained_source: bool
    eligible: bool
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "doc_id", identifier(self.doc_id, "doc_id"))
        object.__setattr__(
            self,
            "source_sequence",
            exact_integer(self.source_sequence, "source_sequence", 1, 2**63 - 1),
        )
        object.__setattr__(
            self,
            "source_profile_fingerprint",
            digest(self.source_profile_fingerprint, "source_profile_fingerprint"),
        )
        object.__setattr__(
            self,
            "target_profile_name",
            identifier(self.target_profile_name, "target_profile_name"),
        )
        object.__setattr__(
            self,
            "target_profile_fingerprint",
            digest(self.target_profile_fingerprint, "target_profile_fingerprint"),
        )
        if not isinstance(self.retained_source, bool) or not isinstance(self.eligible, bool):
            raise ValueError("retained_source and eligible must be booleans.")
        object.__setattr__(self, "reason", identifier(self.reason, "reason", 200))
        if self.eligible and not self.retained_source:
            raise ValueError("eligible migrations require a retained source.")


@dataclass(frozen=True)
class MigrationTask:
    task_id: str
    owner_id: str
    doc_id: str
    source_sequence: int
    source_profile_fingerprint: str
    target_profile_name: str
    target_profile_fingerprint: str
    state: str
    attempt: int
    created_at: float
    updated_at: float
    lease_owner: str | None = None
    lease_expires_at: float | None = None
    validation_digest: str | None = None
    failure_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", identifier(self.task_id, "task_id", 64))
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "doc_id", identifier(self.doc_id, "doc_id"))
        object.__setattr__(
            self,
            "source_sequence",
            exact_integer(self.source_sequence, "source_sequence", 1, 2**63 - 1),
        )
        object.__setattr__(
            self,
            "source_profile_fingerprint",
            digest(self.source_profile_fingerprint, "source_profile_fingerprint"),
        )
        object.__setattr__(
            self,
            "target_profile_name",
            identifier(self.target_profile_name, "target_profile_name"),
        )
        object.__setattr__(
            self,
            "target_profile_fingerprint",
            digest(self.target_profile_fingerprint, "target_profile_fingerprint"),
        )
        if self.state not in _STATES:
            raise ValueError("migration state is invalid.")
        object.__setattr__(self, "attempt", exact_integer(self.attempt, "attempt", 0, 1_000_000))
        object.__setattr__(self, "created_at", timestamp(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", timestamp(self.updated_at, "updated_at"))
        if self.lease_owner is not None:
            object.__setattr__(self, "lease_owner", identifier(self.lease_owner, "lease_owner", 128))
        if self.lease_expires_at is not None:
            object.__setattr__(
                self,
                "lease_expires_at",
                timestamp(self.lease_expires_at, "lease_expires_at"),
            )
        if self.validation_digest is not None:
            object.__setattr__(
                self,
                "validation_digest",
                digest(self.validation_digest, "validation_digest"),
            )
        if self.failure_type is not None:
            object.__setattr__(
                self,
                "failure_type",
                identifier(self.failure_type, "failure_type", 200),
            )
        if self.state in {"running", "validated"} and (
            self.lease_owner is None or self.lease_expires_at is None
        ):
            raise ValueError("active migration states require a lease.")
        if self.state == "validated" and self.validation_digest is None:
            raise ValueError("validated migrations require a validation digest.")


__all__ = [
    "MigrationCandidate",
    "MigrationTask",
    "digest",
    "exact_integer",
    "identifier",
    "timestamp",
]
