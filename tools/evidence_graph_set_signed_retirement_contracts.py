"""Durable contracts for retiring expired authorization-only publication duplicates."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any

from tools.security import normalize_owner_id

_SCHEMA_VERSION = 1
_STATES = frozenset({"planned", "running", "completed", "failed", "cancelled"})
_PHASES = frozenset(
    {
        "planned",
        "pointer_restore_intent",
        "pointer_safe",
        "authorization_retired",
        "verified",
    }
)
_TERMINAL_STATES = frozenset({"completed", "cancelled"})
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


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


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


def deterministic_signed_retirement_id(
    *,
    owner_id: str,
    publication_operation_id: str,
    graph_set_key: str,
    signed_candidate_set_id: str,
    signed_candidate_set_digest: str,
    authorization_candidate_set_id: str | None,
    signed_authority_digest: str,
) -> str:
    owner = normalize_owner_id(owner_id)
    operation = _digest(publication_operation_id, "publication_operation_id")
    key = _identifier(graph_set_key, "graph_set_key", 500)
    signed_id = _digest(signed_candidate_set_id, "signed_candidate_set_id")
    signed_digest = _digest(
        signed_candidate_set_digest, "signed_candidate_set_digest"
    )
    authorization_id = _optional_digest(
        authorization_candidate_set_id, "authorization_candidate_set_id"
    )
    authority = _digest(signed_authority_digest, "signed_authority_digest")
    return _canonical_digest(
        {
            "scope": "rigorousrag-signed-publication-retirement-v1",
            "owner_id": owner,
            "publication_operation_id": operation,
            "graph_set_key": key,
            "signed_candidate_set_id": signed_id,
            "signed_candidate_set_digest": signed_digest,
            "authorization_candidate_set_id": authorization_id,
            "signed_authority_digest": authority,
        }
    )


@dataclass(frozen=True)
class SignedPublicationRetirementAttempt:
    retirement_id: str
    owner_id: str
    publication_operation_id: str
    graph_set_key: str
    signed_candidate_set_id: str
    signed_candidate_set_digest: str
    authorization_candidate_set_id: str | None
    signed_authority_digest: str
    state: str
    phase: str
    attempt_count: int
    max_attempts: int
    lease_owner: str | None
    lease_expires_at: float | None
    final_pointer_set_id: str | None
    verification_digest: str | None
    failure_type: str | None
    created_at: float
    updated_at: float
    completed_at: float | None
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        operation = _digest(
            self.publication_operation_id, "publication_operation_id"
        )
        key = _identifier(self.graph_set_key, "graph_set_key", 500)
        signed_id = _digest(self.signed_candidate_set_id, "signed_candidate_set_id")
        signed_digest = _digest(
            self.signed_candidate_set_digest, "signed_candidate_set_digest"
        )
        authorization_id = _optional_digest(
            self.authorization_candidate_set_id, "authorization_candidate_set_id"
        )
        authority = _digest(self.signed_authority_digest, "signed_authority_digest")
        expected_id = deterministic_signed_retirement_id(
            owner_id=owner,
            publication_operation_id=operation,
            graph_set_key=key,
            signed_candidate_set_id=signed_id,
            signed_candidate_set_digest=signed_digest,
            authorization_candidate_set_id=authorization_id,
            signed_authority_digest=authority,
        )
        if _digest(self.retirement_id, "retirement_id") != expected_id:
            raise ValueError("retirement_id differs from immutable retirement scope.")
        state = _identifier(self.state, "state", 30)
        phase = _identifier(self.phase, "phase", 40)
        if state not in _STATES or phase not in _PHASES:
            raise ValueError("retirement state or phase is unsupported.")
        attempts = _integer(self.attempt_count, "attempt_count", 0, 1_000_000)
        maximum = _integer(self.max_attempts, "max_attempts", 1, 1_000_000)
        if attempts > maximum:
            raise ValueError("attempt_count may not exceed max_attempts.")
        lease_owner = None if self.lease_owner is None else _identifier(
            self.lease_owner, "lease_owner", 200
        )
        lease_expires = None if self.lease_expires_at is None else _timestamp(
            self.lease_expires_at, "lease_expires_at"
        )
        final_pointer = _optional_digest(
            self.final_pointer_set_id, "final_pointer_set_id"
        )
        verification = _optional_digest(
            self.verification_digest, "verification_digest"
        )
        failure = None if self.failure_type is None else _identifier(
            self.failure_type, "failure_type", 200
        )
        created = _timestamp(self.created_at, "created_at")
        updated = _timestamp(self.updated_at, "updated_at")
        completed = None if self.completed_at is None else _timestamp(
            self.completed_at, "completed_at"
        )
        if updated < created or (completed is not None and completed < created):
            raise ValueError("retirement timestamps are out of order.")
        if state == "running":
            if lease_owner is None or lease_expires is None:
                raise ValueError("running retirement attempts require an active lease.")
        elif lease_owner is not None or lease_expires is not None:
            raise ValueError("only running retirement attempts may retain a lease.")
        if phase in {"planned", "pointer_restore_intent", "pointer_safe"}:
            if final_pointer is not None or verification is not None:
                raise ValueError("early retirement phases may not contain outcome data.")
        if phase == "authorization_retired" and verification is not None:
            raise ValueError("authorization_retired phase may not be verified yet.")
        if phase == "verified":
            if state != "completed" or verification is None:
                raise ValueError("verified retirement attempts must be completed.")
        if state == "completed":
            if phase != "verified" or completed is None or failure is not None:
                raise ValueError("completed retirement attempt is inconsistent.")
        elif state == "cancelled":
            if phase != "planned" or completed is None or failure is not None:
                raise ValueError("cancelled retirement attempt is inconsistent.")
        else:
            if completed is not None:
                raise ValueError("nonterminal retirement attempt has completed_at.")
            if state == "failed" and failure is None:
                raise ValueError("failed retirement attempt requires failure_type.")
            if state != "failed" and failure is not None:
                raise ValueError("failure_type is valid only for failed attempts.")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("retirement schema is unsupported.")
        object.__setattr__(self, "retirement_id", expected_id)
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "publication_operation_id", operation)
        object.__setattr__(self, "graph_set_key", key)
        object.__setattr__(self, "signed_candidate_set_id", signed_id)
        object.__setattr__(self, "signed_candidate_set_digest", signed_digest)
        object.__setattr__(self, "authorization_candidate_set_id", authorization_id)
        object.__setattr__(self, "signed_authority_digest", authority)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "attempt_count", attempts)
        object.__setattr__(self, "max_attempts", maximum)
        object.__setattr__(self, "lease_owner", lease_owner)
        object.__setattr__(self, "lease_expires_at", lease_expires)
        object.__setattr__(self, "final_pointer_set_id", final_pointer)
        object.__setattr__(self, "verification_digest", verification)
        object.__setattr__(self, "failure_type", failure)
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)
        object.__setattr__(self, "completed_at", completed)

    @property
    def immutable_digest(self) -> str:
        return _canonical_digest(
            {
                "retirement_id": self.retirement_id,
                "owner_id": self.owner_id,
                "publication_operation_id": self.publication_operation_id,
                "graph_set_key": self.graph_set_key,
                "signed_candidate_set_id": self.signed_candidate_set_id,
                "signed_candidate_set_digest": self.signed_candidate_set_digest,
                "authorization_candidate_set_id": self.authorization_candidate_set_id,
                "signed_authority_digest": self.signed_authority_digest,
                "schema_version": self.schema_version,
            }
        )

    @classmethod
    def create(
        cls,
        *,
        owner_id: str,
        publication_operation_id: str,
        graph_set_key: str,
        signed_candidate_set_id: str,
        signed_candidate_set_digest: str,
        authorization_candidate_set_id: str | None,
        signed_authority_digest: str,
        max_attempts: int = 3,
        now: float | None = None,
    ) -> "SignedPublicationRetirementAttempt":
        timestamp = _timestamp(time.time() if now is None else now, "now")
        retirement_id = deterministic_signed_retirement_id(
            owner_id=owner_id,
            publication_operation_id=publication_operation_id,
            graph_set_key=graph_set_key,
            signed_candidate_set_id=signed_candidate_set_id,
            signed_candidate_set_digest=signed_candidate_set_digest,
            authorization_candidate_set_id=authorization_candidate_set_id,
            signed_authority_digest=signed_authority_digest,
        )
        return cls(
            retirement_id=retirement_id,
            owner_id=owner_id,
            publication_operation_id=publication_operation_id,
            graph_set_key=graph_set_key,
            signed_candidate_set_id=signed_candidate_set_id,
            signed_candidate_set_digest=signed_candidate_set_digest,
            authorization_candidate_set_id=authorization_candidate_set_id,
            signed_authority_digest=signed_authority_digest,
            state="planned",
            phase="planned",
            attempt_count=0,
            max_attempts=max_attempts,
            lease_owner=None,
            lease_expires_at=None,
            final_pointer_set_id=None,
            verification_digest=None,
            failure_type=None,
            created_at=timestamp,
            updated_at=timestamp,
            completed_at=None,
        )


__all__ = [
    "SignedPublicationRetirementAttempt",
    "_MAX_LIMIT",
    "_PHASES",
    "_STATES",
    "_TERMINAL_STATES",
    "_canonical_digest",
    "_digest",
    "_identifier",
    "_integer",
    "_optional_digest",
    "_timestamp",
    "deterministic_signed_retirement_id",
]
