"""Durable contracts for one-serial custody timestamp issuance."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _integer,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_snapshot import _path
from tools.security import normalize_owner_id

_SCHEMA_VERSION = 1
_STATES = frozenset({"planned", "running", "completed", "failed", "cancelled"})
_PHASES = frozenset({"planned", "output_published", "verified"})


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


def timestamp_output_path_digest(value: str | os.PathLike[str]) -> str:
    selected = _path(value, label="output_path")
    return hashlib.sha256(str(selected).encode("utf-8")).hexdigest()


def deterministic_timestamp_issuance_id(
    *,
    owner_id: str,
    authority_id: str,
    key_id: str,
    serial: str,
    output_path_digest: str,
) -> str:
    return _canonical_digest(
        {
            "scope": "rigorousrag-custody-timestamp-issuance-v1",
            "owner_id": normalize_owner_id(owner_id),
            "authority_id": _identifier(authority_id, "authority_id", 200),
            "key_id": _identifier(key_id, "key_id", 200),
            "serial": _digest(serial, "serial"),
            "output_path_digest": _digest(
                output_path_digest,
                "output_path_digest",
            ),
        }
    )


@dataclass(frozen=True)
class CustodyTimestampIssuanceAttempt:
    issuance_id: str
    owner_id: str
    authority_id: str
    key_id: str
    serial: str
    attestation_digest: str
    output_path_digest: str
    state: str
    phase: str
    attempt_count: int
    max_attempts: int
    lease_owner: str | None
    lease_expires_at: float | None
    verification_digest: str | None
    failure_type: str | None
    created_at: float
    updated_at: float
    completed_at: float | None
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        authority = _identifier(self.authority_id, "authority_id", 200)
        key_id = _identifier(self.key_id, "key_id", 200)
        serial = _digest(self.serial, "serial")
        attestation = _digest(self.attestation_digest, "attestation_digest")
        output = _digest(self.output_path_digest, "output_path_digest")
        issuance = _digest(self.issuance_id, "issuance_id")
        expected = deterministic_timestamp_issuance_id(
            owner_id=owner,
            authority_id=authority,
            key_id=key_id,
            serial=serial,
            output_path_digest=output,
        )
        if issuance != expected:
            raise ValueError("issuance_id differs from timestamp issuance scope.")
        state = _identifier(self.state, "state", 30)
        phase = _identifier(self.phase, "phase", 40)
        if state not in _STATES or phase not in _PHASES:
            raise ValueError("timestamp issuance state or phase is unsupported.")
        attempts = _integer(self.attempt_count, "attempt_count", 0, 1_000_000)
        maximum = _integer(self.max_attempts, "max_attempts", 1, 1_000_000)
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
        verification = (
            None
            if self.verification_digest is None
            else _digest(self.verification_digest, "verification_digest")
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
            raise ValueError("timestamp issuance timestamps are not monotonic.")
        if state == "running":
            if lease_owner is None or lease_expires is None:
                raise ValueError("running timestamp issuance requires a lease.")
        elif lease_owner is not None or lease_expires is not None:
            raise ValueError("non-running timestamp issuance may not retain a lease.")
        if state == "completed":
            if phase != "verified" or verification is None or completed is None:
                raise ValueError("completed timestamp issuance requires verification.")
        elif completed is not None:
            raise ValueError("non-completed timestamp issuance may not have completed_at.")
        if phase == "verified" and state != "completed":
            raise ValueError("verified timestamp issuance must be completed.")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("timestamp issuance schema is unsupported.")
        for name, value in {
            "issuance_id": issuance,
            "owner_id": owner,
            "authority_id": authority,
            "key_id": key_id,
            "serial": serial,
            "attestation_digest": attestation,
            "output_path_digest": output,
            "state": state,
            "phase": phase,
            "attempt_count": attempts,
            "max_attempts": maximum,
            "lease_owner": lease_owner,
            "lease_expires_at": lease_expires,
            "verification_digest": verification,
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
        owner_id: str,
        authority_id: str,
        key_id: str,
        serial: str,
        attestation_digest: str,
        output_path_digest: str,
        max_attempts: int = 3,
        now: float,
    ) -> "CustodyTimestampIssuanceAttempt":
        timestamp = _timestamp(now, "now")
        owner = normalize_owner_id(owner_id)
        authority = _identifier(authority_id, "authority_id", 200)
        selected_key = _identifier(key_id, "key_id", 200)
        selected_serial = _digest(serial, "serial")
        output = _digest(output_path_digest, "output_path_digest")
        return cls(
            issuance_id=deterministic_timestamp_issuance_id(
                owner_id=owner,
                authority_id=authority,
                key_id=selected_key,
                serial=selected_serial,
                output_path_digest=output,
            ),
            owner_id=owner,
            authority_id=authority,
            key_id=selected_key,
            serial=selected_serial,
            attestation_digest=_digest(attestation_digest, "attestation_digest"),
            output_path_digest=output,
            state="planned",
            phase="planned",
            attempt_count=0,
            max_attempts=max_attempts,
            lease_owner=None,
            lease_expires_at=None,
            verification_digest=None,
            failure_type=None,
            created_at=timestamp,
            updated_at=timestamp,
            completed_at=None,
        )

    @property
    def immutable_digest(self) -> str:
        return _canonical_digest(
            {
                "scope": "rigorousrag-custody-timestamp-issuance-immutable-v1",
                "issuance_id": self.issuance_id,
                "owner_id": self.owner_id,
                "authority_id": self.authority_id,
                "key_id": self.key_id,
                "serial": self.serial,
                "attestation_digest": self.attestation_digest,
                "output_path_digest": self.output_path_digest,
                "max_attempts": self.max_attempts,
                "schema_version": self.schema_version,
            }
        )


__all__ = [
    "CustodyTimestampIssuanceAttempt",
    "deterministic_timestamp_issuance_id",
    "timestamp_output_path_digest",
]
