"""Durable contracts for crash-recoverable evidence-graph-set publication.

The contracts retain only immutable proposal, pointer, candidate and outcome
identities. They never store graph text, relation evidence, source paths,
queries, provider responses or unreviewed semantic output.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tools.security import normalize_owner_id

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_MAX_LIMIT = 10_000
_MAX_PROPOSALS = 100_000
_SCHEMA_VERSION = 1
_STATES = frozenset(
    {"planned", "running", "completed", "compensated", "failed", "cancelled"}
)
_PHASES = frozenset(
    {"planned", "candidate_stored", "pointer_activated", "verified", "compensated"}
)


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("publication database path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in rendered)
    ):
        raise ValueError("publication database path is invalid.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for part in (absolute, *absolute.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(
                "publication database path could not be validated."
            ) from exc
        if _redirecting(info):
            raise ValueError("publication database path may not contain redirects.")
    return absolute


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip()
    if (
        not cleaned
        or len(cleaned) > maximum
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in cleaned)
    ):
        raise ValueError(f"{label} is invalid.")
    return cleaned


def _digest(value: Any, label: str) -> str:
    cleaned = _identifier(value, label, 64).lower()
    if len(cleaned) != 64 or any(ch not in "0123456789abcdef" for ch in cleaned):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return cleaned


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
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return result


def _proposal_ids(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("proposal_ids must be an iterable of SHA-256 digests.")
    result: list[str] = []
    try:
        iterator = iter(values)
    except Exception as exc:
        raise ValueError("proposal_ids is not safely iterable.") from exc
    for value in iterator:
        result.append(_digest(value, "proposal_id"))
        if len(result) > _MAX_PROPOSALS:
            raise ValueError("proposal_ids exceeds the item limit.")
    if not result:
        raise ValueError("at least one proposal ID is required.")
    if len(set(result)) != len(result):
        raise ValueError("proposal IDs must be unique.")
    return tuple(sorted(result))


def _generic_errors(values: Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("compensation_errors must be an iterable.")
    result: list[str] = []
    for value in values:
        result.append(_identifier(value, "compensation_error", 200))
        if len(result) > 100:
            raise ValueError("compensation_errors exceeds the item limit.")
    return tuple(result)


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


def deterministic_publication_operation_id(
    *,
    owner_id: str,
    graph_set_key: str,
    proposal_ids: Iterable[str],
    expected_current_set_id: str | None,
) -> str:
    owner = normalize_owner_id(owner_id)
    key = _identifier(graph_set_key, "graph_set_key", 500)
    proposals = _proposal_ids(proposal_ids)
    expected = _optional_digest(expected_current_set_id, "expected_current_set_id")
    return _canonical_digest(
        {
            "scope": "rigorousrag-evidence-graph-set-publication-v1",
            "owner_id": owner,
            "graph_set_key": key,
            "proposal_ids": proposals,
            "expected_current_set_id": expected,
        }
    )


@dataclass(frozen=True)
class EvidenceGraphSetPublicationAttempt:
    operation_id: str
    owner_id: str
    graph_set_key: str
    proposal_ids: tuple[str, ...]
    expected_current_set_id: str | None
    state: str
    phase: str
    attempt_count: int
    max_attempts: int
    lease_owner: str | None
    lease_expires_at: float | None
    previous_graph_set_id: str | None
    previous_graph_set_digest: str | None
    candidate_graph_set_id: str | None
    candidate_graph_set_digest: str | None
    member_count: int | None
    edge_count: int | None
    verification_digest: str | None
    failure_type: str | None
    compensation_errors: tuple[str, ...]
    created_at: float
    updated_at: float
    completed_at: float | None
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        key = _identifier(self.graph_set_key, "graph_set_key", 500)
        proposals = _proposal_ids(self.proposal_ids)
        expected_current = _optional_digest(
            self.expected_current_set_id, "expected_current_set_id"
        )
        expected_operation = deterministic_publication_operation_id(
            owner_id=owner,
            graph_set_key=key,
            proposal_ids=proposals,
            expected_current_set_id=expected_current,
        )
        if _digest(self.operation_id, "operation_id") != expected_operation:
            raise ValueError(
                "operation_id does not match immutable publication identity."
            )
        state = _identifier(self.state, "state", 30)
        phase = _identifier(self.phase, "phase", 30)
        if state not in _STATES or phase not in _PHASES:
            raise ValueError("publication state or phase is unsupported.")
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
        previous_id = _optional_digest(
            self.previous_graph_set_id, "previous_graph_set_id"
        )
        previous_digest = _optional_digest(
            self.previous_graph_set_digest, "previous_graph_set_digest"
        )
        candidate_id = _optional_digest(
            self.candidate_graph_set_id, "candidate_graph_set_id"
        )
        candidate_digest = _optional_digest(
            self.candidate_graph_set_digest, "candidate_graph_set_digest"
        )
        if (previous_id is None) != (previous_digest is None):
            raise ValueError(
                "previous graph-set identity must be complete or absent."
            )
        if (candidate_id is None) != (candidate_digest is None):
            raise ValueError(
                "candidate graph-set identity must be complete or absent."
            )
        member_count = None if self.member_count is None else _integer(
            self.member_count, "member_count", 2, 100_000
        )
        edge_count = None if self.edge_count is None else _integer(
            self.edge_count, "edge_count", 1, 500_000
        )
        if (member_count is None) != (edge_count is None):
            raise ValueError("candidate counts must be complete or absent.")
        verification = _optional_digest(
            self.verification_digest, "verification_digest"
        )
        failure = None if self.failure_type is None else _identifier(
            self.failure_type, "failure_type", 200
        )
        compensation_errors = _generic_errors(self.compensation_errors)
        created = _timestamp(self.created_at, "created_at")
        updated = _timestamp(self.updated_at, "updated_at")
        completed = None if self.completed_at is None else _timestamp(
            self.completed_at, "completed_at"
        )
        if updated < created or (completed is not None and completed < created):
            raise ValueError("publication timestamps are out of order.")
        if state == "running":
            if lease_owner is None or lease_expires is None:
                raise ValueError("running attempts require an active lease.")
        elif lease_owner is not None or lease_expires is not None:
            raise ValueError("only running attempts may retain a lease.")
        if phase == "planned":
            if any(
                value is not None
                for value in (
                    candidate_id,
                    candidate_digest,
                    member_count,
                    edge_count,
                    verification,
                )
            ):
                raise ValueError(
                    "planned attempts may not contain candidate identities."
                )
        elif any(
            value is None
            for value in (candidate_id, candidate_digest, member_count, edge_count)
        ):
            raise ValueError(
                "post-plan phases require complete candidate identities."
            )
        if phase in {"verified", "compensated"} and verification is None:
            raise ValueError(
                "verified/compensated attempts require verification_digest."
            )
        if state == "completed":
            if phase != "verified" or completed is None or verification is None:
                raise ValueError(
                    "completed attempts require verified phase and completion metadata."
                )
            if failure is not None or compensation_errors:
                raise ValueError(
                    "completed attempts may not retain failure metadata."
                )
        elif state == "compensated":
            if phase != "compensated" or completed is None or compensation_errors:
                raise ValueError(
                    "compensated attempts require exact successful compensation."
                )
            if failure is None:
                raise ValueError(
                    "compensated attempts require the triggering failure type."
                )
        elif state == "cancelled":
            if completed is None:
                raise ValueError("cancelled attempts require completed_at.")
            if failure is not None or compensation_errors:
                raise ValueError(
                    "cancelled attempts may not retain failure metadata."
                )
        elif completed is not None:
            raise ValueError("only terminal attempts may contain completed_at.")
        if state == "failed" and failure is None:
            raise ValueError("failed attempts require failure_type.")
        if state not in {"failed", "compensated"} and failure is not None:
            raise ValueError(
                "failure_type is only valid for failed/compensated attempts."
            )
        if compensation_errors and state != "failed":
            raise ValueError(
                "compensation_errors are only valid for failed attempts."
            )
        if phase == "verified" and state != "completed":
            raise ValueError("verified phase is terminal completed state only.")
        if phase == "compensated" and state != "compensated":
            raise ValueError("compensated phase is terminal compensated state only.")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("publication schema is unsupported.")
        object.__setattr__(self, "operation_id", expected_operation)
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "graph_set_key", key)
        object.__setattr__(self, "proposal_ids", proposals)
        object.__setattr__(self, "expected_current_set_id", expected_current)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "attempt_count", attempts)
        object.__setattr__(self, "max_attempts", maximum)
        object.__setattr__(self, "lease_owner", lease_owner)
        object.__setattr__(self, "lease_expires_at", lease_expires)
        object.__setattr__(self, "previous_graph_set_id", previous_id)
        object.__setattr__(self, "previous_graph_set_digest", previous_digest)
        object.__setattr__(self, "candidate_graph_set_id", candidate_id)
        object.__setattr__(self, "candidate_graph_set_digest", candidate_digest)
        object.__setattr__(self, "member_count", member_count)
        object.__setattr__(self, "edge_count", edge_count)
        object.__setattr__(self, "verification_digest", verification)
        object.__setattr__(self, "failure_type", failure)
        object.__setattr__(self, "compensation_errors", compensation_errors)
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)
        object.__setattr__(self, "completed_at", completed)

    @classmethod
    def create(
        cls,
        *,
        owner_id: str,
        graph_set_key: str,
        proposal_ids: Iterable[str],
        expected_current_set_id: str | None,
        max_attempts: int = 3,
        now: float | None = None,
    ) -> "EvidenceGraphSetPublicationAttempt":
        timestamp = _timestamp(time.time() if now is None else now, "now")
        proposals = _proposal_ids(proposal_ids)
        operation_id = deterministic_publication_operation_id(
            owner_id=owner_id,
            graph_set_key=graph_set_key,
            proposal_ids=proposals,
            expected_current_set_id=expected_current_set_id,
        )
        return cls(
            operation_id=operation_id,
            owner_id=owner_id,
            graph_set_key=graph_set_key,
            proposal_ids=proposals,
            expected_current_set_id=expected_current_set_id,
            state="planned",
            phase="planned",
            attempt_count=0,
            max_attempts=max_attempts,
            lease_owner=None,
            lease_expires_at=None,
            previous_graph_set_id=None,
            previous_graph_set_digest=None,
            candidate_graph_set_id=None,
            candidate_graph_set_digest=None,
            member_count=None,
            edge_count=None,
            verification_digest=None,
            failure_type=None,
            compensation_errors=(),
            created_at=timestamp,
            updated_at=timestamp,
            completed_at=None,
        )

    @property
    def immutable_digest(self) -> str:
        return _canonical_digest(
            {
                "schema_version": self.schema_version,
                "operation_id": self.operation_id,
                "owner_id": self.owner_id,
                "graph_set_key": self.graph_set_key,
                "proposal_ids": self.proposal_ids,
                "expected_current_set_id": self.expected_current_set_id,
                "max_attempts": self.max_attempts,
            }
        )


__all__ = [
    "EvidenceGraphSetPublicationAttempt",
    "deterministic_publication_operation_id",
]
