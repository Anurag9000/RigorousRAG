"""Privacy-safe operator audit correlation and export contracts.

Operational audits need enough identity to correlate lifecycle operations, distributed
jobs, generations and traces without exporting source paths, query text, document text,
model prompts, credentials or arbitrary exception strings.  This module uses closed
schemas, typed reason codes and deterministic pseudonymous correlation identifiers.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

_MAX_EVENTS = 1_000_000


def _identifier(value: Any, label: str, maximum: int = 2_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if not result or len(result) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{label} is invalid")
    return result


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _sha256(value: Any, label: str) -> str:
    digest = _identifier(value, label, 64).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class AuditEventKind(str, Enum):
    LIFECYCLE_PLANNED = "lifecycle_planned"
    INDEX_COMMITTED = "index_committed"
    REGISTRY_COMMITTED = "registry_committed"
    CLEANUP_COMMITTED = "cleanup_committed"
    JOB_CLAIMED = "job_claimed"
    JOB_RENEWED = "job_renewed"
    JOB_RETRIED = "job_retried"
    JOB_DEAD_LETTERED = "job_dead_lettered"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    RECONCILIATION = "reconciliation"
    REINDEX = "reindex"
    ADOPTION = "adoption"
    MIGRATION = "migration"
    RETENTION = "retention"
    SECURITY = "security"


class AuditOutcome(str, Enum):
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RETRYABLE = "retryable"
    REJECTED = "rejected"
    NOOP = "noop"


@dataclass(frozen=True)
class InternalAuditEvent:
    event_id: str
    event_kind: AuditEventKind
    outcome: AuditOutcome
    occurred_at: datetime
    tenant_id: str
    owner_id: str
    operation_id: str | None = None
    job_id: str | None = None
    document_id: str | None = None
    generation_id: str | None = None
    trace_id: str | None = None
    reason_code: str = "unspecified"
    component: str = "unknown"
    source_commit: str | None = None
    public_metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("event_id", "tenant_id", "owner_id", "reason_code", "component"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if not isinstance(self.event_kind, AuditEventKind):
            object.__setattr__(self, "event_kind", AuditEventKind(self.event_kind))
        if not isinstance(self.outcome, AuditOutcome):
            object.__setattr__(self, "outcome", AuditOutcome(self.outcome))
        object.__setattr__(self, "occurred_at", _utc(self.occurred_at, "occurred_at"))
        for name in ("operation_id", "job_id", "document_id", "generation_id", "trace_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _identifier(value, name))
        if self.source_commit is not None:
            commit = _identifier(self.source_commit, "source_commit", 64).lower()
            if len(commit) not in {40, 64} or any(ch not in "0123456789abcdef" for ch in commit):
                raise ValueError("source_commit must be a Git SHA-1/SHA-256 object id")
            object.__setattr__(self, "source_commit", commit)
        if not isinstance(self.public_metadata, Mapping) or len(self.public_metadata) > 1_000:
            raise ValueError("public_metadata must be a bounded mapping")
        object.__setattr__(
            self,
            "public_metadata",
            {
                _identifier(key, "public metadata key", 300): _identifier(value, "public metadata value", 10_000)
                for key, value in self.public_metadata.items()
            },
        )


@dataclass(frozen=True)
class AuditExportPolicy:
    policy_id: str
    pseudonymization_key_id: str
    include_generation_pseudonym: bool = True
    include_trace_pseudonym: bool = False
    allowed_metadata_keys: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for name in ("policy_id", "pseudonymization_key_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if not isinstance(self.include_generation_pseudonym, bool) or not isinstance(self.include_trace_pseudonym, bool):
            raise ValueError("policy flags must be boolean")
        object.__setattr__(
            self,
            "allowed_metadata_keys",
            frozenset(_identifier(value, "allowed metadata key", 300) for value in self.allowed_metadata_keys),
        )

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))


@dataclass(frozen=True)
class OperatorAuditEvent:
    event_id: str
    event_kind: AuditEventKind
    outcome: AuditOutcome
    occurred_at: datetime
    tenant_pseudonym: str
    owner_pseudonym: str
    operation_pseudonym: str | None
    job_pseudonym: str | None
    document_pseudonym: str | None
    generation_pseudonym: str | None
    trace_pseudonym: str | None
    reason_code: str
    component: str
    source_commit: str | None
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class AuditExportBundle:
    policy_digest: str
    events: tuple[OperatorAuditEvent, ...]
    first_occurred_at: datetime | None
    last_occurred_at: datetime | None
    event_count: int
    bundle_digest: str


def _pseudonym(secret: bytes, namespace: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(secret, bytes) or len(secret) < 32:
        raise ValueError("pseudonymization secret must contain at least 256 bits")
    payload = f"{namespace}\0{value}".encode("utf-8")
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def sanitize_event(
    event: InternalAuditEvent,
    *,
    policy: AuditExportPolicy,
    pseudonymization_secret: bytes,
) -> OperatorAuditEvent:
    if not isinstance(event, InternalAuditEvent) or not isinstance(policy, AuditExportPolicy):
        raise ValueError("event/policy types are invalid")
    metadata = {
        key: value for key, value in event.public_metadata.items() if key in policy.allowed_metadata_keys
    }
    return OperatorAuditEvent(
        event_id=event.event_id,
        event_kind=event.event_kind,
        outcome=event.outcome,
        occurred_at=event.occurred_at,
        tenant_pseudonym=_pseudonym(pseudonymization_secret, "tenant", event.tenant_id) or "",
        owner_pseudonym=_pseudonym(pseudonymization_secret, "owner", event.owner_id) or "",
        operation_pseudonym=_pseudonym(pseudonymization_secret, "operation", event.operation_id),
        job_pseudonym=_pseudonym(pseudonymization_secret, "job", event.job_id),
        document_pseudonym=_pseudonym(pseudonymization_secret, "document", event.document_id),
        generation_pseudonym=(
            _pseudonym(pseudonymization_secret, "generation", event.generation_id)
            if policy.include_generation_pseudonym
            else None
        ),
        trace_pseudonym=(
            _pseudonym(pseudonymization_secret, "trace", event.trace_id)
            if policy.include_trace_pseudonym
            else None
        ),
        reason_code=event.reason_code,
        component=event.component,
        source_commit=event.source_commit,
        metadata=metadata,
    )


def export_operator_audit(
    events: Sequence[InternalAuditEvent],
    *,
    policy: AuditExportPolicy,
    pseudonymization_secret: bytes,
) -> AuditExportBundle:
    if len(events) > _MAX_EVENTS or any(not isinstance(event, InternalAuditEvent) for event in events):
        raise ValueError("events must be bounded InternalAuditEvent values")
    sanitized = tuple(
        sorted(
            (
                sanitize_event(event, policy=policy, pseudonymization_secret=pseudonymization_secret)
                for event in events
            ),
            key=lambda event: (event.occurred_at, event.event_id),
        )
    )
    first = sanitized[0].occurred_at if sanitized else None
    last = sanitized[-1].occurred_at if sanitized else None
    digest_input = {
        "policy_digest": policy.digest,
        "events": [asdict(event) for event in sanitized],
        "event_count": len(sanitized),
    }
    bundle_digest = canonical_digest(digest_input)
    return AuditExportBundle(policy.digest, sanitized, first, last, len(sanitized), bundle_digest)


__all__ = [
    "AuditEventKind",
    "AuditExportBundle",
    "AuditExportPolicy",
    "AuditOutcome",
    "InternalAuditEvent",
    "OperatorAuditEvent",
    "canonical_digest",
    "export_operator_audit",
    "sanitize_event",
]
