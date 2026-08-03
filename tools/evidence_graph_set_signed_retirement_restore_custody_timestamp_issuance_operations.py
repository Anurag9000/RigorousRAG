"""Read-only custody timestamp issuance audit and retention planning."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _integer,
    _timestamp,
)
from tools.security import normalize_owner_id

_CLASSIFICATIONS = frozenset(
    {
        "planned_ready",
        "running_active",
        "running_expired_reclaimable",
        "running_expired_exhausted",
        "failed_retryable",
        "failed_exhausted",
        "completed",
        "cancelled",
    }
)
_TERMINAL = frozenset({"completed", "cancelled"})


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


def _duration(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(selected) or selected < 0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return selected


def _holds(values: Iterable[str] | None) -> frozenset[str]:
    if values is None:
        return frozenset()
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("held issuance IDs must be an iterable.")
    rendered = {_digest(value, "held_issuance_id") for value in values}
    if len(rendered) > 100_000:
        raise ValueError("held issuance IDs exceed the limit.")
    return frozenset(rendered)


def _classification(value: Any, now: float) -> tuple[str, bool, bool]:
    if value.state == "planned":
        return "planned_ready", False, False
    if value.state == "running":
        active = bool(
            value.lease_expires_at is not None and value.lease_expires_at > now
        )
        if active:
            return "running_active", True, False
        return (
            "running_expired_reclaimable"
            if value.attempt_count < value.max_attempts
            else "running_expired_exhausted",
            False,
            True,
        )
    if value.state == "failed":
        return (
            "failed_retryable"
            if value.attempt_count < value.max_attempts
            else "failed_exhausted",
            False,
            False,
        )
    if value.state == "completed":
        return "completed", False, False
    if value.state == "cancelled":
        return "cancelled", False, False
    raise RuntimeError("timestamp issuance journal returned an unsupported state.")


@dataclass(frozen=True)
class CustodyTimestampIssuanceOperationalItem:
    issuance_id: str
    authority_id: str
    key_id: str
    serial: str
    attestation_digest: str
    output_path_digest: str
    state: str
    phase: str
    attempt_count: int
    max_attempts: int
    lease_owner_present: bool
    lease_expires_at: float | None
    lease_active: bool
    lease_expired: bool
    updated_at: float
    completed_at: float | None
    classification: str
    failure_type: str | None


@dataclass(frozen=True)
class CustodyTimestampIssuanceOperationalReport:
    owner_id: str
    generated_at: float
    item_count: int
    classification_counts: dict[str, int]
    items: tuple[CustodyTimestampIssuanceOperationalItem, ...]
    report_digest: str
    mutation_performed: bool = False
    contains_attestation_signatures: bool = False
    contains_private_key_material: bool = False
    contains_raw_paths: bool = False

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        generated = _timestamp(self.generated_at, "generated_at")
        items = tuple(self.items)
        if self.item_count != len(items):
            raise ValueError("timestamp issuance audit item count differs.")
        identities = tuple(item.issuance_id for item in items)
        if identities != tuple(sorted(identities)) or len(set(identities)) != len(identities):
            raise ValueError("timestamp issuance audit items must be unique and ordered.")
        counts = {name: 0 for name in sorted(_CLASSIFICATIONS)}
        for item in items:
            if item.classification not in _CLASSIFICATIONS:
                raise ValueError("timestamp issuance audit classification is unsupported.")
            counts[item.classification] += 1
        if self.classification_counts != counts:
            raise ValueError("timestamp issuance audit counts differ from items.")
        stable = {
            "scope": "rigorousrag-custody-timestamp-issuance-audit-v1",
            "owner_id": owner,
            "generated_at": generated,
            "item_count": len(items),
            "classification_counts": counts,
            "items": [asdict(item) for item in items],
        }
        if _digest(self.report_digest, "report_digest") != _canonical_digest(stable):
            raise ValueError("report_digest differs from timestamp issuance audit.")
        if any(
            value is not False
            for value in (
                self.mutation_performed,
                self.contains_attestation_signatures,
                self.contains_private_key_material,
                self.contains_raw_paths,
            )
        ):
            raise ValueError("timestamp issuance audit safety flags must be false.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "generated_at", generated)
        object.__setattr__(self, "items", items)


@dataclass(frozen=True)
class CustodyTimestampIssuanceRetentionItem:
    issuance_id: str
    authority_id: str
    key_id: str
    serial: str
    state: str
    phase: str
    completed_at: float
    age_seconds: float
    held: bool
    protected_as_latest: bool
    retention_candidate: bool
    reason: str


@dataclass(frozen=True)
class CustodyTimestampIssuanceRetentionPlan:
    owner_id: str
    generated_at: float
    minimum_age_seconds: float
    retain_latest_per_authority_key: int
    include_completed: bool
    candidate_count: int
    items: tuple[CustodyTimestampIssuanceRetentionItem, ...]
    plan_digest: str
    deletion_performed: bool = False
    contains_attestation_signatures: bool = False
    contains_private_key_material: bool = False
    contains_raw_paths: bool = False

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        generated = _timestamp(self.generated_at, "generated_at")
        minimum_age = _duration(self.minimum_age_seconds, "minimum_age_seconds")
        latest = _integer(
            self.retain_latest_per_authority_key,
            "retain_latest_per_authority_key",
            1,
            100,
        )
        items = tuple(self.items)
        identities = tuple(item.issuance_id for item in items)
        if identities != tuple(sorted(identities)) or len(set(identities)) != len(identities):
            raise ValueError("timestamp issuance retention items must be unique and ordered.")
        candidate_count = sum(item.retention_candidate for item in items)
        if candidate_count != self.candidate_count:
            raise ValueError("timestamp issuance retention candidate count differs.")
        stable = {
            "scope": "rigorousrag-custody-timestamp-issuance-retention-v1",
            "owner_id": owner,
            "generated_at": generated,
            "minimum_age_seconds": minimum_age,
            "retain_latest_per_authority_key": latest,
            "include_completed": self.include_completed,
            "candidate_count": candidate_count,
            "items": [asdict(item) for item in items],
        }
        if _digest(self.plan_digest, "plan_digest") != _canonical_digest(stable):
            raise ValueError("plan_digest differs from timestamp issuance retention plan.")
        if any(
            value is not False
            for value in (
                self.deletion_performed,
                self.contains_attestation_signatures,
                self.contains_private_key_material,
                self.contains_raw_paths,
            )
        ):
            raise ValueError("timestamp issuance retention safety flags must be false.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "generated_at", generated)
        object.__setattr__(self, "minimum_age_seconds", minimum_age)
        object.__setattr__(self, "retain_latest_per_authority_key", latest)
        object.__setattr__(self, "items", items)


def audit_custody_timestamp_issuances(
    *,
    owner_id: str,
    journal: Any,
    now: float | None = None,
    limit: int = 1_000,
) -> CustodyTimestampIssuanceOperationalReport:
    owner = normalize_owner_id(owner_id)
    timestamp = _timestamp(time.time() if now is None else now, "now")
    count = _integer(limit, "limit", 1, 10_000)
    values = tuple(journal.list(owner_id=owner, limit=count))
    if len(values) >= count:
        raise RuntimeError("timestamp issuance audit reached the bounded result limit.")
    seen: set[str] = set()
    rendered: list[CustodyTimestampIssuanceOperationalItem] = []
    counts = {name: 0 for name in sorted(_CLASSIFICATIONS)}
    for value in values:
        issuance_id = _digest(value.issuance_id, "issuance_id")
        if issuance_id in seen:
            raise RuntimeError("timestamp issuance journal returned duplicate IDs.")
        seen.add(issuance_id)
        classification, active, expired = _classification(value, timestamp)
        counts[classification] += 1
        rendered.append(
            CustodyTimestampIssuanceOperationalItem(
                issuance_id=issuance_id,
                authority_id=_identifier(value.authority_id, "authority_id", 200),
                key_id=_identifier(value.key_id, "key_id", 200),
                serial=_digest(value.serial, "serial"),
                attestation_digest=_digest(
                    value.attestation_digest,
                    "attestation_digest",
                ),
                output_path_digest=_digest(
                    value.output_path_digest,
                    "output_path_digest",
                ),
                state=_identifier(value.state, "state", 30),
                phase=_identifier(value.phase, "phase", 40),
                attempt_count=_integer(
                    value.attempt_count,
                    "attempt_count",
                    0,
                    1_000_000,
                ),
                max_attempts=_integer(
                    value.max_attempts,
                    "max_attempts",
                    1,
                    1_000_000,
                ),
                lease_owner_present=value.lease_owner is not None,
                lease_expires_at=(
                    None
                    if value.lease_expires_at is None
                    else _timestamp(value.lease_expires_at, "lease_expires_at")
                ),
                lease_active=active,
                lease_expired=expired,
                updated_at=_timestamp(value.updated_at, "updated_at"),
                completed_at=(
                    None
                    if value.completed_at is None
                    else _timestamp(value.completed_at, "completed_at")
                ),
                classification=classification,
                failure_type=value.failure_type,
            )
        )
    items = tuple(sorted(rendered, key=lambda item: item.issuance_id))
    stable = {
        "scope": "rigorousrag-custody-timestamp-issuance-audit-v1",
        "owner_id": owner,
        "generated_at": timestamp,
        "item_count": len(items),
        "classification_counts": counts,
        "items": [asdict(item) for item in items],
    }
    return CustodyTimestampIssuanceOperationalReport(
        owner_id=owner,
        generated_at=timestamp,
        item_count=len(items),
        classification_counts=counts,
        items=items,
        report_digest=_canonical_digest(stable),
    )


def plan_custody_timestamp_issuance_retention(
    *,
    owner_id: str,
    journal: Any,
    now: float | None = None,
    minimum_age_seconds: float = 180 * 24 * 60 * 60,
    retain_latest_per_authority_key: int = 1,
    include_completed: bool = False,
    held_issuance_ids: Iterable[str] | None = None,
    limit: int = 10_000,
) -> CustodyTimestampIssuanceRetentionPlan:
    owner = normalize_owner_id(owner_id)
    timestamp = _timestamp(time.time() if now is None else now, "now")
    minimum_age = _duration(minimum_age_seconds, "minimum_age_seconds")
    latest_count = _integer(
        retain_latest_per_authority_key,
        "retain_latest_per_authority_key",
        1,
        100,
    )
    if not isinstance(include_completed, bool):
        raise ValueError("include_completed must be boolean.")
    held = _holds(held_issuance_ids)
    count = _integer(limit, "limit", 1, 10_000)
    values = tuple(journal.list(owner_id=owner, limit=count))
    if len(values) >= count:
        raise RuntimeError("timestamp issuance retention reached the bounded result limit.")
    terminal_by_key: dict[tuple[str, str], list[Any]] = {}
    for value in values:
        if value.state in _TERMINAL:
            terminal_by_key.setdefault((value.authority_id, value.key_id), []).append(value)
    protected: set[str] = set()
    for key_values in terminal_by_key.values():
        ordered = sorted(
            key_values,
            key=lambda value: (
                -float(value.completed_at or value.updated_at),
                value.issuance_id,
            ),
        )
        protected.update(value.issuance_id for value in ordered[:latest_count])
    rendered: list[CustodyTimestampIssuanceRetentionItem] = []
    for value in values:
        if value.state not in _TERMINAL:
            continue
        completed_at = value.completed_at or value.updated_at
        completed = _timestamp(completed_at, "completed_at")
        age = max(0.0, timestamp - completed)
        is_held = value.issuance_id in held
        is_latest = value.issuance_id in protected
        eligible_state = value.state == "cancelled" or include_completed
        candidate = bool(
            age >= minimum_age
            and not is_held
            and not is_latest
            and eligible_state
        )
        if is_held:
            reason = "legal_hold"
        elif is_latest:
            reason = "latest_terminal_for_authority_key"
        elif age < minimum_age:
            reason = "younger_than_minimum_age"
        elif value.state == "completed" and not include_completed:
            reason = "completed_issuances_retained_by_default"
        elif candidate:
            reason = "old_terminal_issuance_candidate"
        else:
            reason = "not_retention_candidate"
        rendered.append(
            CustodyTimestampIssuanceRetentionItem(
                issuance_id=value.issuance_id,
                authority_id=value.authority_id,
                key_id=value.key_id,
                serial=value.serial,
                state=value.state,
                phase=value.phase,
                completed_at=completed,
                age_seconds=age,
                held=is_held,
                protected_as_latest=is_latest,
                retention_candidate=candidate,
                reason=reason,
            )
        )
    items = tuple(sorted(rendered, key=lambda item: item.issuance_id))
    candidate_count = sum(item.retention_candidate for item in items)
    stable = {
        "scope": "rigorousrag-custody-timestamp-issuance-retention-v1",
        "owner_id": owner,
        "generated_at": timestamp,
        "minimum_age_seconds": minimum_age,
        "retain_latest_per_authority_key": latest_count,
        "include_completed": include_completed,
        "candidate_count": candidate_count,
        "items": [asdict(item) for item in items],
    }
    return CustodyTimestampIssuanceRetentionPlan(
        owner_id=owner,
        generated_at=timestamp,
        minimum_age_seconds=minimum_age,
        retain_latest_per_authority_key=latest_count,
        include_completed=include_completed,
        candidate_count=candidate_count,
        items=items,
        plan_digest=_canonical_digest(stable),
    )


__all__ = [
    "CustodyTimestampIssuanceOperationalItem",
    "CustodyTimestampIssuanceOperationalReport",
    "CustodyTimestampIssuanceRetentionItem",
    "CustodyTimestampIssuanceRetentionPlan",
    "audit_custody_timestamp_issuances",
    "plan_custody_timestamp_issuance_retention",
]
