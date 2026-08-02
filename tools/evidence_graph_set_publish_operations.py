"""Privacy-safe operational audit and retention planning for publication attempts."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any

from tools.evidence_graph_set_publish_attempts import (
    EvidenceGraphSetPublicationAttempt,
    EvidenceGraphSetPublicationJournal,
)
from tools.security import normalize_owner_id


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


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _classification(
    attempt: EvidenceGraphSetPublicationAttempt, *, now: float
) -> str:
    if attempt.state == "running":
        if attempt.lease_expires_at is not None and attempt.lease_expires_at <= now:
            return (
                "expired_exhausted"
                if attempt.attempt_count >= attempt.max_attempts
                else "expired_reclaimable"
            )
        return "running_active"
    if attempt.state == "failed":
        if attempt.compensation_errors:
            return "compensation_failed"
        return (
            "failed_exhausted"
            if attempt.attempt_count >= attempt.max_attempts
            else "failed_retryable"
        )
    return attempt.state


@dataclass(frozen=True)
class PublicationAuditItem:
    operation_id: str
    graph_set_key: str
    state: str
    phase: str
    classification: str
    attempt_count: int
    max_attempts: int
    lease_expires_at: float | None
    candidate_graph_set_id: str | None
    previous_graph_set_id: str | None
    failure_type: str | None
    compensation_error_count: int
    updated_at: float


@dataclass(frozen=True)
class PublicationAuditReport:
    owner_id: str
    graph_set_key: str | None
    generated_at: float
    total: int
    classification_counts: dict[str, int]
    items: tuple[PublicationAuditItem, ...]
    report_digest: str


@dataclass(frozen=True)
class PublicationRetentionItem:
    operation_id: str
    state: str
    phase: str
    eligible: bool
    reason: str
    completed_at: float | None
    candidate_graph_set_id: str | None
    previous_graph_set_id: str | None


@dataclass(frozen=True)
class PublicationRetentionPlan:
    owner_id: str
    graph_set_key: str | None
    generated_at: float
    cutoff_at: float
    eligible_count: int
    retained_count: int
    items: tuple[PublicationRetentionItem, ...]
    plan_digest: str
    deletion_performed: bool = False


def audit_publication_attempts(
    journal: EvidenceGraphSetPublicationJournal,
    *,
    owner_id: str,
    graph_set_key: str | None = None,
    now: float | None = None,
    limit: int = 10_000,
) -> PublicationAuditReport:
    if not callable(getattr(journal, "list", None)):
        raise ValueError("journal must expose list().")
    owner = normalize_owner_id(owner_id)
    timestamp = _timestamp(time.time() if now is None else now, "now")
    count = _integer(limit, "limit", 1, 10_000)
    attempts = journal.list(
        owner_id=owner,
        graph_set_key=graph_set_key,
        limit=count,
    )
    items: list[PublicationAuditItem] = []
    counts: dict[str, int] = {}
    for attempt in attempts:
        classification = _classification(attempt, now=timestamp)
        counts[classification] = counts.get(classification, 0) + 1
        items.append(
            PublicationAuditItem(
                operation_id=attempt.operation_id,
                graph_set_key=attempt.graph_set_key,
                state=attempt.state,
                phase=attempt.phase,
                classification=classification,
                attempt_count=attempt.attempt_count,
                max_attempts=attempt.max_attempts,
                lease_expires_at=attempt.lease_expires_at,
                candidate_graph_set_id=attempt.candidate_graph_set_id,
                previous_graph_set_id=attempt.previous_graph_set_id,
                failure_type=attempt.failure_type,
                compensation_error_count=len(attempt.compensation_errors),
                updated_at=attempt.updated_at,
            )
        )
    ordered = tuple(
        sorted(items, key=lambda item: (item.classification, item.operation_id))
    )
    ordered_counts = dict(sorted(counts.items()))
    digest = _digest(
        {
            "scope": "rigorousrag-publication-audit-v1",
            "owner_id": owner,
            "graph_set_key": graph_set_key,
            "generated_at": timestamp,
            "classification_counts": ordered_counts,
            "items": [item.__dict__ for item in ordered],
        }
    )
    return PublicationAuditReport(
        owner_id=owner,
        graph_set_key=graph_set_key,
        generated_at=timestamp,
        total=len(ordered),
        classification_counts=ordered_counts,
        items=ordered,
        report_digest=digest,
    )


def plan_publication_retention(
    journal: EvidenceGraphSetPublicationJournal,
    *,
    set_store: Any,
    owner_id: str,
    graph_set_key: str | None = None,
    minimum_age_seconds: int = 30 * 24 * 60 * 60,
    now: float | None = None,
    limit: int = 10_000,
) -> PublicationRetentionPlan:
    if not callable(getattr(journal, "list", None)):
        raise ValueError("journal must expose list().")
    owner = normalize_owner_id(owner_id)
    timestamp = _timestamp(time.time() if now is None else now, "now")
    age = _integer(
        minimum_age_seconds,
        "minimum_age_seconds",
        0,
        10 * 365 * 24 * 60 * 60,
    )
    count = _integer(limit, "limit", 1, 10_000)
    cutoff = timestamp - age
    attempts = journal.list(
        owner_id=owner,
        graph_set_key=graph_set_key,
        limit=count,
    )
    current_by_key: dict[str, str | None] = {}
    items: list[PublicationRetentionItem] = []
    for attempt in attempts:
        if attempt.graph_set_key not in current_by_key:
            current = set_store.current(
                owner_id=owner,
                graph_set_key=attempt.graph_set_key,
            )
            current_by_key[attempt.graph_set_key] = (
                None if current is None else current.graph_set_id
            )
        current_id = current_by_key[attempt.graph_set_key]
        eligible = False
        reason = "nonterminal"
        if attempt.state == "failed":
            reason = "failure_record"
        elif attempt.state not in {"completed", "compensated", "cancelled"}:
            reason = "nonterminal"
        elif attempt.completed_at is None:
            reason = "missing_completion_time"
        elif attempt.completed_at > cutoff:
            reason = "recent_terminal"
        elif (
            current_id
            in {
                attempt.candidate_graph_set_id,
                attempt.previous_graph_set_id,
            }
            and current_id is not None
        ):
            reason = "references_current_pointer"
        elif attempt.compensation_errors:
            reason = "compensation_errors"
        else:
            eligible = True
            reason = "old_terminal_noncurrent"
        items.append(
            PublicationRetentionItem(
                operation_id=attempt.operation_id,
                state=attempt.state,
                phase=attempt.phase,
                eligible=eligible,
                reason=reason,
                completed_at=attempt.completed_at,
                candidate_graph_set_id=attempt.candidate_graph_set_id,
                previous_graph_set_id=attempt.previous_graph_set_id,
            )
        )
    ordered = tuple(
        sorted(items, key=lambda item: (not item.eligible, item.operation_id))
    )
    digest = _digest(
        {
            "scope": "rigorousrag-publication-retention-plan-v1",
            "owner_id": owner,
            "graph_set_key": graph_set_key,
            "generated_at": timestamp,
            "cutoff_at": cutoff,
            "items": [item.__dict__ for item in ordered],
        }
    )
    return PublicationRetentionPlan(
        owner_id=owner,
        graph_set_key=graph_set_key,
        generated_at=timestamp,
        cutoff_at=cutoff,
        eligible_count=sum(item.eligible for item in ordered),
        retained_count=sum(not item.eligible for item in ordered),
        items=ordered,
        plan_digest=digest,
    )


__all__ = [
    "PublicationAuditItem",
    "PublicationAuditReport",
    "PublicationRetentionItem",
    "PublicationRetentionPlan",
    "audit_publication_attempts",
    "plan_publication_retention",
]
