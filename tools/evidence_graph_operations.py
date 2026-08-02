"""Privacy-safe operational audit and retention planning for graph jobs."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from tools.evidence_graph_jobs import EvidenceGraphJob
from tools.security import normalize_owner_id

_STATES = ("planned", "running", "completed", "failed", "cancelled")


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


def _same_source(job: EvidenceGraphJob, generation: Any | None) -> bool:
    return bool(
        generation is not None
        and getattr(generation, "owner_id", None) == job.owner_id
        and getattr(generation, "doc_id", None) == job.doc_id
        and getattr(generation, "sequence", None) == job.source_sequence
        and getattr(generation, "state", None) == job.source_state
        and getattr(generation, "content_sha256", None) == job.content_sha256
        and getattr(generation, "profile_fingerprint", None)
        == job.profile_fingerprint
        and getattr(generation, "sparse_generation", None)
        == job.sparse_generation
    )


def _sorted_unique(values: list[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


@dataclass(frozen=True)
class EvidenceGraphOperationalReport:
    owner_id: str
    scanned_count: int
    state_counts: Mapping[str, int]
    expired_running_job_ids: tuple[str, ...]
    retryable_failed_job_ids: tuple[str, ...]
    dead_letter_job_ids: tuple[str, ...]
    superseded_nonterminal_job_ids: tuple[str, ...]
    current_completed_job_ids: tuple[str, ...]
    stale_completed_job_ids: tuple[str, ...]
    missing_or_mismatched_graph_job_ids: tuple[str, ...]
    generated_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(
            self,
            "scanned_count",
            _integer(self.scanned_count, "scanned_count", 0, 10_000),
        )
        if not isinstance(self.state_counts, Mapping) or set(self.state_counts) != set(_STATES):
            raise ValueError("state_counts must contain every graph job state.")
        clean_counts = {
            state: _integer(self.state_counts[state], f"state_counts.{state}", 0, 10_000)
            for state in _STATES
        }
        if sum(clean_counts.values()) != self.scanned_count:
            raise ValueError("state_counts must sum to scanned_count.")
        object.__setattr__(self, "state_counts", clean_counts)
        for name in (
            "expired_running_job_ids",
            "retryable_failed_job_ids",
            "dead_letter_job_ids",
            "superseded_nonterminal_job_ids",
            "current_completed_job_ids",
            "stale_completed_job_ids",
            "missing_or_mismatched_graph_job_ids",
        ):
            value = tuple(getattr(self, name))
            if value != tuple(sorted(set(value))):
                raise ValueError(f"{name} must be unique and sorted.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "generated_at", _timestamp(self.generated_at, "generated_at"))

    @property
    def report_digest(self) -> str:
        value = asdict(self)
        value.pop("generated_at", None)
        return _canonical_digest(value)


@dataclass(frozen=True)
class EvidenceGraphRetentionCandidate:
    job_id: str
    state: str
    source_sequence: int
    age_seconds: float
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.state not in {"completed", "cancelled"}:
            raise ValueError("retention candidates must be terminal completed/cancelled jobs.")
        object.__setattr__(
            self,
            "source_sequence",
            _integer(self.source_sequence, "source_sequence", 1, 2**63 - 1),
        )
        object.__setattr__(self, "age_seconds", _timestamp(self.age_seconds, "age_seconds"))
        reasons = tuple(sorted(set(self.reason_codes)))
        if not reasons:
            raise ValueError("retention candidates require reason codes.")
        object.__setattr__(self, "reason_codes", reasons)


@dataclass(frozen=True)
class EvidenceGraphRetentionPlan:
    owner_id: str
    min_age_seconds: float
    scanned_count: int
    candidates: tuple[EvidenceGraphRetentionCandidate, ...]
    retained_current_or_recent_job_ids: tuple[str, ...]
    retained_failed_or_running_job_ids: tuple[str, ...]
    retained_missing_or_mismatched_graph_job_ids: tuple[str, ...]
    generated_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(
            self, "min_age_seconds", _timestamp(self.min_age_seconds, "min_age_seconds")
        )
        object.__setattr__(
            self, "scanned_count", _integer(self.scanned_count, "scanned_count", 0, 10_000)
        )
        values = tuple(sorted(self.candidates, key=lambda item: item.job_id))
        if len({item.job_id for item in values}) != len(values):
            raise ValueError("retention candidates must have unique job IDs.")
        object.__setattr__(self, "candidates", values)
        for name in (
            "retained_current_or_recent_job_ids",
            "retained_failed_or_running_job_ids",
            "retained_missing_or_mismatched_graph_job_ids",
        ):
            value = tuple(getattr(self, name))
            if value != tuple(sorted(set(value))):
                raise ValueError(f"{name} must be unique and sorted.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "generated_at", _timestamp(self.generated_at, "generated_at"))

    @property
    def plan_digest(self) -> str:
        value = asdict(self)
        value.pop("generated_at", None)
        return _canonical_digest(value)


def audit_evidence_graph_jobs(
    *,
    owner_id: str,
    journal: Any,
    generations: Any,
    graphs: Any,
    limit: int = 10_000,
    now: float | None = None,
) -> EvidenceGraphOperationalReport:
    owner = normalize_owner_id(owner_id)
    count = _integer(limit, "limit", 1, 10_000)
    timestamp = _timestamp(time.time() if now is None else now, "now")
    jobs = tuple(journal.list(owner_id=owner, limit=count))
    state_counts = {state: 0 for state in _STATES}
    expired: list[str] = []
    retryable: list[str] = []
    dead: list[str] = []
    superseded: list[str] = []
    current_completed: list[str] = []
    stale_completed: list[str] = []
    mismatched: list[str] = []

    authority_cache: dict[str, Any | None] = {}
    graph_current_cache: dict[str, Any | None] = {}
    for job in jobs:
        if not isinstance(job, EvidenceGraphJob) or job.owner_id != owner:
            raise RuntimeError("graph job journal escaped owner scope.")
        state_counts[job.state] += 1
        authoritative = authority_cache.setdefault(
            job.doc_id,
            generations.current(owner_id=owner, doc_id=job.doc_id),
        )
        if job.state == "running" and job.lease_expires_at is not None and job.lease_expires_at <= timestamp:
            expired.append(job.job_id)
        if job.state == "failed":
            (retryable if job.attempt_count < job.max_attempts else dead).append(job.job_id)
        if job.state in {"planned", "running", "failed"} and not _same_source(job, authoritative):
            superseded.append(job.job_id)
        if job.state != "completed":
            continue
        try:
            historical = graphs.get(
                owner_id=owner,
                doc_id=job.doc_id,
                generation=job.source_sequence,
            )
        except (KeyError, RuntimeError):
            historical = None
        if historical is None or getattr(historical, "graph_digest", None) != job.graph_digest:
            mismatched.append(job.job_id)
            continue
        current_graph = graph_current_cache.setdefault(
            job.doc_id,
            graphs.current(owner_id=owner, doc_id=job.doc_id),
        )
        if (
            _same_source(job, authoritative)
            and current_graph is not None
            and getattr(current_graph, "generation", None) == job.source_sequence
            and getattr(current_graph, "graph_digest", None) == job.graph_digest
        ):
            current_completed.append(job.job_id)
        else:
            stale_completed.append(job.job_id)

    return EvidenceGraphOperationalReport(
        owner_id=owner,
        scanned_count=len(jobs),
        state_counts=state_counts,
        expired_running_job_ids=_sorted_unique(expired),
        retryable_failed_job_ids=_sorted_unique(retryable),
        dead_letter_job_ids=_sorted_unique(dead),
        superseded_nonterminal_job_ids=_sorted_unique(superseded),
        current_completed_job_ids=_sorted_unique(current_completed),
        stale_completed_job_ids=_sorted_unique(stale_completed),
        missing_or_mismatched_graph_job_ids=_sorted_unique(mismatched),
        generated_at=timestamp,
    )


def plan_evidence_graph_job_retention(
    *,
    owner_id: str,
    journal: Any,
    generations: Any,
    graphs: Any,
    min_age_seconds: float,
    limit: int = 10_000,
    now: float | None = None,
) -> EvidenceGraphRetentionPlan:
    owner = normalize_owner_id(owner_id)
    count = _integer(limit, "limit", 1, 10_000)
    minimum_age = _timestamp(min_age_seconds, "min_age_seconds")
    timestamp = _timestamp(time.time() if now is None else now, "now")
    jobs = tuple(journal.list(owner_id=owner, limit=count))
    candidates: list[EvidenceGraphRetentionCandidate] = []
    keep_current: list[str] = []
    keep_active: list[str] = []
    keep_corrupt: list[str] = []
    authority_cache: dict[str, Any | None] = {}
    graph_current_cache: dict[str, Any | None] = {}

    for job in jobs:
        if not isinstance(job, EvidenceGraphJob) or job.owner_id != owner:
            raise RuntimeError("graph job journal escaped owner scope.")
        age = max(0.0, timestamp - job.updated_at)
        if job.state not in {"completed", "cancelled"}:
            keep_active.append(job.job_id)
            continue
        authoritative = authority_cache.setdefault(
            job.doc_id,
            generations.current(owner_id=owner, doc_id=job.doc_id),
        )
        current_graph = graph_current_cache.setdefault(
            job.doc_id,
            graphs.current(owner_id=owner, doc_id=job.doc_id),
        )
        current_identity = _same_source(job, authoritative) or bool(
            current_graph is not None
            and getattr(current_graph, "generation", None) == job.source_sequence
        )
        if age < minimum_age or current_identity:
            keep_current.append(job.job_id)
            continue
        if job.state == "completed":
            try:
                historical = graphs.get(
                    owner_id=owner,
                    doc_id=job.doc_id,
                    generation=job.source_sequence,
                )
            except (KeyError, RuntimeError):
                historical = None
            if historical is None or getattr(historical, "graph_digest", None) != job.graph_digest:
                keep_corrupt.append(job.job_id)
                continue
        reasons = ["terminal", "older_than_minimum", "not_authoritative_current"]
        if job.state == "completed":
            reasons.append("historical_graph_digest_verified")
        else:
            reasons.append("cancelled_without_publication")
        candidates.append(
            EvidenceGraphRetentionCandidate(
                job_id=job.job_id,
                state=job.state,
                source_sequence=job.source_sequence,
                age_seconds=age,
                reason_codes=tuple(reasons),
            )
        )

    return EvidenceGraphRetentionPlan(
        owner_id=owner,
        min_age_seconds=minimum_age,
        scanned_count=len(jobs),
        candidates=tuple(candidates),
        retained_current_or_recent_job_ids=_sorted_unique(keep_current),
        retained_failed_or_running_job_ids=_sorted_unique(keep_active),
        retained_missing_or_mismatched_graph_job_ids=_sorted_unique(keep_corrupt),
        generated_at=timestamp,
    )


__all__ = [
    "EvidenceGraphOperationalReport",
    "EvidenceGraphRetentionCandidate",
    "EvidenceGraphRetentionPlan",
    "audit_evidence_graph_jobs",
    "plan_evidence_graph_job_retention",
]
