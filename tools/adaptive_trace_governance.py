"""Governed retention and privacy-safe export for adaptive retrieval traces.

This module deliberately reuses :class:`AdaptiveTraceStore` instead of creating a
parallel lifecycle database. Exports contain only already-sanitized trace metadata,
query hashes, pseudonymous identifiers, and bounded aggregate controls.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import operator
from dataclasses import dataclass
from typing import Any

from tools.adaptive_trace_store import (
    AdaptiveTraceAggregate,
    AdaptiveTraceAttempt,
    AdaptiveTraceStore,
)
from tools.security import normalize_owner_id

_MAX_RETAIN = 1_000_000
_MAX_EXPORT = 1_000
_MIN_SECRET_BYTES = 16


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        parsed = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return parsed


def _secret(value: str | bytes) -> bytes:
    if isinstance(value, str):
        encoded = value.encode("utf-8")
    elif isinstance(value, bytes):
        encoded = value
    else:
        raise ValueError("export_secret must be bytes or a string.")
    if not _MIN_SECRET_BYTES <= len(encoded) <= 4_096:
        raise ValueError("export_secret must contain 16-4096 bytes.")
    return encoded


def _pseudonym(secret: bytes, kind: str, value: str) -> str:
    return hmac.new(secret, f"rigorousrag:{kind}:{value}".encode("utf-8"), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class TraceRetentionPolicy:
    retain_latest: int = 10_000
    export_limit: int = 250
    include_attempts: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "retain_latest", _integer(self.retain_latest, "retain_latest", 0, _MAX_RETAIN))
        object.__setattr__(self, "export_limit", _integer(self.export_limit, "export_limit", 1, _MAX_EXPORT))
        if not isinstance(self.include_attempts, bool):
            raise ValueError("include_attempts must be boolean.")


@dataclass(frozen=True)
class PrivacySafeTraceAttempt:
    sequence: int
    mode: str
    top_k: int
    candidate_pool: int
    use_multi_query: bool
    use_hyde: bool
    reranker: str
    returned_evidence: int
    accumulated_evidence: int
    decision: str
    sufficiency: float
    error_type: str | None

    @classmethod
    def from_attempt(cls, attempt: AdaptiveTraceAttempt) -> "PrivacySafeTraceAttempt":
        return cls(
            sequence=attempt.sequence,
            mode=attempt.mode,
            top_k=attempt.top_k,
            candidate_pool=attempt.candidate_pool,
            use_multi_query=attempt.use_multi_query,
            use_hyde=attempt.use_hyde,
            reranker=attempt.reranker,
            returned_evidence=attempt.returned_evidence,
            accumulated_evidence=attempt.accumulated_evidence,
            decision=attempt.decision,
            sufficiency=attempt.sufficiency,
            error_type=attempt.error_type,
        )


@dataclass(frozen=True)
class PrivacySafeTraceExport:
    owner_pseudonym: str
    run_pseudonym: str
    query_sha256: str
    started_at: float
    completed_at: float
    duration_ms: float
    attempt_count: int
    evidence_count: int
    final_decision: str
    final_sufficiency: float
    abstain: bool
    exhausted: bool
    estimated_cost: int
    error_count: int
    attempts: tuple[PrivacySafeTraceAttempt, ...] = ()


@dataclass(frozen=True)
class TraceGovernanceReport:
    deleted_runs: int
    retained_runs: int
    exportable_runs: int
    aggregate: AdaptiveTraceAggregate


def apply_trace_retention(
    store: AdaptiveTraceStore,
    *,
    owner_id: str,
    policy: TraceRetentionPolicy | None = None,
) -> TraceGovernanceReport:
    """Apply count-bounded retention using the trace store's atomic owner prune."""

    if not isinstance(store, AdaptiveTraceStore):
        raise ValueError("store must be AdaptiveTraceStore.")
    selected = policy or TraceRetentionPolicy()
    if not isinstance(selected, TraceRetentionPolicy):
        raise ValueError("policy must be TraceRetentionPolicy.")
    owner = normalize_owner_id(owner_id)
    deleted = store.prune_owner(owner_id=owner, retain_latest=selected.retain_latest)
    retained = len(store.list_runs(owner_id=owner, limit=min(max(selected.retain_latest, 1), _MAX_EXPORT))) if selected.retain_latest else 0
    aggregate = store.aggregate(owner_id=owner, limit=min(max(selected.retain_latest, 1), _MAX_EXPORT)) if selected.retain_latest else AdaptiveTraceAggregate(0, 0, 0, 0, 0.0, 0.0, (), ())
    return TraceGovernanceReport(
        deleted_runs=deleted,
        retained_runs=retained,
        exportable_runs=min(retained, selected.export_limit),
        aggregate=aggregate,
    )


def export_privacy_safe_traces(
    store: AdaptiveTraceStore,
    *,
    owner_id: str,
    export_secret: str | bytes,
    policy: TraceRetentionPolicy | None = None,
) -> tuple[PrivacySafeTraceExport, ...]:
    """Export sanitized trace diagnostics without raw owner IDs, queries, or evidence."""

    if not isinstance(store, AdaptiveTraceStore):
        raise ValueError("store must be AdaptiveTraceStore.")
    selected = policy or TraceRetentionPolicy()
    if not isinstance(selected, TraceRetentionPolicy):
        raise ValueError("policy must be TraceRetentionPolicy.")
    owner = normalize_owner_id(owner_id)
    secret = _secret(export_secret)
    owner_token = _pseudonym(secret, "owner", owner)
    summaries = store.list_runs(owner_id=owner, limit=selected.export_limit)
    result: list[PrivacySafeTraceExport] = []
    for summary in summaries:
        attempts: tuple[PrivacySafeTraceAttempt, ...] = ()
        if selected.include_attempts:
            record = store.get_run(owner_id=owner, run_id=summary.run_id)
            if record is None:
                continue
            attempts = tuple(PrivacySafeTraceAttempt.from_attempt(item) for item in record.attempts)
        duration = max(0.0, (summary.completed_at - summary.started_at) * 1_000.0)
        if not math.isfinite(duration):
            raise RuntimeError("adaptive trace duration is corrupt.")
        result.append(
            PrivacySafeTraceExport(
                owner_pseudonym=owner_token,
                run_pseudonym=_pseudonym(secret, "run", summary.run_id),
                query_sha256=summary.query_sha256,
                started_at=summary.started_at,
                completed_at=summary.completed_at,
                duration_ms=round(duration, 6),
                attempt_count=summary.attempt_count,
                evidence_count=summary.evidence_count,
                final_decision=summary.final_decision,
                final_sufficiency=summary.final_sufficiency,
                abstain=summary.abstain,
                exhausted=summary.exhausted,
                estimated_cost=summary.estimated_cost,
                error_count=summary.error_count,
                attempts=attempts,
            )
        )
    return tuple(result)


__all__ = [
    "PrivacySafeTraceAttempt",
    "PrivacySafeTraceExport",
    "TraceGovernanceReport",
    "TraceRetentionPolicy",
    "apply_trace_retention",
    "export_privacy_safe_traces",
]
