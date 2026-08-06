"""Read-only reconciliation and exact recovery for graph compaction receipts.

The compaction store intentionally separates a durable ``planned`` intent from a
``completed`` receipt. This module classifies every durable record against the
job journal, authoritative generation pointer, and historical graph store without
mutating any of them. A second exact-confirmation operation may complete only
those intents whose destructive action is already observably complete (or whose
action is audit-only).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from tools.evidence_graph_compaction import (
    EvidenceGraphCompactionRecord,
    EvidenceGraphCompactionStore,
    _digest,
    _identifier,
    _integer,
    _job_matches_current,
    _timestamp,
)
from tools.evidence_graph_jobs import EvidenceGraphJob
from tools.security import normalize_owner_id

_MAX_LIMIT = 10_000
_RECOVERABLE = frozenset(
    {
        "completion_pending_after_delete",
        "audit_only_completion_pending",
    }
)
_CONFLICTS = frozenset(
    {
        "authoritative_current_conflict",
        "current_graph_conflict",
        "journal_identity_conflict",
        "journal_missing_conflict",
        "graph_digest_conflict",
        "completed_graph_present",
        "record_corrupt",
    }
)


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


def _confirmed_ids(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("confirm_job_ids must be an iterable of digests.")
    return tuple(sorted({_digest(value, "confirm_job_id") for value in values}))


@dataclass(frozen=True)
class EvidenceGraphCompactionFinding:
    job_id: str
    owner_id: str
    doc_id: str
    source_sequence: int
    phase: str
    action: str
    status: str
    recoverable: bool
    conflict: bool
    graph_present: bool | None
    journal_present: bool
    observed_graph_digest: str | None = None
    expected_graph_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _digest(self.job_id, "job_id"))
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "doc_id", _identifier(self.doc_id, "doc_id"))
        object.__setattr__(
            self,
            "source_sequence",
            _integer(self.source_sequence, "source_sequence", 1, 2**63 - 1),
        )
        if self.phase not in {"planned", "completed"}:
            raise ValueError("finding phase is unsupported.")
        if self.action not in {"delete_graph_generation", "retain_job_audit_only"}:
            raise ValueError("finding action is unsupported.")
        object.__setattr__(self, "status", _identifier(self.status, "status", 80))
        if not isinstance(self.recoverable, bool) or not isinstance(self.conflict, bool):
            raise ValueError("finding flags must be booleans.")
        if self.recoverable != (self.status in _RECOVERABLE):
            raise ValueError("recoverable flag does not match finding status.")
        if self.conflict != (self.status in _CONFLICTS):
            raise ValueError("conflict flag does not match finding status.")
        if self.graph_present not in {True, False, None}:
            raise ValueError("graph_present must be true, false, or null.")
        if not isinstance(self.journal_present, bool):
            raise ValueError("journal_present must be a boolean.")
        if self.observed_graph_digest is not None:
            object.__setattr__(
                self,
                "observed_graph_digest",
                _digest(self.observed_graph_digest, "observed_graph_digest"),
            )
        if self.expected_graph_digest is not None:
            object.__setattr__(
                self,
                "expected_graph_digest",
                _digest(self.expected_graph_digest, "expected_graph_digest"),
            )


@dataclass(frozen=True)
class EvidenceGraphCompactionReconciliationReport:
    owner_id: str
    findings: tuple[EvidenceGraphCompactionFinding, ...]
    generated_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        if not isinstance(self.findings, tuple):
            raise ValueError("findings must be an immutable tuple.")
        ordered = tuple(sorted(self.findings, key=lambda value: value.job_id))
        if ordered != self.findings or len({value.job_id for value in ordered}) != len(ordered):
            raise ValueError("findings must be unique and sorted by job_id.")
        if any(value.owner_id != self.owner_id for value in ordered):
            raise ValueError("every finding must belong to the report owner.")
        object.__setattr__(self, "generated_at", _timestamp(self.generated_at, "generated_at"))

    @property
    def recoverable_job_ids(self) -> tuple[str, ...]:
        return tuple(value.job_id for value in self.findings if value.recoverable)

    @property
    def conflict_job_ids(self) -> tuple[str, ...]:
        return tuple(value.job_id for value in self.findings if value.conflict)

    @property
    def healthy(self) -> bool:
        return not self.conflict_job_ids

    @property
    def recoverable_count(self) -> int:
        return len(self.recoverable_job_ids)

    @property
    def conflict_count(self) -> int:
        return len(self.conflict_job_ids)

    @property
    def status_counts(self) -> tuple[tuple[str, int], ...]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.status] = counts.get(finding.status, 0) + 1
        return tuple(sorted(counts.items()))

    @property
    def report_digest(self) -> str:
        value = asdict(self)
        value.pop("generated_at", None)
        return _canonical_digest(value)


@dataclass(frozen=True)
class EvidenceGraphCompactionRecoveryResult:
    owner_id: str
    report_digest: str
    completed_job_ids: tuple[str, ...]
    already_completed_job_ids: tuple[str, ...]
    completed_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "report_digest", _digest(self.report_digest, "report_digest"))
        for name in ("completed_job_ids", "already_completed_job_ids"):
            values = tuple(_digest(value, name) for value in getattr(self, name))
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be unique and sorted.")
            object.__setattr__(self, name, values)
        if set(self.completed_job_ids) & set(self.already_completed_job_ids):
            raise ValueError("recovery result job sets must not overlap.")
        object.__setattr__(self, "completed_at", _timestamp(self.completed_at, "completed_at"))


def _journal_identity(record: EvidenceGraphCompactionRecord, job: EvidenceGraphJob) -> bool:
    return bool(
        job.job_id == record.job_id
        and job.owner_id == record.owner_id
        and job.doc_id == record.doc_id
        and job.source_sequence == record.source_sequence
        and job.state == record.job_state
        and job.graph_digest == record.graph_digest
    )


def _finding(
    record: EvidenceGraphCompactionRecord,
    *,
    status: str,
    graph_present: bool | None,
    journal_present: bool,
    observed_graph_digest: str | None = None,
) -> EvidenceGraphCompactionFinding:
    return EvidenceGraphCompactionFinding(
        job_id=record.job_id,
        owner_id=record.owner_id,
        doc_id=record.doc_id,
        source_sequence=record.source_sequence,
        phase=record.phase,
        action=record.action,
        status=status,
        recoverable=status in _RECOVERABLE,
        conflict=status in _CONFLICTS,
        graph_present=graph_present,
        journal_present=journal_present,
        observed_graph_digest=observed_graph_digest,
        expected_graph_digest=record.graph_digest,
    )


def reconcile_evidence_graph_compactions(
    *,
    owner_id: str,
    compactions: EvidenceGraphCompactionStore,
    journal: Any,
    generations: Any,
    graphs: Any,
    limit: int = 100,
    now: float | None = None,
) -> EvidenceGraphCompactionReconciliationReport:
    """Classify durable compaction records without changing any store."""

    owner = normalize_owner_id(owner_id)
    if not isinstance(compactions, EvidenceGraphCompactionStore):
        raise ValueError("compactions must be EvidenceGraphCompactionStore.")
    count = _integer(limit, "limit", 1, _MAX_LIMIT)
    timestamp = _timestamp(time.time() if now is None else now, "now")
    records = compactions.list(owner_id=owner, limit=count)
    authority_cache: dict[str, Any | None] = {}
    current_graph_cache: dict[str, Any | None] = {}
    findings: list[EvidenceGraphCompactionFinding] = []

    for record in records:
        try:
            job = journal.get(record.job_id)
        except Exception:
            job = None
        journal_present = job is not None
        if job is None:
            findings.append(
                _finding(
                    record,
                    status="journal_missing_conflict",
                    graph_present=None,
                    journal_present=False,
                )
            )
            continue
        if not isinstance(job, EvidenceGraphJob) or not _journal_identity(record, job):
            findings.append(
                _finding(
                    record,
                    status="journal_identity_conflict",
                    graph_present=None,
                    journal_present=True,
                )
            )
            continue

        if record.doc_id not in authority_cache:
            authority_cache[record.doc_id] = generations.current(
                owner_id=owner,
                doc_id=record.doc_id,
            )
        if _job_matches_current(job, authority_cache[record.doc_id]):
            findings.append(
                _finding(
                    record,
                    status="authoritative_current_conflict",
                    graph_present=None,
                    journal_present=True,
                )
            )
            continue

        if record.action == "retain_job_audit_only":
            status = (
                "audit_only_completion_pending"
                if record.phase == "planned"
                else "audit_only_consistent"
            )
            findings.append(
                _finding(
                    record,
                    status=status,
                    graph_present=None,
                    journal_present=journal_present,
                )
            )
            continue

        if record.doc_id not in current_graph_cache:
            current_graph_cache[record.doc_id] = graphs.current(
                owner_id=owner,
                doc_id=record.doc_id,
            )
        current_graph = current_graph_cache[record.doc_id]
        if (
            current_graph is not None
            and getattr(current_graph, "generation", None) == record.source_sequence
        ):
            findings.append(
                _finding(
                    record,
                    status="current_graph_conflict",
                    graph_present=True,
                    journal_present=journal_present,
                    observed_graph_digest=getattr(current_graph, "graph_digest", None),
                )
            )
            continue

        try:
            historical = graphs.get(
                owner_id=owner,
                doc_id=record.doc_id,
                generation=record.source_sequence,
            )
        except KeyError:
            historical = None
        except Exception:
            findings.append(
                _finding(
                    record,
                    status="record_corrupt",
                    graph_present=None,
                    journal_present=journal_present,
                )
            )
            continue

        if historical is None:
            status = (
                "completion_pending_after_delete"
                if record.phase == "planned"
                else "completed_consistent"
            )
            findings.append(
                _finding(
                    record,
                    status=status,
                    graph_present=False,
                    journal_present=journal_present,
                )
            )
            continue

        observed = getattr(historical, "graph_digest", None)
        try:
            observed_digest = _digest(observed, "observed_graph_digest")
        except ValueError:
            findings.append(
                _finding(
                    record,
                    status="record_corrupt",
                    graph_present=True,
                    journal_present=journal_present,
                )
            )
            continue
        if observed_digest != record.graph_digest:
            findings.append(
                _finding(
                    record,
                    status="graph_digest_conflict",
                    graph_present=True,
                    journal_present=journal_present,
                    observed_graph_digest=observed_digest,
                )
            )
            continue
        status = "deletion_pending" if record.phase == "planned" else "completed_graph_present"
        findings.append(
            _finding(
                record,
                status=status,
                graph_present=True,
                journal_present=journal_present,
                observed_graph_digest=observed_digest,
            )
        )

    return EvidenceGraphCompactionReconciliationReport(
        owner_id=owner,
        findings=tuple(sorted(findings, key=lambda value: value.job_id)),
        generated_at=timestamp,
    )


def recover_reconciled_compaction_receipts(
    *,
    report: EvidenceGraphCompactionReconciliationReport,
    compactions: EvidenceGraphCompactionStore,
    journal: Any,
    generations: Any,
    graphs: Any,
    confirm_report_digest: str,
    confirm_job_ids: Iterable[str],
    now: float | None = None,
) -> EvidenceGraphCompactionRecoveryResult:
    """Complete only revalidated interrupted receipts with exact confirmation."""

    if not isinstance(report, EvidenceGraphCompactionReconciliationReport):
        raise ValueError("report must be EvidenceGraphCompactionReconciliationReport.")
    selected_digest = _digest(confirm_report_digest, "confirm_report_digest")
    if selected_digest != report.report_digest:
        raise ValueError("confirmation must exactly match report_digest.")
    confirmed = _confirmed_ids(confirm_job_ids)
    expected = report.recoverable_job_ids
    if confirmed != expected:
        raise ValueError("job confirmations must exactly match every recoverable finding.")
    timestamp = _timestamp(time.time() if now is None else now, "now")
    if timestamp < report.generated_at:
        raise ValueError("recovery time may not precede report generation.")

    current = reconcile_evidence_graph_compactions(
        owner_id=report.owner_id,
        compactions=compactions,
        journal=journal,
        generations=generations,
        graphs=graphs,
        limit=max(1, len(report.findings)),
        now=timestamp,
    )
    current_by_id = {value.job_id: value for value in current.findings}
    completed: list[str] = []
    already: list[str] = []
    for job_id in confirmed:
        finding = current_by_id.get(job_id)
        if finding is None:
            raise RuntimeError("recoverable compaction finding disappeared.")
        if finding.status in {"completed_consistent", "audit_only_consistent"}:
            already.append(job_id)
            continue
        if finding.status not in _RECOVERABLE:
            raise RuntimeError("recoverable compaction state changed after confirmation.")
        record = compactions.get(job_id)
        if record is None:
            raise RuntimeError("compaction intent disappeared before recovery.")
        compactions.complete(
            job_id,
            owner_id=report.owner_id,
            plan_digest=record.plan_digest,
            now=timestamp,
        )
        completed.append(job_id)

    return EvidenceGraphCompactionRecoveryResult(
        owner_id=report.owner_id,
        report_digest=report.report_digest,
        completed_job_ids=tuple(sorted(completed)),
        already_completed_job_ids=tuple(sorted(already)),
        completed_at=timestamp,
    )


__all__ = [
    "EvidenceGraphCompactionFinding",
    "EvidenceGraphCompactionReconciliationReport",
    "EvidenceGraphCompactionRecoveryResult",
    "reconcile_evidence_graph_compactions",
    "recover_reconciled_compaction_receipts",
]
