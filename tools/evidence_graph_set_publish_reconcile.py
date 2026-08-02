"""Crash-recoverable reviewed graph-set publication over a durable phase journal."""

from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable

from tools.evidence_graph_authority import resolve_evidence_graph
from tools.evidence_graph_relation_review import approved_relations
from tools.evidence_graph_set_pointer import clear_current_graph_set_pointer
from tools.evidence_graph_set_publish_attempts import (
    EvidenceGraphSetPublicationAttempt,
    EvidenceGraphSetPublicationJournal,
)
from tools.evidence_graph_set_store import assess_graph_set_authority
from tools.evidence_graph_sets import build_evidence_graph_set
from tools.index_coordinator import _document_lock


class EvidenceGraphSetPublicationRecoveryError(RuntimeError):
    """Raised when publication fails; durable state contains the recovery outcome."""

    def __init__(
        self,
        message: str,
        *,
        operation_id: str,
        state: str,
        phase: str,
        compensation_errors: tuple[str, ...] = (),
    ) -> None:
        self.operation_id = operation_id
        self.state = state
        self.phase = phase
        self.compensation_errors = tuple(compensation_errors)
        suffix = (
            " Compensation errors: " + ", ".join(self.compensation_errors)
            if self.compensation_errors
            else ""
        )
        super().__init__(message + suffix)


@dataclass(frozen=True)
class EvidenceGraphSetPublicationExecution:
    operation_id: str
    state: str
    phase: str
    graph_set_key: str
    candidate_graph_set_id: str | None
    candidate_graph_set_digest: str | None
    previous_graph_set_id: str | None
    member_count: int | None
    edge_count: int | None
    verification_digest: str | None
    attempt_count: int
    pointer_current_set_id: str | None
    graph_set_mutation_performed: bool
    authoritative_mutation_performed: bool = False


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


def _digest_payload(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _pointer_id(
    store: Any, attempt: EvidenceGraphSetPublicationAttempt
) -> str | None:
    current = store.current(
        owner_id=attempt.owner_id,
        graph_set_key=attempt.graph_set_key,
    )
    return None if current is None else current.graph_set_id


def _generic_failure(exc: BaseException) -> str:
    name = type(exc).__name__
    return name if len(name) <= 200 else "PublicationFailure"


def _hook(
    callback: Callable[[str, EvidenceGraphSetPublicationAttempt], None] | None,
    name: str,
    attempt: EvidenceGraphSetPublicationAttempt,
) -> None:
    if callback is not None:
        callback(name, attempt)


def _proposals_and_docs(
    attempt: EvidenceGraphSetPublicationAttempt, ledger: Any
) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    proposals = tuple(ledger.get_proposal(value) for value in attempt.proposal_ids)
    docs: set[str] = set()
    for proposal in proposals:
        if (
            proposal.owner_id != attempt.owner_id
            or proposal.graph_set_key != attempt.graph_set_key
        ):
            raise RuntimeError(
                "publication proposal escaped immutable operation scope."
            )
        docs.add(proposal.source.doc_id)
        docs.add(proposal.target.doc_id)
    if len(docs) < 2:
        raise RuntimeError("publication requires at least two distinct documents.")
    return proposals, tuple(sorted(docs))


def _load_candidate(
    attempt: EvidenceGraphSetPublicationAttempt, set_store: Any
) -> Any:
    if (
        attempt.candidate_graph_set_id is None
        or attempt.candidate_graph_set_digest is None
    ):
        raise RuntimeError("publication candidate identity is unavailable.")
    candidate = set_store.get(
        owner_id=attempt.owner_id,
        graph_set_id=attempt.candidate_graph_set_id,
    )
    if (
        candidate.graph_set_digest != attempt.candidate_graph_set_digest
        or candidate.graph_set_key != attempt.graph_set_key
        or candidate.owner_id != attempt.owner_id
        or len(candidate.members) != attempt.member_count
        or len(candidate.edges) != attempt.edge_count
    ):
        raise RuntimeError("stored publication candidate identity is corrupt.")
    return candidate


def _load_previous(
    attempt: EvidenceGraphSetPublicationAttempt, set_store: Any
) -> Any | None:
    if attempt.previous_graph_set_id is None:
        if attempt.previous_graph_set_digest is not None:
            raise RuntimeError("previous graph-set identity is incomplete.")
        return None
    previous = set_store.get(
        owner_id=attempt.owner_id,
        graph_set_id=attempt.previous_graph_set_id,
    )
    if previous.graph_set_digest != attempt.previous_graph_set_digest:
        raise RuntimeError("stored previous graph-set identity is corrupt.")
    if previous.graph_set_key != attempt.graph_set_key:
        raise RuntimeError("previous graph set escaped operation key scope.")
    return previous


def _outcome_digest(
    attempt: EvidenceGraphSetPublicationAttempt,
    *,
    outcome: str,
    current_set_id: str | None,
    authority_digest: str | None,
    failure_type: str | None,
) -> str:
    return _digest_payload(
        {
            "scope": "rigorousrag-evidence-graph-set-publication-outcome-v1",
            "operation_id": attempt.operation_id,
            "outcome": outcome,
            "expected_current_set_id": attempt.expected_current_set_id,
            "previous_graph_set_id": attempt.previous_graph_set_id,
            "candidate_graph_set_id": attempt.candidate_graph_set_id,
            "current_set_id": current_set_id,
            "authority_digest": authority_digest,
            "failure_type": failure_type,
        }
    )


def _compensate(
    attempt: EvidenceGraphSetPublicationAttempt,
    *,
    set_store: Any,
    now: float,
) -> tuple[tuple[str, ...], str | None]:
    candidate_id = attempt.candidate_graph_set_id
    if candidate_id is None:
        return ("candidate:missing",), _pointer_id(set_store, attempt)
    errors: list[str] = []
    try:
        actual_id = _pointer_id(set_store, attempt)
        if actual_id == candidate_id:
            if attempt.previous_graph_set_id is None:
                cleared = clear_current_graph_set_pointer(
                    set_store,
                    owner_id=attempt.owner_id,
                    graph_set_key=attempt.graph_set_key,
                    expected_current_set_id=candidate_id,
                )
                if not cleared:
                    errors.append("pointer:missing")
            elif attempt.previous_graph_set_id != candidate_id:
                previous = _load_previous(attempt, set_store)
                set_store.commit(
                    previous,
                    make_current=True,
                    expected_current_set_id=candidate_id,
                    now=now,
                )
        elif actual_id != attempt.previous_graph_set_id:
            errors.append("pointer:external_change")
    except Exception as exc:
        errors.append(f"pointer:{_generic_failure(exc)}")
    try:
        actual_id = _pointer_id(set_store, attempt)
        if actual_id != attempt.previous_graph_set_id:
            errors.append("pointer:verification")
    except Exception as exc:
        actual_id = None
        errors.append(f"verification:{_generic_failure(exc)}")
    return tuple(errors), actual_id


def _execution(
    attempt: EvidenceGraphSetPublicationAttempt,
    *,
    set_store: Any,
    mutated: bool,
) -> EvidenceGraphSetPublicationExecution:
    try:
        pointer = _pointer_id(set_store, attempt)
    except Exception:
        pointer = None
    return EvidenceGraphSetPublicationExecution(
        operation_id=attempt.operation_id,
        state=attempt.state,
        phase=attempt.phase,
        graph_set_key=attempt.graph_set_key,
        candidate_graph_set_id=attempt.candidate_graph_set_id,
        candidate_graph_set_digest=attempt.candidate_graph_set_digest,
        previous_graph_set_id=attempt.previous_graph_set_id,
        member_count=attempt.member_count,
        edge_count=attempt.edge_count,
        verification_digest=attempt.verification_digest,
        attempt_count=attempt.attempt_count,
        pointer_current_set_id=pointer,
        graph_set_mutation_performed=mutated,
    )


def _prepare_candidate(
    attempt: EvidenceGraphSetPublicationAttempt,
    *,
    worker_id: str,
    journal: EvidenceGraphSetPublicationJournal,
    ledger: Any,
    set_store: Any,
    generations: Any,
    graphs: Any,
    now: float,
    phase_hook: Callable[[str, EvidenceGraphSetPublicationAttempt], None] | None,
) -> EvidenceGraphSetPublicationAttempt:
    _, doc_ids = _proposals_and_docs(attempt, ledger)
    previous = set_store.current(
        owner_id=attempt.owner_id,
        graph_set_key=attempt.graph_set_key,
    )
    previous_id = None if previous is None else previous.graph_set_id
    if previous_id != attempt.expected_current_set_id:
        raise RuntimeError(
            "graph set current pointer differs from immutable expectation."
        )
    views = tuple(
        resolve_evidence_graph(
            owner_id=attempt.owner_id,
            doc_id=doc_id,
            graphs=graphs,
            generations=generations,
        )
        for doc_id in doc_ids
    )
    relations = approved_relations(
        owner_id=attempt.owner_id,
        graph_set_key=attempt.graph_set_key,
        proposal_ids=attempt.proposal_ids,
        authority_views=views,
        ledger=ledger,
    )
    candidate = build_evidence_graph_set(
        owner_id=attempt.owner_id,
        graph_set_key=attempt.graph_set_key,
        authority_views=views,
        relations=relations,
        now=now,
    )
    set_store.commit(candidate, make_current=False, now=now)
    _hook(phase_hook, "candidate_committed", attempt)
    recorded = journal.record_candidate(
        attempt.operation_id,
        worker_id=worker_id,
        previous_graph_set_id=previous_id,
        previous_graph_set_digest=(
            None if previous is None else previous.graph_set_digest
        ),
        candidate_graph_set_id=candidate.graph_set_id,
        candidate_graph_set_digest=candidate.graph_set_digest,
        member_count=len(candidate.members),
        edge_count=len(candidate.edges),
        now=now,
    )
    _hook(phase_hook, "candidate_recorded", recorded)
    return recorded


def _activate_candidate(
    attempt: EvidenceGraphSetPublicationAttempt,
    *,
    worker_id: str,
    journal: EvidenceGraphSetPublicationJournal,
    set_store: Any,
    generations: Any,
    graphs: Any,
    now: float,
    phase_hook: Callable[[str, EvidenceGraphSetPublicationAttempt], None] | None,
) -> EvidenceGraphSetPublicationAttempt:
    candidate = _load_candidate(attempt, set_store)
    actual_id = _pointer_id(set_store, attempt)
    if actual_id == candidate.graph_set_id:
        recorded = journal.record_pointer_activated(
            attempt.operation_id, worker_id=worker_id, now=now
        )
        _hook(phase_hook, "pointer_recorded", recorded)
        return recorded
    if actual_id != attempt.previous_graph_set_id:
        raise RuntimeError(
            "graph set current pointer changed outside this operation."
        )
    before = assess_graph_set_authority(
        candidate, generations=generations, graphs=graphs
    )
    if not before.authoritative_current:
        raise RuntimeError("candidate members changed before pointer activation.")
    set_store.commit(
        candidate,
        make_current=True,
        expected_current_set_id=attempt.previous_graph_set_id,
        now=now,
    )
    _hook(phase_hook, "pointer_committed", attempt)
    recorded = journal.record_pointer_activated(
        attempt.operation_id, worker_id=worker_id, now=now
    )
    _hook(phase_hook, "pointer_recorded", recorded)
    return recorded


def _verify_recovered_activation(
    attempt: EvidenceGraphSetPublicationAttempt,
    *,
    worker_id: str,
    journal: EvidenceGraphSetPublicationJournal,
    set_store: Any,
    generations: Any,
    graphs: Any,
    now: float,
    phase_hook: Callable[[str, EvidenceGraphSetPublicationAttempt], None] | None,
) -> EvidenceGraphSetPublicationAttempt:
    candidate = _load_candidate(attempt, set_store)
    actual_id = _pointer_id(set_store, attempt)
    if (
        actual_id == attempt.previous_graph_set_id
        and actual_id != candidate.graph_set_id
    ):
        verification = _outcome_digest(
            attempt,
            outcome="compensated_before_journal",
            current_set_id=actual_id,
            authority_digest=None,
            failure_type="RecoveredCompensation",
        )
        return journal.mark_compensated(
            attempt.operation_id,
            worker_id=worker_id,
            verification_digest=verification,
            failure_type="RecoveredCompensation",
            now=now,
        )
    if actual_id != candidate.graph_set_id:
        raise RuntimeError(
            "publication pointer is neither candidate nor expected previous set."
        )
    _hook(phase_hook, "before_final_verification", attempt)
    report = assess_graph_set_authority(
        candidate, generations=generations, graphs=graphs
    )
    if not report.authoritative_current:
        errors, current_id = _compensate(
            attempt, set_store=set_store, now=now
        )
        _hook(phase_hook, "pointer_compensated", attempt)
        verification = _outcome_digest(
            attempt,
            outcome="compensated_stale_candidate",
            current_set_id=current_id,
            authority_digest=report.authority_digest,
            failure_type="StaleCandidate",
        )
        if errors:
            return journal.fail(
                attempt.operation_id,
                worker_id=worker_id,
                failure_type="CompensationFailure",
                compensation_errors=errors,
                now=now,
            )
        return journal.mark_compensated(
            attempt.operation_id,
            worker_id=worker_id,
            verification_digest=verification,
            failure_type="StaleCandidate",
            now=now,
        )
    completed = journal.complete(
        attempt.operation_id,
        worker_id=worker_id,
        verification_digest=report.authority_digest,
        now=now,
    )
    _hook(phase_hook, "completed", completed)
    return completed


def _handle_exception(
    attempt: EvidenceGraphSetPublicationAttempt,
    exc: Exception,
    *,
    worker_id: str,
    journal: EvidenceGraphSetPublicationJournal,
    set_store: Any,
    now: float,
    phase_hook: Callable[[str, EvidenceGraphSetPublicationAttempt], None] | None,
) -> EvidenceGraphSetPublicationAttempt:
    failure = _generic_failure(exc)
    current = journal.get(attempt.operation_id)
    if current.state != "running":
        return current
    try:
        actual_id = _pointer_id(set_store, current)
    except Exception:
        actual_id = None
    activated = bool(
        current.candidate_graph_set_id is not None
        and actual_id == current.candidate_graph_set_id
        and current.candidate_graph_set_id != current.previous_graph_set_id
    )
    if current.phase == "pointer_activated" or activated:
        if activated and current.phase == "candidate_stored":
            try:
                current = journal.record_pointer_activated(
                    current.operation_id, worker_id=worker_id, now=now
                )
            except (KeyError, RuntimeError):
                # The exact pointer state remains the source of truth. Compensation
                # still runs even when the lease expired before phase persistence.
                current = journal.get(current.operation_id)
        errors, current_id = _compensate(
            current, set_store=set_store, now=now
        )
        _hook(phase_hook, "pointer_compensated", current)
        verification = _outcome_digest(
            current,
            outcome=(
                "compensated_exception" if not errors else "compensation_failed"
            ),
            current_set_id=current_id,
            authority_digest=None,
            failure_type=failure,
        )
        try:
            if not errors:
                return journal.mark_compensated(
                    current.operation_id,
                    worker_id=worker_id,
                    verification_digest=verification,
                    failure_type=failure,
                    now=now,
                )
            return journal.fail(
                current.operation_id,
                worker_id=worker_id,
                failure_type=failure,
                compensation_errors=errors,
                now=now,
            )
        except (KeyError, RuntimeError):
            # The lease may have expired while compensation was being completed.
            # Leave the durable phase for the next claimant to reconcile.
            return journal.get(current.operation_id)
    try:
        return journal.fail(
            current.operation_id,
            worker_id=worker_id,
            failure_type=failure,
            now=now,
        )
    except (KeyError, RuntimeError):
        return journal.get(current.operation_id)


def execute_publication_attempt(
    operation_id: str,
    *,
    worker_id: str,
    lease_seconds: int,
    journal: EvidenceGraphSetPublicationJournal,
    ledger: Any,
    set_store: Any,
    generations: Any,
    graphs: Any,
    now: float | None = None,
    _phase_hook: Callable[
        [str, EvidenceGraphSetPublicationAttempt], None
    ]
    | None = None,
) -> EvidenceGraphSetPublicationExecution:
    """Execute or recover one attempt without mutating authoritative stores."""

    def clock() -> float:
        return _timestamp(time.time() if now is None else now, "now")

    existing = journal.get(operation_id)
    if existing.state in {"completed", "compensated", "cancelled"}:
        return _execution(existing, set_store=set_store, mutated=False)
    claimed = journal.claim(
        operation_id,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        now=clock(),
    )
    _, doc_ids = _proposals_and_docs(claimed, ledger)
    mutated = False
    try:
        with ExitStack() as stack:
            for doc_id in doc_ids:
                stack.enter_context(_document_lock(claimed.owner_id, doc_id))
            while True:
                phase_now = clock()
                current = journal.renew(
                    claimed.operation_id,
                    worker_id=worker_id,
                    lease_seconds=lease_seconds,
                    now=phase_now,
                )
                if current.state != "running":
                    return _execution(
                        current, set_store=set_store, mutated=mutated
                    )
                if current.phase == "planned":
                    current = _prepare_candidate(
                        current,
                        worker_id=worker_id,
                        journal=journal,
                        ledger=ledger,
                        set_store=set_store,
                        generations=generations,
                        graphs=graphs,
                        now=phase_now,
                        phase_hook=_phase_hook,
                    )
                    mutated = True
                    continue
                if current.phase == "candidate_stored":
                    before_id = _pointer_id(set_store, current)
                    current = _activate_candidate(
                        current,
                        worker_id=worker_id,
                        journal=journal,
                        set_store=set_store,
                        generations=generations,
                        graphs=graphs,
                        now=phase_now,
                        phase_hook=_phase_hook,
                    )
                    mutated = (
                        mutated
                        or _pointer_id(set_store, current) != before_id
                    )
                    continue
                if current.phase == "pointer_activated":
                    current = _verify_recovered_activation(
                        current,
                        worker_id=worker_id,
                        journal=journal,
                        set_store=set_store,
                        generations=generations,
                        graphs=graphs,
                        now=phase_now,
                        phase_hook=_phase_hook,
                    )
                    return _execution(
                        current, set_store=set_store, mutated=True
                    )
                raise RuntimeError(
                    "publication attempt entered an unsupported recoverable phase."
                )
    except Exception as exc:
        outcome = _handle_exception(
            claimed,
            exc,
            worker_id=worker_id,
            journal=journal,
            set_store=set_store,
            now=clock(),
            phase_hook=_phase_hook,
        )
        raise EvidenceGraphSetPublicationRecoveryError(
            f"graph-set publication attempt failed ({_generic_failure(exc)}).",
            operation_id=outcome.operation_id,
            state=outcome.state,
            phase=outcome.phase,
            compensation_errors=outcome.compensation_errors,
        ) from exc


def execute_next_publication_attempt(
    *,
    owner_id: str,
    worker_id: str,
    lease_seconds: int,
    journal: EvidenceGraphSetPublicationJournal,
    ledger: Any,
    set_store: Any,
    generations: Any,
    graphs: Any,
    now: float | None = None,
) -> EvidenceGraphSetPublicationExecution | None:
    timestamp = _timestamp(time.time() if now is None else now, "now")
    operation_id = journal.next_claimable_id(
        owner_id=owner_id, now=timestamp
    )
    if operation_id is None:
        return None
    return execute_publication_attempt(
        operation_id,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        journal=journal,
        ledger=ledger,
        set_store=set_store,
        generations=generations,
        graphs=graphs,
        now=timestamp,
    )


__all__ = [
    "EvidenceGraphSetPublicationExecution",
    "EvidenceGraphSetPublicationRecoveryError",
    "execute_next_publication_attempt",
    "execute_publication_attempt",
]
