"""Crash-recoverable retirement of expired authorization-only publication duplicates."""

from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Callable

from tools.evidence_graph_set_signed_retirement import (
    SignedPublicationRetirementPreflight,
    preflight_expired_signed_publication_duplicate_retirement,
)
from tools.evidence_graph_set_signed_retirement_contracts import (
    SignedPublicationRetirementAttempt,
    _digest,
    _integer,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_journal import (
    SignedPublicationRetirementJournal,
)
from tools.evidence_graph_set_signed_retirement_mutation import (
    claim_or_renew_authorization_publication_retirement_lease,
    retire_claimed_authorization_publication_attempt,
)
from tools.evidence_graph_set_store import assess_graph_set_authority
from tools.index_coordinator import _document_lock


class SignedPublicationRetirementRecoveryError(RuntimeError):
    """Raised when durable state records a recoverable retirement failure."""

    def __init__(
        self,
        message: str,
        *,
        retirement_id: str,
        state: str,
        phase: str,
    ) -> None:
        self.retirement_id = retirement_id
        self.state = state
        self.phase = phase
        super().__init__(message)


@dataclass(frozen=True)
class SignedPublicationRetirementExecution:
    retirement_id: str
    publication_operation_id: str
    state: str
    phase: str
    graph_set_key: str
    signed_candidate_set_id: str
    authorization_candidate_set_id: str | None
    final_pointer_set_id: str | None
    verification_digest: str | None
    attempt_count: int
    pointer_mutation_performed: bool
    authorization_journal_mutation_performed: bool
    authoritative_store_mutation_performed: bool = False
    source_text_returned: bool = False


def _clock(value: float | None) -> float:
    if value is not None:
        return _timestamp(value, "now")
    return _timestamp(time.time(), "now")


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


def _generic_failure(exc: BaseException) -> str:
    name = type(exc).__name__
    return name if len(name) <= 200 else "RetirementFailure"


def _pointer_id(set_store: Any, attempt: SignedPublicationRetirementAttempt) -> str | None:
    current = set_store.current(
        owner_id=attempt.owner_id,
        graph_set_key=attempt.graph_set_key,
    )
    return None if current is None else _digest(
        current.graph_set_id, "current_pointer_set_id"
    )


def _publication_attempts(
    attempt: SignedPublicationRetirementAttempt,
    *,
    authorization_journal: Any,
    signed_journal: Any,
) -> tuple[Any, Any]:
    common = authorization_journal.get(attempt.publication_operation_id)
    signed = signed_journal.get(attempt.publication_operation_id)
    if (
        common.owner_id != attempt.owner_id
        or signed.owner_id != attempt.owner_id
        or common.operation_id != attempt.publication_operation_id
        or signed.operation_id != attempt.publication_operation_id
        or common.graph_set_key != attempt.graph_set_key
        or signed.graph_set_key != attempt.graph_set_key
        or tuple(common.proposal_ids) != tuple(signed.proposal_ids)
        or common.expected_current_set_id != signed.expected_current_set_id
        or common.candidate_graph_set_id != attempt.authorization_candidate_set_id
        or signed.candidate_graph_set_id != attempt.signed_candidate_set_id
        or signed.candidate_graph_set_digest != attempt.signed_candidate_set_digest
    ):
        raise RuntimeError("publication attempts differ from immutable retirement scope.")
    if signed.state != "completed" or signed.phase != "verified":
        raise RuntimeError("signed publication attempt is not completed and verified.")
    return common, signed


def _signed_candidate(
    attempt: SignedPublicationRetirementAttempt,
    *,
    set_store: Any,
) -> Any:
    candidate = set_store.get(
        owner_id=attempt.owner_id,
        graph_set_id=attempt.signed_candidate_set_id,
    )
    if (
        candidate.owner_id != attempt.owner_id
        or candidate.graph_set_key != attempt.graph_set_key
        or candidate.graph_set_id != attempt.signed_candidate_set_id
        or candidate.graph_set_digest != attempt.signed_candidate_set_digest
    ):
        raise RuntimeError("stored signed candidate identity is corrupt.")
    return candidate


def _authority(
    attempt: SignedPublicationRetirementAttempt,
    candidate: Any,
    *,
    generations: Any,
    graphs: Any,
) -> Any:
    report = assess_graph_set_authority(
        candidate,
        generations=generations,
        graphs=graphs,
    )
    if not report.authoritative_current:
        raise RuntimeError("signed publication candidate is no longer authoritative.")
    if report.authority_digest != attempt.signed_authority_digest:
        raise RuntimeError("signed publication authority digest changed.")
    return report


def _execution(
    attempt: SignedPublicationRetirementAttempt,
    *,
    pointer_mutated: bool,
    authorization_mutated: bool,
) -> SignedPublicationRetirementExecution:
    return SignedPublicationRetirementExecution(
        retirement_id=attempt.retirement_id,
        publication_operation_id=attempt.publication_operation_id,
        state=attempt.state,
        phase=attempt.phase,
        graph_set_key=attempt.graph_set_key,
        signed_candidate_set_id=attempt.signed_candidate_set_id,
        authorization_candidate_set_id=attempt.authorization_candidate_set_id,
        final_pointer_set_id=attempt.final_pointer_set_id,
        verification_digest=attempt.verification_digest,
        attempt_count=attempt.attempt_count,
        pointer_mutation_performed=pointer_mutated,
        authorization_journal_mutation_performed=authorization_mutated,
    )


def _hook(
    callback: Callable[[str, SignedPublicationRetirementAttempt], None] | None,
    name: str,
    attempt: SignedPublicationRetirementAttempt,
) -> None:
    if callback is not None:
        callback(name, attempt)


def seed_signed_publication_retirement(
    *,
    owner_id: str,
    publication_operation_id: str,
    authorization_journal: Any,
    signed_journal: Any,
    retirement_journal: SignedPublicationRetirementJournal,
    set_store: Any,
    generations: Any,
    graphs: Any,
    max_attempts: int = 3,
    now: float | None = None,
) -> tuple[SignedPublicationRetirementAttempt, SignedPublicationRetirementPreflight]:
    timestamp = _clock(now)
    maximum = _integer(max_attempts, "max_attempts", 1, 1_000_000)
    preflight = preflight_expired_signed_publication_duplicate_retirement(
        owner_id=owner_id,
        operation_id=publication_operation_id,
        authorization_journal=authorization_journal,
        signed_journal=signed_journal,
        set_store=set_store,
        generations=generations,
        graphs=graphs,
        now=timestamp,
    )
    if not preflight.eligible:
        raise RuntimeError(
            f"retirement preflight is not eligible ({preflight.disposition})."
        )
    signed = signed_journal.get(preflight.operation_id)
    if signed.candidate_graph_set_digest is None or preflight.signed_authority_digest is None:
        raise RuntimeError("signed retirement identity is incomplete.")
    attempt = SignedPublicationRetirementAttempt.create(
        owner_id=preflight.owner_id,
        publication_operation_id=preflight.operation_id,
        graph_set_key=preflight.graph_set_key,
        signed_candidate_set_id=preflight.signed_candidate_set_id,
        signed_candidate_set_digest=signed.candidate_graph_set_digest,
        authorization_candidate_set_id=preflight.authorization_candidate_set_id,
        signed_authority_digest=preflight.signed_authority_digest,
        max_attempts=maximum,
        now=timestamp,
    )
    return retirement_journal.seed(attempt), preflight


def execute_signed_publication_retirement(
    retirement_id: str,
    *,
    worker_id: str,
    lease_seconds: int,
    retirement_journal: SignedPublicationRetirementJournal,
    authorization_journal: Any,
    signed_journal: Any,
    set_store: Any,
    generations: Any,
    graphs: Any,
    now: float | None = None,
    _phase_hook: Callable[
        [str, SignedPublicationRetirementAttempt], None
    ]
    | None = None,
) -> SignedPublicationRetirementExecution:
    """Execute or recover one retirement without restoring weaker graph state."""

    duration = _integer(lease_seconds, "lease_seconds", 1, 86_400)

    def clock() -> float:
        return _clock(now)

    existing = retirement_journal.get(retirement_id)
    if existing.state in {"completed", "cancelled"}:
        return _execution(
            existing,
            pointer_mutated=False,
            authorization_mutated=False,
        )
    claimed = retirement_journal.claim(
        retirement_id,
        worker_id=worker_id,
        lease_seconds=duration,
        now=clock(),
    )
    candidate = _signed_candidate(claimed, set_store=set_store)
    doc_ids = tuple(sorted({member.doc_id for member in candidate.members}))
    if not doc_ids:
        raise RuntimeError("signed candidate has no member documents.")
    pointer_mutated = False
    authorization_mutated = False
    try:
        with ExitStack() as stack:
            for doc_id in doc_ids:
                stack.enter_context(_document_lock(claimed.owner_id, doc_id))
            while True:
                phase_now = clock()
                current = retirement_journal.renew(
                    claimed.retirement_id,
                    worker_id=worker_id,
                    lease_seconds=duration,
                    now=phase_now,
                )
                if current.state != "running":
                    return _execution(
                        current,
                        pointer_mutated=pointer_mutated,
                        authorization_mutated=authorization_mutated,
                    )
                common, _signed = _publication_attempts(
                    current,
                    authorization_journal=authorization_journal,
                    signed_journal=signed_journal,
                )
                candidate = _signed_candidate(current, set_store=set_store)
                _authority(
                    current,
                    candidate,
                    generations=generations,
                    graphs=graphs,
                )
                if current.phase == "planned":
                    pointer = _pointer_id(set_store, current)
                    allowed = {current.signed_candidate_set_id}
                    if current.authorization_candidate_set_id is not None:
                        allowed.add(current.authorization_candidate_set_id)
                    if pointer not in allowed:
                        raise RuntimeError(
                            "retirement pointer changed before durable intent."
                        )
                    current = retirement_journal.record_pointer_restore_intent(
                        current.retirement_id,
                        worker_id=worker_id,
                        now=phase_now,
                    )
                    _hook(_phase_hook, "pointer_restore_intent", current)
                    continue
                if current.phase == "pointer_restore_intent":
                    common = claim_or_renew_authorization_publication_retirement_lease(
                        authorization_journal,
                        operation_id=current.publication_operation_id,
                        retirement_id=current.retirement_id,
                        owner_id=current.owner_id,
                        graph_set_key=current.graph_set_key,
                        expected_candidate_set_id=current.authorization_candidate_set_id,
                        lease_seconds=duration,
                        now=phase_now,
                    )
                    pointer = _pointer_id(set_store, current)
                    if (
                        current.authorization_candidate_set_id is not None
                        and pointer == current.authorization_candidate_set_id
                    ):
                        set_store.commit(
                            candidate,
                            make_current=True,
                            expected_current_set_id=current.authorization_candidate_set_id,
                            now=phase_now,
                        )
                        pointer_mutated = True
                        _hook(_phase_hook, "signed_pointer_committed", current)
                    elif pointer == current.signed_candidate_set_id:
                        pass
                    else:
                        # Durable intent proves any different pointer appeared after the
                        # operation accepted a signed/authorization candidate pointer.
                        # Preserve it; never overwrite a newer external publication.
                        _hook(_phase_hook, "external_pointer_preserved", current)
                    current = retirement_journal.record_pointer_safe(
                        current.retirement_id,
                        worker_id=worker_id,
                        now=phase_now,
                    )
                    _hook(_phase_hook, "pointer_safe_recorded", current)
                    if common.state == "cancelled":
                        continue
                    continue
                if current.phase == "pointer_safe":
                    claim_or_renew_authorization_publication_retirement_lease(
                        authorization_journal,
                        operation_id=current.publication_operation_id,
                        retirement_id=current.retirement_id,
                        owner_id=current.owner_id,
                        graph_set_key=current.graph_set_key,
                        expected_candidate_set_id=current.authorization_candidate_set_id,
                        lease_seconds=duration,
                        now=phase_now,
                    )
                    pointer = _pointer_id(set_store, current)
                    if (
                        current.authorization_candidate_set_id is not None
                        and pointer == current.authorization_candidate_set_id
                    ):
                        raise RuntimeError(
                            "authorization-only candidate became current after pointer safety."
                        )
                    _retired, mutated = retire_claimed_authorization_publication_attempt(
                        authorization_journal,
                        operation_id=current.publication_operation_id,
                        retirement_id=current.retirement_id,
                        owner_id=current.owner_id,
                        graph_set_key=current.graph_set_key,
                        expected_candidate_set_id=current.authorization_candidate_set_id,
                        now=phase_now,
                    )
                    authorization_mutated = authorization_mutated or mutated
                    _hook(_phase_hook, "authorization_attempt_retired", current)
                    final_pointer = _pointer_id(set_store, current)
                    current = retirement_journal.record_authorization_retired(
                        current.retirement_id,
                        worker_id=worker_id,
                        final_pointer_set_id=final_pointer,
                        now=phase_now,
                    )
                    _hook(_phase_hook, "authorization_retirement_recorded", current)
                    continue
                if current.phase == "authorization_retired":
                    common, _signed = _publication_attempts(
                        current,
                        authorization_journal=authorization_journal,
                        signed_journal=signed_journal,
                    )
                    if common.state != "cancelled":
                        raise RuntimeError(
                            "authorization-only publication was not durably retired."
                        )
                    final_pointer = _pointer_id(set_store, current)
                    if (
                        current.authorization_candidate_set_id is not None
                        and final_pointer == current.authorization_candidate_set_id
                    ):
                        raise RuntimeError(
                            "retired authorization-only candidate is current again."
                        )
                    authority = _authority(
                        current,
                        candidate,
                        generations=generations,
                        graphs=graphs,
                    )
                    verification = _canonical_digest(
                        {
                            "scope": "rigorousrag-signed-publication-retirement-outcome-v1",
                            "retirement_id": current.retirement_id,
                            "publication_operation_id": current.publication_operation_id,
                            "signed_candidate_set_id": current.signed_candidate_set_id,
                            "authorization_candidate_set_id": (
                                current.authorization_candidate_set_id
                            ),
                            "final_pointer_set_id": final_pointer,
                            "signed_authority_digest": authority.authority_digest,
                            "authorization_state": common.state,
                        }
                    )
                    completed = retirement_journal.complete(
                        current.retirement_id,
                        worker_id=worker_id,
                        verification_digest=verification,
                        final_pointer_set_id=final_pointer,
                        now=phase_now,
                    )
                    _hook(_phase_hook, "completed", completed)
                    return _execution(
                        completed,
                        pointer_mutated=pointer_mutated,
                        authorization_mutated=authorization_mutated,
                    )
                raise RuntimeError("retirement entered an unsupported recovery phase.")
    except Exception as exc:
        failure = _generic_failure(exc)
        current = retirement_journal.get(claimed.retirement_id)
        if current.state == "running":
            try:
                current = retirement_journal.fail(
                    current.retirement_id,
                    worker_id=worker_id,
                    failure_type=failure,
                    now=clock(),
                )
            except (KeyError, RuntimeError):
                current = retirement_journal.get(claimed.retirement_id)
        raise SignedPublicationRetirementRecoveryError(
            f"signed publication retirement failed ({failure}).",
            retirement_id=current.retirement_id,
            state=current.state,
            phase=current.phase,
        ) from exc


def execute_next_signed_publication_retirement(
    *,
    owner_id: str,
    worker_id: str,
    lease_seconds: int,
    retirement_journal: SignedPublicationRetirementJournal,
    authorization_journal: Any,
    signed_journal: Any,
    set_store: Any,
    generations: Any,
    graphs: Any,
    now: float | None = None,
) -> SignedPublicationRetirementExecution | None:
    timestamp = _clock(now)
    retirement_id = retirement_journal.next_claimable_id(
        owner_id=owner_id,
        now=timestamp,
    )
    if retirement_id is None:
        return None
    return execute_signed_publication_retirement(
        retirement_id,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        retirement_journal=retirement_journal,
        authorization_journal=authorization_journal,
        signed_journal=signed_journal,
        set_store=set_store,
        generations=generations,
        graphs=graphs,
        now=timestamp,
    )


__all__ = [
    "SignedPublicationRetirementExecution",
    "SignedPublicationRetirementRecoveryError",
    "execute_next_signed_publication_retirement",
    "execute_signed_publication_retirement",
    "seed_signed_publication_retirement",
]
