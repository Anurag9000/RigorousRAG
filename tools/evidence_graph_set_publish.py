"""Compensating publication of reviewed cross-document graph-set versions."""

from __future__ import annotations

import math
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Iterable

from tools.evidence_graph_authority import resolve_evidence_graph
from tools.evidence_graph_relation_review import approved_relations
from tools.evidence_graph_set_pointer import clear_current_graph_set_pointer
from tools.evidence_graph_set_store import assess_graph_set_authority
from tools.evidence_graph_sets import build_evidence_graph_set
from tools.index_coordinator import _document_lock
from tools.security import normalize_owner_id


class EvidenceGraphSetPublishError(RuntimeError):
    """Raised when reviewed graph-set publication fails or compensation is incomplete."""

    def __init__(self, message: str, *, compensation_errors: tuple[str, ...] = ()) -> None:
        self.compensation_errors = tuple(compensation_errors)
        suffix = (
            " Compensation errors: " + ", ".join(self.compensation_errors)
            if self.compensation_errors
            else ""
        )
        super().__init__(message + suffix)


@dataclass(frozen=True)
class EvidenceGraphSetPublishResult:
    graph_set_id: str
    graph_set_digest: str
    graph_set_key: str
    previous_graph_set_id: str | None
    member_count: int
    edge_count: int
    authority_digest: str
    proposal_ids: tuple[str, ...]
    pointer_changed: bool
    compensation_performed: bool
    published_at: float


def _proposal_ids(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("proposal IDs must be strings.")
        cleaned = value.strip().lower()
        if len(cleaned) != 64 or any(character not in "0123456789abcdef" for character in cleaned):
            raise ValueError("proposal IDs must be SHA-256 digests.")
        result.append(cleaned)
        if len(result) > 100_000:
            raise ValueError("proposal ID list exceeds the limit.")
    if not result:
        raise ValueError("at least one approved proposal ID is required.")
    if len(set(result)) != len(result):
        raise ValueError("proposal IDs must be unique.")
    return tuple(sorted(result))


def _compensate_pointer(
    *,
    owner_id: str,
    graph_set_key: str,
    activated_set_id: str,
    previous: Any | None,
    store: Any,
    now: float,
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        if previous is None:
            cleared = clear_current_graph_set_pointer(
                store,
                owner_id=owner_id,
                graph_set_key=graph_set_key,
                expected_current_set_id=activated_set_id,
            )
            if not cleared:
                errors.append("pointer:missing")
        else:
            store.commit(
                previous,
                make_current=True,
                expected_current_set_id=activated_set_id,
                now=now,
            )
    except Exception as exc:
        errors.append(f"pointer:{type(exc).__name__}")
    try:
        actual = store.current(owner_id=owner_id, graph_set_key=graph_set_key)
        expected_id = None if previous is None else previous.graph_set_id
        actual_id = None if actual is None else actual.graph_set_id
        if actual_id != expected_id:
            errors.append("pointer:verification")
    except Exception as exc:
        errors.append(f"verification:{type(exc).__name__}")
    return tuple(errors)


def publish_approved_graph_set(
    *,
    owner_id: str,
    graph_set_key: str,
    proposal_ids: Iterable[str],
    expected_current_set_id: str | None,
    ledger: Any,
    set_store: Any,
    generations: Any,
    graphs: Any,
    now: float | None = None,
) -> EvidenceGraphSetPublishResult:
    """Publish reviewed relations with exact authority and pointer compensation."""

    owner = normalize_owner_id(owner_id)
    if not isinstance(graph_set_key, str) or not graph_set_key.strip():
        raise ValueError("graph_set_key is required.")
    key = graph_set_key.strip()
    ids = _proposal_ids(proposal_ids)
    if expected_current_set_id is not None:
        if not isinstance(expected_current_set_id, str):
            raise ValueError("expected_current_set_id must be a SHA-256 digest or None.")
        expected_current_set_id = expected_current_set_id.strip().lower()
        if len(expected_current_set_id) != 64 or any(
            character not in "0123456789abcdef" for character in expected_current_set_id
        ):
            raise ValueError("expected_current_set_id must be a SHA-256 digest or None.")
    if isinstance(now, bool):
        raise ValueError("now must be finite and non-negative.")
    try:
        timestamp = float(time.time() if now is None else now)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("now must be finite and non-negative.") from exc
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("now must be finite and non-negative.")

    proposals = [ledger.get_proposal(proposal_id) for proposal_id in ids]
    doc_ids: set[str] = set()
    for proposal in proposals:
        if proposal.owner_id != owner or proposal.graph_set_key != key:
            raise EvidenceGraphSetPublishError("proposal escaped publication scope.")
        doc_ids.add(proposal.source.doc_id)
        doc_ids.add(proposal.target.doc_id)
    if len(doc_ids) < 2:
        raise EvidenceGraphSetPublishError("publication requires at least two documents.")

    activated = False
    previous = None
    candidate = None
    with ExitStack() as stack:
        for doc_id in sorted(doc_ids):
            stack.enter_context(_document_lock(owner, doc_id))
        try:
            previous = set_store.current(owner_id=owner, graph_set_key=key)
            previous_id = None if previous is None else previous.graph_set_id
            if previous_id != expected_current_set_id:
                raise EvidenceGraphSetPublishError(
                    "graph set current pointer differs from the explicit expectation."
                )
            views = tuple(
                resolve_evidence_graph(
                    owner_id=owner,
                    doc_id=doc_id,
                    graphs=graphs,
                    generations=generations,
                )
                for doc_id in sorted(doc_ids)
            )
            relations = approved_relations(
                owner_id=owner,
                graph_set_key=key,
                proposal_ids=ids,
                authority_views=views,
                ledger=ledger,
            )
            candidate = build_evidence_graph_set(
                owner_id=owner,
                graph_set_key=key,
                authority_views=views,
                relations=relations,
                now=timestamp,
            )
            set_store.commit(candidate, make_current=False, now=timestamp)
            before = assess_graph_set_authority(
                candidate,
                generations=generations,
                graphs=graphs,
            )
            if not before.authoritative_current:
                raise EvidenceGraphSetPublishError(
                    "graph-set members changed before pointer activation."
                )
            set_store.commit(
                candidate,
                make_current=True,
                expected_current_set_id=previous_id,
                now=timestamp,
            )
            activated = True
            after = assess_graph_set_authority(
                candidate,
                generations=generations,
                graphs=graphs,
            )
            if not after.authoritative_current:
                raise EvidenceGraphSetPublishError(
                    "graph-set members changed after pointer activation."
                )
            current = set_store.current(owner_id=owner, graph_set_key=key)
            if current is None or current.graph_set_id != candidate.graph_set_id:
                raise EvidenceGraphSetPublishError(
                    "graph-set pointer verification failed after activation."
                )
            return EvidenceGraphSetPublishResult(
                graph_set_id=candidate.graph_set_id,
                graph_set_digest=candidate.graph_set_digest,
                graph_set_key=key,
                previous_graph_set_id=previous_id,
                member_count=len(candidate.members),
                edge_count=len(candidate.edges),
                authority_digest=after.authority_digest,
                proposal_ids=ids,
                pointer_changed=previous_id != candidate.graph_set_id,
                compensation_performed=False,
                published_at=timestamp,
            )
        except Exception as exc:
            if activated and candidate is not None:
                errors = _compensate_pointer(
                    owner_id=owner,
                    graph_set_key=key,
                    activated_set_id=candidate.graph_set_id,
                    previous=previous,
                    store=set_store,
                    now=timestamp,
                )
                raise EvidenceGraphSetPublishError(
                    f"reviewed graph-set publication failed ({type(exc).__name__}).",
                    compensation_errors=errors,
                ) from exc
            if isinstance(exc, EvidenceGraphSetPublishError):
                raise
            raise EvidenceGraphSetPublishError(
                f"reviewed graph-set publication failed ({type(exc).__name__})."
            ) from exc


__all__ = [
    "EvidenceGraphSetPublishError",
    "EvidenceGraphSetPublishResult",
    "publish_approved_graph_set",
]
