"""Governed operator boundary for reviewed graph-set publication."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import replace
from typing import Any, Iterable

from tools.evidence_graph_relation_authorization_runtime import (
    get_relation_review_authorization_store,
)
from tools.evidence_graph_set_publish import publish_approved_graph_set
from tools.evidence_graph_set_publish_reconcile import execute_publication_attempt
from tools.security import normalize_owner_id


def _ids(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("proposal_ids must be an iterable.")
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("proposal IDs must be strings.")
        selected = value.strip().lower()
        if len(selected) != 64 or any(c not in "0123456789abcdef" for c in selected):
            raise ValueError("proposal IDs must be SHA-256 digests.")
        result.append(selected)
        if len(result) > 100_000:
            raise ValueError("proposal ID list exceeds the limit.")
    if not result or len(set(result)) != len(result):
        raise ValueError("proposal IDs must be non-empty and unique.")
    return tuple(result)


def _timestamp(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("now must be finite and non-negative.")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("now must be finite and non-negative.") from exc
    if not math.isfinite(selected) or selected < 0:
        raise ValueError("now must be finite and non-negative.")
    return selected


class GovernedPublicationLedger:
    """Read-only ledger view carrying committed authorization provenance."""

    def __init__(
        self,
        *,
        ledger: Any,
        authorization_store: Any,
        owner_id: str,
        graph_set_key: str,
        proposal_ids: Iterable[str],
    ) -> None:
        if not callable(getattr(ledger, "get_proposal", None)) or not callable(
            getattr(ledger, "get_decision", None)
        ):
            raise ValueError("ledger lacks the required reviewed-proposal boundary.")
        if not callable(getattr(authorization_store, "get", None)):
            raise ValueError("authorization_store lacks the required read boundary.")
        self._ledger = ledger
        self._authorization_store = authorization_store
        self.owner_id = normalize_owner_id(owner_id)
        if not isinstance(graph_set_key, str) or not graph_set_key.strip():
            raise ValueError("graph_set_key is required.")
        self.graph_set_key = graph_set_key.strip()
        self.proposal_ids = _ids(proposal_ids)
        self._proposals: dict[str, Any] = {}
        self._decisions: dict[str, Any] = {}
        self._receipts: dict[str, Any] = {}
        self._validate()

    def _validate(self) -> None:
        for proposal_id in self.proposal_ids:
            proposal = self._ledger.get_proposal(proposal_id)
            decision = self._ledger.get_decision(proposal_id)
            if (
                proposal.owner_id != self.owner_id
                or proposal.graph_set_key != self.graph_set_key
            ):
                raise RuntimeError("governed proposal escaped publication scope.")
            if decision is None or decision.decision != "approved":
                raise RuntimeError("proposal is not governed and approved.")
            receipt = self._authorization_store.get(decision.decision_id)
            if receipt is None or receipt.state != "committed":
                raise RuntimeError(
                    "approved proposal lacks a committed authorization receipt."
                )
            authorization = receipt.authorization
            if (
                authorization.proposal_id != proposal.proposal_id
                or authorization.decision_id != decision.decision_id
                or authorization.owner_id != self.owner_id
                or authorization.graph_set_key != self.graph_set_key
                or authorization.decision != "approved"
                or authorization.reviewer_id != decision.reviewer_id
                or authorization.separation_of_duties_enforced is not True
            ):
                raise RuntimeError(
                    "authorization receipt differs from approved proposal."
                )
            self._proposals[proposal_id] = proposal
            self._decisions[proposal_id] = decision
            self._receipts[proposal_id] = receipt

    @property
    def authorization_digest(self) -> str:
        payload = [
            self._receipts[value].authorization.authorization_digest
            for value in sorted(self.proposal_ids)
        ]
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def get_proposal(self, proposal_id: str) -> Any:
        if not isinstance(proposal_id, str):
            raise ValueError("proposal_id must be a string.")
        selected = proposal_id.strip().lower()
        proposal = self._proposals.get(selected)
        if proposal is None:
            return self._ledger.get_proposal(proposal_id)
        receipt = self._receipts[selected]
        authorization = receipt.authorization
        return replace(
            proposal,
            metadata={
                **dict(proposal.metadata),
                "review_authorization_digest": authorization.authorization_digest,
                "review_policy_digest": authorization.policy_digest,
                "review_grant_digest": authorization.grant_digest,
                "review_authorization_state": receipt.state,
                "review_separation_of_duties": True,
            },
        )

    def get_decision(self, proposal_id: str) -> Any:
        if not isinstance(proposal_id, str):
            raise ValueError("proposal_id must be a string.")
        selected = proposal_id.strip().lower()
        if selected in self._decisions:
            return self._decisions[selected]
        return self._ledger.get_decision(proposal_id)


def governed_publication_ledger(
    *,
    owner_id: str,
    graph_set_key: str,
    proposal_ids: Iterable[str],
    ledger: Any,
    authorization_store: Any | None = None,
) -> GovernedPublicationLedger:
    store = (
        get_relation_review_authorization_store()
        if authorization_store is None
        else authorization_store
    )
    return GovernedPublicationLedger(
        ledger=ledger,
        authorization_store=store,
        owner_id=owner_id,
        graph_set_key=graph_set_key,
        proposal_ids=proposal_ids,
    )


def publish_governed_approved_graph_set(
    *,
    owner_id: str,
    graph_set_key: str,
    proposal_ids: Iterable[str],
    expected_current_set_id: str | None,
    ledger: Any,
    set_store: Any,
    generations: Any,
    graphs: Any,
    authorization_store: Any | None = None,
    now: float | None = None,
):
    governed = governed_publication_ledger(
        owner_id=owner_id,
        graph_set_key=graph_set_key,
        proposal_ids=proposal_ids,
        ledger=ledger,
        authorization_store=authorization_store,
    )
    return publish_approved_graph_set(
        owner_id=owner_id,
        graph_set_key=graph_set_key,
        proposal_ids=governed.proposal_ids,
        expected_current_set_id=expected_current_set_id,
        ledger=governed,
        set_store=set_store,
        generations=generations,
        graphs=graphs,
        now=now,
    )


def execute_governed_publication_attempt(
    operation_id: str,
    *,
    worker_id: str,
    lease_seconds: int,
    journal: Any,
    ledger: Any,
    set_store: Any,
    generations: Any,
    graphs: Any,
    authorization_store: Any | None = None,
    now: float | None = None,
):
    attempt = journal.get(operation_id)
    governed = governed_publication_ledger(
        owner_id=attempt.owner_id,
        graph_set_key=attempt.graph_set_key,
        proposal_ids=attempt.proposal_ids,
        ledger=ledger,
        authorization_store=authorization_store,
    )
    return execute_publication_attempt(
        operation_id,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        journal=journal,
        ledger=governed,
        set_store=set_store,
        generations=generations,
        graphs=graphs,
        now=now,
    )


def execute_next_governed_publication_attempt(
    *,
    owner_id: str,
    worker_id: str,
    lease_seconds: int,
    journal: Any,
    ledger: Any,
    set_store: Any,
    generations: Any,
    graphs: Any,
    authorization_store: Any | None = None,
    now: float | None = None,
):
    timestamp = _timestamp(time.time() if now is None else now)
    operation_id = journal.next_claimable_id(owner_id=owner_id, now=timestamp)
    if operation_id is None:
        return None
    return execute_governed_publication_attempt(
        operation_id,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        journal=journal,
        ledger=ledger,
        set_store=set_store,
        generations=generations,
        graphs=graphs,
        authorization_store=authorization_store,
        now=timestamp,
    )


__all__ = [
    "GovernedPublicationLedger",
    "execute_governed_publication_attempt",
    "execute_next_governed_publication_attempt",
    "governed_publication_ledger",
    "publish_governed_approved_graph_set",
]
