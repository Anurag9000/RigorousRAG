"""Bind committed signed actor-use provenance into reviewed graph publication."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from tools.evidence_graph_relation_actor_use_runtime import get_signed_actor_use_store
from tools.evidence_graph_set_governed_publish import (
    GovernedPublicationLedger,
    governed_publication_ledger,
)
from tools.evidence_graph_set_publish import publish_approved_graph_set
from tools.evidence_graph_set_publish_reconcile import execute_publication_attempt

_MAX_ACTOR_USES_PER_DECISION = 10_000


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class _SignedActorProposalView:
    """Delegate proposal identity while adding committed actor-use provenance."""

    def __init__(self, proposal: Any, metadata: dict[str, Any]) -> None:
        self._proposal = proposal
        self.metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._proposal, name)


class SignedActorPublicationLedger:
    """Read-only governed ledger that verifies signed actor-use completion."""

    def __init__(
        self,
        *,
        governed_ledger: GovernedPublicationLedger,
        actor_use_store: Any,
    ) -> None:
        if not isinstance(governed_ledger, GovernedPublicationLedger):
            raise ValueError("governed_ledger must be GovernedPublicationLedger.")
        if not callable(getattr(actor_use_store, "list", None)):
            raise ValueError("actor_use_store lacks the required read boundary.")
        self._governed = governed_ledger
        self._actor_use_store = actor_use_store
        self.owner_id = governed_ledger.owner_id
        self.graph_set_key = governed_ledger.graph_set_key
        self.proposal_ids = governed_ledger.proposal_ids
        self._uses: dict[str, tuple[Any, ...]] = {}
        self._validate()

    def _validate(self) -> None:
        for proposal_id in self.proposal_ids:
            proposal = self._governed.get_proposal(proposal_id)
            decision = self._governed.get_decision(proposal_id)
            values = tuple(
                self._actor_use_store.list(
                    owner_id=self.owner_id,
                    decision_id=decision.decision_id,
                    limit=_MAX_ACTOR_USES_PER_DECISION,
                )
            )
            if len(values) >= _MAX_ACTOR_USES_PER_DECISION:
                raise RuntimeError(
                    "signed actor-use provenance reached the publication ceiling."
                )
            for value in values:
                if (
                    value.state != "committed"
                    or value.decision_id != decision.decision_id
                    or value.proposal_id != proposal.proposal_id
                    or value.owner_id != proposal.owner_id
                    or value.graph_set_key != proposal.graph_set_key
                    or value.decision != decision.decision
                    or value.actor_id != decision.reviewer_id
                ):
                    raise RuntimeError(
                        "signed actor-use provenance differs from the governed decision."
                    )
            self._uses[proposal_id] = tuple(
                sorted(values, key=lambda value: value.assertion_digest)
            )

    @property
    def signed_actor_use_digest(self) -> str:
        return _sha256(
            {
                "scope": "rigorousrag-reviewed-publication-actor-uses-v1",
                "proposals": [
                    {
                        "proposal_id": proposal_id,
                        "use_digests": [
                            value.use_digest for value in self._uses[proposal_id]
                        ],
                    }
                    for proposal_id in sorted(self.proposal_ids)
                ],
            }
        )

    def get_proposal(self, proposal_id: str) -> Any:
        proposal = self._governed.get_proposal(proposal_id)
        values = self._uses.get(proposal.proposal_id, ())
        aggregate = _sha256(
            {
                "scope": "rigorousrag-reviewed-relation-actor-uses-v1",
                "proposal_id": proposal.proposal_id,
                "use_digests": [value.use_digest for value in values],
            }
        )
        return _SignedActorProposalView(
            proposal,
            {
                **dict(proposal.metadata),
                "review_signed_actor_use_count": len(values),
                "review_signed_actor_use_digest": aggregate,
                "review_signed_actor_use_required": bool(values),
            },
        )

    def get_decision(self, proposal_id: str) -> Any:
        return self._governed.get_decision(proposal_id)


def signed_actor_publication_ledger(
    *,
    owner_id: str,
    graph_set_key: str,
    proposal_ids: Iterable[str],
    ledger: Any,
    authorization_store: Any,
    actor_use_store: Any | None = None,
) -> SignedActorPublicationLedger:
    governed = governed_publication_ledger(
        owner_id=owner_id,
        graph_set_key=graph_set_key,
        proposal_ids=proposal_ids,
        ledger=ledger,
        authorization_store=authorization_store,
    )
    selected_actor_uses = (
        get_signed_actor_use_store()
        if actor_use_store is None
        else actor_use_store
    )
    return SignedActorPublicationLedger(
        governed_ledger=governed,
        actor_use_store=selected_actor_uses,
    )


def publish_signed_actor_governed_graph_set(
    *,
    owner_id: str,
    graph_set_key: str,
    proposal_ids: Iterable[str],
    expected_current_set_id: str | None,
    ledger: Any,
    authorization_store: Any,
    actor_use_store: Any,
    set_store: Any,
    generations: Any,
    graphs: Any,
    now: float | None = None,
):
    governed = signed_actor_publication_ledger(
        owner_id=owner_id,
        graph_set_key=graph_set_key,
        proposal_ids=proposal_ids,
        ledger=ledger,
        authorization_store=authorization_store,
        actor_use_store=actor_use_store,
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


def execute_signed_actor_publication_attempt(
    operation_id: str,
    *,
    worker_id: str,
    lease_seconds: int,
    journal: Any,
    ledger: Any,
    authorization_store: Any,
    actor_use_store: Any,
    set_store: Any,
    generations: Any,
    graphs: Any,
    now: float | None = None,
):
    attempt = journal.get(operation_id)
    governed = signed_actor_publication_ledger(
        owner_id=attempt.owner_id,
        graph_set_key=attempt.graph_set_key,
        proposal_ids=attempt.proposal_ids,
        ledger=ledger,
        authorization_store=authorization_store,
        actor_use_store=actor_use_store,
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


def execute_next_signed_actor_publication_attempt(
    *,
    owner_id: str,
    worker_id: str,
    lease_seconds: int,
    journal: Any,
    ledger: Any,
    authorization_store: Any,
    actor_use_store: Any,
    set_store: Any,
    generations: Any,
    graphs: Any,
    now: float | None = None,
):
    operation_id = journal.next_claimable_id(owner_id=owner_id, now=now)
    if operation_id is None:
        return None
    return execute_signed_actor_publication_attempt(
        operation_id,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        journal=journal,
        ledger=ledger,
        authorization_store=authorization_store,
        actor_use_store=actor_use_store,
        set_store=set_store,
        generations=generations,
        graphs=graphs,
        now=now,
    )


__all__ = [
    "SignedActorPublicationLedger",
    "execute_next_signed_actor_publication_attempt",
    "execute_signed_actor_publication_attempt",
    "publish_signed_actor_governed_graph_set",
    "signed_actor_publication_ledger",
]
