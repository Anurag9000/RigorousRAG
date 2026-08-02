from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools import evidence_graph_set_signed_actor_provenance as provenance
from tools.evidence_graph_set_governed_publish import GovernedPublicationLedger


class Ledger:
    def __init__(self, proposal, decision):
        self.proposal = proposal
        self.decision = decision

    def get_proposal(self, proposal_id):
        assert proposal_id == self.proposal.proposal_id
        return self.proposal

    def get_decision(self, proposal_id):
        assert proposal_id == self.proposal.proposal_id
        return self.decision


class AuthorizationStore:
    def __init__(self, receipt):
        self.receipt = receipt

    def get(self, decision_id):
        assert decision_id == self.receipt.authorization.decision_id
        return self.receipt


class ActorUseStore:
    def __init__(self, values):
        self.values = tuple(values)
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        return self.values[: kwargs["limit"]]


def fixture(values=()):
    proposal = SimpleNamespace(
        proposal_id="1" * 64,
        owner_id="alice",
        graph_set_key="review",
        metadata={"existing": "retained"},
    )
    decision = SimpleNamespace(
        decision_id="2" * 64,
        proposal_id=proposal.proposal_id,
        owner_id="alice",
        decision="approved",
        reviewer_id="reviewer",
    )
    authorization = SimpleNamespace(
        proposal_id=proposal.proposal_id,
        decision_id=decision.decision_id,
        owner_id="alice",
        graph_set_key="review",
        decision="approved",
        reviewer_id="reviewer",
        separation_of_duties_enforced=True,
        authorization_digest="3" * 64,
        policy_digest="4" * 64,
        grant_digest="5" * 64,
    )
    receipt = SimpleNamespace(state="committed", authorization=authorization)
    governed = GovernedPublicationLedger(
        ledger=Ledger(proposal, decision),
        authorization_store=AuthorizationStore(receipt),
        owner_id="alice",
        graph_set_key="review",
        proposal_ids=(proposal.proposal_id,),
    )
    store = ActorUseStore(values)
    return proposal, decision, governed, store


def actor_use(*, state="committed", actor_id="reviewer", digit="6"):
    return SimpleNamespace(
        assertion_digest=digit * 64,
        decision_id="2" * 64,
        proposal_id="1" * 64,
        owner_id="alice",
        graph_set_key="review",
        decision="approved",
        actor_id=actor_id,
        use_digest=("7" if digit == "6" else "8") * 64,
        state=state,
    )


def test_direct_actor_decision_gets_deterministic_zero_use_metadata():
    proposal, _decision, governed, store = fixture()
    selected = provenance.SignedActorPublicationLedger(
        governed_ledger=governed,
        actor_use_store=store,
    )

    value = selected.get_proposal(proposal.proposal_id)

    assert value.proposal_id == proposal.proposal_id
    assert value.metadata["existing"] == "retained"
    assert value.metadata["review_signed_actor_use_count"] == 0
    assert value.metadata["review_signed_actor_use_required"] is False
    assert len(value.metadata["review_signed_actor_use_digest"]) == 64
    assert len(selected.signed_actor_use_digest) == 64


def test_committed_signed_actor_uses_are_aggregated_deterministically():
    values = (actor_use(digit="9"), actor_use(digit="6"))
    proposal, decision, governed, store = fixture(values)
    selected = provenance.SignedActorPublicationLedger(
        governed_ledger=governed,
        actor_use_store=store,
    )

    value = selected.get_proposal(proposal.proposal_id)

    assert value.metadata["review_signed_actor_use_count"] == 2
    assert value.metadata["review_signed_actor_use_required"] is True
    assert len(value.metadata["review_signed_actor_use_digest"]) == 64
    assert store.calls == [
        {
            "owner_id": "alice",
            "decision_id": decision.decision_id,
            "limit": 10_000,
        }
    ]


def test_reserved_or_mismatched_actor_uses_fail_closed():
    _proposal, _decision, governed, store = fixture(
        (actor_use(state="reserved"),)
    )
    with pytest.raises(RuntimeError, match="differs"):
        provenance.SignedActorPublicationLedger(
            governed_ledger=governed,
            actor_use_store=store,
        )

    _proposal, _decision, governed, store = fixture(
        (actor_use(actor_id="other-reviewer"),)
    )
    with pytest.raises(RuntimeError, match="differs"):
        provenance.SignedActorPublicationLedger(
            governed_ledger=governed,
            actor_use_store=store,
        )


def test_immediate_publication_delegates_with_signed_actor_ledger(monkeypatch):
    proposal, _decision, governed, store = fixture((actor_use(),))
    marker = object()
    observed = {}
    monkeypatch.setattr(
        provenance,
        "governed_publication_ledger",
        lambda **kwargs: governed,
    )

    def publish(**kwargs):
        observed.update(kwargs)
        return marker

    monkeypatch.setattr(provenance, "publish_approved_graph_set", publish)
    result = provenance.publish_signed_actor_governed_graph_set(
        owner_id="alice",
        graph_set_key="review",
        proposal_ids=(proposal.proposal_id,),
        expected_current_set_id=None,
        ledger=object(),
        authorization_store=object(),
        actor_use_store=store,
        set_store=object(),
        generations=object(),
        graphs=object(),
        now=12.0,
    )

    assert result is marker
    assert isinstance(observed["ledger"], provenance.SignedActorPublicationLedger)
    assert observed["proposal_ids"] == (proposal.proposal_id,)
    assert observed["now"] == 12.0


def test_durable_execution_delegates_after_actor_use_validation(monkeypatch):
    proposal, _decision, governed, store = fixture((actor_use(),))
    attempt = SimpleNamespace(
        owner_id="alice",
        graph_set_key="review",
        proposal_ids=(proposal.proposal_id,),
    )
    journal = SimpleNamespace(get=lambda operation_id: attempt)
    marker = object()
    observed = {}
    monkeypatch.setattr(
        provenance,
        "governed_publication_ledger",
        lambda **kwargs: governed,
    )

    def execute(operation_id, **kwargs):
        observed["operation_id"] = operation_id
        observed.update(kwargs)
        return marker

    monkeypatch.setattr(provenance, "execute_publication_attempt", execute)
    result = provenance.execute_signed_actor_publication_attempt(
        "9" * 64,
        worker_id="worker",
        lease_seconds=30,
        journal=journal,
        ledger=object(),
        authorization_store=object(),
        actor_use_store=store,
        set_store=object(),
        generations=object(),
        graphs=object(),
        now=20.0,
    )

    assert result is marker
    assert observed["operation_id"] == "9" * 64
    assert isinstance(observed["ledger"], provenance.SignedActorPublicationLedger)
    assert observed["worker_id"] == "worker"
