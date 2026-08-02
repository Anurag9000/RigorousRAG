from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools import evidence_graph_set_governed_publish as governed_publish
from tools.evidence_graph_relation_authorization_store import (
    RelationReviewAuthorizationStore,
)
from tools.evidence_graph_relation_policy import (
    GovernedRelationReviewService,
    RelationReviewPolicy,
)
from tools.evidence_graph_relation_review import (
    CrossDocumentRelationProposal,
    RelationEndpoint,
    RelationReviewDecision,
    RelationReviewLedger,
)


def endpoint(doc_id: str, digit: str) -> RelationEndpoint:
    return RelationEndpoint(
        doc_id=doc_id,
        generation=1,
        graph_digest=digit * 64,
        node_id=("c" if digit == "a" else "d") * 64,
        provenance_digest=("e" if digit == "a" else "f") * 64,
    )


def setup_review(tmp_path, *, commit_receipt: bool = True):
    ledger = RelationReviewLedger(tmp_path / "reviews.sqlite3")
    proposal = ledger.submit(
        CrossDocumentRelationProposal.create(
            owner_id="alice",
            graph_set_key="review",
            relation_key="a-b",
            source=endpoint("doc-a", "a"),
            target=endpoint("doc-b", "b"),
            edge_type="supports",
            proposer_kind="human",
            proposer_id="annotator",
            evidence_digest="1" * 64,
            created_at=1.0,
        )
    )
    decision = RelationReviewDecision.create(
        proposal_id=proposal.proposal_id,
        owner_id="alice",
        decision="approved",
        reviewer_id="reviewer",
        reason_code="verified",
        replacement_proposal_id=None,
        decided_at=2.0,
    )
    policy = RelationReviewPolicy.from_mapping(
        {
            "schema_version": 1,
            "reviewers": [
                {
                    "reviewer_id": "reviewer",
                    "owners": ["alice"],
                    "graph_set_keys": ["review"],
                    "decisions": ["approved"],
                }
            ],
        }
    )
    authorizations = RelationReviewAuthorizationStore(
        tmp_path / "authorizations.sqlite3"
    )
    service = GovernedRelationReviewService(
        ledger=ledger,
        policy=policy,
        authorization_store=authorizations,
        clock=lambda: 10.0,
    )
    if commit_receipt:
        service.decide(decision)
    else:
        authorization = service._authorization(proposal, decision, now=10.0)
        authorizations.prepare(authorization, now=10.0)
        ledger.decide(decision)
    return ledger, authorizations, proposal, decision


def test_governed_ledger_enriches_converter_view_without_changing_identity(tmp_path):
    ledger, authorizations, proposal, decision = setup_review(tmp_path)

    view = governed_publish.governed_publication_ledger(
        owner_id="alice",
        graph_set_key="review",
        proposal_ids=(proposal.proposal_id,),
        ledger=ledger,
        authorization_store=authorizations,
    )
    selected = view.get_proposal(proposal.proposal_id)
    receipt = authorizations.get(decision.decision_id)

    assert selected.proposal_id == proposal.proposal_id
    assert selected._proposal is proposal
    assert proposal.metadata == {}
    assert selected.metadata["review_authorization_digest"] == (
        receipt.authorization.authorization_digest
    )
    assert selected.metadata["review_policy_digest"] == (
        receipt.authorization.policy_digest
    )
    assert selected.metadata["review_grant_digest"] == (
        receipt.authorization.grant_digest
    )
    assert selected.metadata["review_authorization_state"] == "committed"
    assert selected.metadata["review_separation_of_duties"] is True
    assert len(view.authorization_digest) == 64


def test_ungoverned_and_uncommitted_approvals_cannot_publish(tmp_path):
    ledger = RelationReviewLedger(tmp_path / "legacy.sqlite3")
    proposal = ledger.submit(
        CrossDocumentRelationProposal.create(
            owner_id="alice",
            graph_set_key="review",
            relation_key="legacy",
            source=endpoint("doc-a", "a"),
            target=endpoint("doc-b", "b"),
            edge_type="supports",
            proposer_kind="human",
            proposer_id="annotator",
            evidence_digest="2" * 64,
            created_at=1.0,
        )
    )
    ledger.decide(
        RelationReviewDecision.create(
            proposal_id=proposal.proposal_id,
            owner_id="alice",
            decision="approved",
            reviewer_id="reviewer",
            reason_code="legacy",
            replacement_proposal_id=None,
            decided_at=2.0,
        )
    )
    empty = RelationReviewAuthorizationStore(tmp_path / "empty.sqlite3")
    with pytest.raises(RuntimeError, match="committed authorization"):
        governed_publish.governed_publication_ledger(
            owner_id="alice",
            graph_set_key="review",
            proposal_ids=(proposal.proposal_id,),
            ledger=ledger,
            authorization_store=empty,
        )

    ledger2, pending, proposal2, _decision = setup_review(
        tmp_path / "pending", commit_receipt=False
    )
    with pytest.raises(RuntimeError, match="committed authorization"):
        governed_publish.governed_publication_ledger(
            owner_id="alice",
            graph_set_key="review",
            proposal_ids=(proposal2.proposal_id,),
            ledger=ledger2,
            authorization_store=pending,
        )


def test_immediate_publication_receives_only_governed_ledger(
    tmp_path, monkeypatch
):
    ledger, authorizations, proposal, _decision = setup_review(tmp_path)
    observed = {}
    marker = object()

    def publish(**kwargs):
        observed.update(kwargs)
        return marker

    monkeypatch.setattr(governed_publish, "publish_approved_graph_set", publish)
    result = governed_publish.publish_governed_approved_graph_set(
        owner_id="alice",
        graph_set_key="review",
        proposal_ids=(proposal.proposal_id,),
        expected_current_set_id=None,
        ledger=ledger,
        authorization_store=authorizations,
        set_store=object(),
        generations=object(),
        graphs=object(),
        now=12.0,
    )

    assert result is marker
    assert isinstance(observed["ledger"], governed_publish.GovernedPublicationLedger)
    assert observed["proposal_ids"] == (proposal.proposal_id,)
    assert observed["now"] == 12.0


def test_durable_execution_validates_receipts_before_raw_executor(
    tmp_path, monkeypatch
):
    ledger, authorizations, proposal, _decision = setup_review(tmp_path)
    attempt = SimpleNamespace(
        operation_id="9" * 64,
        owner_id="alice",
        graph_set_key="review",
        proposal_ids=(proposal.proposal_id,),
    )
    journal = SimpleNamespace(get=lambda operation_id: attempt)
    observed = {}
    marker = object()

    def execute(operation_id, **kwargs):
        observed["operation_id"] = operation_id
        observed.update(kwargs)
        return marker

    monkeypatch.setattr(governed_publish, "execute_publication_attempt", execute)
    result = governed_publish.execute_governed_publication_attempt(
        attempt.operation_id,
        worker_id="worker",
        lease_seconds=30,
        journal=journal,
        ledger=ledger,
        authorization_store=authorizations,
        set_store=object(),
        generations=object(),
        graphs=object(),
        now=20.0,
    )

    assert result is marker
    assert observed["operation_id"] == attempt.operation_id
    assert isinstance(observed["ledger"], governed_publish.GovernedPublicationLedger)
    assert observed["worker_id"] == "worker"


def test_next_governed_execution_is_idle_without_loading_receipts(monkeypatch):
    journal = SimpleNamespace(next_claimable_id=lambda **kwargs: None)
    monkeypatch.setattr(
        governed_publish,
        "execute_governed_publication_attempt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("executor should not run")
        ),
    )

    assert governed_publish.execute_next_governed_publication_attempt(
        owner_id="alice",
        worker_id="worker",
        lease_seconds=30,
        journal=journal,
        ledger=object(),
        authorization_store=object(),
        set_store=object(),
        generations=object(),
        graphs=object(),
        now=1.0,
    ) is None
