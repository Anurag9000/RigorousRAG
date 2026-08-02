from __future__ import annotations

import json

from tools import evidence_graph_relation_cli as cli
from tools.evidence_graph_relation_actor import (
    ReviewActorBinding,
    require_relation_review_actor,
)
from tools.evidence_graph_relation_actor_use_store import (
    SignedActorUseRecord,
    SignedActorUseStore,
)
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


def policy():
    return RelationReviewPolicy.from_mapping(
        {
            "schema_version": 1,
            "reviewers": [
                {
                    "reviewer_id": "reviewer-1",
                    "owners": ["alice"],
                    "graph_set_keys": ["review"],
                    "decisions": ["approved", "rejected", "superseded"],
                }
            ],
        }
    )


def proposal(ledger, digit="1"):
    return ledger.submit(
        CrossDocumentRelationProposal.create(
            owner_id="alice",
            graph_set_key="review",
            relation_key=f"relation-{digit}",
            source=RelationEndpoint(
                doc_id="doc-a",
                generation=1,
                graph_digest="a" * 64,
                node_id="c" * 64,
                provenance_digest="e" * 64,
            ),
            target=RelationEndpoint(
                doc_id="doc-b",
                generation=1,
                graph_digest="b" * 64,
                node_id="d" * 64,
                provenance_digest="f" * 64,
            ),
            edge_type="supports",
            proposer_kind="human",
            proposer_id="annotator",
            evidence_digest=digit * 64,
            created_at=1.0,
        )
    )


def binding(digit="7", loaded_at=10.0):
    return ReviewActorBinding.create(
        actor_id="reviewer-1",
        binding_method="hmac_assertion",
        assertion_digest=digit * 64,
        issuer="review-control-plane",
        expires_at=100.0,
        loaded_at=loaded_at,
    )


def decision(value, decided_at=2.0):
    return RelationReviewDecision.create(
        proposal_id=value.proposal_id,
        owner_id="alice",
        decision="approved",
        reviewer_id="reviewer-1",
        reason_code="verified",
        replacement_proposal_id=None,
        decided_at=decided_at,
    )


def args(value):
    return [
        "decide", value.proposal_id,
        "--owner-id", "alice",
        "--decision", "approved",
        "--reviewer-id", "reviewer-1",
        "--reason-code", "verified",
    ]


def install(tmp_path, monkeypatch, selected_binding):
    ledger = RelationReviewLedger(tmp_path / "reviews.sqlite3")
    authorizations = RelationReviewAuthorizationStore(
        tmp_path / "authorizations.sqlite3"
    )
    uses = SignedActorUseStore(tmp_path / "actor-uses.sqlite3")
    monkeypatch.setattr(cli, "get_relation_review_ledger", lambda: ledger)
    monkeypatch.setattr(
        cli, "get_relation_review_authorization_store", lambda: authorizations
    )
    monkeypatch.setattr(cli, "get_signed_actor_use_store", lambda: uses)
    monkeypatch.setattr(cli, "get_relation_review_policy", policy)
    monkeypatch.setattr(
        cli,
        "require_relation_review_actor",
        lambda requested: require_relation_review_actor(
            requested, binding=selected_binding[0]
        ),
    )
    return ledger, authorizations, uses


def read(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def test_signed_decision_commits_use_and_replays_idempotently(
    tmp_path, monkeypatch, capsys
):
    selected = [binding()]
    ledger, _authorizations, uses = install(tmp_path, monkeypatch, selected)
    value = proposal(ledger)

    assert cli.main(args(value)) == 0
    first, error = read(capsys)
    assert error is None
    assert first["review_actor_binding"]["binding_method"] == "hmac_assertion"
    assert first["signed_actor_uses"][0]["state"] == "committed"
    assert first["signed_actor_uses"][0]["contains_signature"] is False
    assert first["signed_actor_uses"][0]["contains_key_material"] is False

    assert cli.main(args(value)) == 0
    second, error = read(capsys)
    assert error is None
    assert second["review"]["decision_id"] == first["review"]["decision_id"]
    records = uses.list(owner_id="alice", decision_id=first["review"]["decision_id"])
    assert len(records) == 1 and records[0].state == "committed"


def test_same_assertion_cannot_authorize_another_decision(
    tmp_path, monkeypatch, capsys
):
    selected = [binding()]
    ledger, _authorizations, _uses = install(tmp_path, monkeypatch, selected)
    first = proposal(ledger, "1")
    second = proposal(ledger, "2")
    assert cli.main(args(first)) == 0
    read(capsys)

    assert cli.main(args(second)) == 2
    output, error = read(capsys)
    assert output is None
    assert error == {"error": "invalid_or_unavailable"}
    assert ledger.get_decision(second.proposal_id) is None


def test_signed_assertion_cannot_be_backfilled_onto_existing_decision(
    tmp_path, monkeypatch, capsys
):
    selected = [binding()]
    ledger, authorizations, _uses = install(tmp_path, monkeypatch, selected)
    value = proposal(ledger)
    service = GovernedRelationReviewService(
        ledger=ledger,
        policy=policy(),
        authorization_store=authorizations,
        clock=lambda: 10.0,
    )
    service.decide(decision(value))

    assert cli.main(args(value)) == 2
    output, error = read(capsys)
    assert output is None
    assert error == {"error": "invalid_or_unavailable"}


def test_fresh_assertion_recovers_only_after_prior_signed_reservation(
    tmp_path, monkeypatch, capsys
):
    selected = [binding("7")]
    ledger, authorizations, uses = install(tmp_path, monkeypatch, selected)
    value = proposal(ledger)
    original_decision = decision(value, decided_at=2.0)
    original_use = SignedActorUseRecord.create(
        binding=selected[0],
        proposal=value,
        decision=original_decision,
        reserved_at=10.0,
    )
    uses.reserve(original_use)
    service = GovernedRelationReviewService(
        ledger=ledger,
        policy=policy(),
        authorization_store=authorizations,
        clock=lambda: 10.0,
    )
    authorization = service._authorization(value, original_decision, now=10.0)
    authorizations.prepare(authorization, now=10.0)
    ledger.decide(original_decision)

    selected[0] = binding("8", loaded_at=20.0)
    assert cli.main(args(value)) == 0
    recovered, error = read(capsys)
    assert error is None
    assert recovered["review"]["decided_at"] == 2.0
    assert len(recovered["signed_actor_uses"]) == 2
    records = uses.list(
        owner_id="alice",
        decision_id=original_decision.decision_id,
    )
    assert len(records) == 2
    assert {record.state for record in records} == {"committed"}
    assert authorizations.get(original_decision.decision_id).state == "committed"
