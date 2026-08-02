from __future__ import annotations

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


def proposal(ledger: RelationReviewLedger):
    return ledger.submit(
        CrossDocumentRelationProposal.create(
            owner_id="alice",
            graph_set_key="review",
            relation_key="relation",
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
            evidence_digest="1" * 64,
            created_at=1.0,
        )
    )


def policy():
    return RelationReviewPolicy.from_mapping(
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


def decision(value, *, decided_at: float, reason="verified"):
    return RelationReviewDecision.create(
        proposal_id=value.proposal_id,
        owner_id="alice",
        decision="approved",
        reviewer_id="reviewer",
        reason_code=reason,
        replacement_proposal_id=None,
        decided_at=decided_at,
    )


def test_stable_terminal_identity_replays_across_audit_timestamps(tmp_path):
    ledger = RelationReviewLedger(tmp_path / "reviews.sqlite3")
    authorizations = RelationReviewAuthorizationStore(
        tmp_path / "authorizations.sqlite3"
    )
    selected = proposal(ledger)
    service = GovernedRelationReviewService(
        ledger=ledger,
        policy=policy(),
        authorization_store=authorizations,
        clock=lambda: 10.0,
    )
    original = decision(selected, decided_at=2.0)
    authorization = service._authorization(selected, original, now=10.0)
    authorizations.prepare(authorization, now=10.0)
    ledger.decide(original)

    replay = decision(selected, decided_at=999.0)
    stored, receipt = service.decide(replay)

    assert replay.decision_id == original.decision_id
    assert stored == original
    assert stored.decided_at == 2.0
    assert receipt.state == "committed"


def test_stable_replay_refuses_changed_governed_content(tmp_path):
    ledger = RelationReviewLedger(tmp_path / "reviews.sqlite3")
    authorizations = RelationReviewAuthorizationStore(
        tmp_path / "authorizations.sqlite3"
    )
    selected = proposal(ledger)
    service = GovernedRelationReviewService(
        ledger=ledger,
        policy=policy(),
        authorization_store=authorizations,
        clock=lambda: 10.0,
    )
    original = decision(selected, decided_at=2.0)
    service.decide(original)

    changed = decision(selected, decided_at=3.0, reason="different")
    try:
        service.decide(changed)
    except RuntimeError as exc:
        assert "different terminal decision" in str(exc)
    else:
        raise AssertionError("changed terminal decision replay was accepted")
