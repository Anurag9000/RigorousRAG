from __future__ import annotations

import sqlite3

import pytest

from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_relation_actor_use_store import (
    SignedActorUseRecord,
    SignedActorUseStore,
)
from tools.evidence_graph_relation_review import (
    CrossDocumentRelationProposal,
    RelationEndpoint,
    RelationReviewDecision,
)


def proposal(digit="1"):
    return CrossDocumentRelationProposal.create(
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


def decision(value, reason="verified"):
    return RelationReviewDecision.create(
        proposal_id=value.proposal_id,
        owner_id="alice",
        decision="approved",
        reviewer_id="reviewer-1",
        reason_code=reason,
        replacement_proposal_id=None,
        decided_at=2.0,
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


def test_reserve_commit_and_exact_replay(tmp_path):
    store = SignedActorUseStore(tmp_path / "uses.sqlite3")
    selected_proposal = proposal()
    selected_decision = decision(selected_proposal)
    value = SignedActorUseRecord.create(
        binding=binding(),
        proposal=selected_proposal,
        decision=selected_decision,
        reserved_at=10.0,
    )

    reserved = store.reserve(value)
    replay = store.reserve(value)
    committed = store.mark_committed(
        value.assertion_digest,
        decision_id=selected_decision.decision_id,
        now=20.0,
    )
    repeated = store.mark_committed(
        value.assertion_digest,
        decision_id=selected_decision.decision_id,
        now=30.0,
    )

    assert reserved.state == replay.state == "reserved"
    assert committed.state == repeated.state == "committed"
    assert committed.committed_at == repeated.committed_at == 20.0
    assert store.get(value.assertion_digest) == committed
    assert store.list(owner_id="alice", decision_id=selected_decision.decision_id) == (
        committed,
    )


def test_assertion_cannot_be_reserved_for_another_decision(tmp_path):
    store = SignedActorUseStore(tmp_path / "uses.sqlite3")
    first_proposal = proposal("1")
    first_decision = decision(first_proposal)
    first = SignedActorUseRecord.create(
        binding=binding(),
        proposal=first_proposal,
        decision=first_decision,
        reserved_at=10.0,
    )
    store.reserve(first)

    second_proposal = proposal("2")
    second_decision = decision(second_proposal)
    second = SignedActorUseRecord.create(
        binding=binding(),
        proposal=second_proposal,
        decision=second_decision,
        reserved_at=11.0,
    )
    with pytest.raises(RuntimeError, match="another decision"):
        store.reserve(second)


def test_scope_expiry_and_direct_bindings_fail_closed(tmp_path):
    selected_proposal = proposal()
    selected_decision = decision(selected_proposal)
    with pytest.raises(PermissionError, match="expired"):
        SignedActorUseRecord.create(
            binding=binding(loaded_at=99.0),
            proposal=selected_proposal,
            decision=selected_decision,
            reserved_at=101.0,
        )

    direct = ReviewActorBinding.create(
        actor_id="reviewer-1",
        binding_method="process_environment",
        loaded_at=1.0,
    )
    with pytest.raises(ValueError, match="signed actor"):
        SignedActorUseRecord.create(
            binding=direct,
            proposal=selected_proposal,
            decision=selected_decision,
            reserved_at=1.0,
        )

    mismatched = RelationReviewDecision.create(
        proposal_id=selected_proposal.proposal_id,
        owner_id="alice",
        decision="approved",
        reviewer_id="other-reviewer",
        reason_code="verified",
        replacement_proposal_id=None,
        decided_at=2.0,
    )
    with pytest.raises(PermissionError, match="scope"):
        SignedActorUseRecord.create(
            binding=binding(),
            proposal=selected_proposal,
            decision=mismatched,
            reserved_at=10.0,
        )


def test_store_detects_payload_and_database_identity_tampering(tmp_path):
    store = SignedActorUseStore(tmp_path / "uses.sqlite3")
    selected_proposal = proposal()
    selected_decision = decision(selected_proposal)
    value = SignedActorUseRecord.create(
        binding=binding(),
        proposal=selected_proposal,
        decision=selected_decision,
        reserved_at=10.0,
    )
    store.reserve(value)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE signed_review_actor_uses SET payload_json='{}' "
            "WHERE assertion_digest=?",
            (value.assertion_digest,),
        )
    with pytest.raises(RuntimeError, match="corrupt"):
        store.get(value.assertion_digest)

    path = tmp_path / "identity.sqlite3"
    guarded = SignedActorUseStore(path)
    path.rename(tmp_path / "old.sqlite3")
    path.write_bytes(b"")
    with pytest.raises(RuntimeError, match="identity changed"):
        guarded.list(owner_id="alice")
