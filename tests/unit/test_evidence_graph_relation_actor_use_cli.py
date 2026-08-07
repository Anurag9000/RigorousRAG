from __future__ import annotations

import json

from tools import evidence_graph_relation_actor_use_cli as cli
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


def record():
    proposal = CrossDocumentRelationProposal.create(
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
    decision = RelationReviewDecision.create(
        proposal_id=proposal.proposal_id,
        owner_id="alice",
        decision="approved",
        reviewer_id="reviewer-1",
        reason_code="verified",
        replacement_proposal_id=None,
        decided_at=2.0,
    )
    binding = ReviewActorBinding.create(
        actor_id="reviewer-1",
        binding_method="hmac_assertion",
        assertion_digest="7" * 64,
        issuer="review-control-plane",
        expires_at=100.0,
        loaded_at=10.0,
    )
    return SignedActorUseRecord.create(
        binding=binding,
        proposal=proposal,
        decision=decision,
        reserved_at=10.0,
    )


def read(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def test_status_and_list_are_read_only_and_secret_free(
    tmp_path, monkeypatch, capsys
):
    store = SignedActorUseStore(tmp_path / "uses.sqlite3")
    selected = record()
    store.reserve(selected)
    store.mark_committed(
        selected.assertion_digest,
        decision_id=selected.decision_id,
        now=20.0,
    )
    monkeypatch.setattr(cli, "get_signed_actor_use_store", lambda: store)

    assert cli.main(["status", selected.assertion_digest]) == 0
    status, error = read(capsys)
    assert error is None
    assert status["state"] == "committed"
    assert status["assertion_digest"] == selected.assertion_digest
    assert status["contains_signature"] is False
    assert status["contains_key_material"] is False
    assert status["contains_source_text"] is False
    assert status["mutation_performed"] is False

    assert cli.main([
        "list",
        "--owner-id", "alice",
        "--decision-id", selected.decision_id,
        "--state", "committed",
    ]) == 0
    listing, error = read(capsys)
    assert error is None
    assert listing["count"] == 1
    assert listing["actor_uses"][0]["use_digest"] == selected.use_digest
    rendered = json.dumps(listing).lower()
    assert listing["contains_signature"] is False
    assert listing["contains_key_material"] is False
    assert listing["contains_source_text"] is False
    assert "private text" not in rendered


def test_missing_and_invalid_actor_uses_are_bounded(
    tmp_path, monkeypatch, capsys
):
    store = SignedActorUseStore(tmp_path / "uses.sqlite3")
    monkeypatch.setattr(cli, "get_signed_actor_use_store", lambda: store)

    assert cli.main(["status", "f" * 64]) == 1
    output, error = read(capsys)
    assert output is None
    assert error == {"error": "not_found"}

    assert cli.main(["list", "--owner-id", "alice", "--limit", "0"]) == 2
    output, error = read(capsys)
    assert output is None
    assert error == {"error": "invalid_or_unavailable"}
