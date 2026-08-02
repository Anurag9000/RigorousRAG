from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from tools.evidence_graph_relation_review import (
    CrossDocumentRelationProposal,
    RelationEndpoint,
    RelationReviewDecision,
    RelationReviewLedger,
    approved_relations,
)
from tools.evidence_graph_types import (
    EvidenceGraphBatch,
    EvidenceNode,
    deterministic_node_id,
)

A = "a" * 64
B = "b" * 64


def graph(doc_id, generation=1):
    document = EvidenceNode(
        node_id=deterministic_node_id(
            owner_id="alice",
            doc_id=doc_id,
            generation=generation,
            node_type="document",
            natural_key="document",
        ),
        owner_id="alice",
        doc_id=doc_id,
        generation=generation,
        node_type="document",
        natural_key="document",
        label=doc_id,
    )
    claim = EvidenceNode(
        node_id=deterministic_node_id(
            owner_id="alice",
            doc_id=doc_id,
            generation=generation,
            node_type="claim",
            natural_key="claim",
        ),
        owner_id="alice",
        doc_id=doc_id,
        generation=generation,
        node_type="claim",
        natural_key="claim",
        label=f"claim-{doc_id}",
    )
    return EvidenceGraphBatch(
        owner_id="alice",
        doc_id=doc_id,
        generation=generation,
        content_sha256=A,
        profile_fingerprint=B,
        nodes=(document, claim),
        edges=(),
        created_at=1.0,
    )


def claim(value):
    return next(item for item in value.nodes if item.node_type == "claim")


def endpoint(value):
    selected = claim(value)
    return RelationEndpoint(
        doc_id=value.doc_id,
        generation=value.generation,
        graph_digest=value.graph_digest,
        node_id=selected.node_id,
        provenance_digest=(
            getattr(selected, "provenance_digest", None)
            or __import__("hashlib").sha256(
                __import__("json").dumps(
                    asdict(selected),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
        ),
    )


def proposal(a, b, key="a-b", proposer_kind="human"):
    kwargs = {}
    if proposer_kind != "human":
        kwargs = {"extractor_name": "extractor", "extractor_version": "v1"}
    return CrossDocumentRelationProposal.create(
        owner_id="alice",
        graph_set_key="review",
        relation_key=key,
        source=endpoint(a),
        target=endpoint(b),
        edge_type="supports",
        proposer_kind=proposer_kind,
        proposer_id="annotator-1",
        evidence_digest="e" * 64,
        metadata={"review_batch": "batch-1"},
        created_at=1.0,
        **kwargs,
    )


def view(value, current=True):
    return SimpleNamespace(
        batch=value,
        authoritative_current=current,
        authority_digest="f" * 64,
    )


def test_proposals_are_deterministic_text_free_and_extractor_governed():
    a = graph("doc-a")
    b = graph("doc-b")
    first = proposal(a, b)
    second = proposal(a, b)
    assert first.proposal_id == second.proposal_id
    assert len(first.proposal_digest) == 64
    assert "text" not in str(asdict(first)).lower()
    with pytest.raises(ValueError, match="require extractor"):
        CrossDocumentRelationProposal.create(
            owner_id="alice",
            graph_set_key="review",
            relation_key="bad",
            source=endpoint(a),
            target=endpoint(b),
            edge_type="supports",
            proposer_kind="model",
            proposer_id="model-1",
            evidence_digest="e" * 64,
            created_at=1.0,
        )


def test_ledger_submission_approval_and_immutability(tmp_path):
    a = graph("doc-a")
    b = graph("doc-b")
    ledger = RelationReviewLedger(tmp_path / "reviews.sqlite3")
    value = proposal(a, b)
    assert ledger.submit(value) == value
    assert ledger.submit(value) == value
    decision = RelationReviewDecision.create(
        proposal_id=value.proposal_id,
        owner_id="alice",
        decision="approved",
        reviewer_id="reviewer-1",
        reason_code="evidence_verified",
        replacement_proposal_id=None,
        decided_at=2.0,
    )
    assert ledger.decide(decision) == decision
    assert ledger.decide(decision) == decision
    conflicting = RelationReviewDecision.create(
        proposal_id=value.proposal_id,
        owner_id="alice",
        decision="rejected",
        reviewer_id="reviewer-2",
        reason_code="unsupported",
        replacement_proposal_id=None,
        decided_at=3.0,
    )
    with pytest.raises(RuntimeError, match="different terminal"):
        ledger.decide(conflicting)
    listed = ledger.list(owner_id="alice", graph_set_key="review", decision="approved")
    assert listed == ((value, decision),)


def test_supersession_requires_an_existing_replacement(tmp_path):
    a = graph("doc-a")
    b = graph("doc-b")
    ledger = RelationReviewLedger(tmp_path / "reviews.sqlite3")
    old = ledger.submit(proposal(a, b, "old"))
    replacement = proposal(a, b, "replacement")
    decision = RelationReviewDecision.create(
        proposal_id=old.proposal_id,
        owner_id="alice",
        decision="superseded",
        reviewer_id="reviewer",
        reason_code="corrected_relation",
        replacement_proposal_id=replacement.proposal_id,
        decided_at=2.0,
    )
    with pytest.raises(RuntimeError, match="replacement"):
        ledger.decide(decision)
    ledger.submit(replacement)
    assert ledger.decide(decision).replacement_proposal_id == replacement.proposal_id


def test_only_approved_current_endpoints_convert_to_relations(tmp_path):
    a = graph("doc-a")
    b = graph("doc-b")
    ledger = RelationReviewLedger(tmp_path / "reviews.sqlite3")
    value = ledger.submit(proposal(a, b))
    with pytest.raises(RuntimeError, match="not approved"):
        approved_relations(
            owner_id="alice",
            graph_set_key="review",
            proposal_ids=(value.proposal_id,),
            authority_views=(view(a), view(b)),
            ledger=ledger,
        )
    ledger.decide(
        RelationReviewDecision.create(
            proposal_id=value.proposal_id,
            owner_id="alice",
            decision="approved",
            reviewer_id="reviewer",
            reason_code="verified",
            replacement_proposal_id=None,
            decided_at=2.0,
        )
    )
    relations = approved_relations(
        owner_id="alice",
        graph_set_key="review",
        proposal_ids=(value.proposal_id,),
        authority_views=(view(a), view(b)),
        ledger=ledger,
    )
    assert len(relations) == 1
    assert relations[0].metadata["proposal_id"] == value.proposal_id
    moved = graph("doc-a", generation=2)
    with pytest.raises(RuntimeError, match="stale or missing"):
        approved_relations(
            owner_id="alice",
            graph_set_key="review",
            proposal_ids=(value.proposal_id,),
            authority_views=(view(moved), view(b)),
            ledger=ledger,
        )


def test_rejected_and_pending_filters_are_explicit(tmp_path):
    a = graph("doc-a")
    b = graph("doc-b")
    ledger = RelationReviewLedger(tmp_path / "reviews.sqlite3")
    pending = ledger.submit(proposal(a, b, "pending"))
    rejected = ledger.submit(proposal(a, b, "rejected"))
    ledger.decide(
        RelationReviewDecision.create(
            proposal_id=rejected.proposal_id,
            owner_id="alice",
            decision="rejected",
            reviewer_id="reviewer",
            reason_code="insufficient_evidence",
            replacement_proposal_id=None,
            decided_at=2.0,
        )
    )
    assert ledger.list(owner_id="alice", graph_set_key="review", decision="pending") == (
        (pending, None),
    )
    assert ledger.list(owner_id="alice", graph_set_key="review", decision="rejected")[0][0] == rejected
