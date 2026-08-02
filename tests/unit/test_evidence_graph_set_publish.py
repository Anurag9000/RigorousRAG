from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from tools import evidence_graph_set_publish as publish
from tools.evidence_graph_relation_review import (
    CrossDocumentRelationProposal,
    RelationEndpoint,
    RelationReviewDecision,
    RelationReviewLedger,
)
from tools.evidence_graph_set_pointer import clear_current_graph_set_pointer
from tools.evidence_graph_set_store import (
    EvidenceGraphSetAuthorityReport,
    EvidenceGraphSetStore,
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
    provenance = getattr(selected, "provenance_digest", None)
    if provenance is None:
        from tools.evidence_graph_sets import _sha256

        provenance = _sha256(asdict(selected))
    return RelationEndpoint(
        doc_id=value.doc_id,
        generation=value.generation,
        graph_digest=value.graph_digest,
        node_id=selected.node_id,
        provenance_digest=provenance,
    )


class Generations:
    def __init__(self, values):
        self.values = values

    def current(self, **kwargs):
        graph_value = self.values.get(kwargs["doc_id"])
        if graph_value is None:
            return None
        return SimpleNamespace(
            owner_id="alice",
            doc_id=graph_value.doc_id,
            sequence=graph_value.generation,
            state="active",
            content_sha256=graph_value.content_sha256,
            profile_fingerprint=graph_value.profile_fingerprint,
        )


class Graphs:
    def __init__(self, values):
        self.values = values

    def current(self, **kwargs):
        return self.values.get(kwargs["doc_id"])

    def get(self, **kwargs):
        value = self.values.get(kwargs["doc_id"])
        if value is None or value.generation != kwargs["generation"]:
            raise KeyError(kwargs)
        return value


def approved_ledger(tmp_path, a, b, key="r1"):
    ledger = RelationReviewLedger(tmp_path / f"{key}.sqlite3")
    proposal = ledger.submit(
        CrossDocumentRelationProposal.create(
            owner_id="alice",
            graph_set_key="review",
            relation_key=key,
            source=endpoint(a),
            target=endpoint(b),
            edge_type="supports",
            proposer_kind="human",
            proposer_id="annotator",
            evidence_digest="e" * 64,
            created_at=1.0,
        )
    )
    ledger.decide(
        RelationReviewDecision.create(
            proposal_id=proposal.proposal_id,
            owner_id="alice",
            decision="approved",
            reviewer_id="reviewer",
            reason_code="verified",
            replacement_proposal_id=None,
            decided_at=2.0,
        )
    )
    return ledger, proposal


def test_publish_first_reviewed_set_and_idempotent_replay(tmp_path):
    a, b = graph("doc-a"), graph("doc-b")
    ledger, proposal = approved_ledger(tmp_path, a, b)
    store = EvidenceGraphSetStore(tmp_path / "sets.sqlite3")
    generations = Generations({"doc-a": a, "doc-b": b})
    graphs = Graphs({"doc-a": a, "doc-b": b})
    first = publish.publish_approved_graph_set(
        owner_id="alice",
        graph_set_key="review",
        proposal_ids=(proposal.proposal_id,),
        expected_current_set_id=None,
        ledger=ledger,
        set_store=store,
        generations=generations,
        graphs=graphs,
        now=10.0,
    )
    assert first.previous_graph_set_id is None
    assert first.pointer_changed is True
    assert first.compensation_performed is False
    assert (
        store.current(owner_id="alice", graph_set_key="review").graph_set_id
        == first.graph_set_id
    )
    second = publish.publish_approved_graph_set(
        owner_id="alice",
        graph_set_key="review",
        proposal_ids=(proposal.proposal_id,),
        expected_current_set_id=first.graph_set_id,
        ledger=ledger,
        set_store=store,
        generations=generations,
        graphs=graphs,
        now=11.0,
    )
    assert second.graph_set_id == first.graph_set_id
    assert second.pointer_changed is False


def test_publish_requires_explicit_current_pointer_expectation(tmp_path):
    a, b = graph("doc-a"), graph("doc-b")
    ledger, proposal = approved_ledger(tmp_path, a, b)
    store = EvidenceGraphSetStore(tmp_path / "sets.sqlite3")
    generations = Generations({"doc-a": a, "doc-b": b})
    graphs = Graphs({"doc-a": a, "doc-b": b})
    first = publish.publish_approved_graph_set(
        owner_id="alice",
        graph_set_key="review",
        proposal_ids=(proposal.proposal_id,),
        expected_current_set_id=None,
        ledger=ledger,
        set_store=store,
        generations=generations,
        graphs=graphs,
    )
    with pytest.raises(publish.EvidenceGraphSetPublishError, match="expectation"):
        publish.publish_approved_graph_set(
            owner_id="alice",
            graph_set_key="review",
            proposal_ids=(proposal.proposal_id,),
            expected_current_set_id=None,
            ledger=ledger,
            set_store=store,
            generations=generations,
            graphs=graphs,
        )
    assert (
        store.current(owner_id="alice", graph_set_key="review").graph_set_id
        == first.graph_set_id
    )


def test_unapproved_proposal_never_publishes(tmp_path):
    a, b = graph("doc-a"), graph("doc-b")
    ledger = RelationReviewLedger(tmp_path / "reviews.sqlite3")
    proposal = ledger.submit(
        CrossDocumentRelationProposal.create(
            owner_id="alice",
            graph_set_key="review",
            relation_key="pending",
            source=endpoint(a),
            target=endpoint(b),
            edge_type="supports",
            proposer_kind="human",
            proposer_id="annotator",
            evidence_digest="e" * 64,
            created_at=1.0,
        )
    )
    store = EvidenceGraphSetStore(tmp_path / "sets.sqlite3")
    with pytest.raises(publish.EvidenceGraphSetPublishError):
        publish.publish_approved_graph_set(
            owner_id="alice",
            graph_set_key="review",
            proposal_ids=(proposal.proposal_id,),
            expected_current_set_id=None,
            ledger=ledger,
            set_store=store,
            generations=Generations({"doc-a": a, "doc-b": b}),
            graphs=Graphs({"doc-a": a, "doc-b": b}),
        )
    assert store.current(owner_id="alice", graph_set_key="review") is None


def test_post_activation_authority_failure_clears_first_pointer(
    tmp_path, monkeypatch
):
    a, b = graph("doc-a"), graph("doc-b")
    ledger, proposal = approved_ledger(tmp_path, a, b)
    store = EvidenceGraphSetStore(tmp_path / "sets.sqlite3")
    generations = Generations({"doc-a": a, "doc-b": b})
    graphs = Graphs({"doc-a": a, "doc-b": b})
    calls = {"count": 0}

    def assessment(value, **kwargs):
        calls["count"] += 1
        current = calls["count"] == 1
        return EvidenceGraphSetAuthorityReport(
            graph_set_id=value.graph_set_id,
            graph_set_digest=value.graph_set_digest,
            authoritative_current=current,
            stale_member_doc_ids=() if current else ("doc-a",),
            missing_member_doc_ids=(),
            authority_digest=("a" if current else "b") * 64,
        )

    monkeypatch.setattr(publish, "assess_graph_set_authority", assessment)
    with pytest.raises(publish.EvidenceGraphSetPublishError) as error:
        publish.publish_approved_graph_set(
            owner_id="alice",
            graph_set_key="review",
            proposal_ids=(proposal.proposal_id,),
            expected_current_set_id=None,
            ledger=ledger,
            set_store=store,
            generations=generations,
            graphs=graphs,
        )
    assert error.value.compensation_errors == ()
    assert store.current(owner_id="alice", graph_set_key="review") is None
    assert len(store.history(owner_id="alice", graph_set_key="review")) == 1


def test_clear_current_requires_exact_compare_and_swap(tmp_path):
    a, b = graph("doc-a"), graph("doc-b")
    ledger, proposal = approved_ledger(tmp_path, a, b)
    store = EvidenceGraphSetStore(tmp_path / "sets.sqlite3")
    result = publish.publish_approved_graph_set(
        owner_id="alice",
        graph_set_key="review",
        proposal_ids=(proposal.proposal_id,),
        expected_current_set_id=None,
        ledger=ledger,
        set_store=store,
        generations=Generations({"doc-a": a, "doc-b": b}),
        graphs=Graphs({"doc-a": a, "doc-b": b}),
    )
    with pytest.raises(RuntimeError, match="concurrently"):
        clear_current_graph_set_pointer(
            store,
            owner_id="alice",
            graph_set_key="review",
            expected_current_set_id="f" * 64,
        )
    assert clear_current_graph_set_pointer(
        store,
        owner_id="alice",
        graph_set_key="review",
        expected_current_set_id=result.graph_set_id,
    ) is True
    assert store.current(owner_id="alice", graph_set_key="review") is None


def test_post_activation_failure_restores_previous_pointer(tmp_path, monkeypatch):
    a, b = graph("doc-a"), graph("doc-b")
    ledger, first_proposal = approved_ledger(tmp_path, a, b, key="first")
    second_proposal = ledger.submit(
        CrossDocumentRelationProposal.create(
            owner_id="alice",
            graph_set_key="review",
            relation_key="second",
            source=endpoint(a),
            target=endpoint(b),
            edge_type="contradicts",
            proposer_kind="human",
            proposer_id="annotator",
            evidence_digest="d" * 64,
            created_at=2.0,
        )
    )
    ledger.decide(
        RelationReviewDecision.create(
            proposal_id=second_proposal.proposal_id,
            owner_id="alice",
            decision="approved",
            reviewer_id="reviewer",
            reason_code="verified",
            replacement_proposal_id=None,
            decided_at=3.0,
        )
    )
    store = EvidenceGraphSetStore(tmp_path / "sets.sqlite3")
    generations = Generations({"doc-a": a, "doc-b": b})
    graphs = Graphs({"doc-a": a, "doc-b": b})
    first = publish.publish_approved_graph_set(
        owner_id="alice",
        graph_set_key="review",
        proposal_ids=(first_proposal.proposal_id,),
        expected_current_set_id=None,
        ledger=ledger,
        set_store=store,
        generations=generations,
        graphs=graphs,
    )
    calls = {"count": 0}

    def assessment(value, **kwargs):
        calls["count"] += 1
        current = calls["count"] == 1
        return EvidenceGraphSetAuthorityReport(
            graph_set_id=value.graph_set_id,
            graph_set_digest=value.graph_set_digest,
            authoritative_current=current,
            stale_member_doc_ids=() if current else ("doc-b",),
            missing_member_doc_ids=(),
            authority_digest=("a" if current else "b") * 64,
        )

    monkeypatch.setattr(publish, "assess_graph_set_authority", assessment)
    with pytest.raises(publish.EvidenceGraphSetPublishError) as error:
        publish.publish_approved_graph_set(
            owner_id="alice",
            graph_set_key="review",
            proposal_ids=(second_proposal.proposal_id,),
            expected_current_set_id=first.graph_set_id,
            ledger=ledger,
            set_store=store,
            generations=generations,
            graphs=graphs,
        )
    assert error.value.compensation_errors == ()
    assert (
        store.current(owner_id="alice", graph_set_key="review").graph_set_id
        == first.graph_set_id
    )
