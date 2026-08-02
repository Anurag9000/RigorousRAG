from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tools import evidence_graph_rag as rag
from tools.evidence_graph_set_store import EvidenceGraphSetAuthorityReport
from tools.evidence_graph_sets import (
    ExplicitCrossDocumentRelation,
    build_evidence_graph_set,
)
from tools.evidence_graph_types import (
    EvidenceEdge,
    EvidenceGraphBatch,
    EvidenceNode,
    deterministic_edge_id,
    deterministic_node_id,
)

A = "a" * 64
B = "b" * 64


def node(doc_id, generation, node_type, key, label, text=""):
    return EvidenceNode(
        node_id=deterministic_node_id(
            owner_id="alice",
            doc_id=doc_id,
            generation=generation,
            node_type=node_type,
            natural_key=key,
        ),
        owner_id="alice",
        doc_id=doc_id,
        generation=generation,
        node_type=node_type,
        natural_key=key,
        label=label,
        text=text,
        page_number=1,
        section="Results",
    )


def graph(doc_id, generation=1, with_internal=False):
    document = node(doc_id, generation, "document", "document", doc_id)
    claim = node(
        doc_id,
        generation,
        "claim",
        "claim",
        f"Claim {doc_id}",
        f"alpha evidence {doc_id}",
    )
    method = node(
        doc_id,
        generation,
        "method",
        "method",
        f"Method {doc_id}",
        f"method details {doc_id}",
    )
    edges = ()
    if with_internal:
        edge = EvidenceEdge(
            edge_id=deterministic_edge_id(
                owner_id="alice",
                doc_id=doc_id,
                generation=generation,
                source_node_id=claim.node_id,
                target_node_id=method.node_id,
                edge_type="uses_method",
                relation_key="claim-method",
            ),
            owner_id="alice",
            doc_id=doc_id,
            generation=generation,
            source_node_id=claim.node_id,
            target_node_id=method.node_id,
            edge_type="uses_method",
            relation_key="claim-method",
            weight=0.5,
        )
        edges = (edge,)
    return EvidenceGraphBatch(
        owner_id="alice",
        doc_id=doc_id,
        generation=generation,
        content_sha256=A,
        profile_fingerprint=B,
        nodes=(document, claim, method),
        edges=edges,
        created_at=1.0,
    )


def claim(batch):
    return next(value for value in batch.nodes if value.node_type == "claim")


def method(batch):
    return next(value for value in batch.nodes if value.node_type == "method")


def view(batch):
    return SimpleNamespace(
        batch=batch,
        authoritative_current=True,
        authority_digest=(batch.doc_id[-1] * 64),
    )


def graph_set(a, b):
    return build_evidence_graph_set(
        owner_id="alice",
        graph_set_key="review",
        authority_views=(view(a), view(b)),
        relations=(
            ExplicitCrossDocumentRelation(
                relation_key="a-b",
                source_doc_id=a.doc_id,
                source_node_id=claim(a).node_id,
                target_doc_id=b.doc_id,
                target_node_id=claim(b).node_id,
                edge_type="supports",
                weight=0.75,
            ),
        ),
        now=1.0,
    )


class Graphs:
    def __init__(self, values):
        self.values = values

    def get(self, **kwargs):
        return self.values[kwargs["doc_id"]]


def authority(value, current=True):
    return EvidenceGraphSetAuthorityReport(
        graph_set_id=value.graph_set_id,
        graph_set_digest=value.graph_set_digest,
        authoritative_current=current,
        stale_member_doc_ids=() if current else ("doc-a",),
        missing_member_doc_ids=(),
        authority_digest="f" * 64,
    )


def hit(value, score=10.0, terms=("alpha",)):
    return SimpleNamespace(node=value, score=score, matched_terms=terms)


def test_lexical_selection_is_bounded_and_generates_no_answer_or_citations(monkeypatch):
    a, b = graph("doc-a"), graph("doc-b")
    value = graph_set(a, b)

    def search(batch, query, **kwargs):
        return (
            hit(claim(batch), score=10.0 if batch.doc_id == "doc-a" else 8.0),
        )

    monkeypatch.setattr(rag, "search_nodes", search)
    monkeypatch.setattr(rag, "outgoing_neighbors", lambda *args, **kwargs: ())
    result = rag.select_graph_set_evidence(
        value,
        authority(value),
        query="alpha",
        graphs=Graphs({"doc-a": a, "doc-b": b}),
        max_cross_depth=0,
    )
    assert [item.doc_id for item in result.items] == ["doc-a", "doc-b"]
    assert all(item.origin == "lexical" for item in result.items)
    assert result.query_digest != "alpha"
    assert result.answer_generated is False
    assert result.citation_conversion_performed is False
    assert result.abstained is False
    assert len(result.selection_digest) == 64


def test_cross_document_expansion_materializes_reviewed_target(monkeypatch):
    a, b = graph("doc-a"), graph("doc-b")
    value = graph_set(a, b)

    def search(batch, query, **kwargs):
        return (hit(claim(a)),) if batch.doc_id == "doc-a" else ()

    monkeypatch.setattr(rag, "search_nodes", search)
    monkeypatch.setattr(rag, "outgoing_neighbors", lambda *args, **kwargs: ())
    result = rag.select_graph_set_evidence(
        value,
        authority(value),
        query="alpha",
        graphs=Graphs({"doc-a": a, "doc-b": b}),
        max_cross_depth=2,
    )
    assert {(item.doc_id, item.origin) for item in result.items} == {
        ("doc-a", "lexical"),
        ("doc-b", "cross_document"),
    }
    expanded = next(item for item in result.items if item.origin == "cross_document")
    assert expanded.lineage_step_digests
    assert len(result.traversals) == 1
    assert result.traversals[0].edge_type == "supports"


def test_within_document_expansion_uses_only_stored_edges(monkeypatch):
    a, b = graph("doc-a", with_internal=True), graph("doc-b")
    value = graph_set(a, b)

    def search(batch, query, **kwargs):
        return (hit(claim(a)),) if batch.doc_id == "doc-a" else ()

    def neighbors(batch, node_id, **kwargs):
        if batch.doc_id == "doc-a" and node_id == claim(a).node_id:
            return ((a.edges[0], method(a)),)
        return ()

    monkeypatch.setattr(rag, "search_nodes", search)
    monkeypatch.setattr(rag, "outgoing_neighbors", neighbors)
    result = rag.select_graph_set_evidence(
        value,
        authority(value),
        query="alpha",
        graphs=Graphs({"doc-a": a, "doc-b": b}),
        max_cross_depth=0,
    )
    assert {(item.node_type, item.origin) for item in result.items} == {
        ("claim", "lexical"),
        ("method", "within_document"),
    }
    assert result.traversals[0].traversal_kind == "within_document"


def test_lexical_hit_wins_over_duplicate_cross_expansion(monkeypatch):
    a, b = graph("doc-a"), graph("doc-b")
    value = graph_set(a, b)

    def search(batch, query, **kwargs):
        return (hit(claim(batch), score=10.0),)

    monkeypatch.setattr(rag, "search_nodes", search)
    monkeypatch.setattr(rag, "outgoing_neighbors", lambda *args, **kwargs: ())
    result = rag.select_graph_set_evidence(
        value,
        authority(value),
        query="alpha",
        graphs=Graphs({"doc-a": a, "doc-b": b}),
    )
    b_item = next(item for item in result.items if item.doc_id == "doc-b")
    assert b_item.origin == "lexical"
    assert b_item.lineage_step_digests == ()


def test_stale_authority_and_member_graph_drift_fail_closed(monkeypatch):
    a, b = graph("doc-a"), graph("doc-b")
    value = graph_set(a, b)
    monkeypatch.setattr(rag, "search_nodes", lambda *args, **kwargs: ())
    monkeypatch.setattr(rag, "outgoing_neighbors", lambda *args, **kwargs: ())
    with pytest.raises(RuntimeError, match="not authoritative"):
        rag.select_graph_set_evidence(
            value,
            authority(value, False),
            query="alpha",
            graphs=Graphs({"doc-a": a, "doc-b": b}),
        )
    fake_b = SimpleNamespace(**vars(b))
    fake_b.graph_digest = "0" * 64
    with pytest.raises(RuntimeError, match="differs"):
        rag.select_graph_set_evidence(
            value,
            authority(value),
            query="alpha",
            graphs=Graphs({"doc-a": a, "doc-b": fake_b}),
        )


def test_cross_edge_target_provenance_mismatch_fails_closed(monkeypatch):
    a, b = graph("doc-a"), graph("doc-b")
    value = graph_set(a, b)
    changed_claim = replace(claim(b), label="Changed claim")
    fake_b = SimpleNamespace(
        owner_id=b.owner_id,
        doc_id=b.doc_id,
        generation=b.generation,
        content_sha256=b.content_sha256,
        profile_fingerprint=b.profile_fingerprint,
        graph_digest=b.graph_digest,
        nodes=(b.nodes[0], changed_claim, b.nodes[2]),
        edges=b.edges,
    )

    def search(batch, query, **kwargs):
        return (hit(claim(a)),) if batch.doc_id == "doc-a" else ()

    monkeypatch.setattr(rag, "search_nodes", search)
    monkeypatch.setattr(rag, "outgoing_neighbors", lambda *args, **kwargs: ())
    with pytest.raises(RuntimeError, match="provenance changed"):
        rag.select_graph_set_evidence(
            value,
            authority(value),
            query="alpha",
            graphs=Graphs({"doc-a": a, "doc-b": fake_b}),
        )


def test_empty_search_abstains_without_traversal(monkeypatch):
    a, b = graph("doc-a"), graph("doc-b")
    value = graph_set(a, b)
    monkeypatch.setattr(rag, "search_nodes", lambda *args, **kwargs: ())
    monkeypatch.setattr(rag, "outgoing_neighbors", lambda *args, **kwargs: ())
    result = rag.select_graph_set_evidence(
        value,
        authority(value),
        query="no-match",
        graphs=Graphs({"doc-a": a, "doc-b": b}),
    )
    assert result.abstained is True
    assert result.items == ()
    assert result.traversals == ()


def test_unsupported_filters_fail_closed(monkeypatch):
    a, b = graph("doc-a"), graph("doc-b")
    value = graph_set(a, b)
    monkeypatch.setattr(rag, "search_nodes", lambda *args, **kwargs: ())
    monkeypatch.setattr(rag, "outgoing_neighbors", lambda *args, **kwargs: ())
    with pytest.raises(ValueError, match="unsupported"):
        rag.select_graph_set_evidence(
            value,
            authority(value),
            query="alpha",
            graphs=Graphs({"doc-a": a, "doc-b": b}),
            cross_edge_types=("invented",),
        )


def test_current_selection_rechecks_graph_set_after_selection(monkeypatch):
    a, b = graph("doc-a"), graph("doc-b")
    value = graph_set(a, b)
    changed = graph_set(graph("doc-a", generation=2), b)

    class SetStore:
        def __init__(self):
            self.calls = 0

        def resolve_current(self, **kwargs):
            self.calls += 1
            if self.calls < 3:
                return value, authority(value)
            return changed, authority(changed)

    monkeypatch.setattr(rag, "search_nodes", lambda *args, **kwargs: ())
    monkeypatch.setattr(rag, "outgoing_neighbors", lambda *args, **kwargs: ())
    with pytest.raises(RuntimeError, match="during evidence selection"):
        rag.select_current_graph_set_evidence(
            owner_id="alice",
            graph_set_key="review",
            query="alpha",
            set_store=SetStore(),
            generations=object(),
            graphs=Graphs({"doc-a": a, "doc-b": b}),
        )
