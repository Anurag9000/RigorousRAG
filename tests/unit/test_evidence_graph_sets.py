from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from tools.evidence_graph_sets import (
    ExplicitCrossDocumentRelation,
    build_evidence_graph_set,
    cross_document_neighbors,
    find_cross_document_paths,
)
from tools.evidence_graph_types import (
    EvidenceGraphBatch,
    EvidenceNode,
    deterministic_node_id,
)

PROFILE = "b" * 64


def graph(doc_id, generation=1, label="Claim"):
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
        label=label,
        text=f"private text for {doc_id}",
        page_number=1,
        section="Results",
    )
    return EvidenceGraphBatch(
        owner_id="alice",
        doc_id=doc_id,
        generation=generation,
        content_sha256=(doc_id[-1] * 64 if doc_id[-1] in "abcdef" else "a" * 64),
        profile_fingerprint=PROFILE,
        nodes=(document, claim),
        edges=(),
        created_at=1.0,
    )


def view(batch, current=True):
    return SimpleNamespace(
        batch=batch,
        authoritative_current=current,
        authority_digest=(batch.doc_id[-1] * 64 if batch.doc_id[-1] in "abcdef" else "f" * 64),
    )


def node(batch, kind="claim"):
    return next(item for item in batch.nodes if item.node_type == kind)


def relation(source, target, key="r1", edge_type="supports"):
    return ExplicitCrossDocumentRelation(
        relation_key=key,
        source_doc_id=source.doc_id,
        source_node_id=node(source).node_id,
        target_doc_id=target.doc_id,
        target_node_id=node(target).node_id,
        edge_type=edge_type,
    )


def test_graph_set_is_deterministic_and_preserves_member_generations():
    a = graph("doc-a")
    b = graph("doc-b")
    first = build_evidence_graph_set(
        owner_id="alice",
        graph_set_key="review-1",
        authority_views=(view(a), view(b)),
        relations=(relation(a, b),),
        now=1.0,
    )
    second = build_evidence_graph_set(
        owner_id="alice",
        graph_set_key="review-1",
        authority_views=(view(b), view(a)),
        relations=(relation(a, b),),
        now=2.0,
    )
    assert first.graph_set_id == second.graph_set_id
    assert first.graph_set_digest == second.graph_set_digest
    assert first.created_at != second.created_at
    assert [(item.doc_id, item.generation) for item in first.members] == [
        ("doc-a", 1),
        ("doc-b", 1),
    ]
    assert first.edges[0].metadata["explicit_cross_document_relation"] is True


def test_graph_set_references_never_copy_node_text():
    a = graph("doc-a")
    b = graph("doc-b")
    value = build_evidence_graph_set(
        owner_id="alice",
        graph_set_key="review-1",
        authority_views=(view(a), view(b)),
        relations=(relation(a, b),),
        now=1.0,
    )
    rendered = json.dumps(asdict(value))
    assert "private text" not in rendered
    assert value.edges[0].source.page_number == 1
    assert value.edges[0].source.section == "Results"


def test_only_authoritative_current_unique_members_are_accepted():
    a = graph("doc-a")
    b = graph("doc-b")
    with pytest.raises(ValueError, match="authoritative current"):
        build_evidence_graph_set(
            owner_id="alice",
            graph_set_key="review",
            authority_views=(view(a), view(b, False)),
            relations=(),
        )
    with pytest.raises(ValueError, match="duplicate"):
        build_evidence_graph_set(
            owner_id="alice",
            graph_set_key="review",
            authority_views=(view(a), view(a)),
            relations=(),
        )


def test_relations_are_explicit_cross_document_and_endpoint_checked():
    a = graph("doc-a")
    b = graph("doc-b")
    with pytest.raises(ValueError, match="different documents"):
        ExplicitCrossDocumentRelation(
            "bad", "doc-a", node(a).node_id, "doc-a", node(a).node_id, "supports"
        )
    with pytest.raises(ValueError, match="unsupported"):
        ExplicitCrossDocumentRelation(
            "bad", "doc-a", node(a).node_id, "doc-b", node(b).node_id, "contains"
        )
    missing = ExplicitCrossDocumentRelation(
        "missing", "doc-a", "f" * 64, "doc-b", node(b).node_id, "supports"
    )
    with pytest.raises(ValueError, match="unknown"):
        build_evidence_graph_set(
            owner_id="alice",
            graph_set_key="review",
            authority_views=(view(a), view(b)),
            relations=(missing,),
        )


def test_generation_change_produces_a_new_set_identity():
    a1 = graph("doc-a", 1)
    a2 = graph("doc-a", 2)
    b = graph("doc-b")
    first = build_evidence_graph_set(
        owner_id="alice",
        graph_set_key="review",
        authority_views=(view(a1), view(b)),
        relations=(),
    )
    second = build_evidence_graph_set(
        owner_id="alice",
        graph_set_key="review",
        authority_views=(view(a2), view(b)),
        relations=(),
    )
    assert first.graph_set_id != second.graph_set_id


def test_bounded_explicit_cross_document_paths_and_neighbors():
    a = graph("doc-a")
    b = graph("doc-b")
    c = graph("doc-c")
    value = build_evidence_graph_set(
        owner_id="alice",
        graph_set_key="review",
        authority_views=(view(a), view(b), view(c)),
        relations=(
            relation(a, b, "a-b", "cites"),
            relation(b, c, "b-c", "supports"),
        ),
    )
    neighbors = cross_document_neighbors(
        value,
        doc_id="doc-a",
        node_id=node(a).node_id,
    )
    assert [(edge.edge_type, target.doc_id) for edge, target in neighbors] == [
        ("cites", "doc-b")
    ]
    paths = find_cross_document_paths(
        value,
        source_doc_id="doc-a",
        source_node_id=node(a).node_id,
        target_doc_id="doc-c",
        target_node_id=node(c).node_id,
    )
    assert len(paths) == 1
    assert [item.doc_id for item in paths[0].nodes] == ["doc-a", "doc-b", "doc-c"]
    assert [item.edge_type for item in paths[0].edges] == ["cites", "supports"]
    assert len(paths[0].path_digest) == 64
    assert find_cross_document_paths(
        value,
        source_doc_id="doc-a",
        source_node_id=node(a).node_id,
        target_doc_id="doc-c",
        target_node_id=node(c).node_id,
        edge_types=("cites",),
    ) == ()
