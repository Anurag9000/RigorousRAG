from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tools.evidence_graph_analysis import analyze_evidence_graph
from tools.evidence_graph_builder import (
    ExplicitGraphRelation,
    GraphAnnotation,
    build_evidence_graph,
)
from tools.evidence_graph_retrieval import find_paths, outgoing_neighbors, search_nodes
from tools.evidence_graph_runtime import (
    clear_evidence_graph_store_cache,
    get_evidence_graph_store,
)
from tools.evidence_graph_store import EvidenceGraphStore
from tools.evidence_graph_types import EvidencePath

PROFILE = "b" * 64


def document():
    text = "Finalized privacy-safe text."
    return SimpleNamespace(
        id="doc-1",
        title="Scientific paper",
        filename="paper.pdf",
        text=text,
        metadata={"content_sha256": hashlib.sha256(text.encode()).hexdigest()},
        sections=[
            SimpleNamespace(
                title="Methods",
                content="We used an explicit randomized protocol.",
                page_number=1,
                metadata={"field_type": "body"},
            ),
            SimpleNamespace(
                title="Results",
                content="The measured outcome increased.",
                page_number=2,
                metadata={},
            ),
        ],
    )


def annotations():
    return (
        GraphAnnotation(
            "claim-1",
            "claim",
            "Primary outcome increased",
            "The measured outcome increased.",
            section_index=1,
        ),
        GraphAnnotation("method-1", "method", "Randomized protocol", section_index=0),
        GraphAnnotation("entity-1", "entity", "Population A", section_index=0),
        GraphAnnotation("citation-1", "citation", "Reference 12", section_index=1),
    )


def graph(generation=5, *, conflict=False, now=1.0):
    extra_annotations = ()
    extra_relations = ()
    if conflict:
        extra_annotations = (
            GraphAnnotation("citation-2", "citation", "Reference 13", section_index=1),
        )
        extra_relations = (
            ExplicitGraphRelation(
                "claim-contradicted-by-reference",
                "citation-2",
                "claim-1",
                "contradicts",
            ),
        )
    return build_evidence_graph(
        document(),
        owner_id="alice",
        generation=generation,
        profile_fingerprint=PROFILE,
        annotations=annotations() + extra_annotations,
        relations=(
            ExplicitGraphRelation(
                "claim-supported-by-results",
                "section:1",
                "claim-1",
                "supports",
            ),
            ExplicitGraphRelation(
                "claim-uses-method",
                "claim-1",
                "method-1",
                "uses_method",
            ),
            ExplicitGraphRelation(
                "claim-cites-reference",
                "claim-1",
                "citation-1",
                "cites",
            ),
        )
        + extra_relations,
        now=now,
    )


def node_by_key(value, key):
    natural = key if key.startswith("section:") or key == "document" else f"annotation:{key}"
    return next(node for node in value.nodes if node.natural_key == natural)


def test_explicit_builder_is_deterministic_and_preserves_provenance():
    first = graph(now=1)
    second = graph(now=2)
    assert first.graph_digest == second.graph_digest
    assert first.created_at != second.created_at
    assert len(first.nodes) == 7
    assert {edge.edge_type for edge in first.edges if edge.metadata.get("explicit_relation")} == {
        "supports",
        "uses_method",
        "cites",
    }
    claim = node_by_key(first, "claim-1")
    assert claim.page_number == 2 and claim.section == "Results"


def test_text_does_not_create_implicit_support_or_contradiction():
    value = build_evidence_graph(
        document(),
        owner_id="alice",
        generation=5,
        profile_fingerprint=PROFILE,
        annotations=(
            GraphAnnotation("claim-a", "claim", "Outcome increased", section_index=1),
            GraphAnnotation("claim-b", "claim", "Outcome did not increase", section_index=1),
        ),
        now=1,
    )
    assert {edge.edge_type for edge in value.edges} == {"contains"}
    assert analyze_evidence_graph(value).claim_clusters == ()


def test_builder_rejects_bad_hash_unknown_relations_and_duplicate_keys():
    changed = document()
    changed.metadata["content_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="content hash"):
        build_evidence_graph(
            changed,
            owner_id="alice",
            generation=5,
            profile_fingerprint=PROFILE,
        )
    with pytest.raises(ValueError, match="unknown"):
        build_evidence_graph(
            document(),
            owner_id="alice",
            generation=5,
            profile_fingerprint=PROFILE,
            annotations=annotations(),
            relations=(
                ExplicitGraphRelation("missing", "claim-1", "unknown", "supports"),
            ),
        )
    with pytest.raises(ValueError, match="unique"):
        build_evidence_graph(
            document(),
            owner_id="alice",
            generation=5,
            profile_fingerprint=PROFILE,
            annotations=(annotations()[0], annotations()[0]),
        )
    with pytest.raises(ValueError, match="explicit edge_type"):
        ExplicitGraphRelation("implicit", "document", "section:0", "contains")


def test_lexical_search_filters_and_ranks_without_semantic_inference():
    value = graph()
    results = search_nodes(value, "primary outcome increased")
    assert results[0].node.natural_key == "annotation:claim-1"
    methods = search_nodes(value, "randomized protocol", node_types=("method",))
    assert len(methods) == 1 and methods[0].node.natural_key == "annotation:method-1"
    assert search_nodes(value, "randomized protocol", node_types=("claim",)) == ()
    with pytest.raises(ValueError, match="query"):
        search_nodes(value, " ")


def test_directed_paths_are_cycle_safe_and_type_filterable():
    value = graph()
    source = node_by_key(value, "document")
    target = node_by_key(value, "method-1")
    paths = find_paths(
        value,
        source_node_id=source.node_id,
        target_node_id=target.node_id,
    )
    assert [node.natural_key for node in paths[0].nodes] == [
        "document",
        "section:1",
        "annotation:claim-1",
        "annotation:method-1",
    ]
    assert [edge.edge_type for edge in paths[0].edges] == [
        "contains",
        "contains",
        "uses_method",
    ]
    assert find_paths(
        value,
        source_node_id=source.node_id,
        target_node_id=target.node_id,
        edge_types=("contains",),
    ) == ()
    assert isinstance(paths[0], EvidencePath)


def test_outgoing_neighbors_are_deterministic():
    value = graph()
    claim = node_by_key(value, "claim-1")
    assert [(edge.edge_type, node.node_type) for edge, node in outgoing_neighbors(value, claim.node_id)] == [
        ("cites", "citation"),
        ("uses_method", "method"),
    ]


def test_explicit_conflict_clusters_preserve_incoming_provenance():
    value = graph(conflict=True)
    analysis = analyze_evidence_graph(value)
    assert analysis.edge_counts["supports"] == 1
    assert analysis.edge_counts["contradicts"] == 1
    assert len(analysis.claim_clusters) == 1
    cluster = analysis.claim_clusters[0]
    assert cluster.claim.natural_key == "annotation:claim-1"
    assert cluster.has_conflict is True
    assert len(cluster.supporting_nodes) == 1
    assert len(cluster.contradicting_nodes) == 1
    assert len(cluster.cluster_digest) == 64


def test_store_commit_current_history_and_idempotency(tmp_path):
    store = EvidenceGraphStore(tmp_path / "graphs.sqlite3")
    first = graph(1, now=1)
    second = graph(2, conflict=True, now=2)
    assert store.commit(first, expected_current_generation=0) == first
    assert store.commit(graph(1, now=3)) == first
    store.commit(second, expected_current_generation=1)
    assert store.current(owner_id="alice", doc_id="doc-1") == second
    assert store.history(owner_id="alice", doc_id="doc-1") == (second, first)


def test_store_collision_pointer_and_exact_deletion_boundaries(tmp_path):
    store = EvidenceGraphStore(tmp_path / "graphs.sqlite3")
    first = graph(1)
    second = graph(2, conflict=True)
    store.commit(first, expected_current_generation=0)
    with pytest.raises(RuntimeError, match="collision"):
        store.commit(graph(1, conflict=True))
    store.commit(second, expected_current_generation=1)
    with pytest.raises(RuntimeError, match="current"):
        store.delete_generation(
            owner_id="alice",
            doc_id="doc-1",
            generation=2,
            confirm_graph_digest=second.graph_digest,
        )
    with pytest.raises(RuntimeError, match="confirmation"):
        store.delete_generation(
            owner_id="alice",
            doc_id="doc-1",
            generation=1,
            confirm_graph_digest="f" * 64,
        )
    assert store.delete_generation(
        owner_id="alice",
        doc_id="doc-1",
        generation=1,
        confirm_graph_digest=first.graph_digest,
    ) is True


def test_store_strict_json_and_database_identity_defenses(tmp_path):
    path = tmp_path / "graphs.sqlite3"
    store = EvidenceGraphStore(path)
    store.commit(graph(1), expected_current_generation=0)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE graph_generations SET batch_json=? WHERE owner_id='alice'",
            ('{"owner_id":"alice","owner_id":"bob"}',),
        )
    with pytest.raises(RuntimeError, match="corrupt"):
        store.get(owner_id="alice", doc_id="doc-1", generation=1)

    path2 = tmp_path / "identity.sqlite3"
    guarded = EvidenceGraphStore(path2)
    path2.rename(tmp_path / "old.sqlite3")
    path2.write_bytes(b"")
    with pytest.raises(RuntimeError, match="identity changed"):
        guarded.current(owner_id="alice", doc_id="doc-1")


def test_runtime_cache_is_path_scoped(tmp_path):
    clear_evidence_graph_store_cache()
    first = get_evidence_graph_store(tmp_path / "one.sqlite3")
    again = get_evidence_graph_store(tmp_path / "one.sqlite3")
    second = get_evidence_graph_store(tmp_path / "two.sqlite3")
    assert first is again and first is not second
