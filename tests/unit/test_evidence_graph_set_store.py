from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from tools.evidence_graph_set_store import (
    EvidenceGraphSetAuthorityError,
    EvidenceGraphSetStore,
    assess_graph_set_authority,
)
from tools.evidence_graph_sets import (
    ExplicitCrossDocumentRelation,
    build_evidence_graph_set,
)
from tools.evidence_graph_types import (
    EvidenceGraphBatch,
    EvidenceNode,
    deterministic_node_id,
)

PROFILE = "b" * 64


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
        content_sha256=(doc_id[-1] * 64),
        profile_fingerprint=PROFILE,
        nodes=(document, claim),
        edges=(),
        created_at=1.0,
    )


def claim(value):
    return next(item for item in value.nodes if item.node_type == "claim")


def view(value):
    return SimpleNamespace(
        batch=value,
        authoritative_current=True,
        authority_digest=(value.doc_id[-1] * 64),
    )


def graph_set(generation=1, now=1.0):
    a = graph("doc-a", generation)
    b = graph("doc-b", generation)
    return build_evidence_graph_set(
        owner_id="alice",
        graph_set_key="review",
        authority_views=(view(a), view(b)),
        relations=(
            ExplicitCrossDocumentRelation(
                "a-b",
                "doc-a",
                claim(a).node_id,
                "doc-b",
                claim(b).node_id,
                "supports",
            ),
        ),
        now=now,
    )


class Generations:
    def __init__(self, generation=1):
        self.generation = generation

    def current(self, **kwargs):
        doc_id = kwargs["doc_id"]
        return SimpleNamespace(
            sequence=self.generation,
            content_sha256=doc_id[-1] * 64,
            profile_fingerprint=PROFILE,
        )


class Graphs:
    def __init__(self, generation=1):
        self.values = {
            doc: graph(doc, generation) for doc in ("doc-a", "doc-b")
        }

    def current(self, **kwargs):
        return self.values.get(kwargs["doc_id"])


def test_store_commit_current_history_and_idempotency(tmp_path):
    store = EvidenceGraphSetStore(tmp_path / "sets.sqlite3")
    first = graph_set(1, 1.0)
    second = graph_set(2, 2.0)
    assert store.commit(first, expected_current_set_id=None) == first
    assert store.commit(graph_set(1, 3.0)) == first
    store.commit(second, expected_current_set_id=first.graph_set_id)
    assert store.current(owner_id="alice", graph_set_key="review") == second
    assert store.history(owner_id="alice", graph_set_key="review") == (
        second,
        first,
    )


def test_store_optimistic_pointer_and_exact_deletion(tmp_path):
    store = EvidenceGraphSetStore(tmp_path / "sets.sqlite3")
    first = graph_set(1)
    second = graph_set(2)
    store.commit(first, expected_current_set_id=None)
    with pytest.raises(RuntimeError, match="concurrently"):
        store.commit(second, expected_current_set_id=None)
    store.commit(second, expected_current_set_id=first.graph_set_id)
    with pytest.raises(RuntimeError, match="current"):
        store.delete(
            owner_id="alice",
            graph_set_id=second.graph_set_id,
            confirm_graph_set_digest=second.graph_set_digest,
        )
    with pytest.raises(RuntimeError, match="confirmation"):
        store.delete(
            owner_id="alice",
            graph_set_id=first.graph_set_id,
            confirm_graph_set_digest="f" * 64,
        )
    assert store.delete(
        owner_id="alice",
        graph_set_id=first.graph_set_id,
        confirm_graph_set_digest=first.graph_set_digest,
    ) is True


def test_authority_requires_every_member_current():
    value = graph_set(1)
    report = assess_graph_set_authority(
        value,
        generations=Generations(1),
        graphs=Graphs(1),
    )
    assert report.authoritative_current is True
    stale = assess_graph_set_authority(
        value,
        generations=Generations(2),
        graphs=Graphs(2),
    )
    assert stale.authoritative_current is False
    assert stale.stale_member_doc_ids == ("doc-a", "doc-b")


def test_resolve_current_fails_closed_when_any_member_moves(tmp_path):
    store = EvidenceGraphSetStore(tmp_path / "sets.sqlite3")
    value = graph_set(1)
    store.commit(value, expected_current_set_id=None)
    resolved, report = store.resolve_current(
        owner_id="alice",
        graph_set_key="review",
        generations=Generations(1),
        graphs=Graphs(1),
    )
    assert resolved == value and report.authoritative_current is True
    with pytest.raises(EvidenceGraphSetAuthorityError):
        store.resolve_current(
            owner_id="alice",
            graph_set_key="review",
            generations=Generations(2),
            graphs=Graphs(2),
        )


def test_store_detects_payload_and_database_identity_tampering(tmp_path):
    path = tmp_path / "sets.sqlite3"
    store = EvidenceGraphSetStore(path)
    value = graph_set(1)
    store.commit(value, expected_current_set_id=None)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE evidence_graph_sets SET payload_json=?",
            ('{"owner_id":"alice","owner_id":"bob"}',),
        )
    with pytest.raises(RuntimeError, match="corrupt"):
        store.get(owner_id="alice", graph_set_id=value.graph_set_id)

    guarded_path = tmp_path / "guarded.sqlite3"
    guarded = EvidenceGraphSetStore(guarded_path)
    guarded_path.rename(tmp_path / "old.sqlite3")
    guarded_path.write_bytes(b"")
    with pytest.raises(RuntimeError, match="identity changed"):
        guarded.current(owner_id="alice", graph_set_key="review")
