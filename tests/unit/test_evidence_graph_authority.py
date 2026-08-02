from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.evidence_graph_authority import (
    EvidenceGraphAuthorityError,
    assess_graph_authority,
    resolve_evidence_graph,
)
from tools.evidence_graph_types import (
    EvidenceGraphBatch,
    EvidenceNode,
    deterministic_node_id,
)

A = "a" * 64
B = "b" * 64


def batch(sequence=3, content=A, profile=B, deleted=False):
    metadata = (
        {"derived_tombstone": True, "authoritative_state": "deleted"}
        if deleted
        else {}
    )
    node = EvidenceNode(
        node_id=deterministic_node_id(
            owner_id="alice",
            doc_id="doc-1",
            generation=sequence,
            node_type="document",
            natural_key="document",
        ),
        owner_id="alice",
        doc_id="doc-1",
        generation=sequence,
        node_type="document",
        natural_key="document",
        label="doc-1",
        metadata=metadata,
    )
    return EvidenceGraphBatch(
        owner_id="alice",
        doc_id="doc-1",
        generation=sequence,
        content_sha256=content,
        profile_fingerprint=profile,
        nodes=(node,),
        edges=(),
        created_at=1.0,
    )


def record(sequence=3, state="active", content=A, profile=B):
    return SimpleNamespace(
        owner_id="alice",
        doc_id="doc-1",
        sequence=sequence,
        state=state,
        content_sha256=content,
        profile_fingerprint=profile,
    )


class Graphs:
    def __init__(self, current, historical=None):
        self.value = current
        self.historical = historical or {current.generation: current}

    def current(self, **kwargs):
        return self.value

    def get(self, **kwargs):
        return self.historical[kwargs["generation"]]


class Generations:
    def __init__(self, value):
        self.value = value

    def current(self, **kwargs):
        return self.value


def test_current_graph_requires_exact_authoritative_identity():
    view = resolve_evidence_graph(
        owner_id="alice",
        doc_id="doc-1",
        graphs=Graphs(batch()),
        generations=Generations(record()),
    )
    assert view.authoritative_current is True
    assert len(view.authority_digest) == 64
    for current in (
        record(sequence=4),
        record(content="c" * 64),
        record(profile="d" * 64),
    ):
        with pytest.raises(EvidenceGraphAuthorityError):
            resolve_evidence_graph(
                owner_id="alice",
                doc_id="doc-1",
                graphs=Graphs(batch()),
                generations=Generations(current),
            )


def test_deleted_current_requires_tombstone_shape():
    good = batch(deleted=True)
    assert assess_graph_authority(
        good, record(state="deleted")
    ).authoritative_current is True
    with pytest.raises(EvidenceGraphAuthorityError):
        resolve_evidence_graph(
            owner_id="alice",
            doc_id="doc-1",
            graphs=Graphs(batch()),
            generations=Generations(record(state="deleted")),
        )


def test_explicit_historical_graph_is_inspectable_but_not_current():
    old = batch(sequence=2)
    current = batch(sequence=3)
    view = resolve_evidence_graph(
        owner_id="alice",
        doc_id="doc-1",
        generation=2,
        graphs=Graphs(current, {2: old, 3: current}),
        generations=Generations(record(sequence=3)),
    )
    assert view.batch.generation == 2
    assert view.authoritative_current is False


def test_missing_authoritative_or_graph_is_not_found():
    with pytest.raises(KeyError):
        resolve_evidence_graph(
            owner_id="alice",
            doc_id="doc-1",
            graphs=Graphs(batch()),
            generations=Generations(None),
        )
    graphs = Graphs(batch())
    graphs.value = None
    with pytest.raises(KeyError):
        resolve_evidence_graph(
            owner_id="alice",
            doc_id="doc-1",
            graphs=graphs,
            generations=Generations(record()),
        )
