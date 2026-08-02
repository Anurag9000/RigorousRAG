from __future__ import annotations

import json
from types import SimpleNamespace

from tools import evidence_graph_set_cli as cli
from tools.evidence_graph_set_store import EvidenceGraphSetStore
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
        content_sha256=doc_id[-1] * 64,
        profile_fingerprint=PROFILE,
        nodes=(document, claim),
        edges=(),
        created_at=1.0,
    )


def claim(value):
    return next(item for item in value.nodes if item.node_type == "claim")


def graph_set():
    a = graph("doc-a")
    b = graph("doc-b")
    value = build_evidence_graph_set(
        owner_id="alice",
        graph_set_key="review",
        authority_views=(
            SimpleNamespace(batch=a, authoritative_current=True, authority_digest="a" * 64),
            SimpleNamespace(batch=b, authoritative_current=True, authority_digest="b" * 64),
        ),
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
        now=1.0,
    )
    return value, a, b


class Generations:
    def __init__(self, generation=1):
        self.generation = generation

    def current(self, **kwargs):
        return SimpleNamespace(
            sequence=self.generation,
            content_sha256=kwargs["doc_id"][-1] * 64,
            profile_fingerprint=PROFILE,
        )


class Graphs:
    def __init__(self, a, b):
        self.values = {"doc-a": a, "doc-b": b}

    def current(self, **kwargs):
        return self.values[kwargs["doc_id"]]


def read(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def install(tmp_path, monkeypatch):
    value, a, b = graph_set()
    store = EvidenceGraphSetStore(tmp_path / "sets.sqlite3")
    store.commit(value, expected_current_set_id=None)
    generations = Generations()
    graphs = Graphs(a, b)
    monkeypatch.setattr(cli, "get_evidence_graph_set_store", lambda: store)
    monkeypatch.setattr(cli, "get_generation_store", lambda: generations)
    monkeypatch.setattr(cli, "get_evidence_graph_store", lambda: graphs)
    return value, generations


def test_status_and_neighbors_are_read_only_and_text_free(tmp_path, monkeypatch, capsys):
    value, _generations = install(tmp_path, monkeypatch)
    assert cli.main(["status", "--owner-id", "alice", "--graph-set-key", "review"]) == 0
    output, error = read(capsys)
    assert error is None and output["authoritative_current"] is True
    assert output["mutation_performed"] is False
    rendered = json.dumps(output).lower()
    assert "private text" not in rendered
    source = value.edges[0].source
    assert (
        cli.main(
            [
                "neighbors",
                "--owner-id",
                "alice",
                "--graph-set-key",
                "review",
                "--doc-id",
                source.doc_id,
                "--node-id",
                source.node_id,
            ]
        )
        == 0
    )
    output, _error = read(capsys)
    assert output["result_count"] == 1


def test_current_set_fails_closed_when_member_moves(tmp_path, monkeypatch, capsys):
    _value, generations = install(tmp_path, monkeypatch)
    generations.generation = 2
    assert cli.main(["status", "--owner-id", "alice", "--graph-set-key", "review"]) == 1
    _output, error = read(capsys)
    assert error == {"error": "stale_graph_set"}


def test_history_remains_inspectable_when_stale(tmp_path, monkeypatch, capsys):
    _value, generations = install(tmp_path, monkeypatch)
    generations.generation = 2
    assert cli.main(["history", "--owner-id", "alice", "--graph-set-key", "review"]) == 0
    output, error = read(capsys)
    assert error is None
    assert output["versions"][0]["authoritative_current"] is False
