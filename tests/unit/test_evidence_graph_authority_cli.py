from __future__ import annotations

import json
from types import SimpleNamespace

from tools import evidence_graph_cli as cli
from tools.evidence_graph_types import (
    EvidenceGraphBatch,
    EvidenceNode,
    deterministic_node_id,
)

A = "a" * 64
B = "b" * 64


def graph(sequence=3):
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
        label="doc",
    )
    return EvidenceGraphBatch(
        owner_id="alice",
        doc_id="doc-1",
        generation=sequence,
        content_sha256=A,
        profile_fingerprint=B,
        nodes=(node,),
        edges=(),
        created_at=1.0,
    )


class Store:
    def __init__(self, value):
        self.value = value

    def current(self, **kwargs):
        return self.value

    def get(self, **kwargs):
        return self.value

    def history(self, **kwargs):
        return (self.value,)


class Generations:
    def __init__(self, sequence=3):
        self.value = SimpleNamespace(
            owner_id="alice",
            doc_id="doc-1",
            sequence=sequence,
            state="active",
            content_sha256=A,
            profile_fingerprint=B,
        )

    def current(self, **kwargs):
        return self.value


def parse(capsys):
    value = capsys.readouterr()
    return (
        json.loads(value.out) if value.out else None,
        json.loads(value.err) if value.err else None,
    )


def test_current_cli_reports_authority(monkeypatch, capsys):
    monkeypatch.setattr(cli, "get_evidence_graph_store", lambda: Store(graph()))
    monkeypatch.setattr(cli, "get_generation_store", lambda: Generations())
    assert (
        cli.main(["status", "--owner-id", "alice", "--doc-id", "doc-1"])
        == 0
    )
    output, error = parse(capsys)
    assert error is None
    assert output["authoritative_current"] is True
    assert len(output["authority_digest"]) == 64


def test_stale_current_cli_fails_closed_but_history_is_inspectable(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        cli, "get_evidence_graph_store", lambda: Store(graph(sequence=2))
    )
    monkeypatch.setattr(cli, "get_generation_store", lambda: Generations(sequence=3))
    assert (
        cli.main(["status", "--owner-id", "alice", "--doc-id", "doc-1"])
        == 1
    )
    _output, error = parse(capsys)
    assert error == {"error": "stale_graph"}
    assert (
        cli.main(["history", "--owner-id", "alice", "--doc-id", "doc-1"])
        == 0
    )
    output, error = parse(capsys)
    assert error is None
    assert output["generations"][0]["authoritative_current"] is False
