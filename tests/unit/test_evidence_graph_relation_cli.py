from __future__ import annotations

import json

from tools import evidence_graph_relation_cli as cli
from tools.evidence_graph_relation_review import RelationReviewLedger

D1 = "a" * 64
D2 = "b" * 64
G1 = "c" * 64
G2 = "d" * 64
N1 = "e" * 64
N2 = "f" * 64
P1 = "1" * 64
P2 = "2" * 64
E = "3" * 64


def install(tmp_path, monkeypatch):
    ledger = RelationReviewLedger(tmp_path / "reviews.sqlite3")
    monkeypatch.setattr(cli, "get_relation_review_ledger", lambda: ledger)
    return ledger


def proposal_args(kind="human"):
    values = [
        "propose",
        "--owner-id", "alice",
        "--graph-set-key", "review",
        "--relation-key", "a-b",
        "--source-doc-id", "doc-a",
        "--source-generation", "1",
        "--source-graph-digest", G1,
        "--source-node-id", N1,
        "--source-provenance-digest", P1,
        "--target-doc-id", "doc-b",
        "--target-generation", "1",
        "--target-graph-digest", G2,
        "--target-node-id", N2,
        "--target-provenance-digest", P2,
        "--edge-type", "supports",
        "--proposer-kind", kind,
        "--proposer-id", "proposer-1",
        "--evidence-digest", E,
    ]
    if kind != "human":
        values.extend(["--extractor-name", "extractor", "--extractor-version", "v1"])
    return values


def read(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def test_propose_decide_status_and_list_are_text_free(tmp_path, monkeypatch, capsys):
    install(tmp_path, monkeypatch)
    assert cli.main(proposal_args()) == 0
    proposed, error = read(capsys)
    assert error is None
    assert proposed["review"] is None
    assert proposed["contains_source_text"] is False
    assert proposed["automatic_approval_performed"] is False
    proposal_id = proposed["proposal_id"]
    assert cli.main(["status", proposal_id]) == 0
    status, _error = read(capsys)
    assert status["proposal_id"] == proposal_id
    assert (
        cli.main(
            [
                "decide", proposal_id,
                "--owner-id", "alice",
                "--decision", "approved",
                "--reviewer-id", "reviewer-1",
                "--reason-code", "verified",
            ]
        )
        == 0
    )
    decided, _error = read(capsys)
    assert decided["review"]["decision"] == "approved"
    assert cli.main(["list", "--owner-id", "alice", "--graph-set-key", "review", "--decision", "approved"]) == 0
    listing, _error = read(capsys)
    rendered = json.dumps(listing).lower()
    assert listing["count"] == 1
    assert "source_text" in rendered and "private text" not in rendered


def test_model_proposal_requires_extractor_identity(tmp_path, monkeypatch, capsys):
    install(tmp_path, monkeypatch)
    values = proposal_args("model")
    name_index = values.index("--extractor-name")
    del values[name_index:name_index + 4]
    assert cli.main(values) == 2
    _output, error = read(capsys)
    assert error == {"error": "invalid_or_unavailable"}


def test_missing_proposal_is_bounded(tmp_path, monkeypatch, capsys):
    install(tmp_path, monkeypatch)
    assert cli.main(["status", "f" * 64]) == 1
    _output, error = read(capsys)
    assert error == {"error": "not_found"}
