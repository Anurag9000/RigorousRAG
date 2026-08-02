from __future__ import annotations

import json

from tools import evidence_graph_set_signed_publish_cli as cli
from tools.evidence_graph_set_publish import (
    EvidenceGraphSetPublishError,
    EvidenceGraphSetPublishResult,
)


def read(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def install(monkeypatch):
    monkeypatch.setattr(cli, "get_relation_review_ledger", lambda: "ledger")
    monkeypatch.setattr(
        cli, "get_relation_review_authorization_store", lambda: "authorizations"
    )
    monkeypatch.setattr(cli, "get_signed_actor_use_store", lambda: "actor-uses")
    monkeypatch.setattr(cli, "get_evidence_graph_set_store", lambda: "sets")
    monkeypatch.setattr(cli, "get_generation_store", lambda: "generations")
    monkeypatch.setattr(cli, "get_evidence_graph_store", lambda: "graphs")


def test_signed_publish_passes_all_governance_dependencies(
    monkeypatch, capsys
):
    install(monkeypatch)
    observed = {}

    def publish(**kwargs):
        observed.update(kwargs)
        return EvidenceGraphSetPublishResult(
            graph_set_id="a" * 64,
            graph_set_digest="b" * 64,
            graph_set_key="review",
            previous_graph_set_id=None,
            member_count=2,
            edge_count=1,
            authority_digest="c" * 64,
            proposal_ids=("d" * 64,),
            pointer_changed=True,
            compensation_performed=False,
            published_at=1.0,
        )

    monkeypatch.setattr(cli, "publish_signed_actor_governed_graph_set", publish)
    assert cli.main([
        "publish-approved",
        "--owner-id", "alice",
        "--graph-set-key", "review",
        "--proposal-id", "d" * 64,
        "--expect-no-current",
    ]) == 0
    output, error = read(capsys)

    assert error is None
    assert observed["expected_current_set_id"] is None
    assert observed["ledger"] == "ledger"
    assert observed["authorization_store"] == "authorizations"
    assert observed["actor_use_store"] == "actor-uses"
    assert output["committed_review_authorizations_required"] is True
    assert output["signed_actor_use_provenance_validated"] is True
    assert output["source_text_returned"] is False


def test_signed_publish_reports_compensation_without_secret_data(
    monkeypatch, capsys
):
    install(monkeypatch)

    def fail(**kwargs):
        raise EvidenceGraphSetPublishError(
            "failed", compensation_errors=("pointer:verification",)
        )

    monkeypatch.setattr(cli, "publish_signed_actor_governed_graph_set", fail)
    assert cli.main([
        "publish-approved",
        "--owner-id", "alice",
        "--graph-set-key", "review",
        "--proposal-id", "d" * 64,
        "--expect-no-current",
    ]) == 1
    output, error = read(capsys)

    assert output is None
    assert error["error"] == "publication_failed"
    assert error["compensation_complete"] is False
    assert error["compensation_errors"] == ["pointer:verification"]
    assert error["signed_actor_use_provenance_validated"] is False
    assert "signature" not in json.dumps(error).lower()
    assert "key" not in json.dumps(error).lower()
