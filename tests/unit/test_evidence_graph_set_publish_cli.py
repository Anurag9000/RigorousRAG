from __future__ import annotations

import json

from tools import evidence_graph_set_publish_cli as cli
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


def install_dependencies(monkeypatch):
    monkeypatch.setattr(cli, "get_relation_review_ledger", lambda: object())
    monkeypatch.setattr(cli, "get_evidence_graph_set_store", lambda: object())
    monkeypatch.setattr(cli, "get_generation_store", lambda: object())
    monkeypatch.setattr(cli, "get_evidence_graph_store", lambda: object())


def test_publish_cli_requires_explicit_first_publication_expectation(
    monkeypatch, capsys
):
    install_dependencies(monkeypatch)
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

    monkeypatch.setattr(cli, "publish_approved_graph_set", publish)
    assert (
        cli.main(
            [
                "publish-approved",
                "--owner-id",
                "alice",
                "--graph-set-key",
                "review",
                "--proposal-id",
                "d" * 64,
                "--expect-no-current",
            ]
        )
        == 0
    )
    output, error = read(capsys)
    assert error is None
    assert observed["expected_current_set_id"] is None
    assert output["reviewed_proposals_required"] is True
    assert output["automatic_approval_performed"] is False
    assert output["source_text_returned"] is False
    assert "private text" not in json.dumps(output).lower()


def test_publish_cli_passes_expected_pointer(monkeypatch, capsys):
    install_dependencies(monkeypatch)
    observed = {}

    def publish(**kwargs):
        observed.update(kwargs)
        return EvidenceGraphSetPublishResult(
            graph_set_id="e" * 64,
            graph_set_digest="f" * 64,
            graph_set_key="review",
            previous_graph_set_id="a" * 64,
            member_count=2,
            edge_count=1,
            authority_digest="b" * 64,
            proposal_ids=("d" * 64,),
            pointer_changed=True,
            compensation_performed=False,
            published_at=1.0,
        )

    monkeypatch.setattr(cli, "publish_approved_graph_set", publish)
    assert (
        cli.main(
            [
                "publish-approved",
                "--owner-id",
                "alice",
                "--graph-set-key",
                "review",
                "--proposal-id",
                "d" * 64,
                "--expected-current-set-id",
                "a" * 64,
            ]
        )
        == 0
    )
    read(capsys)
    assert observed["expected_current_set_id"] == "a" * 64


def test_publication_failure_reports_compensation_status(monkeypatch, capsys):
    install_dependencies(monkeypatch)

    def fail(**kwargs):
        raise EvidenceGraphSetPublishError(
            "failed", compensation_errors=("pointer:verification",)
        )

    monkeypatch.setattr(cli, "publish_approved_graph_set", fail)
    assert (
        cli.main(
            [
                "publish-approved",
                "--owner-id",
                "alice",
                "--graph-set-key",
                "review",
                "--proposal-id",
                "d" * 64,
                "--expect-no-current",
            ]
        )
        == 1
    )
    output, error = read(capsys)
    assert output is None
    assert error["error"] == "publication_failed"
    assert error["compensation_complete"] is False
    assert error["compensation_errors"] == ["pointer:verification"]
