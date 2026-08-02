from __future__ import annotations

import json

from tools import evidence_graph_relation_authorization_cli as cli
from tools.evidence_graph_relation_authorization_store import (
    RelationReviewAuthorizationStore,
)
from tools.evidence_graph_relation_policy import ReviewAuthorization
from tools.evidence_graph_relation_policy_integrity import (
    deterministic_review_authorization_digest,
)


def authorization() -> ReviewAuthorization:
    values = {
        "proposal_id": "1" * 64,
        "decision_id": "2" * 64,
        "owner_id": "alice",
        "graph_set_key": "review",
        "decision": "approved",
        "reviewer_id": "reviewer",
        "policy_digest": "3" * 64,
        "grant_digest": "4" * 64,
        "separation_of_duties_enforced": True,
        "replacement_scope_validated": False,
    }
    return ReviewAuthorization(
        **values,
        authorization_digest=deterministic_review_authorization_digest(**values),
        authorized_at=1.0,
    )


def read(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def test_status_and_list_are_read_only_and_text_free(
    tmp_path, monkeypatch, capsys
):
    store = RelationReviewAuthorizationStore(tmp_path / "authorizations.sqlite3")
    value = authorization()
    store.prepare(value, now=2.0)
    store.mark_committed(value.decision_id, now=3.0)
    monkeypatch.setattr(
        cli, "get_relation_review_authorization_store", lambda: store
    )

    assert cli.main(["status", value.decision_id]) == 0
    status, error = read(capsys)
    assert error is None
    assert status["state"] == "committed"
    assert status["authorization_digest"] == value.authorization_digest
    assert status["separation_of_duties_enforced"] is True
    assert status["contains_source_text"] is False
    assert status["mutation_performed"] is False

    assert cli.main([
        "list",
        "--owner-id", "alice",
        "--graph-set-key", "review",
        "--state", "committed",
    ]) == 0
    listing, error = read(capsys)
    assert error is None
    assert listing["count"] == 1
    assert listing["authorizations"][0]["decision_id"] == value.decision_id
    rendered = json.dumps(listing).lower()
    assert "source_text" in rendered
    assert "private text" not in rendered


def test_missing_and_invalid_receipts_are_bounded(tmp_path, monkeypatch, capsys):
    store = RelationReviewAuthorizationStore(tmp_path / "authorizations.sqlite3")
    monkeypatch.setattr(
        cli, "get_relation_review_authorization_store", lambda: store
    )

    assert cli.main(["status", "f" * 64]) == 1
    output, error = read(capsys)
    assert output is None
    assert error == {"error": "not_found"}

    assert cli.main([
        "list", "--owner-id", "alice", "--limit", "0"
    ]) == 2
    output, error = read(capsys)
    assert output is None
    assert error == {"error": "invalid_or_unavailable"}
