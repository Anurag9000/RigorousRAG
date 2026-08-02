from __future__ import annotations

import json

from tools import evidence_graph_relation_cli as cli
from tools.evidence_graph_relation_actor import (
    ReviewActorBinding,
    require_relation_review_actor,
)
from tools.evidence_graph_relation_authorization_store import (
    RelationReviewAuthorizationStore,
)
from tools.evidence_graph_relation_policy import RelationReviewPolicy
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


def review_policy() -> RelationReviewPolicy:
    return RelationReviewPolicy.from_mapping(
        {
            "schema_version": 1,
            "reviewers": [
                {
                    "reviewer_id": "reviewer-1",
                    "owners": ["alice"],
                    "graph_set_keys": ["review"],
                    "decisions": ["approved", "rejected", "superseded"],
                }
            ],
        }
    )


def install(tmp_path, monkeypatch):
    ledger = RelationReviewLedger(tmp_path / "reviews.sqlite3")
    authorizations = RelationReviewAuthorizationStore(
        tmp_path / "review-authorizations.sqlite3"
    )
    actor = ReviewActorBinding.create(
        actor_id="reviewer-1",
        binding_method="process_environment",
        loaded_at=1.0,
    )
    monkeypatch.setattr(cli, "get_relation_review_ledger", lambda: ledger)
    monkeypatch.setattr(
        cli,
        "get_relation_review_authorization_store",
        lambda: authorizations,
    )
    monkeypatch.setattr(cli, "get_relation_review_policy", review_policy)
    monkeypatch.setattr(
        cli,
        "require_relation_review_actor",
        lambda requested: require_relation_review_actor(requested, binding=actor),
    )
    return ledger, authorizations


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
    assert proposed["governed_review"] is False
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
    assert decided["governed_review"] is True
    assert decided["review_authorization"]["state"] == "committed"
    assert decided["review_authorization"]["separation_of_duties_enforced"] is True
    assert decided["review_actor_binding"]["actor_id"] == "reviewer-1"
    assert decided["review_actor_binding"]["binding_method"] == "process_environment"
    assert decided["review_actor_binding"]["durable_receipt_field"] is False
    assert cli.main(["status", proposal_id]) == 0
    governed_status, _error = read(capsys)
    assert governed_status["governed_review"] is True
    assert cli.main([
        "list",
        "--owner-id", "alice",
        "--graph-set-key", "review",
        "--decision", "approved",
    ]) == 0
    listing, _error = read(capsys)
    rendered = json.dumps(listing).lower()
    assert listing["count"] == 1
    assert listing["proposals"][0]["governed_review"] is True
    assert "source_text" in rendered and "private text" not in rendered


def test_reviewer_argument_must_match_process_actor(tmp_path, monkeypatch, capsys):
    install(tmp_path, monkeypatch)
    assert cli.main(proposal_args()) == 0
    proposed, _error = read(capsys)
    assert cli.main([
        "decide", proposed["proposal_id"],
        "--owner-id", "alice",
        "--decision", "approved",
        "--reviewer-id", "other-reviewer",
        "--reason-code", "verified",
    ]) == 2
    output, error = read(capsys)
    assert output is None
    assert error == {"error": "invalid_or_unavailable"}


def test_model_proposal_requires_extractor_identity(tmp_path, monkeypatch, capsys):
    install(tmp_path, monkeypatch)
    values = proposal_args("model")
    name_index = values.index("--extractor-name")
    del values[name_index:name_index + 4]
    assert cli.main(values) == 2
    _output, error = read(capsys)
    assert error == {"error": "invalid_or_unavailable"}


def test_self_review_and_missing_policy_fail_closed(tmp_path, monkeypatch, capsys):
    install(tmp_path, monkeypatch)
    values = proposal_args()
    values[values.index("--proposer-id") + 1] = "reviewer-1"
    assert cli.main(values) == 0
    proposed, _error = read(capsys)
    assert cli.main([
        "decide", proposed["proposal_id"],
        "--owner-id", "alice",
        "--decision", "approved",
        "--reviewer-id", "reviewer-1",
        "--reason-code", "verified",
    ]) == 2
    _output, error = read(capsys)
    assert error == {"error": "invalid_or_unavailable"}

    monkeypatch.setattr(
        cli,
        "get_relation_review_policy",
        lambda: (_ for _ in ()).throw(RuntimeError("missing policy")),
    )
    other = proposal_args()
    other[other.index("--relation-key") + 1] = "other"
    other[other.index("--evidence-digest") + 1] = "4" * 64
    assert cli.main(other) == 0
    proposed, _error = read(capsys)
    assert cli.main([
        "decide", proposed["proposal_id"],
        "--owner-id", "alice",
        "--decision", "approved",
        "--reviewer-id", "reviewer-1",
        "--reason-code", "verified",
    ]) == 2
    _output, error = read(capsys)
    assert error == {"error": "invalid_or_unavailable"}


def test_missing_proposal_is_bounded(tmp_path, monkeypatch, capsys):
    install(tmp_path, monkeypatch)
    assert cli.main(["status", "f" * 64]) == 1
    _output, error = read(capsys)
    assert error == {"error": "not_found"}
