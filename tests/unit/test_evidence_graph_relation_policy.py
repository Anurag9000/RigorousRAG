from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from tools.evidence_graph_relation_authorization_store import (
    RelationReviewAuthorizationStore,
)
from tools.evidence_graph_relation_policy import (
    GovernedRelationReviewService,
    RelationReviewPolicy,
    load_relation_review_policy,
)
from tools.evidence_graph_relation_review import (
    CrossDocumentRelationProposal,
    RelationEndpoint,
    RelationReviewDecision,
    RelationReviewLedger,
)

G1 = "a" * 64
G2 = "b" * 64
N1 = "c" * 64
N2 = "d" * 64
P1 = "e" * 64
P2 = "f" * 64
E = "1" * 64


def endpoint(doc_id: str, *, second: bool = False) -> RelationEndpoint:
    return RelationEndpoint(
        doc_id=doc_id,
        generation=1,
        graph_digest=G2 if second else G1,
        node_id=N2 if second else N1,
        provenance_digest=P2 if second else P1,
    )


def submit(
    ledger: RelationReviewLedger,
    *,
    relation_key: str = "a-b",
    graph_set_key: str = "review",
    proposer_id: str = "proposer-1",
    evidence_digest: str = E,
) -> CrossDocumentRelationProposal:
    return ledger.submit(
        CrossDocumentRelationProposal.create(
            owner_id="alice",
            graph_set_key=graph_set_key,
            relation_key=relation_key,
            source=endpoint("doc-a"),
            target=endpoint("doc-b", second=True),
            edge_type="supports",
            proposer_kind="human",
            proposer_id=proposer_id,
            evidence_digest=evidence_digest,
            created_at=1.0,
        )
    )


def policy(
    *,
    owners=("alice",),
    keys=("review",),
    decisions=("approved", "rejected", "superseded"),
    expires_at=None,
) -> RelationReviewPolicy:
    reviewer = {
        "reviewer_id": "reviewer-1",
        "owners": list(owners),
        "graph_set_keys": list(keys),
        "decisions": list(decisions),
    }
    if expires_at is not None:
        reviewer["expires_at"] = expires_at
    return RelationReviewPolicy.from_mapping(
        {"schema_version": 1, "reviewers": [reviewer]}
    )


def service(
    tmp_path: Path,
    ledger: RelationReviewLedger,
    selected_policy=None,
    now=10.0,
):
    store = RelationReviewAuthorizationStore(tmp_path / "authorizations.sqlite3")
    return (
        GovernedRelationReviewService(
            ledger=ledger,
            policy=selected_policy or policy(),
            authorization_store=store,
            clock=lambda: now,
        ),
        store,
    )


def decision(proposal, *, kind="approved", replacement=None, reviewer="reviewer-1"):
    return RelationReviewDecision.create(
        proposal_id=proposal.proposal_id,
        owner_id=proposal.owner_id,
        decision=kind,
        reviewer_id=reviewer,
        reason_code="verified",
        replacement_proposal_id=(
            None if replacement is None else replacement.proposal_id
        ),
        decided_at=2.0,
    )


def test_authorized_decision_is_journaled_committed_and_replayable(tmp_path):
    ledger = RelationReviewLedger(tmp_path / "reviews.sqlite3")
    proposal = submit(ledger)
    governed, store = service(tmp_path, ledger)
    value = decision(proposal)

    stored, receipt = governed.decide(value)

    assert stored == value
    assert receipt.state == "committed"
    assert receipt.authorization.policy_digest == policy().policy_digest
    assert receipt.authorization.separation_of_duties_enforced is True
    assert store.get(value.decision_id) == receipt

    replayed, replay_receipt = governed.decide(value)
    assert replayed == value
    assert replay_receipt == receipt


def test_crash_window_after_decision_recovers_from_authorized_receipt(tmp_path):
    ledger = RelationReviewLedger(tmp_path / "reviews.sqlite3")
    proposal = submit(ledger)
    governed, store = service(tmp_path, ledger)
    value = decision(proposal)
    authorization = governed._authorization(proposal, value, now=10.0)
    assert store.prepare(authorization, now=10.0).state == "authorized"
    ledger.decide(value)

    stored, receipt = governed.decide(value)

    assert stored == value
    assert receipt.state == "committed"


def test_self_review_scope_and_expiry_fail_closed(tmp_path):
    ledger = RelationReviewLedger(tmp_path / "reviews.sqlite3")
    self_authored = submit(ledger, proposer_id="reviewer-1")
    governed, _store = service(tmp_path, ledger)
    with pytest.raises(PermissionError, match="own proposal"):
        governed.decide(decision(self_authored))

    other = submit(ledger, relation_key="other", evidence_digest="2" * 64)
    wrong_owner, _store = service(
        tmp_path / "wrong-owner",
        ledger,
        policy(owners=("bob",)),
    )
    with pytest.raises(PermissionError, match="scope"):
        wrong_owner.decide(decision(other))

    expired, _store = service(
        tmp_path / "expired",
        ledger,
        policy(expires_at=9.0),
        now=10.0,
    )
    with pytest.raises(PermissionError, match="scope"):
        expired.decide(decision(other))


def test_supersession_requires_same_relation_scope_and_independent_reviewer(tmp_path):
    ledger = RelationReviewLedger(tmp_path / "reviews.sqlite3")
    original = submit(ledger)
    replacement = submit(
        ledger,
        proposer_id="proposer-2",
        evidence_digest="2" * 64,
    )
    governed, _store = service(tmp_path, ledger)

    stored, receipt = governed.decide(
        decision(original, kind="superseded", replacement=replacement)
    )
    assert stored.decision == "superseded"
    assert receipt.authorization.replacement_scope_validated is True

    ledger2 = RelationReviewLedger(tmp_path / "reviews-2.sqlite3")
    original2 = submit(ledger2)
    wrong_key = submit(
        ledger2,
        graph_set_key="other-set",
        evidence_digest="3" * 64,
    )
    governed2, _store = service(tmp_path / "other", ledger2)
    with pytest.raises(PermissionError, match="same relation scope"):
        governed2.decide(
            decision(original2, kind="superseded", replacement=wrong_key)
        )


def test_existing_ungoverned_decision_is_not_retroactively_trusted(tmp_path):
    ledger = RelationReviewLedger(tmp_path / "reviews.sqlite3")
    proposal = submit(ledger)
    value = decision(proposal)
    ledger.decide(value)
    governed, _store = service(tmp_path, ledger)

    with pytest.raises(RuntimeError, match="lacks a governed authorization"):
        governed.decide(value)


def test_policy_loader_is_strict_and_file_backed(tmp_path, monkeypatch):
    raw = {
        "schema_version": 1,
        "reviewers": [
            {
                "reviewer_id": "reviewer-1",
                "owners": ["alice"],
                "graph_set_keys": ["review"],
                "decisions": ["approved"],
            }
        ],
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert load_relation_review_policy(path=path).policy_digest == policy(
        decisions=("approved",)
    ).policy_digest

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_relation_review_policy(
            json_text='{"schema_version":1,"schema_version":1,"reviewers":[]}'
        )
    with pytest.raises(ValueError, match="either inline or file"):
        load_relation_review_policy(json_text="{}", path=path)
    monkeypatch.delenv("EVIDENCE_GRAPH_REVIEW_POLICY_JSON", raising=False)
    monkeypatch.delenv("EVIDENCE_GRAPH_REVIEW_POLICY_PATH", raising=False)
    with pytest.raises(RuntimeError, match="not configured"):
        load_relation_review_policy()


def test_authorization_store_detects_payload_tampering(tmp_path):
    ledger = RelationReviewLedger(tmp_path / "reviews.sqlite3")
    proposal = submit(ledger)
    governed, store = service(tmp_path, ledger)
    value = decision(proposal)
    authorization = governed._authorization(proposal, value, now=10.0)
    store.prepare(authorization, now=10.0)

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE relation_review_authorizations SET payload_json='{}' "
            "WHERE decision_id=?",
            (value.decision_id,),
        )
    with pytest.raises(RuntimeError, match="corrupt"):
        store.get(value.decision_id)
