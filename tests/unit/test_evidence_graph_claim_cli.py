from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from tools import evidence_graph_claim_cli as cli
from tools.evidence_graph_claim_extraction import extract_scientific_claim_proposals
from tools.evidence_graph_claim_runtime import (
    clear_scientific_claim_review_runtime_cache,
    get_scientific_claim_review_store,
)
from tools.evidence_graph_claim_submission import submit_scientific_claim_proposals


@dataclass
class Section:
    content: str
    page_number: int = 1


@dataclass
class Document:
    id: str
    text: str
    sections: list[Section]
    metadata: dict


def prepare(tmp_path, monkeypatch):
    text = "Drug A reduced mortality in the randomized cohort."
    document = Document(
        id="doc1",
        text=text,
        sections=[Section(text)],
        metadata={"content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()},
    )
    raw = {
        "schema_version": 1,
        "claims": [
            {
                "claim_key": "claim-1",
                "claim_text": "Drug A reduced mortality.",
                "claim_type": "finding",
                "modality": "asserted",
                "section_index": 0,
                "page_number": 1,
                "char_start": 0,
                "char_end": text.index(" in the"),
                "confidence": 0.9,
            }
        ],
    }
    proposal = extract_scientific_claim_proposals(
        document,
        raw,
        owner_id="alice",
        generation=1,
        profile_fingerprint="b" * 64,
        proposer_id="extractor",
        extractor_name="claims",
        extractor_version="1",
        now=1.0,
    ).proposals[0]
    db_path = tmp_path / "claims.sqlite3"
    monkeypatch.setenv("EVIDENCE_GRAPH_CLAIM_REVIEW_DB_PATH", str(db_path))
    monkeypatch.setenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ID", "reviewer")
    monkeypatch.delenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ID_PATH", raising=False)
    monkeypatch.delenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ASSERTION_PATH", raising=False)
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_CLAIM_REVIEW_POLICY_JSON",
        json.dumps(
            {
                "schema_version": 1,
                "reviewers": [
                    {
                        "reviewer_id": "reviewer",
                        "owners": ["alice"],
                        "doc_ids": ["doc1"],
                        "decisions": ["approved", "rejected", "superseded"],
                    }
                ],
            }
        ),
    )
    monkeypatch.delenv("EVIDENCE_GRAPH_CLAIM_REVIEW_POLICY_PATH", raising=False)
    clear_scientific_claim_review_runtime_cache()
    store = get_scientific_claim_review_store()
    submit_scientific_claim_proposals(store, (proposal,))
    return proposal, store


def read(capsys):
    captured = capsys.readouterr()
    return (
        None if not captured.out else json.loads(captured.out),
        None if not captured.err else json.loads(captured.err),
        captured.out + captured.err,
    )


def test_status_and_list_are_text_free_and_non_mutating(tmp_path, monkeypatch, capsys):
    proposal, _store = prepare(tmp_path, monkeypatch)

    assert cli.main(["status", proposal.proposal_id]) == 0
    output, error, rendered = read(capsys)
    assert error is None
    assert output["mutation_performed"] is False
    assert output["source_text_returned"] is False
    assert output["claim_text_length"] == len(proposal.claim_text)
    assert output["claim_text_sha256"] == hashlib.sha256(
        proposal.claim_text.encode("utf-8")
    ).hexdigest()
    assert proposal.claim_text not in rendered

    assert cli.main([
        "list",
        "--owner-id", "alice",
        "--doc-id", "doc1",
        "--decision", "pending",
    ]) == 0
    output, error, rendered = read(capsys)
    assert error is None
    assert output["item_count"] == 1
    assert output["mutation_performed"] is False
    assert proposal.claim_text not in rendered


def test_decide_is_actor_bound_and_atomic(tmp_path, monkeypatch, capsys):
    proposal, store = prepare(tmp_path, monkeypatch)

    assert cli.main([
        "decide",
        proposal.proposal_id,
        "--owner-id", "alice",
        "--decision", "approved",
        "--reviewer-id", "reviewer",
        "--reason-code", "scientific-review-complete",
    ]) == 0
    output, error, rendered = read(capsys)
    assert error is None
    assert output["decision"] == "approved"
    assert output["atomic_decision_authorization_commit"] is True
    assert output["mutation_performed"] is True
    assert proposal.claim_text not in rendered
    assert store.get_decision(proposal.proposal_id) is not None
    assert store.get_authorization(proposal.proposal_id) is not None


def test_annotation_export_is_text_free_and_does_not_mutate_graph(tmp_path, monkeypatch, capsys):
    proposal, _store = prepare(tmp_path, monkeypatch)
    assert cli.main([
        "decide",
        proposal.proposal_id,
        "--owner-id", "alice",
        "--decision", "approved",
        "--reviewer-id", "reviewer",
        "--reason-code", "scientific-review-complete",
    ]) == 0
    read(capsys)

    assert cli.main([
        "annotations",
        "--owner-id", "alice",
        "--doc-id", "doc1",
        "--generation", "1",
        "--content-sha256", proposal.content_sha256,
        "--profile-fingerprint", proposal.profile_fingerprint,
        "--proposal-id", proposal.proposal_id,
    ]) == 0
    output, error, rendered = read(capsys)
    assert error is None
    assert output["annotation_count"] == 1
    assert output["mutation_performed"] is False
    assert output["graph_mutation_performed"] is False
    assert output["semantic_relation_inference_performed"] is False
    assert output["source_text_returned"] is False
    assert proposal.claim_text not in rendered


def test_wrong_process_actor_and_missing_policy_fail_generically(tmp_path, monkeypatch, capsys):
    proposal, _store = prepare(tmp_path, monkeypatch)
    monkeypatch.setenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ID", "other-reviewer")
    assert cli.main([
        "decide",
        proposal.proposal_id,
        "--owner-id", "alice",
        "--decision", "approved",
        "--reviewer-id", "reviewer",
        "--reason-code", "scientific-review-complete",
    ]) == 2
    output, error, rendered = read(capsys)
    assert output is None
    assert error == {"error": "invalid_or_unavailable"}
    assert proposal.claim_text not in rendered

    monkeypatch.setenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ID", "reviewer")
    monkeypatch.delenv("EVIDENCE_GRAPH_CLAIM_REVIEW_POLICY_JSON", raising=False)
    assert cli.main([
        "decide",
        proposal.proposal_id,
        "--owner-id", "alice",
        "--decision", "approved",
        "--reviewer-id", "reviewer",
        "--reason-code", "scientific-review-complete",
    ]) == 2
    output, error, rendered = read(capsys)
    assert output is None
    assert error == {"error": "invalid_or_unavailable"}
    assert proposal.claim_text not in rendered
