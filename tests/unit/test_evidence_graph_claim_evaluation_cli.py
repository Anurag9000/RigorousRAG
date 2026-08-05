from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import pytest

from tools import evidence_graph_claim_evaluation_cli as cli
from tools.evidence_graph_claim_contracts import ClaimEvidenceLocator, ScientificClaimProposal
from tools.evidence_graph_claim_evaluation import ScientificClaimGold


def fixture():
    locator = ClaimEvidenceLocator(
        section_index=0,
        page_number=1,
        char_start=0,
        char_end=10,
        evidence_sha256=hashlib.sha256(b"evidence").hexdigest(),
    )
    gold = ScientificClaimGold(
        gold_id="g1",
        owner_id="alice",
        doc_id="doc1",
        generation=1,
        content_sha256="a" * 64,
        profile_fingerprint="b" * 64,
        claim_text="Drug A helps",
        claim_type="finding",
        modality="asserted",
        locator=locator,
    )
    proposal = ScientificClaimProposal.create(
        owner_id="alice",
        doc_id="doc1",
        generation=1,
        content_sha256="a" * 64,
        profile_fingerprint="b" * 64,
        claim_key="p1",
        claim_text="Drug A helps",
        claim_type="finding",
        modality="asserted",
        locator=locator,
        proposer_kind="model",
        proposer_id="extractor",
        extractor_name="claims",
        extractor_version="1",
        confidence=0.8,
        created_at=1.0,
    )
    return {
        "schema_version": 1,
        "minimum_span_iou": 0.5,
        "minimum_claim_token_f1": 0.5,
        "gold": [asdict(gold)],
        "proposals": [asdict(proposal)],
    }


def test_fixture_cli_is_text_free_verified_and_non_mutating(tmp_path, capsys):
    path = tmp_path / "claims.json"
    path.write_text(json.dumps(fixture()), encoding="utf-8")

    assert cli.main([str(path)]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert captured.err == ""
    assert payload["matched_count"] == 1
    assert payload["report_digest"]
    assert payload["contains_claim_text"] is False
    assert payload["contains_evidence_text"] is False
    assert payload["semantic_entailment_evaluated"] is False
    assert payload["mutation_performed"] is False
    assert payload["source_text_returned"] is False
    assert "Drug A helps" not in captured.out


def test_fixture_cli_rejects_duplicate_keys_and_nonfinite_values(tmp_path, capsys):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":1,"schema_version":1,"minimum_span_iou":0.5,'
        '"minimum_claim_token_f1":0.5,"gold":[],"proposals":[]}',
        encoding="utf-8",
    )
    assert cli.main([str(duplicate)]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {"error": "invalid_or_unavailable"}

    invalid = fixture()
    invalid["minimum_span_iou"] = float("nan")
    path = tmp_path / "nan.json"
    path.write_text(json.dumps(invalid), encoding="utf-8")
    assert cli.main([str(path)]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {"error": "invalid_or_unavailable"}


def test_fixture_cli_rejects_redirects(tmp_path, capsys):
    target = tmp_path / "claims.json"
    target.write_text(json.dumps(fixture()), encoding="utf-8")
    link = tmp_path / "claims-link.json"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    assert cli.main([str(link)]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {"error": "invalid_or_unavailable"}
