from __future__ import annotations

import hashlib
import json

import pytest

from tools.evidence_graph_claim_contracts import (
    ClaimEvidenceLocator,
    ScientificClaimProposal,
)
from tools.evidence_graph_claim_evaluation import (
    ScientificClaimGold,
    evaluate_scientific_claim_extraction,
)


def locator(start=0, end=20, *, evidence="Drug A reduced deaths"):
    return ClaimEvidenceLocator(
        section_index=0,
        page_number=1,
        char_start=start,
        char_end=end,
        evidence_sha256=hashlib.sha256(evidence.encode("utf-8")).hexdigest(),
    )


def gold(*, gold_id="g1", text="Drug A reduced deaths", kind="finding", modality="asserted", loc=None):
    return ScientificClaimGold(
        gold_id=gold_id,
        owner_id="alice",
        doc_id="doc1",
        generation=1,
        content_sha256="a" * 64,
        profile_fingerprint="b" * 64,
        claim_text=text,
        claim_type=kind,
        modality=modality,
        locator=loc or locator(),
    )


def proposal(*, key="p1", text="Drug A reduced deaths", kind="finding", modality="asserted", loc=None, confidence=0.9):
    return ScientificClaimProposal.create(
        owner_id="alice",
        doc_id="doc1",
        generation=1,
        content_sha256="a" * 64,
        profile_fingerprint="b" * 64,
        claim_key=key,
        claim_text=text,
        claim_type=kind,
        modality=modality,
        locator=loc or locator(),
        proposer_kind="model",
        proposer_id="extractor",
        extractor_name="claims",
        extractor_version="1",
        confidence=confidence,
        created_at=1.0,
    )


def test_perfect_match_has_exact_metrics_and_text_free_report():
    report = evaluate_scientific_claim_extraction(
        gold=(gold(),),
        proposals=(proposal(),),
    )

    assert report.precision == report.recall == report.f1 == 1.0
    assert report.exact_evidence_accuracy == 1.0
    assert report.exact_locator_accuracy == 1.0
    assert report.mean_span_iou == 1.0
    assert report.mean_claim_token_f1 == 1.0
    assert report.claim_type_accuracy == 1.0
    assert report.modality_accuracy == 1.0
    assert report.confidence_brier_score == pytest.approx(0.01)
    rendered = json.dumps(report, default=lambda value: value.__dict__)
    assert "Drug A" not in rendered
    assert report.contains_claim_text is False
    assert report.contains_evidence_text is False
    assert report.semantic_entailment_evaluated is False


def test_matching_is_one_to_one_deterministic_and_reports_unmatched_ids():
    predictions = (
        proposal(key="p2", confidence=0.2),
        proposal(key="p1", confidence=0.8),
    )
    first = evaluate_scientific_claim_extraction(
        gold=(gold(),), proposals=predictions
    )
    second = evaluate_scientific_claim_extraction(
        gold=(gold(),), proposals=reversed(predictions)
    )

    assert first.report_digest == second.report_digest
    assert first.matched_count == 1
    assert first.precision == 0.5
    assert first.recall == 1.0
    assert len(first.unmatched_proposal_ids) == 1
    assert first.matches[0].proposal_id == min(
        value.proposal_id for value in predictions
    )


def test_span_and_text_thresholds_prevent_weak_matches():
    weak = proposal(
        text="An unrelated conclusion",
        loc=locator(100, 120, evidence="different evidence"),
    )
    report = evaluate_scientific_claim_extraction(
        gold=(gold(),), proposals=(weak,)
    )

    assert report.matched_count == 0
    assert report.precision == report.recall == report.f1 == 0.0
    assert report.unmatched_gold_ids == ("g1",)
    assert report.unmatched_proposal_ids == (weak.proposal_id,)
    assert report.confidence_brier_score == pytest.approx(0.81)


def test_taxonomy_metrics_are_separate_from_detection_match():
    predicted = proposal(kind="hypothesis", modality="uncertain")
    report = evaluate_scientific_claim_extraction(
        gold=(gold(),), proposals=(predicted,)
    )

    assert report.matched_count == 1
    assert report.claim_type_accuracy == 0.0
    assert report.modality_accuracy == 0.0
    assert report.exact_evidence_accuracy == 1.0


def test_scope_duplicates_and_invalid_thresholds_fail_closed():
    other_scope = ScientificClaimProposal.create(
        owner_id="alice",
        doc_id="other",
        generation=1,
        content_sha256="a" * 64,
        profile_fingerprint="b" * 64,
        claim_key="p",
        claim_text="Drug A reduced deaths",
        claim_type="finding",
        modality="asserted",
        locator=locator(),
        proposer_kind="model",
        proposer_id="extractor",
        extractor_name="claims",
        extractor_version="1",
        confidence=0.9,
        created_at=1.0,
    )
    with pytest.raises(PermissionError, match="generation scope"):
        evaluate_scientific_claim_extraction(
            gold=(gold(),), proposals=(other_scope,)
        )

    value = proposal()
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_scientific_claim_extraction(
            gold=(gold(),), proposals=(value, value)
        )
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_scientific_claim_extraction(
            gold=(gold(),), proposals=(), minimum_span_iou=2.0
        )
