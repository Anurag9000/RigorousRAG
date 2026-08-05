from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from tools.evidence_graph_claim_contracts import ClaimEvidenceLocator, ScientificClaimProposal
from tools.evidence_graph_claim_evaluation import (
    ScientificClaimGold,
    evaluate_scientific_claim_extraction,
)
from tools.evidence_graph_claim_evaluation_verification import (
    verify_scientific_claim_evaluation_report,
)


def report():
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
    return evaluate_scientific_claim_extraction(
        gold=(gold,), proposals=(proposal,)
    )


def test_report_verification_accepts_exact_digest_and_thresholds():
    value = report()
    assert verify_scientific_claim_evaluation_report(value) is value


def test_report_verification_rejects_metric_and_threshold_tampering():
    value = report()
    with pytest.raises(ValueError, match="digest"):
        verify_scientific_claim_evaluation_report(
            replace(value, precision=0.0)
        )
    with pytest.raises(ValueError, match="digest"):
        verify_scientific_claim_evaluation_report(
            value,
            minimum_span_iou=0.6,
        )
