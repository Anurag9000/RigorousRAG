from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from tools.evidence_graph_claim_contracts import ClaimEvidenceLocator, ScientificClaimProposal
from tools.evidence_graph_claim_evaluation import (
    ScientificClaimGold,
    evaluate_scientific_claim_extraction,
)
from tools.evidence_graph_claim_extractor_benchmark import (
    aggregate_scientific_claim_extractor_benchmark,
    build_scientific_claim_extractor_benchmark_case,
)
from tools.evidence_graph_claim_extractor_registry import (
    SCIENTIFIC_CLAIM_OUTPUT_SCHEMA_SHA256,
    ScientificClaimExtractorRecord,
)
from tools.evidence_graph_relation_actor import ReviewActorBinding


def actor():
    return ReviewActorBinding.create(
        actor_id="admin",
        binding_method="process_environment",
        loaded_at=1.0,
    )


def record(version="1"):
    return ScientificClaimExtractorRecord.active(
        owner_id="alice",
        extractor_name="claims",
        extractor_version=version,
        extractor_kind="model",
        implementation_sha256="a" * 64,
        configuration_sha256="b" * 64,
        supported_claim_types=("finding",),
        supported_modalities=("asserted",),
        supported_languages=("en",),
        actor=actor(),
        now=1.0,
    )


def locator(*, evidence="evidence", start=0, end=10):
    return ClaimEvidenceLocator(
        section_index=0,
        page_number=1,
        char_start=start,
        char_end=end,
        evidence_sha256=hashlib.sha256(evidence.encode("utf-8")).hexdigest(),
    )


def proposal(value, *, key="p1", confidence=0.8):
    return ScientificClaimProposal.create(
        owner_id="alice",
        doc_id="doc1",
        generation=1,
        content_sha256="c" * 64,
        profile_fingerprint="d" * 64,
        claim_key=key,
        claim_text="Drug A helps",
        claim_type="finding",
        modality="asserted",
        locator=locator(),
        proposer_kind="model",
        proposer_id="extractor",
        extractor_name=value.extractor_name,
        extractor_version=value.extractor_version,
        confidence=confidence,
        metadata={
            "extractor_registry_record_digest": value.record_digest,
            "extractor_output_schema_sha256": SCIENTIFIC_CLAIM_OUTPUT_SCHEMA_SHA256,
        },
        created_at=1.0,
    )


def evaluation(value, predictions):
    gold = ScientificClaimGold(
        gold_id="g1",
        owner_id="alice",
        doc_id="doc1",
        generation=1,
        content_sha256="c" * 64,
        profile_fingerprint="d" * 64,
        claim_text="Drug A helps",
        claim_type="finding",
        modality="asserted",
        locator=locator(),
    )
    return evaluate_scientific_claim_extraction(
        gold=(gold,),
        proposals=tuple(predictions),
    )


def case(case_id="case-1", dataset_digit="e", *, predictions=None, version="1"):
    extractor = record(version)
    values = tuple(predictions or (proposal(extractor),))
    report = evaluation(extractor, values)
    return build_scientific_claim_extractor_benchmark_case(
        case_id=case_id,
        dataset_digest=dataset_digit * 64,
        evaluation_report=report,
        proposals=values,
        extractor_record=extractor,
    )


def test_case_requires_exact_registered_proposal_identity_and_is_text_free():
    value = case()

    assert value.matched_count == 1
    assert value.extractor_record_digest == record().record_digest
    assert value.contains_claim_text is False
    assert value.contains_evidence_text is False
    assert len(value.case_digest) == 64

    wrong = replace(
        proposal(record()),
        metadata={"extractor_registry_record_digest": "f" * 64},
    )
    report = evaluation(record(), (wrong,))
    with pytest.raises(PermissionError, match="registry digest"):
        build_scientific_claim_extractor_benchmark_case(
            case_id="wrong",
            dataset_digest="e" * 64,
            evaluation_report=report,
            proposals=(wrong,),
            extractor_record=record(),
        )


def test_suite_aggregates_detection_and_quality_metrics_deterministically():
    first = case("case-a", "e")
    extractor = record()
    unmatched = proposal(extractor, key="p2", confidence=0.2)
    second = case(
        "case-b",
        "f",
        predictions=(proposal(extractor, key="p1"), unmatched),
    )

    suite = aggregate_scientific_claim_extractor_benchmark(
        benchmark_id="benchmark-1",
        cases=(second, first),
    )
    replay = aggregate_scientific_claim_extractor_benchmark(
        benchmark_id="benchmark-1",
        cases=(first, second),
    )

    assert suite == replay
    assert suite.case_count == 2
    assert suite.gold_count == 2
    assert suite.proposal_count == 3
    assert suite.matched_count == 2
    assert suite.precision == pytest.approx(2 / 3)
    assert suite.recall == 1.0
    assert suite.exact_evidence_accuracy == 1.0
    assert suite.contains_claim_text is False
    assert suite.contains_evidence_text is False


def test_suite_refuses_duplicate_dataset_and_cross_version_cases():
    first = case("case-a", "e", version="1")
    duplicate_dataset = case("case-b", "e", version="1")
    with pytest.raises(ValueError, match="duplicate dataset"):
        aggregate_scientific_claim_extractor_benchmark(
            benchmark_id="benchmark-1",
            cases=(first, duplicate_dataset),
        )

    other_version = case("case-b", "f", version="2")
    with pytest.raises(PermissionError, match="extractor scope"):
        aggregate_scientific_claim_extractor_benchmark(
            benchmark_id="benchmark-1",
            cases=(first, other_version),
        )


def test_case_rejects_evaluation_report_and_proposal_identity_drift():
    extractor = record()
    first = proposal(extractor, key="p1")
    second = proposal(extractor, key="p2")
    report = evaluation(extractor, (first,))

    with pytest.raises(ValueError, match="proposal count"):
        build_scientific_claim_extractor_benchmark_case(
            case_id="case-1",
            dataset_digest="e" * 64,
            evaluation_report=report,
            proposals=(first, second),
            extractor_record=extractor,
        )
