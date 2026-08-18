from __future__ import annotations

import hashlib

import pytest

from evaluation.active_learning_adapters import calibration_drift_candidate, structured_support_candidate
from evaluation.cross_profile_calibration import CalibrationQualificationPolicy, qualify_calibrator
from evaluation.cross_profile_calibration_drift import CalibrationDriftPolicy, build_calibration_drift_reference, evaluate_calibration_drift
from evaluation.semantic_support import SemanticLabel
from evaluation.structured_data_support import NumericClaim, NumericOperator, Quantity, evaluate_numeric_claim, table_quantity_evidence
from evaluation.structured_review_acquisition import structured_authority_candidate
from scientific.document_structure import BoundingBox, DocumentRegion, RegionKind, SourceAnchor, StructuredDocument, StructuredTable, TableCell
from scientific.structured_data_quality import StructuredDataAuthorityPolicy, evaluate_table_quantity_authority
from tools.cross_profile_fusion import CalibrationContract, RetrieverScoreProfile, ScoreCalibrationExample, fit_isotonic_calibrator


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def table_document(confidence=0.4):
    anchor = SourceAnchor("doc", "gen", 1, "layout")
    region = DocumentRegion("table", RegionKind.TABLE, BoundingBox(0.1, 0.1, 0.9, 0.5), anchor)
    table = StructuredTable("table", (TableCell("cell", "table", 0, 1, 0, 1, "10.0", confidence=confidence),), 1, 1)
    return StructuredDocument("doc", "gen", (region,), tables=(table,))


def test_structured_neutral_support_becomes_abstaining_acquisition_candidate() -> None:
    document = table_document(confidence=1.0)
    evidence = table_quantity_evidence(document, table_region_id="table", cell_id="cell", quantity=Quantity(10.0, "m"), value_extraction_sha256=sha("parser"))
    claim = NumericClaim("claim", sha("claim"), NumericOperator.GT, "s", value=5.0)
    score = evaluate_numeric_claim(claim, evidence)
    assert score.label is SemanticLabel.NEUTRAL
    candidate = structured_support_candidate(score, owner_id="alice", group_id="table")
    assert candidate.signals.abstained
    assert candidate.item_sha256 == score.claim_sha256
    assert candidate.evidence_sha256s == (score.evidence_sha256,)


def test_review_required_authority_decision_routes_to_active_learning() -> None:
    document = table_document(confidence=0.2)
    evidence = table_quantity_evidence(document, table_region_id="table", cell_id="cell", quantity=Quantity(10.0, "m"), value_extraction_sha256=sha("parser"))
    decision = evaluate_table_quantity_authority(document, evidence, policy=StructuredDataAuthorityPolicy(min_table_cell_confidence=0.8))
    assert decision.action == "review_required"
    candidate = structured_authority_candidate(decision, owner_id="alice", item_sha256=sha("claim"), group_id="table")
    assert candidate.signals.abstained
    assert candidate.signals.uncertainty >= 0.75
    assert decision.decision_sha256 in candidate.evidence_sha256s


def test_authoritative_structured_evidence_is_not_sent_to_review_adapter() -> None:
    document = table_document(confidence=1.0)
    evidence = table_quantity_evidence(document, table_region_id="table", cell_id="cell", quantity=Quantity(10.0, "m"), value_extraction_sha256=sha("parser"))
    decision = evaluate_table_quantity_authority(document, evidence, policy=StructuredDataAuthorityPolicy(min_table_cell_confidence=0.8))
    assert decision.action == "authoritative"
    with pytest.raises(ValueError, match="does not require review"):
        structured_authority_candidate(decision, owner_id="alice", item_sha256=sha("claim"), group_id="table")


def test_calibration_requalification_decision_becomes_high_drift_candidate() -> None:
    profile = RetrieverScoreProfile("dense", "dense", sha("score"), sha("model"))
    contract = CalibrationContract(sha("dataset"), sha("split"), sha("relevance"), sha("universe"), "science")
    artifact = fit_isotonic_calibrator(
        profile=profile,
        contract=contract,
        examples=(ScoreCalibrationExample(0.0, False), ScoreCalibrationExample(0.2, False), ScoreCalibrationExample(0.8, True), ScoreCalibrationExample(1.0, True)),
    )
    qualification = qualify_calibrator(
        artifact,
        (ScoreCalibrationExample(0.1, False), ScoreCalibrationExample(0.9, True)),
        policy=CalibrationQualificationPolicy(min_examples=2, min_positive_examples=1, min_negative_examples=1, max_brier=1.0, max_ece=1.0, ece_bin_count=2),
    )
    reference = build_calibration_drift_reference(artifact, qualification, (0.1, 0.1, 0.9, 0.9), qualified_at=1.0, bin_count=2)
    decision = evaluate_calibration_drift(
        reference,
        artifact,
        (0.9,),
        observed_at=100.0,
        policy=CalibrationDriftPolicy(max_qualification_age_seconds=10.0, min_live_scores=4, min_labeled_examples=1, max_population_stability_index=1.0, max_jensen_shannon_divergence=1.0, max_brier=1.0, max_ece=1.0),
    )
    candidate = calibration_drift_candidate(decision, owner_id="alice")
    assert candidate.item_sha256 == artifact.artifact_sha256
    assert candidate.signals.abstained
    assert candidate.signals.drift == 1.0
