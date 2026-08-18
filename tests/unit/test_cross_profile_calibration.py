from __future__ import annotations

import hashlib

import pytest

from evaluation.cross_profile_calibration import (
    CalibrationQualificationPolicy,
    qualify_calibrator,
    run_qualified_calibrated_fusion,
)
from tools.corpus_fusion import RetrievalCandidate
from tools.cross_profile_fusion import (
    CalibrationContract,
    CrossProfileFusionMode,
    CrossProfileFusionPolicy,
    ProfileRankedList,
    RetrieverScoreProfile,
    ScoreCalibrationExample,
    fit_isotonic_calibrator,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def contract(tag: str = "shared") -> CalibrationContract:
    return CalibrationContract(
        dataset_manifest_sha256=sha(f"dataset:{tag}"),
        split_sha256=sha(f"split:{tag}"),
        relevance_contract_sha256=sha(f"relevance:{tag}"),
        candidate_universe_sha256=sha(f"universe:{tag}"),
        domain_id="scientific",
    )


def profile(profile_id: str) -> RetrieverScoreProfile:
    return RetrieverScoreProfile(
        profile_id=profile_id,
        family=profile_id,
        scoring_contract_sha256=sha(f"score:{profile_id}"),
        model_profile_sha256=sha(f"model:{profile_id}"),
    )


def fit(p: RetrieverScoreProfile, c: CalibrationContract):
    return fit_isotonic_calibrator(
        profile=p,
        contract=c,
        examples=(
            ScoreCalibrationExample(0.0, False),
            ScoreCalibrationExample(0.1, False),
            ScoreCalibrationExample(0.9, True),
            ScoreCalibrationExample(1.0, True),
        ),
    )


def evaluation_examples():
    return (
        ScoreCalibrationExample(0.05, False),
        ScoreCalibrationExample(0.15, False),
        ScoreCalibrationExample(0.85, True),
        ScoreCalibrationExample(0.95, True),
    )


def qualification_policy() -> CalibrationQualificationPolicy:
    return CalibrationQualificationPolicy(
        min_examples=4,
        min_positive_examples=2,
        min_negative_examples=2,
        max_brier=0.25,
        max_ece=0.25,
        ece_bin_count=2,
    )


def ranked(profile_id: str, p: RetrieverScoreProfile, score: float):
    return ProfileRankedList(
        f"{profile_id}-list",
        p,
        (
            RetrievalCandidate(
                candidate_id=f"{profile_id}-candidate",
                corpus_id="papers",
                retriever_id=profile_id,
                document_id="doc",
                chunk_id="chunk",
                rank=1,
                raw_score=score,
            ),
        ),
    )


def test_qualification_requires_class_support_even_when_fit_is_valid() -> None:
    p = profile("dense")
    artifact = fit(p, contract())
    decision = qualify_calibrator(
        artifact,
        (
            ScoreCalibrationExample(0.8, True),
            ScoreCalibrationExample(0.9, True),
            ScoreCalibrationExample(1.0, True),
        ),
        policy=CalibrationQualificationPolicy(
            min_examples=3,
            min_positive_examples=2,
            min_negative_examples=1,
            max_brier=1.0,
            max_ece=1.0,
            ece_bin_count=2,
        ),
    )
    assert not decision.eligible
    assert "insufficient_negative_examples" in decision.reason_codes


def test_good_heldout_calibration_is_qualified_with_content_addressed_receipt() -> None:
    p = profile("dense")
    artifact = fit(p, contract())
    first = qualify_calibrator(artifact, evaluation_examples(), policy=qualification_policy())
    second = qualify_calibrator(artifact, evaluation_examples(), policy=qualification_policy())
    assert first.eligible
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.artifact_sha256 == artifact.artifact_sha256
    assert first.profile_sha256 == p.profile_sha256


def test_qualified_fusion_requires_exact_artifact_coverage() -> None:
    c = contract()
    p = profile("dense")
    artifact = fit(p, c)
    receipt = qualify_calibrator(artifact, evaluation_examples(), policy=qualification_policy())
    other = fit(p, contract("other"))
    policy = CrossProfileFusionPolicy(mode=CrossProfileFusionMode.CALIBRATED_LOGIT)
    with pytest.raises(ValueError, match="does not cover"):
        run_qualified_calibrated_fusion(
            (ranked("dense", p, 0.9),),
            calibrators={"dense": other},
            qualifications={"dense": receipt},
            policy=policy,
        )


def test_qualified_fusion_requires_same_qualification_policy_across_profiles() -> None:
    c = contract()
    dense = profile("dense")
    sparse = profile("sparse")
    dense_artifact = fit(dense, c)
    sparse_artifact = fit(sparse, c)
    first_policy = qualification_policy()
    second_policy = CalibrationQualificationPolicy(
        min_examples=4,
        min_positive_examples=2,
        min_negative_examples=2,
        max_brier=0.30,
        max_ece=0.25,
        ece_bin_count=2,
    )
    dense_receipt = qualify_calibrator(dense_artifact, evaluation_examples(), policy=first_policy)
    sparse_receipt = qualify_calibrator(sparse_artifact, evaluation_examples(), policy=second_policy)
    with pytest.raises(ValueError, match="qualification policy"):
        run_qualified_calibrated_fusion(
            (
                ranked("dense", dense, 0.9),
                ranked("sparse", sparse, 0.9),
            ),
            calibrators={"dense": dense_artifact, "sparse": sparse_artifact},
            qualifications={"dense": dense_receipt, "sparse": sparse_receipt},
            policy=CrossProfileFusionPolicy(mode=CrossProfileFusionMode.CALIBRATED_LOGIT),
        )


def test_qualified_fusion_accepts_compatible_promoted_calibrators() -> None:
    c = contract()
    dense = profile("dense")
    sparse = profile("sparse")
    dense_artifact = fit(dense, c)
    sparse_artifact = fit(sparse, c)
    qpolicy = qualification_policy()
    qualifications = {
        "dense": qualify_calibrator(dense_artifact, evaluation_examples(), policy=qpolicy),
        "sparse": qualify_calibrator(sparse_artifact, evaluation_examples(), policy=qpolicy),
    }
    run = run_qualified_calibrated_fusion(
        (
            ranked("dense", dense, 0.9),
            ranked("sparse", sparse, 0.9),
        ),
        calibrators={"dense": dense_artifact, "sparse": sparse_artifact},
        qualifications=qualifications,
        policy=CrossProfileFusionPolicy(mode=CrossProfileFusionMode.CALIBRATED_LOGIT),
    )
    assert run.result.mode is CrossProfileFusionMode.CALIBRATED_LOGIT
    assert run.result.candidates[0].fused_probability is not None
    assert run.receipt.calibration_contract_sha256 == c.contract_sha256
