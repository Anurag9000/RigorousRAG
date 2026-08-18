from __future__ import annotations

import hashlib

import pytest

from evaluation.cross_profile_calibration import CalibrationQualificationPolicy, qualify_calibrator
from tools.corpus_fusion import RetrievalCandidate
from tools.cross_profile_fusion import CalibrationContract, ProfileRankedList, RetrieverScoreProfile, ScoreCalibrationExample, fit_isotonic_calibrator
from tools.learned_cross_profile_policy import LearnedFusionExecutionReceipt, run_learned_cross_profile_fusion
from training.cross_profile_fusion_fitting import FusionWeightExample, FusionWeightTrainingConfig, FusionWeightTrainingSpec, fit_fusion_weights


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def profile(name: str) -> RetrieverScoreProfile:
    return RetrieverScoreProfile(name, name, sha(f"score:{name}"), sha(f"model:{name}"))


def contract(tag: str = "shared") -> CalibrationContract:
    return CalibrationContract(sha(f"dataset:{tag}"), sha(f"split:{tag}"), sha(f"relevance:{tag}"), sha(f"universe:{tag}"), "scientific")


def calibrator(p: RetrieverScoreProfile, c: CalibrationContract):
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


def qualification(artifact):
    return qualify_calibrator(
        artifact,
        (
            ScoreCalibrationExample(0.05, False),
            ScoreCalibrationExample(0.15, False),
            ScoreCalibrationExample(0.85, True),
            ScoreCalibrationExample(0.95, True),
        ),
        policy=CalibrationQualificationPolicy(
            min_examples=4,
            min_positive_examples=2,
            min_negative_examples=2,
            max_brier=0.25,
            max_ece=0.25,
            ece_bin_count=2,
        ),
    )


def learned_artifact(dense_calibrator, sparse_calibrator):
    spec = FusionWeightTrainingSpec(
        profile_ids=("dense", "sparse"),
        calibration_contract_sha256=dense_calibrator.calibration_contract_sha256,
        calibration_artifact_sha256s=(
            ("dense", dense_calibrator.artifact_sha256),
            ("sparse", sparse_calibrator.artifact_sha256),
        ),
        train_split_sha256=sha("fusion-train"),
        validation_split_sha256=sha("fusion-validation"),
        source_revision="0123456789abcdef0123456789abcdef01234567",
        config=FusionWeightTrainingConfig(epochs=20, batch_size=2, learning_rate=0.2, patience=5, seed=3),
    )
    train = (
        FusionWeightExample({"dense": 0.95, "sparse": 0.6}, True),
        FusionWeightExample({"dense": 0.90, "sparse": 0.4}, True),
        FusionWeightExample({"dense": 0.10, "sparse": 0.6}, False),
        FusionWeightExample({"dense": 0.05, "sparse": 0.4}, False),
    )
    valid = (
        FusionWeightExample({"dense": 0.92, "sparse": 0.55}, True),
        FusionWeightExample({"dense": 0.08, "sparse": 0.45}, False),
    )
    return fit_fusion_weights(spec, train, valid)


def ranked(p: RetrieverScoreProfile, score: float):
    return ProfileRankedList(
        f"{p.profile_id}-list",
        p,
        (
            RetrievalCandidate(
                candidate_id=f"{p.profile_id}-candidate",
                corpus_id="papers",
                retriever_id=p.profile_id,
                document_id="doc",
                chunk_id="chunk",
                rank=1,
                raw_score=score,
            ),
        ),
    )


def test_learned_runtime_binding_accepts_exact_training_lineage() -> None:
    c = contract()
    dense = profile("dense")
    sparse = profile("sparse")
    dense_cal = calibrator(dense, c)
    sparse_cal = calibrator(sparse, c)
    learned = learned_artifact(dense_cal, sparse_cal)
    execution = run_learned_cross_profile_fusion(
        (ranked(dense, 0.9), ranked(sparse, 0.9)),
        learned_weights=learned,
        calibrators={"dense": dense_cal, "sparse": sparse_cal},
        qualifications={"dense": qualification(dense_cal), "sparse": qualification(sparse_cal)},
    )
    assert execution.run.result.candidates
    assert execution.receipt.learned_artifact_sha256 == learned.artifact_sha256
    assert execution.receipt.governed_receipt_sha256 == execution.run.receipt.receipt_sha256


def test_runtime_binding_rejects_calibrator_not_used_to_train_weights() -> None:
    c = contract()
    dense = profile("dense")
    sparse = profile("sparse")
    dense_cal = calibrator(dense, c)
    sparse_cal = calibrator(sparse, c)
    learned = learned_artifact(dense_cal, sparse_cal)
    replacement = calibrator(dense, contract("replacement"))
    with pytest.raises(ValueError, match="training lineage"):
        run_learned_cross_profile_fusion(
            (ranked(dense, 0.9), ranked(sparse, 0.9)),
            learned_weights=learned,
            calibrators={"dense": replacement, "sparse": sparse_cal},
            qualifications={"dense": qualification(replacement), "sparse": qualification(sparse_cal)},
        )


def test_runtime_binding_rejects_changed_profile_population() -> None:
    c = contract()
    dense = profile("dense")
    sparse = profile("sparse")
    dense_cal = calibrator(dense, c)
    sparse_cal = calibrator(sparse, c)
    learned = learned_artifact(dense_cal, sparse_cal)
    with pytest.raises(ValueError, match="profile set"):
        run_learned_cross_profile_fusion(
            (ranked(dense, 0.9),),
            learned_weights=learned,
            calibrators={"dense": dense_cal, "sparse": sparse_cal},
            qualifications={"dense": qualification(dense_cal), "sparse": qualification(sparse_cal)},
        )


def test_execution_receipt_is_tamper_evident() -> None:
    c = contract()
    dense = profile("dense")
    sparse = profile("sparse")
    dense_cal = calibrator(dense, c)
    sparse_cal = calibrator(sparse, c)
    learned = learned_artifact(dense_cal, sparse_cal)
    execution = run_learned_cross_profile_fusion(
        (ranked(dense, 0.9), ranked(sparse, 0.9)),
        learned_weights=learned,
        calibrators={"dense": dense_cal, "sparse": sparse_cal},
        qualifications={"dense": qualification(dense_cal), "sparse": qualification(sparse_cal)},
    )
    receipt = execution.receipt
    with pytest.raises(ValueError, match="does not match"):
        LearnedFusionExecutionReceipt(
            learned_artifact_sha256=receipt.learned_artifact_sha256,
            governed_receipt_sha256=receipt.governed_receipt_sha256,
            calibration_contract_sha256=receipt.calibration_contract_sha256,
            profile_artifact_sha256s=receipt.profile_artifact_sha256s,
            receipt_sha256=sha("tampered"),
        )
