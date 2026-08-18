from __future__ import annotations

import hashlib
import json
import os

import pytest

from evaluation.cross_profile_calibration import CalibrationQualificationPolicy, qualify_calibrator
from evaluation.fusion_weight_promotion import FusionWeightPromotionPolicy, qualify_learned_fusion_weights
from tools.cross_profile_fusion import CalibrationContract, RetrieverScoreProfile, ScoreCalibrationExample, fit_isotonic_calibrator
from training.cross_profile_artifact_io import load_cross_profile_artifact, save_cross_profile_artifact
from training.cross_profile_fusion_fitting import FusionWeightExample, FusionWeightTrainingConfig, FusionWeightTrainingSpec, LearnedFusionWeightArtifact, advance_training, initialize_training_state
from training.cross_profile_listwise_fusion import FusionRankingCandidate, FusionRankingQuery, ListwiseFusionTrainingConfig, ListwiseFusionTrainingSpec, LearnedListwiseFusionArtifact, advance_listwise_training, initialize_listwise_training


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def profile(name: str = "dense") -> RetrieverScoreProfile:
    return RetrieverScoreProfile(name, name, sha(f"score:{name}"), sha(f"model:{name}"))


def contract() -> CalibrationContract:
    return CalibrationContract(sha("dataset"), sha("split"), sha("relevance"), sha("universe"), "science")


def calibrator():
    return fit_isotonic_calibrator(
        profile=profile(),
        contract=contract(),
        examples=(ScoreCalibrationExample(0.0, False), ScoreCalibrationExample(0.2, False), ScoreCalibrationExample(0.8, True), ScoreCalibrationExample(1.0, True)),
    )


def pointwise_spec() -> FusionWeightTrainingSpec:
    return FusionWeightTrainingSpec(
        profile_ids=("dense", "sparse"),
        calibration_contract_sha256=sha("contract"),
        calibration_artifact_sha256s=(("dense", sha("dense-cal")), ("sparse", sha("sparse-cal"))),
        train_split_sha256=sha("train"),
        validation_split_sha256=sha("valid"),
        source_revision="0123456789abcdef0123456789abcdef01234567",
        config=FusionWeightTrainingConfig(epochs=3, batch_size=2, patience=2),
    )


def point_examples():
    return (
        FusionWeightExample({"dense": 0.9, "sparse": 0.6}, True),
        FusionWeightExample({"dense": 0.1, "sparse": 0.4}, False),
    )


def ranking_query(tag: str) -> FusionRankingQuery:
    return FusionRankingQuery(
        sha(tag),
        (
            FusionRankingCandidate("good", {"dense": 0.9, "sparse": 0.6}, 2.0),
            FusionRankingCandidate("bad", {"dense": 0.1, "sparse": 0.4}, 0.0),
        ),
    )


def listwise_spec() -> ListwiseFusionTrainingSpec:
    return ListwiseFusionTrainingSpec(
        profile_ids=("dense", "sparse"),
        calibration_contract_sha256=sha("contract"),
        calibration_artifact_sha256s=(("dense", sha("dense-cal")), ("sparse", sha("sparse-cal"))),
        train_split_sha256=sha("list-train"),
        validation_split_sha256=sha("list-valid"),
        source_revision="0123456789abcdef0123456789abcdef01234567",
        config=ListwiseFusionTrainingConfig(epochs=3, batch_size=1, patience=2),
    )


def test_calibrator_and_qualification_round_trip_through_canonical_envelope(tmp_path) -> None:
    artifact = calibrator()
    path = tmp_path / "calibrator.json"
    digest = save_cross_profile_artifact(path, artifact)
    loaded = load_cross_profile_artifact(path)
    assert loaded == artifact
    assert len(digest) == 64

    receipt = qualify_calibrator(
        artifact,
        (ScoreCalibrationExample(0.1, False), ScoreCalibrationExample(0.9, True)),
        policy=CalibrationQualificationPolicy(min_examples=2, min_positive_examples=1, min_negative_examples=1, max_brier=1.0, max_ece=1.0, ece_bin_count=2),
    )
    receipt_path = tmp_path / "qualification.json"
    save_cross_profile_artifact(receipt_path, receipt)
    assert load_cross_profile_artifact(receipt_path) == receipt


def test_pointwise_training_state_round_trip_preserves_exact_resume_identity(tmp_path) -> None:
    spec = pointwise_spec()
    examples = point_examples()
    state = initialize_training_state(spec, examples, examples)
    state = advance_training(spec, state, examples, examples, max_batches=1)
    path = tmp_path / "point-state.json"
    save_cross_profile_artifact(path, state)
    loaded = load_cross_profile_artifact(path)
    assert loaded == state
    resumed = advance_training(spec, loaded, examples, examples, max_batches=1)
    direct = advance_training(spec, state, examples, examples, max_batches=1)
    assert resumed == direct


def test_listwise_training_state_round_trip_preserves_exact_resume_identity(tmp_path) -> None:
    spec = listwise_spec()
    train = (ranking_query("train-a"), ranking_query("train-b"))
    valid = (ranking_query("valid"),)
    state = initialize_listwise_training(spec, train, valid)
    state = advance_listwise_training(spec, state, train, valid, max_batches=1)
    path = tmp_path / "list-state.json"
    save_cross_profile_artifact(path, state)
    loaded = load_cross_profile_artifact(path)
    assert loaded == state
    resumed = advance_listwise_training(spec, loaded, train, valid, max_batches=1)
    direct = advance_listwise_training(spec, state, train, valid, max_batches=1)
    assert resumed == direct


def test_learned_artifact_and_promotion_round_trip(tmp_path) -> None:
    spec = pointwise_spec()
    examples = point_examples()
    state = initialize_training_state(spec, examples, examples)
    state = advance_training(spec, state, examples, examples)
    learned = LearnedFusionWeightArtifact.build(spec=spec, state=state)
    learned_path = tmp_path / "learned.json"
    save_cross_profile_artifact(learned_path, learned)
    assert load_cross_profile_artifact(learned_path) == learned

    promotion = qualify_learned_fusion_weights(
        learned,
        examples,
        evaluation_split_sha256=sha("promotion"),
        policy=FusionWeightPromotionPolicy(min_examples=2, min_positive_examples=1, min_negative_examples=1, max_log_loss=1.0, max_brier=1.0, max_single_profile_weight=1.0),
    )
    promotion_path = tmp_path / "promotion.json"
    save_cross_profile_artifact(promotion_path, promotion)
    assert load_cross_profile_artifact(promotion_path) == promotion


def test_listwise_learned_artifact_round_trip(tmp_path) -> None:
    spec = listwise_spec()
    train = (ranking_query("train-a"), ranking_query("train-b"))
    valid = (ranking_query("valid"),)
    state = initialize_listwise_training(spec, train, valid)
    state = advance_listwise_training(spec, state, train, valid)
    learned = LearnedListwiseFusionArtifact.build(spec, state)
    path = tmp_path / "list-learned.json"
    save_cross_profile_artifact(path, learned)
    assert load_cross_profile_artifact(path) == learned


def test_loader_rejects_noncanonical_or_digest_tampered_payload(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    save_cross_profile_artifact(path, calibrator())
    envelope = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical"):
        load_cross_profile_artifact(path)

    save_cross_profile_artifact(path, calibrator())
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["content_sha256"] = sha("tampered")
    path.write_text(json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest verification failed"):
        load_cross_profile_artifact(path)


def test_loader_rejects_symlink_path_when_supported(tmp_path) -> None:
    target = tmp_path / "target.json"
    save_cross_profile_artifact(target, calibrator())
    link = tmp_path / "link.json"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="symbolic link|redirecting"):
        load_cross_profile_artifact(link)
