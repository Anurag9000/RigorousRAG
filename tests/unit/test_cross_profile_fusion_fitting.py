from __future__ import annotations

import hashlib

import pytest

from training.cross_profile_fusion_fitting import (
    FusionWeightExample,
    FusionWeightTrainingConfig,
    FusionWeightTrainingSpec,
    LearnedFusionWeightArtifact,
    advance_training,
    fit_fusion_weights,
    initialize_training_state,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def spec(*, source_revision: str = "0123456789abcdef0123456789abcdef01234567") -> FusionWeightTrainingSpec:
    return FusionWeightTrainingSpec(
        profile_ids=("dense", "sparse"),
        calibration_contract_sha256=sha("calibration-contract"),
        calibration_artifact_sha256s=(
            ("dense", sha("dense-calibrator")),
            ("sparse", sha("sparse-calibrator")),
        ),
        train_split_sha256=sha("train-split"),
        validation_split_sha256=sha("validation-split"),
        source_revision=source_revision,
        config=FusionWeightTrainingConfig(
            epochs=40,
            batch_size=2,
            learning_rate=0.2,
            l2=1e-4,
            gradient_clip_norm=5.0,
            patience=8,
            min_delta=1e-8,
            seed=11,
        ),
    )


def examples():
    return (
        FusionWeightExample({"dense": 0.95, "sparse": 0.55}, True),
        FusionWeightExample({"dense": 0.90, "sparse": 0.45}, True),
        FusionWeightExample({"dense": 0.10, "sparse": 0.55}, False),
        FusionWeightExample({"dense": 0.05, "sparse": 0.45}, False),
    )


def validation():
    return (
        FusionWeightExample({"dense": 0.92, "sparse": 0.52}, True),
        FusionWeightExample({"dense": 0.08, "sparse": 0.48}, False),
    )


def test_learned_weights_favor_more_predictive_calibrated_profile() -> None:
    artifact = fit_fusion_weights(spec(), examples(), validation())
    weights = dict(artifact.profile_weights)
    assert weights["dense"] > weights["sparse"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert artifact.probability({"dense": 0.95, "sparse": 0.5}) > artifact.probability(
        {"dense": 0.05, "sparse": 0.5}
    )


def test_mid_epoch_resume_is_exactly_deterministic() -> None:
    training_spec = spec()
    train = examples()
    valid = validation()
    direct = fit_fusion_weights(training_spec, train, valid)

    state = initialize_training_state(training_spec, train, valid)
    while not state.completed:
        state = advance_training(
            training_spec,
            state,
            train,
            valid,
            max_batches=1,
        )
    resumed = LearnedFusionWeightArtifact.build(spec=training_spec, state=state)
    assert resumed.artifact_sha256 == direct.artifact_sha256
    assert resumed.profile_weights == direct.profile_weights
    assert resumed.validation_loss == direct.validation_loss


def test_resume_rejects_changed_training_examples() -> None:
    training_spec = spec()
    train = examples()
    valid = validation()
    state = initialize_training_state(training_spec, train, valid)
    changed = train + (FusionWeightExample({"dense": 0.7, "sparse": 0.7}, True),)
    with pytest.raises(ValueError, match="differ from resumable state"):
        advance_training(training_spec, state, changed, valid, max_batches=1)


def test_training_spec_accepts_native_and_sha256_git_object_ids() -> None:
    native = spec(source_revision="0123456789abcdef0123456789abcdef01234567")
    extended = spec(source_revision="a" * 64)
    assert len(native.source_revision) == 40
    assert len(extended.source_revision) == 64


def test_training_spec_rejects_incomplete_calibration_artifact_coverage() -> None:
    with pytest.raises(ValueError, match="exactly cover"):
        FusionWeightTrainingSpec(
            profile_ids=("dense", "sparse"),
            calibration_contract_sha256=sha("contract"),
            calibration_artifact_sha256s=(("dense", sha("dense")),),
            train_split_sha256=sha("train"),
            validation_split_sha256=sha("validation"),
            source_revision="0123456789abcdef0123456789abcdef01234567",
            config=FusionWeightTrainingConfig(),
        )
