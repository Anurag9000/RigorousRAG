from __future__ import annotations

import hashlib

from training.cross_profile_fusion_fitting import (
    FusionWeightExample,
    FusionWeightTrainingConfig,
    FusionWeightTrainingSpec,
    FusionWeightTrainingState,
    advance_training,
    examples_sha256,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def spec() -> FusionWeightTrainingSpec:
    return FusionWeightTrainingSpec(
        profile_ids=("dense", "sparse"),
        calibration_contract_sha256=sha("contract"),
        calibration_artifact_sha256s=(("dense", sha("dense-cal")), ("sparse", sha("sparse-cal"))),
        train_split_sha256=sha("train"),
        validation_split_sha256=sha("validation"),
        source_revision="0123456789abcdef0123456789abcdef01234567",
        config=FusionWeightTrainingConfig(
            epochs=2,
            batch_size=2,
            learning_rate=0.1,
            l2=0.5,
            gradient_clip_norm=100.0,
            patience=2,
            seed=3,
        ),
    )


def rows(scale: float):
    return (
        FusionWeightExample({"dense": 0.9, "sparse": 0.4}, True, weight=1.0 * scale),
        FusionWeightExample({"dense": 0.2, "sparse": 0.7}, False, weight=2.0 * scale),
    )


def state(training_spec: FusionWeightTrainingSpec, train, validation) -> FusionWeightTrainingState:
    return FusionWeightTrainingState(
        spec_sha256=training_spec.spec_sha256,
        train_examples_sha256=examples_sha256(train, training_spec.profile_ids),
        validation_examples_sha256=examples_sha256(validation, training_spec.profile_ids),
        epoch=0,
        batch_index=0,
        theta=(1.0, -1.0),
        best_theta=(1.0, -1.0),
        best_validation_loss=None,
        best_epoch=None,
        stale_epochs=0,
        completed=False,
    )


def test_uniform_sample_weight_scaling_does_not_change_l2_strength() -> None:
    training_spec = spec()
    unit = rows(1.0)
    scaled = rows(10.0)

    unit_after = advance_training(
        training_spec,
        state(training_spec, unit, unit),
        unit,
        unit,
        max_batches=1,
    )
    scaled_after = advance_training(
        training_spec,
        state(training_spec, scaled, scaled),
        scaled,
        scaled,
        max_batches=1,
    )

    assert scaled_after.theta == unit_after.theta
