import hashlib

import pytest

from tools.training_lineage import (
    TrainingLineage,
    TrainingOutcome,
    TrainingRequest,
    build_privacy_safe_replay_manifest,
    continual_transfer_metrics,
)


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def test_training_lineage_is_deterministic_and_binds_parent_data_code_seed_and_output():
    request_a = TrainingRequest(
        run_id="run-1",
        parent_artifact_sha256=_digest("parent"),
        dataset_sha256=_digest("dataset"),
        code_revision="abc123",
        seed=17,
        config={"lr": 0.001, "epochs": 3},
    )
    request_b = TrainingRequest(
        run_id="run-1",
        parent_artifact_sha256=_digest("parent"),
        dataset_sha256=_digest("dataset"),
        code_revision="abc123",
        seed=17,
        config={"epochs": 3, "lr": 0.001},
    )
    outcome = TrainingOutcome(_digest("adapter"), (_digest("eval-a"),), "provider-run-9")

    lineage_a = TrainingLineage.bind(request_a, outcome)
    lineage_b = TrainingLineage.bind(request_b, outcome)

    assert request_a.config_sha256 == request_b.config_sha256
    assert request_a.request_sha256 == request_b.request_sha256
    assert lineage_a.lineage_sha256 == lineage_b.lineage_sha256
    assert len(lineage_a.lineage_sha256) == 64


def test_training_lineage_changes_when_seed_or_output_changes():
    base = dict(
        run_id="run",
        parent_artifact_sha256=_digest("parent"),
        dataset_sha256=_digest("data"),
        code_revision="rev",
        config={"x": 1},
    )
    a = TrainingRequest(seed=1, **base)
    b = TrainingRequest(seed=2, **base)
    out_a = TrainingOutcome(_digest("out-a"))
    out_b = TrainingOutcome(_digest("out-b"))
    assert a.request_sha256 != b.request_sha256
    assert TrainingLineage.bind(a, out_a).lineage_sha256 != TrainingLineage.bind(a, out_b).lineage_sha256


def test_replay_manifest_contains_no_raw_ids_and_is_owner_scoped():
    salt = b"0123456789abcdef0123456789abcdef"
    manifest_a = build_privacy_safe_replay_manifest(
        owner_id="tenant-a", example_ids=["patient@example.com", "document-42"], secret_salt=salt
    )
    manifest_b = build_privacy_safe_replay_manifest(
        owner_id="tenant-b", example_ids=["patient@example.com", "document-42"], secret_salt=salt
    )
    serialized = repr(manifest_a)
    assert "patient@example.com" not in serialized
    assert "document-42" not in serialized
    assert manifest_a.example_digests != manifest_b.example_digests
    assert len(manifest_a.manifest_sha256) == 64


def test_replay_manifest_is_order_independent_and_rejects_duplicates():
    salt = b"0123456789abcdef"
    a = build_privacy_safe_replay_manifest(owner_id="o", example_ids=["a", "b"], secret_salt=salt)
    b = build_privacy_safe_replay_manifest(owner_id="o", example_ids=["b", "a"], secret_salt=salt)
    assert a.manifest_sha256 == b.manifest_sha256
    with pytest.raises(ValueError, match="duplicate replay"):
        build_privacy_safe_replay_manifest(owner_id="o", example_ids=["a", "a"], secret_salt=salt)


def test_continual_metrics_measure_forgetting_and_forward_transfer():
    report = continual_transfer_metrics(
        [
            [0.80, 0.30, 0.20],
            [0.75, 0.82, 0.25],
            [0.70, 0.78, 0.85],
        ],
        baseline=[0.10, 0.20, 0.15],
    )
    assert report.per_task_forgetting == pytest.approx((0.10, 0.04))
    assert report.average_forgetting == pytest.approx(0.07)
    assert report.per_task_forward_transfer == pytest.approx((0.10, 0.10))
    assert report.average_forward_transfer == pytest.approx(0.10)


def test_continual_metrics_fail_closed_on_malformed_inputs():
    with pytest.raises(ValueError):
        continual_transfer_metrics([], baseline=[])
    with pytest.raises(ValueError):
        continual_transfer_metrics([[1.0, 0.2]], baseline=[0.0, 0.0])
    with pytest.raises(ValueError):
        continual_transfer_metrics([[1.0], [float("nan")]], baseline=[0.0])


def test_digest_validation_rejects_unbound_artifacts():
    with pytest.raises(ValueError, match="SHA-256"):
        TrainingRequest(
            run_id="x",
            parent_artifact_sha256="not-a-digest",
            dataset_sha256=_digest("data"),
            code_revision="rev",
            seed=1,
            config={},
        )
