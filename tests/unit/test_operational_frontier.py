from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tools.disaster_recovery import (
    CanaryObservation,
    create_backup,
    evaluate_canary,
    restore_backup,
    verify_backup,
)
from tools.distributed_coordination import InMemoryLeaseCoordinator, SQLLeaseCoordinator
from tools.feedback_promotion import (
    CandidateMetrics,
    PromotionPolicy,
    build_feedback_batch,
    evaluate_promotion,
)
from tools.feedback_store import ActiveLearningExample
from tools.release_supply_chain import (
    ReleaseProvenance,
    VulnerabilitySummary,
    build_minimal_sbom,
    evaluate_supply_chain,
    verify_ed25519_signature,
    vulnerability_summary_from_records,
)
from tools.service_slo import SLOObjective, StageObservation, TelemetryRecorder, evaluate_slo


def _example(kind: str, subject: str, weight: float = 1.0) -> ActiveLearningExample:
    return ActiveLearningExample(
        kind=kind,
        subject_id=subject,
        weight=weight,
        metadata={"source": "review"},
        query_sha256="a" * 64,
        evidence_sha256="b" * 64,
    )


def test_feedback_batch_is_order_independent_and_privacy_preserving() -> None:
    first = _example("answer_correct", "answer-1")
    second = _example("answer_incorrect", "answer-2")
    left = build_feedback_batch(owner_id="owner", examples=[first, second])
    right = build_feedback_batch(owner_id="owner", examples=[second, first])
    assert left == right
    assert left.example_count == 2
    assert left.positive_weight == 1.0
    assert left.negative_weight == 1.0


def test_promotion_binds_feedback_and_resource_gates() -> None:
    batch = build_feedback_batch(
        owner_id="owner",
        examples=[_example("answer_correct", f"a-{index}") for index in range(18)]
        + [_example("answer_incorrect", f"b-{index}") for index in range(2)],
    )
    decision = evaluate_promotion(
        batch=batch,
        baseline_version="v1",
        candidate_version="v2",
        baseline=CandidateMetrics(quality=0.80, p95_latency_ms=100, estimated_cost=1.0),
        candidate=CandidateMetrics(quality=0.82, p95_latency_ms=110, estimated_cost=1.05),
        policy=PromotionPolicy(min_examples=20, min_negative_weight_fraction=0.10, min_quality_gain=0.01),
    )
    assert decision.eligible
    assert not decision.reason_codes


def test_promotion_refuses_insufficient_feedback() -> None:
    batch = build_feedback_batch(owner_id="owner", examples=[_example("answer_correct", "a")])
    decision = evaluate_promotion(
        batch=batch,
        baseline_version="v1",
        candidate_version="v2",
        baseline=CandidateMetrics(0.8, 100, 1),
        candidate=CandidateMetrics(0.9, 100, 1),
    )
    assert not decision.eligible
    assert "insufficient_feedback_examples" in decision.reason_codes


def test_telemetry_records_success_and_failure() -> None:
    recorder = TelemetryRecorder()
    with recorder.stage("retrieval", trace_id="trace") as span:
        span["tokens"] = 12
        span["estimated_cost"] = 0.01
    with pytest.raises(RuntimeError), recorder.stage("generation", trace_id="trace"):
        raise RuntimeError("boom")
    assert [item.success for item in recorder.observations] == [True, False]
    assert recorder.observations[0].tokens == 12


def test_slo_evaluates_availability_latency_and_error_budget() -> None:
    observations = [
        StageObservation("t1", "request", 50, True),
        StageObservation("t2", "request", 250, True),
        StageObservation("t3", "request", 50, False),
    ]
    report = evaluate_slo(
        observations,
        SLOObjective(availability_target=0.66, latency_target_ms=200, latency_success_fraction=0.33),
    )
    assert report.request_count == 3
    assert report.availability == pytest.approx(2 / 3)
    assert report.within_slo
    assert report.error_budget_consumed == 1


def test_in_memory_leases_fence_expired_holders() -> None:
    now = [10.0]
    coordinator = InMemoryLeaseCoordinator(clock=lambda: now[0])
    first = coordinator.acquire(name="graph", holder="a", ttl_seconds=1)
    assert first is not None
    assert coordinator.acquire(name="graph", holder="b", ttl_seconds=1) is None
    now[0] = 12.0
    second = coordinator.acquire(name="graph", holder="b", ttl_seconds=1)
    assert second is not None
    assert second.token > first.token
    assert not coordinator.release(first)


def test_sql_leases_are_durable_and_fenced(tmp_path: Path) -> None:
    path = tmp_path / "leases.sqlite3"
    now = [100.0]
    one = SQLLeaseCoordinator(path, clock=lambda: now[0])
    two = SQLLeaseCoordinator(path, clock=lambda: now[0])
    first = one.acquire(name="migration", holder="one", ttl_seconds=1)
    assert first is not None
    assert two.acquire(name="migration", holder="two", ttl_seconds=1) is None
    now[0] += 2
    second = two.acquire(name="migration", holder="two", ttl_seconds=1)
    assert second is not None
    assert second.token == first.token + 1
    assert not one.release(first)


def test_supply_chain_gate_requires_signature_sbom_lock_and_vulnerability_budget() -> None:
    provenance = ReleaseProvenance(
        commit_sha="a" * 40,
        dependency_lock_sha256="b" * 64,
        sbom_sha256="c" * 64,
        artifact_sha256="d" * 64,
        image_digest="sha256:" + "e" * 64,
        workflow="release-locks",
        run_id="123",
    )
    bad = evaluate_supply_chain(
        provenance=provenance,
        vulnerabilities=VulnerabilitySummary(high=1),
        signature_verified=False,
        sbom_present=True,
        hashed_lock_verified=True,
    )
    assert not bad.eligible
    assert bad.reason_codes == ("signature_not_verified", "high_vulnerability_budget_exceeded")
    good = evaluate_supply_chain(
        provenance=provenance,
        vulnerabilities=VulnerabilitySummary(),
        signature_verified=True,
        sbom_present=True,
        hashed_lock_verified=True,
    )
    assert good.eligible


def test_minimal_sbom_is_deterministic() -> None:
    left = build_minimal_sbom(components=[{"name": "b", "version": "2"}, {"name": "a", "version": "1"}])
    right = build_minimal_sbom(components=[{"name": "a", "version": "1"}, {"name": "b", "version": "2"}])
    assert left == right


def test_backup_restore_verifies_checksums(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "registry.sqlite3").write_bytes(b"registry")
    (source / "manifest.jsonl").write_bytes(b"manifest")
    backup = tmp_path / "backup"
    manifest = create_backup(
        sources=[source / "registry.sqlite3", source / "manifest.jsonl"],
        destination=backup,
        generation="g1",
        encryption_key_id="kms:key/1",
    )
    assert verify_backup(source=backup, manifest=manifest)
    restored = tmp_path / "restored"
    report = restore_backup(source=backup, destination=restored, manifest=manifest)
    assert report.restored == ("manifest.jsonl", "registry.sqlite3")
    assert (restored / "registry.sqlite3").read_bytes() == b"registry"


def test_backup_refuses_tamper(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"before")
    backup = tmp_path / "backup"
    manifest = create_backup(sources=[source], destination=backup, generation="g1")
    (backup / "source.bin").write_bytes(b"after")
    assert not verify_backup(source=backup, manifest=manifest)
    with pytest.raises(ValueError, match="verification failed"):
        restore_backup(source=backup, destination=tmp_path / "restore", manifest=manifest)


def test_canary_policy_requests_rollback_on_regression() -> None:
    decision = evaluate_canary(
        CanaryObservation(
            requests=100,
            errors=2,
            baseline_p95_latency_ms=100,
            canary_p95_latency_ms=130,
            quality_delta=-0.01,
        )
    )
    assert decision.rollback
    assert not decision.promote
    assert set(decision.reason_codes) == {
        "canary_error_rate_exceeded",
        "canary_latency_regression",
        "canary_quality_regression",
    }


def test_supply_chain_normalizes_scanner_records_and_verifies_signature() -> None:
    summary = vulnerability_summary_from_records(
        [{"severity": "HIGH"}, {"severity": "low"}, {"severity": "high"}]
    )
    assert summary.high == 2
    assert summary.low == 1

    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    payload = b"release-provenance"
    signature = private.sign(payload)
    assert verify_ed25519_signature(public_key=public, payload=payload, signature=signature)
    assert not verify_ed25519_signature(public_key=public, payload=b"tampered", signature=signature)
