from __future__ import annotations

import pytest

from tools.disaster_recovery import (
    CanaryAction,
    CanaryObservation,
    CanaryThresholds,
    RecoveryObjective,
    RestoreRehearsal,
    RollbackState,
    advance_rollback,
    build_backup_manifest,
    evaluate_canary,
    evaluate_recovery,
    prepare_rollback,
    verify_backup_manifest,
)
from tools.distributed_coordination import (
    CoordinationError,
    InMemoryDurableQueue,
    InMemoryLeaseCoordinator,
)
from tools.supply_chain import (
    BuildProvenance,
    Component,
    HMACSigner,
    Severity,
    Vulnerability,
    VulnerabilityPolicy,
    build_manifest,
    build_sbom,
    evaluate_vulnerabilities,
    parse_trivy,
    sha256_bytes,
    verify_manifest,
    verify_provenance_output,
)


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_fenced_leases_reject_stale_writers() -> None:
    clock = Clock()
    coordinator = InMemoryLeaseCoordinator(clock=clock)
    first = coordinator.acquire("index", "worker-a", 5)
    coordinator.assert_fencing_token("index", first.fencing_token)
    clock.advance(6)
    second = coordinator.acquire("index", "worker-b", 5)
    assert second.fencing_token == first.fencing_token + 1
    with pytest.raises(CoordinationError):
        coordinator.assert_valid(first)
    with pytest.raises(CoordinationError):
        coordinator.assert_fencing_token("index", first.fencing_token)


def test_queue_is_idempotent_and_redelivers_then_dead_letters() -> None:
    clock = Clock()
    queue = InMemoryDurableQueue(max_attempts=2, clock=clock)
    first_id = queue.enqueue({"job": "embed"}, idempotency_key="source-1")
    assert queue.enqueue({"job": "ignored"}, idempotency_key="source-1") == first_id
    first = queue.claim("worker-a", visibility_timeout=10)
    assert first is not None and first.attempts == 1
    clock.advance(11)
    second = queue.claim("worker-b", visibility_timeout=10)
    assert second is not None and second.message_id == first_id and second.attempts == 2
    queue.nack(second.receipt)
    assert queue.claim("worker-c", visibility_timeout=10) is None
    assert queue.dead_letters()[0].message_id == first_id


def test_queue_ack_and_retry_delay() -> None:
    clock = Clock()
    queue = InMemoryDurableQueue(clock=clock)
    queue.enqueue({"job": "parse"}, idempotency_key="source-2")
    claimed = queue.claim("worker", visibility_timeout=5)
    assert claimed is not None
    queue.nack(claimed.receipt, retry_delay=10)
    assert queue.claim("worker", visibility_timeout=5) is None
    clock.advance(10)
    retried = queue.claim("worker", visibility_timeout=5)
    assert retried is not None and retried.attempts == 2
    queue.ack(retried.receipt)
    assert queue.claim("worker", visibility_timeout=5) is None


def test_supply_chain_manifest_sbom_provenance_and_signature() -> None:
    files = {"model.bin": b"weights", "config.json": b"{}"}
    manifest = build_manifest(files)
    assert verify_manifest(manifest, files)
    assert not verify_manifest(manifest, {**files, "model.bin": b"tampered"})
    sbom = build_sbom([Component("b", "2"), Component("a", "1"), Component("a", "1")])
    assert [item.name for item in sbom.components] == ["a", "b"]
    output = b"wheel"
    provenance = BuildProvenance(
        "repo",
        "abc",
        "builder",
        "c" * 64,
        manifest.manifest_sha256,
        sha256_bytes(output),
    )
    assert verify_provenance_output(provenance, output)
    assert not verify_provenance_output(provenance, b"other")
    signer = HMACSigner(b"local-test-key")
    signature = signer.sign(provenance.canonical_bytes())
    assert signer.verify(provenance.canonical_bytes(), signature)
    assert not signer.verify(b"tampered", signature)


def test_vulnerability_policy_and_trivy_normalization() -> None:
    payload = {
        "Results": [
            {
                "Vulnerabilities": [
                    {
                        "PkgName": "demo",
                        "VulnerabilityID": "CVE-1",
                        "Severity": "HIGH",
                        "InstalledVersion": "1",
                        "FixedVersion": "2",
                    }
                ]
            }
        ]
    }
    parsed = parse_trivy(payload)
    assert parsed[0].severity == Severity.HIGH
    decision = evaluate_vulnerabilities(parsed, VulnerabilityPolicy(Severity.MEDIUM))
    assert not decision.allowed and decision.blocking == parsed
    assert evaluate_vulnerabilities(
        [Vulnerability("demo", "CVE-low", Severity.LOW)],
        VulnerabilityPolicy(Severity.MEDIUM),
    ).allowed


def test_backup_integrity_and_recovery_objectives() -> None:
    files = {"vector.snapshot": b"v", "graph.snapshot": b"g"}
    manifest = build_backup_manifest(files, created_at=90)
    assert verify_backup_manifest(manifest, files)
    assert not verify_backup_manifest(manifest, {**files, "graph.snapshot": b"bad"})
    rehearsal = RestoreRehearsal(
        incident_at=100,
        backup_at=90,
        restore_started_at=101,
        restore_completed_at=120,
        required_artifacts=("vector", "graph"),
        restored_artifacts=("vector", "graph"),
        integrity_ok=True,
    )
    assert evaluate_recovery(rehearsal, RecoveryObjective(15, 30)).ready
    failed = evaluate_recovery(
        RestoreRehearsal(**{**rehearsal.__dict__, "restored_artifacts": ("vector",)}),
        RecoveryObjective(5, 10),
    )
    assert set(failed.reason_codes) == {
        "rpo_exceeded",
        "rto_exceeded",
        "restore_artifacts_missing",
    }


def test_canary_and_rollback_state_machine() -> None:
    thresholds = CanaryThresholds(
        min_samples=100,
        max_error_rate=0.02,
        max_p95_latency_ms=500,
        min_quality_score=0.8,
    )
    hold = evaluate_canary(CanaryObservation(20, 0, 100, 1), thresholds)
    assert hold.action == CanaryAction.HOLD
    bad = evaluate_canary(CanaryObservation(100, 0.05, 600, 0.7), thresholds)
    assert bad.action == CanaryAction.ROLLBACK
    assert set(bad.reason_codes) == {
        "error_rate_exceeded",
        "latency_budget_exceeded",
        "quality_floor_missed",
    }
    good = evaluate_canary(CanaryObservation(100, 0.01, 400, 0.9), thresholds)
    assert good.action == CanaryAction.PROMOTE

    first = prepare_rollback("generator", "v2", "v1")
    assert first == prepare_rollback("generator", "v2", "v1")
    applied = advance_rollback(first, RollbackState.APPLIED)
    completed = advance_rollback(applied, RollbackState.COMPLETED)
    assert completed.state == RollbackState.COMPLETED
    assert advance_rollback(completed, RollbackState.COMPLETED) == completed
    with pytest.raises(ValueError):
        advance_rollback(first, RollbackState.COMPLETED)
