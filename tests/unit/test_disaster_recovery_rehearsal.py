from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from orchestration.disaster_recovery_rehearsal import (
    CleanupEvidence,
    LocalFileBackupAsset,
    LocalFileRecoveryRehearsalBackend,
    RecoveryObjective,
    RecoveryPoint,
    RecoveryRehearsalSpec,
    RestoreEvidence,
    SQLiteRecoveryRehearsalStore,
    VerificationEvidence,
    advance_recovery_rehearsal,
)
from tools.disaster_recovery import create_backup, manifest_sha256


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def spec(*, rto: float = 20.0, rpo: float = 20.0) -> RecoveryRehearsalSpec:
    return RecoveryRehearsalSpec(
        owner_id="alice",
        incident_at=100.0,
        objective=RecoveryObjective(max_rto_seconds=rto, max_rpo_seconds=rpo),
        recovery_points=(
            RecoveryPoint(
                component="metadata",
                recovery_point_id="metadata-rp-7",
                backup_manifest_sha256=sha("manifest-metadata"),
                source_watermark_at=95.0,
                custody_evidence_sha256=sha("custody-metadata"),
            ),
            RecoveryPoint(
                component="graph",
                recovery_point_id="graph-rp-9",
                backup_manifest_sha256=sha("manifest-graph"),
                source_watermark_at=92.0,
                custody_evidence_sha256=sha("custody-graph"),
            ),
        ),
        policy_sha256=sha("recovery-policy"),
    )


class FakeBackend:
    def __init__(
        self,
        *,
        fail_restore_once: bool = False,
        wrong_manifest: bool = False,
        ready: bool = True,
    ) -> None:
        self.fail_restore_once = fail_restore_once
        self.wrong_manifest = wrong_manifest
        self.ready = ready
        self.restore_keys: list[str] = []
        self.verify_keys: list[str] = []
        self.cleanup_keys: list[str] = []

    def prepare_isolated_target(self, workflow, *, idempotency_key):
        assert idempotency_key == f"{workflow.drill_id}:prepare"
        return "/isolated/rehearsal-target"

    def restore(self, point, *, target_ref, idempotency_key, now):
        assert target_ref == "/isolated/rehearsal-target"
        self.restore_keys.append(idempotency_key)
        if self.fail_restore_once:
            self.fail_restore_once = False
            raise RuntimeError("transient restore outage at private path")
        return RestoreEvidence(
            component=point.component,
            recovery_point_id=point.recovery_point_id,
            restored_manifest_sha256=(
                sha("wrong-manifest")
                if self.wrong_manifest
                else point.backup_manifest_sha256
            ),
            restored_at=now,
            target_digest=sha(f"target:{point.component}"),
        )

    def verify(self, point, *, target_ref, restore, idempotency_key, now):
        assert target_ref == "/isolated/rehearsal-target"
        self.verify_keys.append(idempotency_key)
        return VerificationEvidence(
            component=point.component,
            recovery_point_id=point.recovery_point_id,
            restored_manifest_sha256=restore.restored_manifest_sha256,
            verification_evidence_sha256=sha(f"verification:{point.component}"),
            verified_at=now,
            ready=self.ready,
        )

    def cleanup(self, *, target_ref, idempotency_key, now):
        assert target_ref == "/isolated/rehearsal-target"
        self.cleanup_keys.append(idempotency_key)
        return CleanupEvidence(
            cleaned_at=now,
            target_digest=sha("isolated-root"),
            removed=True,
        )


def advance_to_restore_request(store, workflow, backend) -> None:
    assert advance_recovery_rehearsal(
        store=store, spec=workflow, backend=backend, worker_id="worker-a", now=100.0
    ).state == "prepare_requested"
    assert advance_recovery_rehearsal(
        store=store, spec=workflow, backend=backend, worker_id="worker-a", now=101.0
    ).state == "restoring"
    assert advance_recovery_rehearsal(
        store=store, spec=workflow, backend=backend, worker_id="worker-a", now=102.0
    ).state == "restore_requested"


def test_rehearsal_completes_with_content_addressed_objective_receipt(tmp_path) -> None:
    workflow = spec()
    store = SQLiteRecoveryRehearsalStore(tmp_path / "rehearsal.sqlite3")
    backend = FakeBackend()

    states = []
    for now in range(100, 108):
        states.append(
            advance_recovery_rehearsal(
                store=store,
                spec=workflow,
                backend=backend,
                worker_id="worker-a",
                now=float(now),
            ).state
        )

    assert states[-1] == "completed"
    completed = store.get(workflow.drill_id)
    assert completed.receipt is not None
    assert completed.receipt.verify_digest()
    assert completed.receipt.objective_met is True
    assert completed.receipt.max_observed_rpo_seconds == 8.0
    assert completed.receipt.observed_rto_seconds == 6.0
    assert completed.receipt.reason_codes == ()
    assert backend.cleanup_keys == [f"{workflow.drill_id}:cleanup"]


def test_transient_restore_retry_reuses_same_idempotency_key_and_stays_retryable(tmp_path) -> None:
    workflow = spec()
    store = SQLiteRecoveryRehearsalStore(tmp_path / "rehearsal.sqlite3")
    backend = FakeBackend(fail_restore_once=True)
    advance_to_restore_request(store, workflow, backend)

    failed_attempt = advance_recovery_rehearsal(
        store=store,
        spec=workflow,
        backend=backend,
        worker_id="worker-a",
        now=103.0,
    )
    assert failed_attempt.state == "restore_requested"
    assert failed_attempt.last_error is not None
    assert "/isolated" not in failed_attempt.last_error

    recovered = advance_recovery_rehearsal(
        store=store,
        spec=workflow,
        backend=backend,
        worker_id="worker-a",
        now=104.0,
    )
    assert recovered.state == "verify_requested"
    expected = f"{workflow.drill_id}:graph:restore"
    assert backend.restore_keys == [expected, expected]


def test_wrong_restore_manifest_never_advances_to_verification(tmp_path) -> None:
    workflow = spec()
    store = SQLiteRecoveryRehearsalStore(tmp_path / "rehearsal.sqlite3")
    backend = FakeBackend(wrong_manifest=True)
    advance_to_restore_request(store, workflow, backend)

    record = advance_recovery_rehearsal(
        store=store,
        spec=workflow,
        backend=backend,
        worker_id="worker-a",
        now=103.0,
    )
    assert record.state == "restore_requested"
    assert record.restores == ()
    assert record.last_error is not None


def test_rpo_rto_and_component_readiness_fail_closed_in_receipt(tmp_path) -> None:
    workflow = spec(rto=2.0, rpo=3.0)
    store = SQLiteRecoveryRehearsalStore(tmp_path / "rehearsal.sqlite3")
    backend = FakeBackend(ready=False)

    for now in range(100, 108):
        completed = advance_recovery_rehearsal(
            store=store,
            spec=workflow,
            backend=backend,
            worker_id="worker-a",
            now=float(now),
        )

    assert completed.state == "completed"
    assert completed.receipt is not None
    assert completed.receipt.objective_met is False
    assert "rpo_objective_exceeded" in completed.receipt.reason_codes
    assert "rto_objective_exceeded" in completed.receipt.reason_codes
    assert "component_not_ready:graph" in completed.receipt.reason_codes
    assert "component_not_ready:metadata" in completed.receipt.reason_codes


def test_same_worker_reclaim_fences_its_older_lease(tmp_path) -> None:
    workflow = spec()
    store = SQLiteRecoveryRehearsalStore(tmp_path / "rehearsal.sqlite3")
    record = store.ensure(workflow, now=100.0)
    stale = store.claim(
        workflow.drill_id,
        worker_id="worker-a",
        now=100.0,
        lease_seconds=20.0,
    )
    current = store.claim(
        workflow.drill_id,
        worker_id="worker-a",
        now=101.0,
        lease_seconds=20.0,
    )
    assert current.fencing_token > stale.fencing_token

    with pytest.raises(RuntimeError, match="expired or fenced"):
        store.transition(
            stale,
            expected_state=record.state,
            expected_revision=record.revision,
            state="prepare_requested",
            now=101.0,
        )


def test_component_names_cannot_escape_isolation_root() -> None:
    with pytest.raises(ValueError, match="simple non-path"):
        RecoveryPoint(
            component="../production",
            recovery_point_id="rp",
            backup_manifest_sha256=sha("manifest"),
            source_watermark_at=1.0,
            custody_evidence_sha256=sha("custody"),
        )


def test_local_adapter_requires_exact_recovered_population_and_custody(tmp_path) -> None:
    source_file = tmp_path / "metadata.sqlite3"
    source_file.write_bytes(b"metadata-backup")
    backup_root = tmp_path / "backup"
    manifest = create_backup(
        sources=(source_file,),
        destination=backup_root,
        generation="g-7",
    )
    custody = sha("custody-receipt")
    point = RecoveryPoint(
        component="metadata",
        recovery_point_id="metadata-rp",
        backup_manifest_sha256=manifest_sha256(manifest),
        source_watermark_at=95.0,
        custody_evidence_sha256=custody,
    )
    workflow = RecoveryRehearsalSpec(
        owner_id="alice",
        incident_at=100.0,
        objective=RecoveryObjective(20.0, 10.0),
        recovery_points=(point,),
        policy_sha256=sha("policy"),
    )
    backend = LocalFileRecoveryRehearsalBackend(
        tmp_path / "isolated",
        (
            LocalFileBackupAsset(
                component="metadata",
                source=backup_root,
                manifest=manifest,
                custody_evidence_sha256=custody,
            ),
        ),
    )
    target = backend.prepare_isolated_target(
        workflow,
        idempotency_key=f"{workflow.drill_id}:prepare",
    )
    restored = backend.restore(
        point,
        target_ref=target,
        idempotency_key=f"{workflow.drill_id}:metadata:restore",
        now=101.0,
    )
    assert backend.verify(
        point,
        target_ref=target,
        restore=restored,
        idempotency_key=f"{workflow.drill_id}:metadata:verify",
        now=102.0,
    ).ready is True

    extra = Path(target) / "metadata" / "stale-unmanifested.bin"
    extra.write_bytes(b"stale")
    assert backend.verify(
        point,
        target_ref=target,
        restore=restored,
        idempotency_key=f"{workflow.drill_id}:metadata:verify",
        now=103.0,
    ).ready is False

    cleanup = backend.cleanup(
        target_ref=target,
        idempotency_key=f"{workflow.drill_id}:cleanup",
        now=104.0,
    )
    assert cleanup.removed is True
    assert not Path(target).exists()
