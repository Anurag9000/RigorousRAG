from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_artifact_boundary as boundary,
)
from tools import (
    evidence_graph_set_signed_retirement_restore_custody_artifact_reconcile as reconcile,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_artifact_journal_boundary import (
    GovernedRestoreCustodyArtifactJournal,
)


def path_digest(value):
    return hashlib.sha256(str(Path(value).resolve()).encode("utf-8")).hexdigest()


def install_dependencies(monkeypatch):
    snapshot = SimpleNamespace(owner_id="alice", snapshot_digest="1" * 64)
    monkeypatch.setattr(
        reconcile,
        "verify_signed_retirement_snapshot",
        lambda _path: snapshot,
    )
    monkeypatch.setattr(reconcile, "validate_terminal_snapshot", lambda _value: None)
    monkeypatch.setattr(
        reconcile,
        "canonical_target_path",
        lambda value: Path(value).resolve(),
    )
    monkeypatch.setattr(reconcile, "target_path_digest", path_digest)

    def file_sha(path, *, label):
        payload = Path(path).read_bytes()
        return hashlib.sha256(payload).hexdigest(), len(payload)

    monkeypatch.setattr(reconcile, "_file_sha256", file_sha)

    def verify(*, receipt_path, backup_path):
        receipt = Path(receipt_path)
        backup = Path(backup_path)
        if receipt.read_bytes() != b"receipt":
            raise RuntimeError("receipt mismatch")
        backup_sha, backup_size = file_sha(backup, label="backup")
        return SimpleNamespace(
            owner_id="alice",
            snapshot_digest="1" * 64,
            target_path_digest=path_digest(target),
            backup_sha256=backup_sha,
            backup_size_bytes=backup_size,
            receipt_digest="6" * 64,
            actor_id="actor",
            binding_method="process_environment",
            binding_digest="7" * 64,
        )

    target = None

    def set_target(value):
        nonlocal target
        target = Path(value).resolve()

    monkeypatch.setattr(reconcile, "verify_pre_restore_backup_receipt", verify)
    return snapshot, file_sha, set_target


def prepare(tmp_path, monkeypatch):
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text("{}", encoding="utf-8")
    target = tmp_path / "target.sqlite3"
    target.write_bytes(b"target")
    backup = tmp_path / "backup.sqlite3"
    receipt = tmp_path / "receipt.json"
    _snapshot, file_sha, set_target = install_dependencies(monkeypatch)
    set_target(target)
    journal = GovernedRestoreCustodyArtifactJournal(tmp_path / "attempts.sqlite3")
    seeded = reconcile.seed_restore_custody_artifact_attempt(
        snapshot_path=snapshot_path,
        target_db_path=target,
        backup_output_path=backup,
        receipt_output_path=receipt,
        journal=journal,
        now=1.0,
    )
    return journal, seeded, snapshot_path, target, backup, receipt, file_sha


def test_normal_pair_publication_completes_and_records_provenance(tmp_path, monkeypatch):
    journal, seeded, snapshot_path, target, backup, receipt, _file_sha = prepare(
        tmp_path, monkeypatch
    )

    def create(**kwargs):
        Path(kwargs["backup_output_path"]).write_bytes(b"backup")
        Path(kwargs["receipt_output_path"]).write_bytes(b"receipt")

    monkeypatch.setattr(reconcile, "create_pre_restore_backup_receipt", create)
    result = boundary.execute_restore_custody_artifact_attempt(
        seeded.artifact_id,
        snapshot_path=snapshot_path,
        target_db_path=target,
        backup_output_path=backup,
        receipt_output_path=receipt,
        actor=SimpleNamespace(),
        worker_id="worker",
        lease_seconds=30,
        journal=journal,
        now=2.0,
    )

    assert result.state == "completed"
    assert result.phase == "verified"
    assert result.disposition == "paired"
    assert result.artifact_pair_created is True
    stored = journal.get(seeded.artifact_id)
    assert stored.receipt_actor_id == "actor"
    assert stored.receipt_binding_method == "process_environment"
    assert stored.receipt_binding_digest == "7" * 64
    assert result.artifact_deletion_performed is False
    assert result.artifact_overwrite_performed is False


def test_crash_after_pair_publication_is_completed_by_recovery(tmp_path, monkeypatch):
    journal, seeded, snapshot_path, target, backup, receipt, _file_sha = prepare(
        tmp_path, monkeypatch
    )

    def create(**kwargs):
        Path(kwargs["backup_output_path"]).write_bytes(b"backup")
        Path(kwargs["receipt_output_path"]).write_bytes(b"receipt")

    monkeypatch.setattr(reconcile, "create_pre_restore_backup_receipt", create)

    def hook(phase):
        if phase == "artifacts_published":
            raise RuntimeError("process died after artifact publication")

    result = boundary.execute_restore_custody_artifact_attempt(
        seeded.artifact_id,
        snapshot_path=snapshot_path,
        target_db_path=target,
        backup_output_path=backup,
        receipt_output_path=receipt,
        actor=SimpleNamespace(),
        worker_id="worker",
        lease_seconds=30,
        journal=journal,
        now=2.0,
        _phase_hook=hook,
    )
    assert result.state == "completed"
    assert result.artifact_pair_created is False


def test_backup_only_receipt_only_and_collision_are_terminal_orphans(
    tmp_path,
    monkeypatch,
):
    cases = (
        ("backup", "backup_without_receipt"),
        ("receipt", "receipt_without_backup"),
        ("both", "artifact_collision"),
    )
    for name, disposition in cases:
        case = tmp_path / name
        case.mkdir()
        journal, seeded, snapshot_path, target, backup, receipt, _file_sha = prepare(
            case, monkeypatch
        )
        if name in {"backup", "both"}:
            backup.write_bytes(b"backup")
        if name in {"receipt", "both"}:
            receipt.write_bytes(b"wrong-receipt")
        monkeypatch.setattr(
            reconcile,
            "create_pre_restore_backup_receipt",
            lambda **kwargs: (_ for _ in ()).throw(
                AssertionError("existing artifacts must be observed before publication")
            ),
        )
        result = boundary.execute_restore_custody_artifact_attempt(
            seeded.artifact_id,
            snapshot_path=snapshot_path,
            target_db_path=target,
            backup_output_path=backup,
            receipt_output_path=receipt,
            actor=SimpleNamespace(),
            worker_id="worker",
            lease_seconds=30,
            journal=journal,
            now=2.0,
        )
        assert result.state == "orphaned"
        assert result.phase == "observed"
        assert result.disposition == disposition
        assert result.orphan_recorded is True
        assert backup.exists() == (name in {"backup", "both"})
        assert receipt.exists() == (name in {"receipt", "both"})


def test_completed_pair_is_revalidated_and_tampering_refuses(tmp_path, monkeypatch):
    journal, seeded, snapshot_path, target, backup, receipt, file_sha = prepare(
        tmp_path, monkeypatch
    )

    def create(**kwargs):
        Path(kwargs["backup_output_path"]).write_bytes(b"backup")
        Path(kwargs["receipt_output_path"]).write_bytes(b"receipt")

    monkeypatch.setattr(reconcile, "create_pre_restore_backup_receipt", create)
    boundary.execute_restore_custody_artifact_attempt(
        seeded.artifact_id,
        snapshot_path=snapshot_path,
        target_db_path=target,
        backup_output_path=backup,
        receipt_output_path=receipt,
        actor=SimpleNamespace(),
        worker_id="worker",
        lease_seconds=30,
        journal=journal,
        now=2.0,
    )

    def boundary_verify(*, receipt_path, backup_path):
        if not Path(receipt_path).exists():
            raise FileNotFoundError(receipt_path)
        backup_sha, backup_size = file_sha(backup_path, label="backup")
        return SimpleNamespace(
            owner_id="alice",
            snapshot_digest="1" * 64,
            target_path_digest=path_digest(target),
            backup_sha256=backup_sha,
            backup_size_bytes=backup_size,
            receipt_digest="6" * 64,
            actor_id="actor",
            binding_method="process_environment",
            binding_digest="7" * 64,
        )

    monkeypatch.setattr(boundary, "verify_pre_restore_backup_receipt", boundary_verify)
    monkeypatch.setattr(boundary, "_file_sha256", file_sha)
    assert boundary.execute_restore_custody_artifact_attempt(
        seeded.artifact_id,
        snapshot_path=snapshot_path,
        target_db_path=target,
        backup_output_path=backup,
        receipt_output_path=receipt,
        actor=SimpleNamespace(),
        worker_id="other",
        lease_seconds=30,
        journal=journal,
        now=3.0,
    ).state == "completed"

    receipt.unlink()
    with pytest.raises(boundary.RestoreCustodyArtifactRecoveryError):
        boundary.execute_restore_custody_artifact_attempt(
            seeded.artifact_id,
            snapshot_path=snapshot_path,
            target_db_path=target,
            backup_output_path=backup,
            receipt_output_path=receipt,
            actor=SimpleNamespace(),
            worker_id="other",
            lease_seconds=30,
            journal=journal,
            now=4.0,
        )
    assert journal.get(seeded.artifact_id).state == "completed"


def test_scope_mismatch_fails_before_claim(tmp_path, monkeypatch):
    journal, seeded, snapshot_path, target, backup, receipt, _file_sha = prepare(
        tmp_path, monkeypatch
    )
    with pytest.raises(RuntimeError, match="durable intent"):
        boundary.execute_restore_custody_artifact_attempt(
            seeded.artifact_id,
            snapshot_path=snapshot_path,
            target_db_path=target,
            backup_output_path=tmp_path / "different.sqlite3",
            receipt_output_path=receipt,
            actor=SimpleNamespace(),
            worker_id="worker",
            lease_seconds=30,
            journal=journal,
            now=2.0,
        )
    current = journal.get(seeded.artifact_id)
    assert current.state == "planned"
    assert current.attempt_count == 0
