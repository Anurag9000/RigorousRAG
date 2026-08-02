from __future__ import annotations

import json
import os
import stat
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tools import evidence_graph_set_signed_retirement_restore_custody_cli as cli
from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_contracts import (
    SignedPublicationRetirementAttempt,
)
from tools.evidence_graph_set_signed_retirement_journal import (
    SignedPublicationRetirementJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_boundary import (
    create_post_restore_comparison_receipt,
    create_pre_restore_backup_receipt,
    verify_post_restore_comparison_receipt,
    verify_pre_restore_backup_receipt,
)
from tools.evidence_graph_set_signed_retirement_restore_mutation import (
    target_path_digest,
)
from tools.evidence_graph_set_signed_retirement_snapshot import (
    export_signed_retirement_snapshot,
)


def actor(actor_id="operator"):
    return ReviewActorBinding.create(
        actor_id=actor_id,
        binding_method="process_environment",
        loaded_at=1.0,
    )


def terminal(digit: str):
    value = SignedPublicationRetirementAttempt.create(
        owner_id="alice",
        publication_operation_id=digit * 64,
        graph_set_key="review",
        signed_candidate_set_id="2" * 64,
        signed_candidate_set_digest="3" * 64,
        authorization_candidate_set_id="4" * 64,
        signed_authority_digest="5" * 64,
        now=1.0,
    )
    return replace(
        value,
        state="cancelled",
        updated_at=2.0,
        completed_at=2.0,
    )


class Values:
    def __init__(self, values=()):
        self.values = tuple(values)

    def list(self, **kwargs):
        return self.values[: kwargs["limit"]]


def fixture(tmp_path):
    snapshot_path = tmp_path / "snapshot.json"
    record = terminal("1")
    snapshot = export_signed_retirement_snapshot(
        owner_id="alice",
        journal=Values((record,)),
        output_path=snapshot_path,
        now=10.0,
        limit=100,
    )
    target = SignedPublicationRetirementJournal(tmp_path / "target.sqlite3")
    return snapshot_path, snapshot, record, target


def pre_receipt(tmp_path):
    snapshot_path, snapshot, record, target = fixture(tmp_path)
    backup = tmp_path / "backup.sqlite3"
    receipt_path = tmp_path / "pre.json"
    receipt = create_pre_restore_backup_receipt(
        snapshot_path=snapshot_path,
        target_db_path=target.path,
        backup_output_path=backup,
        receipt_output_path=receipt_path,
        actor=actor(),
        now=11.0,
    )
    return snapshot_path, snapshot, record, target, backup, receipt_path, receipt


def test_pre_restore_backup_is_nonblocking_atomic_and_verifiable(tmp_path):
    (
        _snapshot_path,
        snapshot,
        _record,
        target,
        backup,
        receipt_path,
        receipt,
    ) = pre_receipt(tmp_path)

    verified = verify_pre_restore_backup_receipt(
        receipt_path=receipt_path,
        backup_path=backup,
    )
    assert verified == receipt
    assert receipt.snapshot_digest == snapshot.snapshot_digest
    assert receipt.target_path_digest == target_path_digest(target.path)
    assert receipt.target_record_count == 0
    assert receipt.backup_record_count == 0
    assert receipt.target_schema_digest == receipt.backup_schema_digest
    assert backup.exists() and receipt_path.exists()
    if os.name != "nt":
        assert stat.S_IMODE(backup.stat().st_mode) == 0o600
        assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600


def test_pre_restore_backup_refuses_nonempty_target_and_output_collisions(tmp_path):
    snapshot_path, _snapshot, record, target = fixture(tmp_path)
    target.seed(record)
    with pytest.raises(RuntimeError, match="globally empty"):
        create_pre_restore_backup_receipt(
            snapshot_path=snapshot_path,
            target_db_path=target.path,
            backup_output_path=tmp_path / "backup.sqlite3",
            receipt_output_path=tmp_path / "pre.json",
            actor=actor(),
            now=11.0,
        )

    empty = SignedPublicationRetirementJournal(tmp_path / "empty.sqlite3")
    occupied_receipt = tmp_path / "occupied.json"
    occupied_receipt.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError):
        create_pre_restore_backup_receipt(
            snapshot_path=snapshot_path,
            target_db_path=empty.path,
            backup_output_path=tmp_path / "unused.sqlite3",
            receipt_output_path=occupied_receipt,
            actor=actor(),
            now=11.0,
        )
    assert not (tmp_path / "unused.sqlite3").exists()

    same = tmp_path / "same.sqlite3"
    with pytest.raises(ValueError, match="must be distinct"):
        create_pre_restore_backup_receipt(
            snapshot_path=snapshot_path,
            target_db_path=empty.path,
            backup_output_path=same,
            receipt_output_path=same,
            actor=actor(),
            now=11.0,
        )


def test_pre_receipt_and_backup_tampering_fail_closed(tmp_path):
    (
        _snapshot_path,
        _snapshot,
        _record,
        _target,
        backup,
        receipt_path,
        _receipt,
    ) = pre_receipt(tmp_path)
    raw = json.loads(receipt_path.read_text(encoding="utf-8"))
    raw["created_at"] = 99.0
    receipt_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt_digest"):
        verify_pre_restore_backup_receipt(
            receipt_path=receipt_path,
            backup_path=backup,
        )

    second = pre_receipt(tmp_path / "second")
    second_backup = second[4]
    second_receipt = second[5]
    with second_backup.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(RuntimeError, match="backup artifact differs"):
        verify_pre_restore_backup_receipt(
            receipt_path=second_receipt,
            backup_path=second_backup,
        )


def test_post_restore_comparison_binds_completed_restore_and_exact_target(tmp_path):
    (
        snapshot_path,
        snapshot,
        record,
        target,
        backup,
        pre_path,
        pre,
    ) = pre_receipt(tmp_path)
    target.seed(record)
    restore_id = "a" * 64
    restore_value = SimpleNamespace(
        restore_id=restore_id,
        owner_id="alice",
        snapshot_digest=snapshot.snapshot_digest,
        target_path_digest=target_path_digest(target.path),
        state="completed",
        phase="verified",
    )

    class RestoreJournal:
        def get(self, selected):
            assert selected == restore_id
            return restore_value

    post_path = tmp_path / "post.json"
    post = create_post_restore_comparison_receipt(
        restore_id=restore_id,
        snapshot_path=snapshot_path,
        target_db_path=target.path,
        pre_restore_receipt_path=pre_path,
        backup_path=backup,
        receipt_output_path=post_path,
        restore_journal=RestoreJournal(),
        actor=actor("auditor"),
        now=20.0,
    )
    assert verify_post_restore_comparison_receipt(post_path) == post
    assert post.pre_restore_receipt_digest == pre.receipt_digest
    assert post.backup_sha256 == pre.backup_sha256
    assert post.target_record_count == snapshot.record_count
    assert post.actor_id == "auditor"


def test_post_restore_comparison_refuses_scope_or_target_drift(tmp_path):
    (
        snapshot_path,
        snapshot,
        record,
        target,
        backup,
        pre_path,
        _pre,
    ) = pre_receipt(tmp_path)
    target.seed(record)
    restore_id = "a" * 64

    class WrongRestoreJournal:
        def get(self, selected):
            return SimpleNamespace(
                restore_id=selected,
                owner_id="alice",
                snapshot_digest="f" * 64,
                target_path_digest=target_path_digest(target.path),
                state="completed",
                phase="verified",
            )

    with pytest.raises(RuntimeError, match="completed restore scope"):
        create_post_restore_comparison_receipt(
            restore_id=restore_id,
            snapshot_path=snapshot_path,
            target_db_path=target.path,
            pre_restore_receipt_path=pre_path,
            backup_path=backup,
            receipt_output_path=tmp_path / "wrong.json",
            restore_journal=WrongRestoreJournal(),
            actor=actor(),
            now=20.0,
        )

    target.seed(terminal("2"))

    class CorrectRestoreJournal:
        def get(self, selected):
            return SimpleNamespace(
                restore_id=selected,
                owner_id="alice",
                snapshot_digest=snapshot.snapshot_digest,
                target_path_digest=target_path_digest(target.path),
                state="completed",
                phase="verified",
            )

    with pytest.raises(RuntimeError, match="neither empty nor exact"):
        create_post_restore_comparison_receipt(
            restore_id=restore_id,
            snapshot_path=snapshot_path,
            target_db_path=target.path,
            pre_restore_receipt_path=pre_path,
            backup_path=backup,
            receipt_output_path=tmp_path / "drift.json",
            restore_journal=CorrectRestoreJournal(),
            actor=actor(),
            now=20.0,
        )


def test_custody_cli_confirms_before_actor_and_verify_is_read_only(
    tmp_path,
    monkeypatch,
    capsys,
):
    snapshot_path, snapshot, _record, target = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(
        cli,
        "load_relation_review_actor",
        lambda: calls.append("actor") or actor(),
    )
    assert cli.main([
        "pre-create",
        "--snapshot",
        str(snapshot_path),
        "--target-db-path",
        str(target.path),
        "--backup-output",
        str(tmp_path / "backup.sqlite3"),
        "--receipt-output",
        str(tmp_path / "pre.json"),
        "--confirm-snapshot-digest",
        "f" * 64,
    ]) == 2
    assert calls == []
    assert json.loads(capsys.readouterr().err) == {
        "error": "invalid_or_unavailable"
    }

    receipt_data = pre_receipt(tmp_path / "verified")
    monkeypatch.setattr(
        cli,
        "load_relation_review_actor",
        lambda: (_ for _ in ()).throw(
            AssertionError("verification must not load an actor")
        ),
    )
    monkeypatch.setattr(
        cli,
        "get_signed_retirement_restore_journal",
        lambda: (_ for _ in ()).throw(
            AssertionError("pre verification must not load restore journal")
        ),
    )
    assert cli.main([
        "pre-verify",
        "--receipt",
        str(receipt_data[5]),
        "--backup",
        str(receipt_data[4]),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["receipt_kind"] == "pre_restore_backup"
    assert payload["raw_paths_returned"] is False
    assert payload["target_mutation_performed"] is False
