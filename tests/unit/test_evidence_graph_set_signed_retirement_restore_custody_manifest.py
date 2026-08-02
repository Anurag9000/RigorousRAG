from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest

from tools import evidence_graph_set_signed_retirement_restore_custody_manifest_cli as cli
from tools import evidence_graph_set_signed_retirement_restore_custody_runtime as runtime
from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_contracts import (
    SignedPublicationRetirementAttempt,
)
from tools.evidence_graph_set_signed_retirement_journal import (
    SignedPublicationRetirementJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    SignedRetirementRestoreAttempt,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_boundary import (
    create_post_restore_comparison_receipt,
    create_pre_restore_backup_receipt,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_manifest_boundary import (
    GovernedSignedRetirementRestoreCustodyStore,
)
from tools.evidence_graph_set_signed_retirement_restore_mutation import (
    inspect_restored_target,
    target_path_digest,
)
from tools.evidence_graph_set_signed_retirement_restore_reconcile import (
    seed_signed_retirement_restore,
)
from tools.evidence_graph_set_signed_retirement_restore_journal import (
    SignedRetirementRestoreJournal,
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


class Values:
    def __init__(self, values=()):
        self.values = tuple(values)

    def list(self, **kwargs):
        return self.values[: kwargs["limit"]]


def terminal():
    value = SignedPublicationRetirementAttempt.create(
        owner_id="alice",
        publication_operation_id="1" * 64,
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


def fixture(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    snapshot_path = tmp_path / "snapshot.json"
    record = terminal()
    snapshot = export_signed_retirement_snapshot(
        owner_id="alice",
        journal=Values((record,)),
        output_path=snapshot_path,
        now=10.0,
        limit=100,
    )
    target = SignedPublicationRetirementJournal(tmp_path / "target.sqlite3")
    backup = tmp_path / "backup.sqlite3"
    pre_path = tmp_path / "pre.json"
    pre = create_pre_restore_backup_receipt(
        snapshot_path=snapshot_path,
        target_db_path=target.path,
        backup_output_path=backup,
        receipt_output_path=pre_path,
        actor=actor("backup-operator"),
        now=11.0,
    )
    restores = SignedRetirementRestoreJournal(tmp_path / "restores.sqlite3")
    restore, _target_digest = seed_signed_retirement_restore(
        snapshot_path=str(snapshot_path),
        target_db_path=str(target.path),
        journal=restores,
        confirm_snapshot_digest=snapshot.snapshot_digest,
        now=12.0,
    )
    custody = GovernedSignedRetirementRestoreCustodyStore(
        tmp_path / "custody.sqlite3"
    )
    return {
        "snapshot_path": snapshot_path,
        "snapshot": snapshot,
        "record": record,
        "target": target,
        "backup": backup,
        "pre_path": pre_path,
        "pre": pre,
        "restores": restores,
        "restore": restore,
        "custody": custody,
    }


def test_pre_binding_is_deterministic_replay_stable_and_scope_bound(tmp_path):
    values = fixture(tmp_path)
    bound = values["custody"].bind_pre(
        restore_id=values["restore"].restore_id,
        pre_receipt_path=values["pre_path"],
        backup_path=values["backup"],
        restore_journal=values["restores"],
        actor=actor(),
        now=13.0,
    )
    replay = values["custody"].bind_pre(
        restore_id=values["restore"].restore_id,
        pre_receipt_path=values["pre_path"],
        backup_path=values["backup"],
        restore_journal=values["restores"],
        actor=actor(),
        now=99.0,
    )

    assert replay == bound
    assert replay.pre_bound_at == 13.0
    assert replay.state == "pre_bound"
    assert values["custody"].get_for_restore(
        values["restore"].restore_id
    ) == bound
    with pytest.raises(RuntimeError, match="collision"):
        values["custody"].bind_pre(
            restore_id=values["restore"].restore_id,
            pre_receipt_path=values["pre_path"],
            backup_path=values["backup"],
            restore_journal=values["restores"],
            actor=actor("different"),
            now=14.0,
        )


def test_pre_binding_must_precede_target_work_and_live_files_are_rechecked(tmp_path):
    values = fixture(tmp_path)
    values["restores"].claim(
        values["restore"].restore_id,
        worker_id="worker",
        now=13.0,
    )
    values["restores"].record_target_committed(
        values["restore"].restore_id,
        worker_id="worker",
        target_verification_digest="f" * 64,
        now=14.0,
    )
    values["restores"].fail(
        values["restore"].restore_id,
        worker_id="worker",
        failure_type="Interrupted",
        now=15.0,
    )
    with pytest.raises(RuntimeError, match="before target work"):
        values["custody"].bind_pre(
            restore_id=values["restore"].restore_id,
            pre_receipt_path=values["pre_path"],
            backup_path=values["backup"],
            restore_journal=values["restores"],
            actor=actor(),
            now=16.0,
        )

    second = fixture(tmp_path / "second")
    second["custody"].bind_pre(
        restore_id=second["restore"].restore_id,
        pre_receipt_path=second["pre_path"],
        backup_path=second["backup"],
        restore_journal=second["restores"],
        actor=actor(),
        now=13.0,
    )
    with second["backup"].open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(RuntimeError, match="backup artifact differs"):
        second["custody"].require_pre_bound(
            restore_id=second["restore"].restore_id,
            pre_receipt_path=second["pre_path"],
            backup_path=second["backup"],
            restore_journal=second["restores"],
        )


def test_post_binding_is_monotonic_and_replay_stable(tmp_path):
    values = fixture(tmp_path)
    pre_bound = values["custody"].bind_pre(
        restore_id=values["restore"].restore_id,
        pre_receipt_path=values["pre_path"],
        backup_path=values["backup"],
        restore_journal=values["restores"],
        actor=actor(),
        now=13.0,
    )
    values["target"].seed(values["record"])
    _disposition, verification = inspect_restored_target(
        snapshot=values["snapshot"],
        target_db_path=values["target"].path,
    )
    with values["restores"]._lock, values["restores"]._connect() as connection:
        connection.execute(
            "UPDATE evidence_graph_set_signed_retirement_restores SET "
            "state='completed', phase='verified', target_verification_digest=?, "
            "completed_at=?, updated_at=? WHERE restore_id=?",
            (verification, 20.0, 20.0, values["restore"].restore_id),
        )
    post_path = tmp_path / "post.json"
    post = create_post_restore_comparison_receipt(
        restore_id=values["restore"].restore_id,
        snapshot_path=values["snapshot_path"],
        target_db_path=values["target"].path,
        pre_restore_receipt_path=values["pre_path"],
        backup_path=values["backup"],
        receipt_output_path=post_path,
        restore_journal=values["restores"],
        actor=actor("auditor"),
        now=21.0,
    )
    bound = values["custody"].bind_post(
        restore_id=values["restore"].restore_id,
        post_receipt_path=post_path,
        restore_journal=values["restores"],
        actor=actor("binder"),
        now=22.0,
    )
    replay = values["custody"].bind_post(
        restore_id=values["restore"].restore_id,
        post_receipt_path=post_path,
        restore_journal=values["restores"],
        actor=actor("binder"),
        now=99.0,
    )

    assert pre_bound.state == "pre_bound"
    assert bound.state == "post_bound"
    assert bound.post_receipt_digest == post.receipt_digest
    assert replay == bound
    assert replay.post_bound_at == 22.0
    with pytest.raises(RuntimeError, match="collision"):
        values["custody"].bind_post(
            restore_id=values["restore"].restore_id,
            post_receipt_path=post_path,
            restore_journal=values["restores"],
            actor=actor("different"),
            now=23.0,
        )


def test_manifest_row_tamper_and_custody_target_alias_fail_closed(tmp_path):
    values = fixture(tmp_path)
    bound = values["custody"].bind_pre(
        restore_id=values["restore"].restore_id,
        pre_receipt_path=values["pre_path"],
        backup_path=values["backup"],
        restore_journal=values["restores"],
        actor=actor(),
        now=13.0,
    )
    with values["custody"]._lock, values["custody"]._connect() as connection:
        connection.execute(
            "UPDATE evidence_graph_set_signed_restore_custody "
            "SET backup_size_bytes=backup_size_bytes+1 WHERE custody_id=?",
            (bound.custody_id,),
        )
    with pytest.raises(RuntimeError, match="corrupt"):
        values["custody"].get(bound.custody_id)

    alias_values = fixture(tmp_path / "alias")
    alias_store = GovernedSignedRetirementRestoreCustodyStore(
        alias_values["target"].path
    )
    with pytest.raises(RuntimeError, match="may not be the restore target"):
        alias_store.bind_pre(
            restore_id=alias_values["restore"].restore_id,
            pre_receipt_path=alias_values["pre_path"],
            backup_path=alias_values["backup"],
            restore_journal=alias_values["restores"],
            actor=actor(),
            now=13.0,
        )


def test_custody_runtime_alias_and_manifest_cli_confirmation_boundaries(
    tmp_path,
    monkeypatch,
    capsys,
):
    runtime.clear_signed_retirement_restore_custody_store_cache()
    protected = tmp_path / "restore.sqlite3"
    protected.write_bytes(b"database")
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_RESTORE_DB_PATH",
        str(protected),
    )
    with pytest.raises(RuntimeError, match="must not alias"):
        runtime.get_signed_retirement_restore_custody_store(path=protected)

    calls = []
    monkeypatch.setattr(
        cli,
        "load_relation_review_actor",
        lambda: calls.append("actor") or actor(),
    )
    monkeypatch.setattr(
        cli,
        "get_signed_retirement_restore_custody_store",
        lambda: calls.append("store") or object(),
    )
    assert cli.main([
        "bind-pre",
        "1" * 64,
        "--confirm-restore-id",
        "2" * 64,
        "--pre-receipt",
        "pre.json",
        "--backup",
        "backup.sqlite3",
    ]) == 2
    assert calls == []
    assert json.loads(capsys.readouterr().err) == {
        "error": "invalid_or_unavailable"
    }
