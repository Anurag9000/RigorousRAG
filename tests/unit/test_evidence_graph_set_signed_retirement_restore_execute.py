from __future__ import annotations

from dataclasses import replace

import pytest

from tools.evidence_graph_set_signed_retirement_contracts import (
    SignedPublicationRetirementAttempt,
)
from tools.evidence_graph_set_signed_retirement_journal import (
    SignedPublicationRetirementJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_journal import (
    SignedRetirementRestoreJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_reconcile import (
    SignedRetirementRestoreRecoveryError,
    execute_signed_retirement_restore,
    seed_signed_retirement_restore,
)
from tools.evidence_graph_set_signed_retirement_snapshot import (
    export_signed_retirement_snapshot,
)


def planned(digit: str):
    return SignedPublicationRetirementAttempt.create(
        owner_id="alice",
        publication_operation_id=digit * 64,
        graph_set_key="review",
        signed_candidate_set_id="2" * 64,
        signed_candidate_set_digest="3" * 64,
        authorization_candidate_set_id="4" * 64,
        signed_authority_digest="5" * 64,
        now=1.0,
    )


def cancelled(digit: str):
    return replace(
        planned(digit),
        state="cancelled",
        updated_at=2.0,
        completed_at=2.0,
    )


def completed(digit: str):
    return replace(
        planned(digit),
        state="completed",
        phase="verified",
        final_pointer_set_id="2" * 64,
        verification_digest="6" * 64,
        updated_at=2.0,
        completed_at=2.0,
    )


def failed(digit: str):
    return replace(
        planned(digit),
        state="failed",
        failure_type="Retryable",
        updated_at=2.0,
    )


class Values:
    def __init__(self, values=()):
        self.values = tuple(values)

    def list(self, **kwargs):
        return self.values[: kwargs["limit"]]


def snapshot_file(tmp_path, values):
    path = tmp_path / "snapshot.json"
    snapshot = export_signed_retirement_snapshot(
        owner_id="alice",
        journal=Values(values),
        output_path=path,
        now=10.0,
        limit=100,
    )
    return path, snapshot


def target_journal(tmp_path, values=()):
    journal = SignedPublicationRetirementJournal(
        tmp_path / "target.sqlite3"
    )
    for value in values:
        journal.seed(value)
    return journal


def restore_journal(tmp_path):
    return SignedRetirementRestoreJournal(tmp_path / "restore.sqlite3")


def test_empty_target_restore_completes_and_terminal_replay_is_read_only(tmp_path):
    snapshot_path, snapshot = snapshot_file(
        tmp_path,
        (cancelled("1"), completed("2")),
    )
    target = target_journal(tmp_path)
    restores = restore_journal(tmp_path)
    attempt, _target_digest = seed_signed_retirement_restore(
        snapshot_path=str(snapshot_path),
        target_db_path=str(target.path),
        journal=restores,
        confirm_snapshot_digest=snapshot.snapshot_digest,
        now=11.0,
    )

    result = execute_signed_retirement_restore(
        attempt.restore_id,
        snapshot_path=str(snapshot_path),
        target_db_path=str(target.path),
        worker_id="worker",
        lease_seconds=60,
        journal=restores,
        now=12.0,
    )
    assert result.state == "completed"
    assert result.phase == "verified"
    assert result.target_mutation_performed is True
    assert result.overwrite_performed is False
    assert result.merge_performed is False
    assert len(target.list(owner_id="alice", limit=100)) == 2

    replay = execute_signed_retirement_restore(
        attempt.restore_id,
        snapshot_path=str(snapshot_path),
        target_db_path=str(target.path),
        worker_id="other",
        lease_seconds=60,
        journal=restores,
        now=13.0,
    )
    assert replay.state == "completed"
    assert replay.target_mutation_performed is False
    assert replay.restore_intent_mutation_performed is False


def test_crash_after_target_commit_recovers_without_duplicate_rows(tmp_path):
    snapshot_path, snapshot = snapshot_file(
        tmp_path,
        (cancelled("1"),),
    )
    target = target_journal(tmp_path)
    restores = restore_journal(tmp_path)
    attempt, _ = seed_signed_retirement_restore(
        snapshot_path=str(snapshot_path),
        target_db_path=str(target.path),
        journal=restores,
        confirm_snapshot_digest=snapshot.snapshot_digest,
        now=11.0,
    )

    def crash(name, _value):
        if name == "target_committed":
            raise RuntimeError("process stopped after target commit")

    with pytest.raises(SignedRetirementRestoreRecoveryError):
        execute_signed_retirement_restore(
            attempt.restore_id,
            snapshot_path=str(snapshot_path),
            target_db_path=str(target.path),
            worker_id="worker",
            lease_seconds=60,
            journal=restores,
            now=12.0,
            _phase_hook=crash,
        )
    failed_value = restores.get(attempt.restore_id)
    assert failed_value.state == "failed"
    assert failed_value.phase == "planned"
    assert len(target.list(owner_id="alice", limit=100)) == 1

    restores.retry(
        attempt.restore_id,
        owner_id="alice",
        confirm_restore_id=attempt.restore_id,
        now=13.0,
    )
    recovered = execute_signed_retirement_restore(
        attempt.restore_id,
        snapshot_path=str(snapshot_path),
        target_db_path=str(target.path),
        worker_id="recovery",
        lease_seconds=60,
        journal=restores,
        now=14.0,
    )
    assert recovered.state == "completed"
    assert recovered.target_mutation_performed is False
    assert len(target.list(owner_id="alice", limit=100)) == 1


def test_crash_after_phase_persistence_recovers_from_target_committed(tmp_path):
    snapshot_path, snapshot = snapshot_file(
        tmp_path,
        (cancelled("1"),),
    )
    target = target_journal(tmp_path)
    restores = restore_journal(tmp_path)
    attempt, _ = seed_signed_retirement_restore(
        snapshot_path=str(snapshot_path),
        target_db_path=str(target.path),
        journal=restores,
        confirm_snapshot_digest=snapshot.snapshot_digest,
        now=11.0,
    )

    def crash(name, _value):
        if name == "before_complete":
            raise RuntimeError("process stopped before completion")

    with pytest.raises(SignedRetirementRestoreRecoveryError):
        execute_signed_retirement_restore(
            attempt.restore_id,
            snapshot_path=str(snapshot_path),
            target_db_path=str(target.path),
            worker_id="worker",
            lease_seconds=60,
            journal=restores,
            now=12.0,
            _phase_hook=crash,
        )
    assert restores.get(attempt.restore_id).phase == "target_committed"

    restores.retry(
        attempt.restore_id,
        owner_id="alice",
        confirm_restore_id=attempt.restore_id,
        now=13.0,
    )
    recovered = execute_signed_retirement_restore(
        attempt.restore_id,
        snapshot_path=str(snapshot_path),
        target_db_path=str(target.path),
        worker_id="recovery",
        lease_seconds=60,
        journal=restores,
        now=14.0,
    )
    assert recovered.state == "completed"
    assert recovered.target_mutation_performed is False


def test_seed_refuses_nonterminal_snapshot_and_retroactive_exact_target(tmp_path):
    active_path, active_snapshot = snapshot_file(
        tmp_path / "active",
        (failed("1"),),
    )
    active_target = target_journal(tmp_path / "active")
    with pytest.raises(RuntimeError, match="executable or retryable"):
        seed_signed_retirement_restore(
            snapshot_path=str(active_path),
            target_db_path=str(active_target.path),
            journal=restore_journal(tmp_path / "active"),
            confirm_snapshot_digest=active_snapshot.snapshot_digest,
            now=11.0,
        )

    terminal = cancelled("2")
    exact_path, exact_snapshot = snapshot_file(
        tmp_path / "exact",
        (terminal,),
    )
    exact_target = target_journal(tmp_path / "exact", (terminal,))
    with pytest.raises(RuntimeError, match="new restore intent"):
        seed_signed_retirement_restore(
            snapshot_path=str(exact_path),
            target_db_path=str(exact_target.path),
            journal=restore_journal(tmp_path / "exact"),
            confirm_snapshot_digest=exact_snapshot.snapshot_digest,
            now=11.0,
        )


def test_partial_target_refuses_without_merging_missing_history(tmp_path):
    first = cancelled("1")
    second = cancelled("2")
    snapshot_path, snapshot = snapshot_file(tmp_path, (first, second))
    target = target_journal(tmp_path, (first,))
    restores = restore_journal(tmp_path)

    with pytest.raises(RuntimeError, match="new restore intent"):
        seed_signed_retirement_restore(
            snapshot_path=str(snapshot_path),
            target_db_path=str(target.path),
            journal=restores,
            confirm_snapshot_digest=snapshot.snapshot_digest,
            now=11.0,
        )
    values = target.list(owner_id="alice", limit=100)
    assert values == (first,)
    assert restores.list(owner_id="alice", limit=100) == ()


def test_post_claim_snapshot_or_target_scope_change_is_durably_failed(tmp_path):
    snapshot_path, snapshot = snapshot_file(
        tmp_path,
        (cancelled("1"),),
    )
    target = target_journal(tmp_path)
    alternate = target_journal(tmp_path / "alternate")
    restores = restore_journal(tmp_path)
    attempt, _ = seed_signed_retirement_restore(
        snapshot_path=str(snapshot_path),
        target_db_path=str(target.path),
        journal=restores,
        confirm_snapshot_digest=snapshot.snapshot_digest,
        now=11.0,
    )

    with pytest.raises(SignedRetirementRestoreRecoveryError):
        execute_signed_retirement_restore(
            attempt.restore_id,
            snapshot_path=str(snapshot_path),
            target_db_path=str(alternate.path),
            worker_id="worker",
            lease_seconds=60,
            journal=restores,
            now=12.0,
        )
    failed_value = restores.get(attempt.restore_id)
    assert failed_value.state == "failed"
    assert failed_value.phase == "planned"
    assert target.list(owner_id="alice", limit=100) == ()
    assert alternate.list(owner_id="alice", limit=100) == ()


def test_final_exact_target_lock_refuses_concurrent_additional_history(tmp_path):
    snapshot_path, snapshot = snapshot_file(
        tmp_path,
        (cancelled("1"),),
    )
    target = target_journal(tmp_path)
    restores = restore_journal(tmp_path)
    attempt, _ = seed_signed_retirement_restore(
        snapshot_path=str(snapshot_path),
        target_db_path=str(target.path),
        journal=restores,
        confirm_snapshot_digest=snapshot.snapshot_digest,
        now=11.0,
    )
    injected = False

    def inject(name, _value):
        nonlocal injected
        if name == "before_complete" and not injected:
            injected = True
            target.seed(cancelled("2"))

    with pytest.raises(SignedRetirementRestoreRecoveryError):
        execute_signed_retirement_restore(
            attempt.restore_id,
            snapshot_path=str(snapshot_path),
            target_db_path=str(target.path),
            worker_id="worker",
            lease_seconds=60,
            journal=restores,
            now=12.0,
            _phase_hook=inject,
        )
    failed_value = restores.get(attempt.restore_id)
    assert failed_value.state == "failed"
    assert failed_value.phase == "target_committed"
    assert len(target.list(owner_id="alice", limit=100)) == 2
