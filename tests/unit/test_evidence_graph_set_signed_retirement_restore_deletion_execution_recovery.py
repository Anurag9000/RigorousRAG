from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tools import (
    evidence_graph_set_signed_retirement_restore_deletion_authorizations as authmod,
)
from tools import evidence_graph_set_signed_retirement_restore_deletion_boundary as authboundary
from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    SignedRetirementRestoreAttempt,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_boundary import (
    GovernedSignedRetirementRestoreDeletionAuthorizationStore,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_consumption import (
    get_authorization_consumption,
    reserve_authorization_for_deletion,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_journal import (
    SignedRetirementRestoreDeletionJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_marker_boundary import (
    ensure_active_deletion_marker,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_reconcile import (
    SignedRetirementRestoreDeletionRecoveryError,
    execute_signed_retirement_restore_deletion,
    seed_signed_retirement_restore_deletion,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_boundary import (
    GovernedSignedRetirementRestoreHoldStore,
)
from tools.evidence_graph_set_signed_retirement_restore_journal import (
    SignedRetirementRestoreJournal,
)


class Holds:
    def __init__(self):
        self.restore_ids: frozenset[str] = frozenset()

    def active_restore_ids(self, **kwargs):
        return self.restore_ids


class Custody:
    def __init__(self, value):
        self.value = value

    def get_for_restore(self, restore_id):
        if self.value is None:
            raise KeyError(restore_id)
        return self.value


class Plan:
    def __init__(self, restore, *, digest="4" * 64, candidate=True):
        self.plan_digest = digest
        self.items = (
            SimpleNamespace(
                restore_id=restore.restore_id,
                snapshot_digest=restore.snapshot_digest,
                target_path_digest=restore.target_path_digest,
                retention_candidate=candidate,
            ),
        )


def actor(actor_id="operator"):
    return ReviewActorBinding.create(
        actor_id=actor_id,
        binding_method="process_environment",
        loaded_at=1.0,
    )


def completed_restore():
    value = SignedRetirementRestoreAttempt.create(
        owner_id="alice",
        snapshot_digest="2" * 64,
        target_path_digest="3" * 64,
        snapshot_record_count=1,
        now=1.0,
    )
    return replace(
        value,
        state="completed",
        phase="verified",
        attempt_count=1,
        target_verification_digest="5" * 64,
        updated_at=2.0,
        completed_at=2.0,
    )


def custody(restore):
    return SimpleNamespace(
        custody_id="6" * 64,
        manifest_digest="7" * 64,
        owner_id=restore.owner_id,
        restore_id=restore.restore_id,
        snapshot_digest=restore.snapshot_digest,
        target_path_digest=restore.target_path_digest,
        state="post_bound",
    )


def setup(tmp_path, monkeypatch):
    restore = completed_restore()
    restore_journal = SignedRetirementRestoreJournal(
        tmp_path / "restores.sqlite3"
    )
    restore_journal.seed(restore)
    holds = Holds()
    custody_store = Custody(custody(restore))
    authorization_store = (
        GovernedSignedRetirementRestoreDeletionAuthorizationStore(
            tmp_path / "authorizations.sqlite3"
        )
    )
    monkeypatch.setattr(
        authmod,
        "plan_signed_retirement_restore_retention",
        lambda **kwargs: Plan(restore),
    )
    monkeypatch.setattr(
        authboundary,
        "plan_signed_retirement_restore_retention",
        lambda **kwargs: Plan(restore),
    )
    authorization = authorization_store.authorize(
        owner_id="alice",
        restore_id=restore.restore_id,
        plan_digest="4" * 64,
        plan_generated_at=10.0,
        authorization_key="ticket",
        actor=actor(),
        restore_journal=restore_journal,
        hold_store=holds,
        minimum_age_seconds=1.0,
        retain_latest_per_target=1,
        include_completed=True,
        expires_in_seconds=100.0,
        now=20.0,
    )
    deletion_journal = SignedRetirementRestoreDeletionJournal(
        tmp_path / "deletions.sqlite3"
    )
    deletion, _report = seed_signed_retirement_restore_deletion(
        authorization_id=authorization.authorization_id,
        authorization_store=authorization_store,
        deletion_journal=deletion_journal,
        restore_journal=restore_journal,
        hold_store=holds,
        custody_store=custody_store,
        now=21.0,
    )
    return (
        restore,
        restore_journal,
        holds,
        custody_store,
        authorization_store,
        deletion_journal,
        deletion,
    )


def execute(values, *, worker_id="worker", now=22.0, hook=None):
    (
        _restore,
        restore_journal,
        holds,
        custody_store,
        authorization_store,
        deletion_journal,
        deletion,
    ) = values
    return execute_signed_retirement_restore_deletion(
        deletion.deletion_id,
        worker_id=worker_id,
        lease_seconds=60,
        deletion_journal=deletion_journal,
        authorization_store=authorization_store,
        restore_journal=restore_journal,
        hold_store=holds,
        custody_store=custody_store,
        now=now,
        _phase_hook=hook,
    )


def test_normal_deletion_is_tombstoned_consumed_and_custody_preserved(
    tmp_path, monkeypatch
):
    values = setup(tmp_path, monkeypatch)
    restore, restore_journal, _holds, custody_store, authorization_store, deletion_journal, deletion = values
    result = execute(values)

    assert result.state == "completed"
    assert result.restore_row_deleted is True
    assert result.authorization_consumed is True
    assert result.custody_preserved is True
    assert result.custody_deleted is False
    assert result.holds_deleted is False
    with pytest.raises(KeyError):
        restore_journal.get(restore.restore_id)
    assert custody_store.get_for_restore(restore.restore_id).custody_id == "6" * 64
    assert get_authorization_consumption(
        authorization_store, deletion.authorization_id
    ).state == "consumed"
    assert deletion_journal.get(deletion.deletion_id).tombstone_digest == result.tombstone_digest


def test_crash_after_row_deletion_recovers_without_restoring_row(
    tmp_path, monkeypatch
):
    values = setup(tmp_path, monkeypatch)
    restore, restore_journal, _holds, _custody, _authorization, deletion_journal, deletion = values

    def hook(name, value):
        if name == "restore_deleted":
            raise RuntimeError("simulated process death")

    with pytest.raises(SignedRetirementRestoreDeletionRecoveryError):
        execute(values, worker_id="one", now=22.0, hook=hook)
    with pytest.raises(KeyError):
        restore_journal.get(restore.restore_id)
    deletion_journal.retry(
        deletion.deletion_id,
        owner_id="alice",
        confirm_deletion_id=deletion.deletion_id,
        now=23.0,
    )
    result = execute(values, worker_id="two", now=24.0)
    assert result.state == "completed"
    with pytest.raises(KeyError):
        restore_journal.get(restore.restore_id)


def test_crash_after_marker_and_reservation_recovers_exactly(
    tmp_path, monkeypatch
):
    values = setup(tmp_path, monkeypatch)
    _restore, restore_journal, _holds, _custody, authorization_store, deletion_journal, deletion = values

    def hook(name, value):
        if name == "authorization_reserved":
            raise RuntimeError("simulated process death")

    with pytest.raises(SignedRetirementRestoreDeletionRecoveryError):
        execute(values, worker_id="one", now=22.0, hook=hook)
    assert restore_journal.get(deletion.restore_id).restore_id == deletion.restore_id
    assert get_authorization_consumption(
        authorization_store, deletion.authorization_id
    ).state == "reserved"
    deletion_journal.retry(
        deletion.deletion_id,
        owner_id="alice",
        confirm_deletion_id=deletion.deletion_id,
        now=23.0,
    )
    assert execute(values, worker_id="two", now=24.0).state == "completed"


def test_hold_committed_before_marker_is_caught_and_marker_can_retry(
    tmp_path, monkeypatch
):
    values = setup(tmp_path, monkeypatch)
    restore, restore_journal, holds, _custody, authorization_store, deletion_journal, deletion = values

    def hook(name, value):
        if name == "authorization_reserved":
            holds.restore_ids = frozenset({restore.restore_id})

    with pytest.raises(SignedRetirementRestoreDeletionRecoveryError):
        execute(values, worker_id="one", now=22.0, hook=hook)
    assert restore_journal.get(restore.restore_id).restore_id == restore.restore_id
    assert get_authorization_consumption(
        authorization_store, deletion.authorization_id
    ) is None

    holds.restore_ids = frozenset()
    deletion_journal.retry(
        deletion.deletion_id,
        owner_id="alice",
        confirm_deletion_id=deletion.deletion_id,
        now=23.0,
    )
    assert execute(values, worker_id="two", now=24.0).state == "completed"


def test_active_marker_blocks_new_governed_hold_and_authorization_revocation(
    tmp_path, monkeypatch
):
    values = setup(tmp_path, monkeypatch)
    restore, restore_journal, _holds, _custody, authorization_store, _journal, deletion = values
    ensure_active_deletion_marker(restore_journal, deletion, now=22.0)

    hold_store = GovernedSignedRetirementRestoreHoldStore(
        tmp_path / "holds.sqlite3"
    )
    with pytest.raises(RuntimeError, match="deletion control"):
        hold_store.place(
            owner_id="alice",
            restore_id=restore.restore_id,
            hold_key="case",
            reason_code="litigation",
            actor=actor("holder"),
            restore_journal=restore_journal,
            now=22.0,
        )
    with pytest.raises(RuntimeError, match="deletion control"):
        authorization_store.revoke(
            deletion.authorization_id,
            owner_id="alice",
            confirm_authorization_id=deletion.authorization_id,
            actor=actor("revoker"),
            restore_journal=restore_journal,
            now=22.0,
        )


def test_authorization_reservation_is_bound_to_one_deletion(
    tmp_path, monkeypatch
):
    values = setup(tmp_path, monkeypatch)
    _restore, _journal, _holds, _custody, authorization_store, _deletions, deletion = values
    reserve_authorization_for_deletion(
        authorization_store,
        authorization_id=deletion.authorization_id,
        deletion_id=deletion.deletion_id,
        now=22.0,
    )
    with pytest.raises(RuntimeError, match="another deletion"):
        reserve_authorization_for_deletion(
            authorization_store,
            authorization_id=deletion.authorization_id,
            deletion_id="f" * 64,
            now=23.0,
        )
