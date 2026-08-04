from __future__ import annotations

from dataclasses import replace

import pytest

from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    SignedRetirementRestoreAttempt,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_execution_contracts import (
    SignedRetirementRestoreDeletionAttempt,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_marker_boundary import (
    ensure_active_deletion_marker,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_mutation import (
    canonical_restore_record_digest,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_boundary import (
    GovernedSignedRetirementRestoreHoldStore,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_permits import (
    acquire_hold_placement_permit,
    active_hold_placement_permit,
    release_hold_placement_permit,
)
from tools.evidence_graph_set_signed_retirement_restore_holds import (
    deterministic_restore_hold_id,
)
from tools.evidence_graph_set_signed_retirement_restore_journal import (
    SignedRetirementRestoreJournal,
)


def actor():
    return ReviewActorBinding.create(
        actor_id="operator",
        binding_method="process_environment",
        loaded_at=1.0,
    )


def cancelled_restore():
    value = SignedRetirementRestoreAttempt.create(
        owner_id="alice",
        snapshot_digest="1" * 64,
        target_path_digest="2" * 64,
        snapshot_record_count=1,
        now=1.0,
    )
    return replace(
        value,
        state="cancelled",
        updated_at=2.0,
        completed_at=2.0,
    )


def deletion(restore):
    return SignedRetirementRestoreDeletionAttempt.create(
        authorization_id="3" * 64,
        authorization_digest="4" * 64,
        owner_id=restore.owner_id,
        restore_id=restore.restore_id,
        snapshot_digest=restore.snapshot_digest,
        target_path_digest=restore.target_path_digest,
        restore_state=restore.state,
        restore_phase=restore.phase,
        restore_record_digest=canonical_restore_record_digest(restore),
        custody_id=None,
        custody_manifest_digest=None,
        now=3.0,
    )


def test_active_permit_blocks_marker_and_active_marker_blocks_new_permit(
    tmp_path,
):
    restore = cancelled_restore()
    journal = SignedRetirementRestoreJournal(tmp_path / "restores.sqlite3")
    journal.seed(restore)
    attempt = deletion(restore)
    hold_id = deterministic_restore_hold_id(
        owner_id="alice",
        restore_id=restore.restore_id,
        hold_key="case-a",
    )

    acquire_hold_placement_permit(
        journal,
        owner_id="alice",
        restore_id=restore.restore_id,
        hold_id=hold_id,
        now=4.0,
    )
    with pytest.raises(RuntimeError, match="permit blocks"):
        ensure_active_deletion_marker(journal, attempt, now=5.0)

    assert release_hold_placement_permit(
        journal,
        owner_id="alice",
        restore_id=restore.restore_id,
        hold_id=hold_id,
        now=6.0,
    )
    ensure_active_deletion_marker(journal, attempt, now=7.0)
    with pytest.raises(RuntimeError, match="deletion control"):
        acquire_hold_placement_permit(
            journal,
            owner_id="alice",
            restore_id=restore.restore_id,
            hold_id=deterministic_restore_hold_id(
                owner_id="alice",
                restore_id=restore.restore_id,
                hold_key="case-b",
            ),
            now=8.0,
        )


def test_same_hold_replay_recovers_active_permit(tmp_path):
    restore = cancelled_restore()
    journal = SignedRetirementRestoreJournal(tmp_path / "restores.sqlite3")
    journal.seed(restore)
    holds = GovernedSignedRetirementRestoreHoldStore(
        tmp_path / "holds.sqlite3"
    )
    hold_id = deterministic_restore_hold_id(
        owner_id="alice",
        restore_id=restore.restore_id,
        hold_key="case-a",
    )
    first_digest = acquire_hold_placement_permit(
        journal,
        owner_id="alice",
        restore_id=restore.restore_id,
        hold_id=hold_id,
        now=4.0,
    )
    second_digest = acquire_hold_placement_permit(
        journal,
        owner_id="alice",
        restore_id=restore.restore_id,
        hold_id=hold_id,
        now=5.0,
    )
    assert first_digest == second_digest

    placed = holds.place(
        owner_id="alice",
        restore_id=restore.restore_id,
        hold_key="case-a",
        reason_code="litigation",
        actor=actor(),
        restore_journal=journal,
        now=6.0,
    )
    assert placed.hold_id == hold_id
    with journal._lock, journal._connect() as connection:
        assert active_hold_placement_permit(
            connection,
            restore.restore_id,
        ) is None


def test_different_hold_cannot_take_existing_permit(tmp_path):
    restore = cancelled_restore()
    journal = SignedRetirementRestoreJournal(tmp_path / "restores.sqlite3")
    journal.seed(restore)
    first = deterministic_restore_hold_id(
        owner_id="alice",
        restore_id=restore.restore_id,
        hold_key="case-a",
    )
    second = deterministic_restore_hold_id(
        owner_id="alice",
        restore_id=restore.restore_id,
        hold_key="case-b",
    )
    acquire_hold_placement_permit(
        journal,
        owner_id="alice",
        restore_id=restore.restore_id,
        hold_id=first,
        now=4.0,
    )
    with pytest.raises(RuntimeError, match="another hold placement"):
        acquire_hold_placement_permit(
            journal,
            owner_id="alice",
            restore_id=restore.restore_id,
            hold_id=second,
            now=5.0,
        )
