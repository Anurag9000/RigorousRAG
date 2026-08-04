from __future__ import annotations

from dataclasses import replace

from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    SignedRetirementRestoreAttempt,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_boundary import (
    GovernedSignedRetirementRestoreHoldStore,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_permit_recovery import (
    _quarantine_hold,
    recover_abandoned_hold_placement_permit,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_permits import (
    acquire_hold_placement_permit,
    active_hold_placement_permit,
)
from tools.evidence_graph_set_signed_retirement_restore_holds import (
    deterministic_restore_hold_id,
)
from tools.evidence_graph_set_signed_retirement_restore_journal import (
    SignedRetirementRestoreJournal,
)


def signed_actor(assertion_digit: str, *, loaded_at: float):
    return ReviewActorBinding.create(
        actor_id="operator",
        binding_method="hmac_assertion",
        loaded_at=loaded_at,
        assertion_digest=assertion_digit * 64,
        issuer="restore-governance",
        expires_at=1_000.0,
    )


def test_existing_quarantine_accepts_fresh_actor_binding_on_replay(tmp_path):
    restore = SignedRetirementRestoreAttempt.create(
        owner_id="alice",
        snapshot_digest="1" * 64,
        target_path_digest="2" * 64,
        snapshot_record_count=1,
        now=1.0,
    )
    restore = replace(
        restore,
        state="cancelled",
        updated_at=2.0,
        completed_at=2.0,
    )
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
    permit_digest = acquire_hold_placement_permit(
        journal,
        owner_id="alice",
        restore_id=restore.restore_id,
        hold_id=hold_id,
        now=4.0,
    )

    with holds._lock, holds._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        quarantine = _quarantine_hold(
            hold_store=holds,
            connection=connection,
            owner_id="alice",
            restore_id=restore.restore_id,
            original_hold_id=hold_id,
            actor=signed_actor("3", loaded_at=5.0),
            now=5.0,
        )
        connection.execute("COMMIT")

    receipt, changed = recover_abandoned_hold_placement_permit(
        restore_journal=journal,
        hold_store=holds,
        owner_id="alice",
        hold_id=hold_id,
        confirm_hold_id=hold_id,
        confirm_permit_digest=permit_digest,
        actor=signed_actor("4", loaded_at=100.0),
        minimum_age_seconds=60,
        now=100.0,
    )

    assert changed is True
    assert receipt.quarantine_hold_id == quarantine.hold_id
    assert receipt.quarantine_hold_digest == quarantine.hold_digest
    assert receipt.actor_binding_digest != quarantine.created_binding_digest
    with journal._lock, journal._connect() as connection:
        assert active_hold_placement_permit(
            connection,
            restore.restore_id,
        ) is None
