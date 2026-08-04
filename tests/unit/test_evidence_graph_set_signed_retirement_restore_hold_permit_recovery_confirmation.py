from __future__ import annotations

from dataclasses import replace

import pytest

from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    SignedRetirementRestoreAttempt,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_boundary import (
    GovernedSignedRetirementRestoreHoldStore,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_permit_recovery import (
    recover_abandoned_hold_placement_permit,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_permits import (
    acquire_hold_placement_permit,
)
from tools.evidence_graph_set_signed_retirement_restore_holds import (
    deterministic_restore_hold_id,
)
from tools.evidence_graph_set_signed_retirement_restore_journal import (
    SignedRetirementRestoreJournal,
)


def test_completed_recovery_replay_requires_original_permit_digest(tmp_path):
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
    actor = ReviewActorBinding.create(
        actor_id="operator",
        binding_method="process_environment",
        loaded_at=1.0,
    )
    recover_abandoned_hold_placement_permit(
        restore_journal=journal,
        hold_store=holds,
        owner_id="alice",
        hold_id=hold_id,
        confirm_hold_id=hold_id,
        confirm_permit_digest=permit_digest,
        actor=actor,
        minimum_age_seconds=60,
        now=100.0,
    )

    with pytest.raises(ValueError, match="permit digest confirmation"):
        recover_abandoned_hold_placement_permit(
            restore_journal=journal,
            hold_store=holds,
            owner_id="alice",
            hold_id=hold_id,
            confirm_hold_id=hold_id,
            confirm_permit_digest="f" * 64,
            actor=actor,
            minimum_age_seconds=60,
            now=101.0,
        )
