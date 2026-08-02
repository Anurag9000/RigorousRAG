from __future__ import annotations

import pytest

from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_hold_boundary import (
    GovernedSignedRetirementRestoreHoldStore,
)
from tools.evidence_graph_set_signed_retirement_restore_holds import (
    SignedRetirementRestoreHold,
)


class RestoreJournal:
    def get(self, restore_id):
        return type("Restore", (), {"owner_id": "alice"})()


def actor():
    return ReviewActorBinding.create(
        actor_id="operator",
        binding_method="process_environment",
        loaded_at=1.0,
    )


def test_governed_hold_boundary_refuses_recomputed_unsupported_actor_method(
    tmp_path,
):
    store = GovernedSignedRetirementRestoreHoldStore(tmp_path / "holds.sqlite3")
    value = store.place(
        owner_id="alice",
        restore_id="1" * 64,
        hold_key="case",
        reason_code="litigation",
        actor=actor(),
        restore_journal=RestoreJournal(),
        now=2.0,
    )
    tampered = SignedRetirementRestoreHold(
        hold_id=value.hold_id,
        owner_id=value.owner_id,
        restore_id=value.restore_id,
        hold_key=value.hold_key,
        reason_code=value.reason_code,
        status=value.status,
        created_actor_id=value.created_actor_id,
        created_binding_method="command_line",
        created_binding_digest=value.created_binding_digest,
        created_at=value.created_at,
    )
    with store._lock, store._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "UPDATE evidence_graph_set_signed_restore_holds "
            "SET created_binding_method=? WHERE hold_id=?",
            (tampered.created_binding_method, value.hold_id),
        )
        connection.execute(
            "UPDATE evidence_graph_set_signed_restore_hold_integrity "
            "SET hold_digest=? WHERE hold_id=?",
            (tampered.hold_digest, value.hold_id),
        )
        connection.execute("COMMIT")

    with pytest.raises(RuntimeError, match="actor method is unsupported"):
        store.get(value.hold_id)
