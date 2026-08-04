"""Canonical current-state and execution gate for restore deletion authorization."""

from __future__ import annotations

import time
from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_authorizations import (
    SignedRetirementRestoreDeletionAuthorizationStore,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_consumption import (
    require_authorization_unconsumed,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_mutation import (
    assert_restore_not_under_deletion,
)
from tools.evidence_graph_set_signed_retirement_restore_operations import (
    plan_signed_retirement_restore_retention,
)
from tools.security import normalize_owner_id


class GovernedSignedRetirementRestoreDeletionAuthorizationStore(
    SignedRetirementRestoreDeletionAuthorizationStore
):
    """Require current candidacy and coordinate authorization with execution."""

    def authorize(
        self,
        *,
        owner_id: str,
        restore_id: str,
        plan_digest: str,
        plan_generated_at: float,
        authorization_key: str,
        actor: Any,
        restore_journal: Any,
        hold_store: Any,
        minimum_age_seconds: float,
        retain_latest_per_target: int,
        include_completed: bool,
        expires_in_seconds: float = 24 * 60 * 60,
        now: float | None = None,
        limit: int = 10_000,
    ):
        owner = normalize_owner_id(owner_id)
        restore = _digest(restore_id, "restore_id")
        plan_time = _timestamp(plan_generated_at, "plan_generated_at")
        current = _timestamp(time.time() if now is None else now, "now")
        if plan_time > current:
            raise ValueError("retention plan generation time is in the future.")
        assert_restore_not_under_deletion(restore_journal, restore)
        restore_value = restore_journal.get(restore)
        if restore_value.owner_id != owner:
            raise RuntimeError(
                "restore escaped deletion-authorization owner scope."
            )
        held = hold_store.active_restore_ids(owner_id=owner, limit=limit)
        if restore in held:
            raise RuntimeError(
                "durable legal hold blocks deletion authorization."
            )
        current_plan = plan_signed_retirement_restore_retention(
            owner_id=owner,
            journal=restore_journal,
            now=current,
            minimum_age_seconds=minimum_age_seconds,
            retain_latest_per_target=retain_latest_per_target,
            include_completed=include_completed,
            held_restore_ids=held,
            limit=limit,
        )
        current_candidates = [
            item
            for item in current_plan.items
            if item.restore_id == restore and item.retention_candidate
        ]
        if len(current_candidates) != 1:
            raise RuntimeError(
                "restore is no longer a current retention candidate."
            )
        candidate = current_candidates[0]
        if (
            candidate.snapshot_digest != restore_value.snapshot_digest
            or candidate.target_path_digest != restore_value.target_path_digest
        ):
            raise RuntimeError(
                "current retention candidate escaped restore scope."
            )
        return super().authorize(
            owner_id=owner,
            restore_id=restore,
            plan_digest=plan_digest,
            plan_generated_at=plan_time,
            authorization_key=authorization_key,
            actor=actor,
            restore_journal=restore_journal,
            hold_store=hold_store,
            minimum_age_seconds=minimum_age_seconds,
            retain_latest_per_target=retain_latest_per_target,
            include_completed=include_completed,
            expires_in_seconds=expires_in_seconds,
            now=current,
            limit=limit,
        )

    def revoke(
        self,
        authorization_id: str,
        *,
        owner_id: str,
        confirm_authorization_id: str,
        actor: Any,
        restore_journal: Any,
        now: float | None = None,
    ):
        authorization = self.get(authorization_id)
        assert_restore_not_under_deletion(
            restore_journal,
            authorization.restore_id,
        )
        require_authorization_unconsumed(
            self,
            authorization.authorization_id,
        )
        return super().revoke(
            authorization_id,
            owner_id=owner_id,
            confirm_authorization_id=confirm_authorization_id,
            actor=actor,
            now=now,
        )


__all__ = [
    "GovernedSignedRetirementRestoreDeletionAuthorizationStore"
]
