"""Canonical schema and deletion-coordination boundary for restore legal holds."""

from __future__ import annotations

from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_deletion_coordination import (
    assert_restore_not_under_deletion,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_integrity import (
    IntegritySignedRetirementRestoreHoldStore,
)

_METHODS = frozenset(
    {"process_environment", "descriptor_file", "hmac_assertion"}
)


class GovernedSignedRetirementRestoreHoldStore(
    IntegritySignedRetirementRestoreHoldStore
):
    """Integrity store with strict actor provenance and deletion coordination."""

    @staticmethod
    def _verified_value(connection: Any, row: Any) -> Any:
        value = IntegritySignedRetirementRestoreHoldStore._verified_value(
            connection,
            row,
        )
        if value.created_binding_method not in _METHODS:
            raise RuntimeError(
                "stored restore hold creation actor method is unsupported."
            )
        if (
            value.released_binding_method is not None
            and value.released_binding_method not in _METHODS
        ):
            raise RuntimeError(
                "stored restore hold release actor method is unsupported."
            )
        return value

    def place(
        self,
        *,
        owner_id: str,
        restore_id: str,
        hold_key: str,
        reason_code: str,
        actor: Any,
        restore_journal: Any,
        now: float | None = None,
    ):
        # A hold that committed immediately before the marker is caught by the
        # executor's post-marker hold recheck. A hold starting afterward fails here.
        assert_restore_not_under_deletion(restore_journal, restore_id)
        return super().place(
            owner_id=owner_id,
            restore_id=restore_id,
            hold_key=hold_key,
            reason_code=reason_code,
            actor=actor,
            restore_journal=restore_journal,
            now=now,
        )


__all__ = ["GovernedSignedRetirementRestoreHoldStore"]
