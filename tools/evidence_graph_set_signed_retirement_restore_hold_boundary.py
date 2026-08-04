"""Canonical schema and deletion-coordination boundary for restore legal holds."""

from __future__ import annotations

import time
from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_hold_integrity import (
    IntegritySignedRetirementRestoreHoldStore,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_permits import (
    acquire_hold_placement_permit,
    release_hold_placement_permit,
)
from tools.evidence_graph_set_signed_retirement_restore_holds import (
    deterministic_restore_hold_id,
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
        timestamp = time.time() if now is None else now
        hold_id = deterministic_restore_hold_id(
            owner_id=owner_id,
            restore_id=restore_id,
            hold_key=hold_key,
        )
        acquire_hold_placement_permit(
            restore_journal,
            owner_id=owner_id,
            restore_id=restore_id,
            hold_id=hold_id,
            now=timestamp,
        )
        try:
            value = super().place(
                owner_id=owner_id,
                restore_id=restore_id,
                hold_key=hold_key,
                reason_code=reason_code,
                actor=actor,
                restore_journal=restore_journal,
                now=timestamp,
            )
        except Exception:
            # Ordinary exceptions imply the hold-store transaction rolled back.
            # A process death leaves the permit active; exact hold replay recovers it.
            try:
                release_hold_placement_permit(
                    restore_journal,
                    owner_id=owner_id,
                    restore_id=restore_id,
                    hold_id=hold_id,
                    now=timestamp,
                )
            except Exception:
                pass
            raise
        release_hold_placement_permit(
            restore_journal,
            owner_id=owner_id,
            restore_id=restore_id,
            hold_id=hold_id,
            now=timestamp,
        )
        return value


__all__ = ["GovernedSignedRetirementRestoreHoldStore"]
