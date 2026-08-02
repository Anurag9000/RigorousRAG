"""Canonical schema boundary for integrity-backed restore legal holds."""

from __future__ import annotations

from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_hold_integrity import (
    IntegritySignedRetirementRestoreHoldStore,
)

_METHODS = frozenset(
    {"process_environment", "descriptor_file", "hmac_assertion"}
)


class GovernedSignedRetirementRestoreHoldStore(
    IntegritySignedRetirementRestoreHoldStore
):
    """Integrity store with strict actor-provenance method validation."""

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


__all__ = ["GovernedSignedRetirementRestoreHoldStore"]
