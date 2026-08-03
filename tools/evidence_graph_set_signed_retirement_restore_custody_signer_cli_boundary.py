"""Canonical non-replayable actor boundary for custody signer administration."""

from __future__ import annotations

from typing import Sequence

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_signer_cli as _base,
)

_DIRECT_ADMIN_METHODS = frozenset({"process_environment", "descriptor_file"})
_ORIGINAL_REQUIRE = getattr(
    _base,
    "_unrestricted_require_relation_review_actor",
    _base.require_relation_review_actor,
)
_base._unrestricted_require_relation_review_actor = _ORIGINAL_REQUIRE


def _require_direct_signer_admin_actor(requested_actor_id, *, binding):
    resolved = _ORIGINAL_REQUIRE(requested_actor_id, binding=binding)
    if resolved.binding_method not in _DIRECT_ADMIN_METHODS:
        raise PermissionError(
            "custody signer administration requires a direct process-owned actor."
        )
    return resolved


_base.require_relation_review_actor = _require_direct_signer_admin_actor


def main(argv: Sequence[str] | None = None) -> int:
    return _base.main(argv)


__all__ = ["main"]
