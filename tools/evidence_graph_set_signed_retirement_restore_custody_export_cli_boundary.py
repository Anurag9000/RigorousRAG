"""Canonical protected-key boundary for custody export commands."""

from __future__ import annotations

from typing import Sequence

from tools import evidence_graph_set_signed_retirement_restore_custody_export_cli as _base
from tools import evidence_graph_set_signed_retirement_restore_custody_export_boundary as _protected

_base.authenticate_restore_chain_of_custody = (
    _protected.authenticate_restore_chain_of_custody
)
_base.export_restore_chain_of_custody = _protected.export_restore_chain_of_custody
_base.verify_authenticated_restore_chain_of_custody = (
    _protected.verify_authenticated_restore_chain_of_custody
)
_base.verify_restore_chain_of_custody = _protected.verify_restore_chain_of_custody


def main(argv: Sequence[str] | None = None) -> int:
    return _base.main(argv)


__all__ = ["main"]
