"""Canonical signed signer-administration boundary with query-only status."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from tools import evidence_graph_set_signed_retirement_restore_custody_signer_admin_cli as _base
from tools import evidence_graph_set_signed_retirement_restore_custody_signer_admin_use_boundary as _credential_boundary  # noqa: F401
from tools.evidence_graph_set_signed_retirement_restore_custody_signer_admin_use_readonly import (
    ReadOnlyCustodySignerAdminUseStore,
)

_DEFAULT_PATH = (
    "data/evidence_graph_set_signed_retirement_custody_signer_admin_uses.sqlite3"
)


def _status_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python scripts/"
            "evidence_graph_set_signed_retirement_restore_custody_signer_admin_governed.py"
        )
    )
    parser.add_argument("command", choices=("status",))
    parser.add_argument("use_id")
    parser.add_argument("--admin-use-db-path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    selected = list(sys.argv[1:] if argv is None else argv)
    if not selected or selected[0] != "status":
        return _base.main(selected)
    parser = _status_parser()
    try:
        args = parser.parse_args(selected)
        path = args.admin_use_db_path or os.getenv(
            "EVIDENCE_GRAPH_RESTORE_CUSTODY_SIGNER_ADMIN_USE_DB_PATH",
            _DEFAULT_PATH,
        )
        value = ReadOnlyCustodySignerAdminUseStore(path).get(args.use_id)
        payload = _base._use_summary(value)
        payload.update(
            {
                "admin_use_mutation_performed": False,
                "registry_mutation_performed": False,
            }
        )
        _base._print(payload)
        return 0
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        _base._print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


__all__ = ["main"]
