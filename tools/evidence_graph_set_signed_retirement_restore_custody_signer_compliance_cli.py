"""Query-only custody signer governance compliance CLI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.evidence_graph_set_signed_retirement_restore_custody_signer_admin_use_audit_readonly import (
    ReadOnlyCustodySignerAdminUseAuditStore,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signer_compliance import (
    audit_custody_signer_compliance,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signer_readonly import (
    ReadOnlyCustodySignerKeyRegistry,
)

_DEFAULT_REGISTRY = "data/evidence_graph_set_signed_retirement_custody_signers.sqlite3"


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "tools.evidence_graph_set_signed_retirement_restore_custody_signer_compliance_cli"
        ),
        description=(
            "Audit custody signer registry records against direct actor methods and "
            "committed one-operation signed-administration reservations."
        ),
    )
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--registry-db-path")
    parser.add_argument("--admin-use-db-path")
    parser.add_argument("--limit", type=int, default=1_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        registry_path = args.registry_db_path or os.getenv(
            "EVIDENCE_GRAPH_RESTORE_CUSTODY_SIGNER_DB_PATH",
            _DEFAULT_REGISTRY,
        )
        admin_path = args.admin_use_db_path or os.getenv(
            "EVIDENCE_GRAPH_RESTORE_CUSTODY_SIGNER_ADMIN_USE_DB_PATH"
        )
        admin_store = (
            None
            if not admin_path
            else ReadOnlyCustodySignerAdminUseAuditStore(admin_path)
        )
        report = audit_custody_signer_compliance(
            owner_id=args.owner_id,
            registry=ReadOnlyCustodySignerKeyRegistry(registry_path),
            admin_use_store=admin_store,
            limit=args.limit,
        )
        payload = asdict(report)
        payload.update(
            {
                "admin_use_source_configured": admin_store is not None,
                "registry_mutation_performed": False,
                "admin_use_mutation_performed": False,
                "key_material_mutation_performed": False,
                "source_text_returned": False,
                "raw_path_returned": False,
            }
        )
        _print(payload)
        return 0
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
