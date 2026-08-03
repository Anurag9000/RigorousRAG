"""Read-only custody timestamp issuance operations CLI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_holds_readonly import (
    ReadOnlyCustodyTimestampIssuanceHoldStore,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_operations import (
    audit_custody_timestamp_issuances,
    plan_custody_timestamp_issuance_retention,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_readonly import (
    ReadOnlyCustodyTimestampIssuanceJournal,
)

_DEFAULT_PATH = (
    "data/evidence_graph_set_signed_retirement_custody_timestamp_issuances.sqlite3"
)
_HOLD_ENV = "EVIDENCE_GRAPH_RESTORE_CUSTODY_TIMESTAMP_ISSUANCE_HOLD_DB_PATH"


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m tools."
            "evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_operations_cli"
        ),
        description=(
            "Audit timestamp issuance queue health or plan conservative retention. "
            "No command retries, cancels, signs, publishes, or deletes anything."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit")
    audit.add_argument("--owner-id", required=True)
    audit.add_argument("--issuance-db-path")
    audit.add_argument("--limit", type=int, default=1_000)

    retention = commands.add_parser("retention-plan")
    retention.add_argument("--owner-id", required=True)
    retention.add_argument("--issuance-db-path")
    retention.add_argument(
        "--minimum-age-seconds",
        type=float,
        default=180 * 24 * 60 * 60,
    )
    retention.add_argument("--retain-latest-per-authority-key", type=int, default=1)
    retention.add_argument("--include-completed", action="store_true")
    retention.add_argument("--hold-issuance-id", action="append")
    retention.add_argument("--hold-db-path")
    retention.add_argument("--limit", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        path = args.issuance_db_path or os.getenv(
            "EVIDENCE_GRAPH_RESTORE_CUSTODY_TIMESTAMP_ISSUANCE_DB_PATH",
            _DEFAULT_PATH,
        )
        journal = ReadOnlyCustodyTimestampIssuanceJournal(path)
        if args.command == "audit":
            report = audit_custody_timestamp_issuances(
                owner_id=args.owner_id,
                journal=journal,
                limit=args.limit,
            )
            payload = asdict(report)
            payload.update(
                {
                    "mutation_performed": False,
                    "retry_performed": False,
                    "cancellation_performed": False,
                    "attestation_created": False,
                    "deletion_performed": False,
                    "contains_attestation_signatures": False,
                    "contains_private_key_material": False,
                    "contains_raw_paths": False,
                }
            )
        else:
            durable_holds: frozenset[str] = frozenset()
            hold_path = args.hold_db_path or os.getenv(_HOLD_ENV)
            if hold_path:
                durable_holds = ReadOnlyCustodyTimestampIssuanceHoldStore(
                    hold_path
                ).active_issuance_ids(owner_id=args.owner_id)
            explicit_holds = frozenset(args.hold_issuance_id or ())
            plan = plan_custody_timestamp_issuance_retention(
                owner_id=args.owner_id,
                journal=journal,
                minimum_age_seconds=args.minimum_age_seconds,
                retain_latest_per_authority_key=(
                    args.retain_latest_per_authority_key
                ),
                include_completed=args.include_completed,
                held_issuance_ids=durable_holds | explicit_holds,
                limit=args.limit,
            )
            payload = asdict(plan)
            payload.update(
                {
                    "durable_hold_registry_checked": hold_path is not None,
                    "durable_active_hold_count": len(durable_holds),
                    "explicit_hold_count": len(explicit_holds),
                    "mutation_performed": False,
                    "deletion_performed": False,
                    "compaction_performed": False,
                    "contains_attestation_signatures": False,
                    "contains_private_key_material": False,
                    "contains_raw_paths": False,
                }
            )
        _print(payload)
        return 0
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
