"""Read-only operator CLI for restore-custody audit and retention planning."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.evidence_graph_set_signed_retirement_restore_custody_operations import (
    audit_restore_custody_operations,
    plan_restore_custody_retention,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_readonly import (
    ReadOnlySignedRetirementRestoreCustodyStore,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_readonly import (
    ReadOnlySignedRetirementRestoreHoldStore,
)

_DEFAULT_CUSTODY_PATH = "data/evidence_graph_set_signed_retirement_custody.sqlite3"


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _custody_path(value: str | None) -> str:
    return value or os.getenv(
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_CUSTODY_DB_PATH",
        _DEFAULT_CUSTODY_PATH,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "tools.evidence_graph_set_signed_retirement_restore_custody_operations_cli"
        ),
        description=(
            "Audit restore custody manifests and produce conservative retention plans. "
            "No command mutates custody, hold, restore, or target state."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit")
    audit.add_argument("--owner-id", required=True)
    audit.add_argument("--custody-db-path")
    audit.add_argument("--restore-id")
    audit.add_argument("--snapshot-digest")
    audit.add_argument("--target-path-digest")
    audit.add_argument("--state", choices=("pre_bound", "post_bound"))
    audit.add_argument("--limit", type=int, default=1_000)

    retention = commands.add_parser("retention-plan")
    retention.add_argument("--owner-id", required=True)
    retention.add_argument("--custody-db-path")
    retention.add_argument(
        "--minimum-age-seconds",
        type=float,
        default=365 * 24 * 60 * 60,
    )
    retention.add_argument("--retain-latest-per-target", type=int, default=1)
    retention.add_argument("--include-post-bound", action="store_true")
    retention.add_argument("--hold-custody-id", action="append")
    retention.add_argument("--durable-hold-db-path")
    retention.add_argument("--limit", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        store = ReadOnlySignedRetirementRestoreCustodyStore(
            _custody_path(args.custody_db_path)
        )
        if args.command == "audit":
            report = audit_restore_custody_operations(
                owner_id=args.owner_id,
                store=store,
                restore_id=args.restore_id,
                snapshot_digest=args.snapshot_digest,
                target_path_digest=args.target_path_digest,
                state=args.state,
                limit=args.limit,
            )
            payload = asdict(report)
            payload.update(
                {
                    "custody_store_mutation_performed": False,
                    "hold_store_mutation_performed": False,
                    "restore_mutation_performed": False,
                    "target_mutation_performed": False,
                    "deletion_performed": False,
                }
            )
            _print(payload)
            return 0
        if args.command == "retention-plan":
            explicit_custody_holds = set(args.hold_custody_id or ())
            durable_hold_path = args.durable_hold_db_path or os.getenv(
                "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_HOLD_DB_PATH"
            )
            durable_restore_holds: frozenset[str] = frozenset()
            if durable_hold_path:
                durable_restore_holds = ReadOnlySignedRetirementRestoreHoldStore(
                    durable_hold_path
                ).active_restore_ids(owner_id=args.owner_id, limit=args.limit)
            plan = plan_restore_custody_retention(
                owner_id=args.owner_id,
                store=store,
                minimum_age_seconds=args.minimum_age_seconds,
                retain_latest_per_target=args.retain_latest_per_target,
                include_post_bound=args.include_post_bound,
                held_custody_ids=tuple(sorted(explicit_custody_holds)),
                held_restore_ids=tuple(sorted(durable_restore_holds)),
                limit=args.limit,
            )
            payload = asdict(plan)
            payload.update(
                {
                    "explicit_custody_hold_count": len(explicit_custody_holds),
                    "durable_restore_hold_count": len(durable_restore_holds),
                    "custody_store_mutation_performed": False,
                    "hold_store_mutation_performed": False,
                    "restore_mutation_performed": False,
                    "target_mutation_performed": False,
                    "deletion_performed": False,
                }
            )
            _print(payload)
            return 0
        raise ValueError("unsupported restore custody operations command.")
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
