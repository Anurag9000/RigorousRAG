"""Read-only CLI for restore-intent operations and retention planning."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.evidence_graph_set_signed_retirement_restore_hold_readonly import (
    ReadOnlySignedRetirementRestoreHoldStore,
)
from tools.evidence_graph_set_signed_retirement_restore_operations import (
    audit_signed_retirement_restore_operations,
    plan_signed_retirement_restore_retention,
)
from tools.evidence_graph_set_signed_retirement_restore_runtime import (
    get_signed_retirement_restore_journal,
)


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "tools.evidence_graph_set_signed_retirement_restore_operations_cli"
        ),
        description=(
            "Audit signed-retirement restore intents and plan conservative "
            "retention. No command mutates or deletes journal rows."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit")
    audit.add_argument("--owner-id", required=True)
    audit.add_argument("--state")
    audit.add_argument("--snapshot-digest")
    audit.add_argument("--target-path-digest")
    audit.add_argument("--limit", type=int, default=1_000)

    retention = commands.add_parser("retention-plan")
    retention.add_argument("--owner-id", required=True)
    retention.add_argument(
        "--minimum-age-seconds",
        type=float,
        default=180 * 24 * 60 * 60,
    )
    retention.add_argument("--retain-latest-per-target", type=int, default=1)
    retention.add_argument("--include-completed", action="store_true")
    retention.add_argument("--hold-restore-id", action="append")
    retention.add_argument("--durable-hold-db-path")
    retention.add_argument("--limit", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        journal = get_signed_retirement_restore_journal()
        if args.command == "audit":
            report = audit_signed_retirement_restore_operations(
                owner_id=args.owner_id,
                journal=journal,
                state=args.state,
                snapshot_digest=args.snapshot_digest,
                target_path_digest=args.target_path_digest,
                limit=args.limit,
            )
            payload = asdict(report)
            payload.update(
                {
                    "journal_mutation_performed": False,
                    "target_mutation_performed": False,
                    "deletion_performed": False,
                    "source_text_returned": False,
                    "raw_paths_returned": False,
                }
            )
            _print(payload)
            return 0
        if args.command == "retention-plan":
            explicit_holds = set(args.hold_restore_id or ())
            durable_holds: frozenset[str] = frozenset()
            if args.durable_hold_db_path is not None:
                durable_holds = ReadOnlySignedRetirementRestoreHoldStore(
                    args.durable_hold_db_path
                ).active_restore_ids(
                    owner_id=args.owner_id,
                    limit=args.limit,
                )
            held_restore_ids = tuple(sorted(explicit_holds | set(durable_holds)))
            plan = plan_signed_retirement_restore_retention(
                owner_id=args.owner_id,
                journal=journal,
                minimum_age_seconds=args.minimum_age_seconds,
                retain_latest_per_target=args.retain_latest_per_target,
                include_completed=args.include_completed,
                held_restore_ids=held_restore_ids,
                limit=args.limit,
            )
            payload = asdict(plan)
            payload.update(
                {
                    "explicit_hold_count": len(explicit_holds),
                    "durable_hold_count": len(durable_holds),
                    "journal_mutation_performed": False,
                    "hold_store_mutation_performed": False,
                    "target_mutation_performed": False,
                    "deletion_performed": False,
                    "source_text_returned": False,
                    "raw_paths_returned": False,
                }
            )
            _print(payload)
            return 0
        raise ValueError("unsupported restore operations command.")
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
