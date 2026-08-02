"""Read-only CLI for signed retirement operations and retention planning."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.evidence_graph_set_signed_retirement_operations import (
    audit_signed_retirement_operations,
    plan_signed_retirement_retention,
)
from tools.evidence_graph_set_signed_retirement_runtime import (
    get_signed_publication_retirement_journal,
)


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_set_signed_retirement_operations_cli",
        description=(
            "Audit signed publication retirement work and produce conservative "
            "retention plans. No command deletes or mutates retirement state."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit")
    audit.add_argument("--owner-id", required=True)
    audit.add_argument("--publication-operation-id")
    audit.add_argument("--limit", type=int, default=1000)

    retention = commands.add_parser("retention-plan")
    retention.add_argument("--owner-id", required=True)
    retention.add_argument("--minimum-age-days", type=float, default=180.0)
    retention.add_argument("--retain-latest-per-operation", type=int, default=1)
    retention.add_argument("--include-completed", action="store_true")
    retention.add_argument("--held-retirement-id", action="append")
    retention.add_argument("--limit", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        journal = get_signed_publication_retirement_journal()
        if args.command == "audit":
            report = audit_signed_retirement_operations(
                owner_id=args.owner_id,
                publication_operation_id=args.publication_operation_id,
                journal=journal,
                limit=args.limit,
            )
            payload = asdict(report)
            payload.update(
                {
                    "journal_mutation_performed": False,
                    "pointer_mutation_performed": False,
                    "deletion_performed": False,
                    "source_text_returned": False,
                }
            )
            _print(payload)
            return 0
        if args.command == "retention-plan":
            report = plan_signed_retirement_retention(
                owner_id=args.owner_id,
                journal=journal,
                minimum_age_seconds=args.minimum_age_days * 24 * 60 * 60,
                retain_latest_per_operation=args.retain_latest_per_operation,
                include_completed=args.include_completed,
                held_retirement_ids=args.held_retirement_id,
                limit=args.limit,
            )
            payload = asdict(report)
            payload.update(
                {
                    "journal_mutation_performed": False,
                    "pointer_mutation_performed": False,
                    "deletion_performed": False,
                    "source_text_returned": False,
                }
            )
            _print(payload)
            return 0
        raise ValueError("unsupported signed retirement operations command.")
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
