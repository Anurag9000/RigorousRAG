"""Read-only CLI for permit-recovery audit and retention planning."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.evidence_graph_set_signed_retirement_restore_hold_permit_recovery_operations import (
    audit_hold_permit_recoveries,
    plan_hold_permit_recovery_retention,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_runtime import (
    get_signed_retirement_restore_hold_store,
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
            "python -m tools.evidence_graph_set_signed_retirement_restore_"
            "hold_permit_recovery_operations_cli"
        ),
        description=(
            "Audit hold-permit recovery receipts and produce conservative "
            "retention plans. No command mutates holds, permits or receipts."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit")
    audit.add_argument("--owner-id", required=True)
    audit.add_argument("--limit", type=int, default=1_000)

    retention = commands.add_parser("retention-plan")
    retention.add_argument("--owner-id", required=True)
    retention.add_argument(
        "--minimum-age-seconds",
        type=float,
        default=365 * 24 * 60 * 60,
    )
    retention.add_argument("--retain-latest-per-restore", type=int, default=1)
    retention.add_argument("--hold-recovery-id", action="append", default=[])
    retention.add_argument("--limit", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        restore_journal = get_signed_retirement_restore_journal()
        hold_store = get_signed_retirement_restore_hold_store()
        if args.command == "audit":
            report = audit_hold_permit_recoveries(
                owner_id=args.owner_id,
                restore_journal=restore_journal,
                hold_store=hold_store,
                limit=args.limit,
            )
            _print(asdict(report))
            return 0
        if args.command == "retention-plan":
            plan = plan_hold_permit_recovery_retention(
                owner_id=args.owner_id,
                restore_journal=restore_journal,
                hold_store=hold_store,
                minimum_age_seconds=args.minimum_age_seconds,
                retain_latest_per_restore=args.retain_latest_per_restore,
                held_recovery_ids=args.hold_recovery_id,
                limit=args.limit,
            )
            _print(asdict(plan))
            return 0
        raise ValueError("unsupported permit-recovery operations command.")
    except KeyError:
        _print({"error": "not_found"}, stream=sys.stderr)
        return 1
    except (OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
