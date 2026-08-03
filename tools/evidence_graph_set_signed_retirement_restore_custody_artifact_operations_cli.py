"""Read-only operational CLI for custody artifact publication attempts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.evidence_graph_set_signed_retirement_restore_custody_artifact_operations import (
    audit_restore_custody_artifacts,
    plan_restore_custody_artifact_retention,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_artifact_readonly import (
    ReadOnlyRestoreCustodyArtifactJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_readonly import (
    ReadOnlySignedRetirementRestoreHoldStore,
)

_DEFAULT_ARTIFACT_PATH = (
    "data/evidence_graph_set_signed_retirement_custody_artifacts.sqlite3"
)


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _artifact_path(value: str | None) -> str:
    return value or os.getenv(
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_CUSTODY_ARTIFACT_DB_PATH",
        _DEFAULT_ARTIFACT_PATH,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "tools.evidence_graph_set_signed_retirement_restore_custody_artifact_operations_cli"
        ),
        description=(
            "Audit custody artifact attempts and produce conservative retention plans. "
            "No command mutates journals or artifacts."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit")
    audit.add_argument("--owner-id", required=True)
    audit.add_argument("--artifact-db-path")
    audit.add_argument("--restore-id")
    audit.add_argument(
        "--state",
        choices=("planned", "running", "completed", "orphaned", "failed", "cancelled"),
    )
    audit.add_argument("--limit", type=int, default=1_000)

    retention = commands.add_parser("retention-plan")
    retention.add_argument("--owner-id", required=True)
    retention.add_argument("--artifact-db-path")
    retention.add_argument(
        "--minimum-age-seconds",
        type=float,
        default=365 * 24 * 60 * 60,
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
        journal = ReadOnlyRestoreCustodyArtifactJournal(
            _artifact_path(args.artifact_db_path)
        )
        if args.command == "audit":
            report = audit_restore_custody_artifacts(
                owner_id=args.owner_id,
                journal=journal,
                restore_id=args.restore_id,
                state=args.state,
                limit=args.limit,
            )
            payload = asdict(report)
            payload.update(
                {
                    "journal_mutation_performed": False,
                    "artifact_mutation_performed": False,
                    "artifact_deletion_performed": False,
                    "artifact_overwrite_performed": False,
                }
            )
            _print(payload)
            return 0
        explicit_holds = set(args.hold_restore_id or ())
        durable_path = args.durable_hold_db_path or os.getenv(
            "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_HOLD_DB_PATH"
        )
        durable_holds: frozenset[str] = frozenset()
        if durable_path:
            durable_holds = ReadOnlySignedRetirementRestoreHoldStore(
                durable_path
            ).active_restore_ids(owner_id=args.owner_id, limit=args.limit)
        all_holds = tuple(sorted(explicit_holds | set(durable_holds)))
        plan = plan_restore_custody_artifact_retention(
            owner_id=args.owner_id,
            journal=journal,
            minimum_age_seconds=args.minimum_age_seconds,
            retain_latest_per_target=args.retain_latest_per_target,
            include_completed=args.include_completed,
            held_restore_ids=all_holds,
            limit=args.limit,
        )
        payload = asdict(plan)
        payload.update(
            {
                "explicit_restore_hold_count": len(explicit_holds),
                "durable_restore_hold_count": len(durable_holds),
                "journal_mutation_performed": False,
                "hold_store_mutation_performed": False,
                "artifact_mutation_performed": False,
                "artifact_deletion_performed": False,
                "artifact_overwrite_performed": False,
            }
        )
        _print(payload)
        return 0
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
