"""Privacy-safe audit and retention planning CLI for evidence-graph jobs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.evidence_graph_job_runtime import get_evidence_graph_job_journal
from tools.evidence_graph_operations import (
    audit_evidence_graph_jobs,
    plan_evidence_graph_job_retention,
)
from tools.evidence_graph_runtime import get_evidence_graph_store
from tools.sparse_runtime import get_generation_store


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_operations_cli",
        description=(
            "Audit derived graph jobs and plan conservative retention. "
            "This CLI never deletes jobs or graph generations."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--owner-id", required=True)
    audit.add_argument("--limit", type=int, default=10_000)
    retention = commands.add_parser("retention-plan")
    retention.add_argument("--owner-id", required=True)
    retention.add_argument("--min-age-seconds", type=float, required=True)
    retention.add_argument("--limit", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        common = {
            "owner_id": args.owner_id,
            "journal": get_evidence_graph_job_journal(),
            "generations": get_generation_store(),
            "graphs": get_evidence_graph_store(),
            "limit": args.limit,
        }
        if args.command == "audit":
            report = audit_evidence_graph_jobs(**common)
            payload = asdict(report)
            payload["report_digest"] = report.report_digest
            payload["mutation_performed"] = False
            payload["contains_graph_text"] = False
            _print(payload)
            return 0
        if args.command == "retention-plan":
            plan = plan_evidence_graph_job_retention(
                **common,
                min_age_seconds=args.min_age_seconds,
            )
            payload = asdict(plan)
            payload["plan_digest"] = plan.plan_digest
            payload["mutation_performed"] = False
            payload["deletion_authorized"] = False
            _print(payload)
            return 0
        raise ValueError("unsupported graph operations command.")
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
