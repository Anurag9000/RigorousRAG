"""Read-only operational CLI for publication attempts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.evidence_graph_set_publish_operations import (
    audit_publication_attempts,
    plan_publication_retention,
)
from tools.evidence_graph_set_publish_runtime import (
    get_evidence_graph_set_publication_journal,
)
from tools.evidence_graph_set_runtime import get_evidence_graph_set_store


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_set_publish_operations_cli",
        description="Audit publication attempts and plan retention without deleting data.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--owner-id", required=True)
    audit.add_argument("--graph-set-key")
    audit.add_argument("--limit", type=int, default=10_000)
    retention = commands.add_parser("retention-plan")
    retention.add_argument("--owner-id", required=True)
    retention.add_argument("--graph-set-key")
    retention.add_argument("--minimum-age-seconds", type=int, default=2_592_000)
    retention.add_argument("--limit", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "audit":
            report = audit_publication_attempts(
                get_evidence_graph_set_publication_journal(),
                owner_id=args.owner_id,
                graph_set_key=args.graph_set_key,
                limit=args.limit,
            )
            payload = asdict(report)
            payload.update(
                {
                    "mutation_performed": False,
                    "source_text_returned": False,
                }
            )
            _print(payload)
            return 0
        if args.command == "retention-plan":
            plan = plan_publication_retention(
                get_evidence_graph_set_publication_journal(),
                set_store=get_evidence_graph_set_store(),
                owner_id=args.owner_id,
                graph_set_key=args.graph_set_key,
                minimum_age_seconds=args.minimum_age_seconds,
                limit=args.limit,
            )
            payload = asdict(plan)
            payload["source_text_returned"] = False
            _print(payload)
            return 0
        raise ValueError("unsupported publication-operations command.")
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
