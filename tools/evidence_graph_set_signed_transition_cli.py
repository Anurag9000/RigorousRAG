"""Read-only operator CLI for signed publication journal transition planning."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.evidence_graph_set_publish_runtime import (
    get_evidence_graph_set_publication_journal,
)
from tools.evidence_graph_set_signed_publication_runtime import (
    get_evidence_graph_set_signed_publication_journal,
)
from tools.evidence_graph_set_signed_transition import (
    assess_signed_publication_transition,
)


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_set_signed_transition_cli",
        description=(
            "Audit authorization-only publication attempts before re-seeding them "
            "through the isolated signed publication journal. This command is read-only."
        ),
    )
    audit = parser.add_subparsers(dest="command", required=True).add_parser("audit")
    audit.add_argument("--owner-id", required=True)
    audit.add_argument("--graph-set-key")
    audit.add_argument("--limit", type=int, default=1_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command != "audit":
            raise ValueError("unsupported signed publication transition command.")
        report = assess_signed_publication_transition(
            owner_id=args.owner_id,
            graph_set_key=args.graph_set_key,
            limit=args.limit,
            authorization_journal=get_evidence_graph_set_publication_journal(),
            signed_journal=get_evidence_graph_set_signed_publication_journal(),
        )
        payload = asdict(report)
        payload.update(
            {
                "automatic_migration_performed": False,
                "publication_mutation_performed": False,
                "journal_mutation_performed": False,
                "source_text_returned": False,
            }
        )
        _print(payload)
        return 0
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
