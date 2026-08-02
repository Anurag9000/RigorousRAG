"""Read-only CLI for signed retirement snapshot restore preflight."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.evidence_graph_set_signed_retirement_readonly import (
    ReadOnlySignedPublicationRetirementJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_preflight import (
    preflight_signed_retirement_snapshot_restore,
)
from tools.evidence_graph_set_signed_retirement_snapshot_boundary import (
    verify_signed_retirement_snapshot,
)


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_set_signed_retirement_restore_cli",
        description=(
            "Compare a verified retirement snapshot with an initialized read-only "
            "target journal. No restore or target mutation is performed."
        ),
    )
    preflight = parser.add_subparsers(
        dest="command", required=True
    ).add_parser("preflight")
    preflight.add_argument("snapshot_path")
    preflight.add_argument("--target-db-path", required=True)
    preflight.add_argument("--limit", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command != "preflight":
            raise ValueError("unsupported retirement restore command.")
        snapshot = verify_signed_retirement_snapshot(args.snapshot_path)
        target = ReadOnlySignedPublicationRetirementJournal(args.target_db_path)
        report = preflight_signed_retirement_snapshot_restore(
            snapshot=snapshot,
            target_journal=target,
            limit=args.limit,
        )
        payload = asdict(report)
        payload.update(
            {
                "target_mutation_performed": False,
                "restore_performed": False,
                "journal_insert_performed": False,
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
