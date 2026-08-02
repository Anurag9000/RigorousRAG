"""CLI for exporting and verifying signed retirement journal snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from tools.evidence_graph_set_signed_retirement_runtime import (
    get_signed_publication_retirement_journal,
)
from tools.evidence_graph_set_signed_retirement_snapshot import (
    export_signed_retirement_snapshot,
)
from tools.evidence_graph_set_signed_retirement_snapshot_boundary import (
    verify_signed_retirement_snapshot,
)


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _summary(snapshot: Any, *, output_path: str | None = None) -> dict[str, Any]:
    result = {
        "owner_id": snapshot.owner_id,
        "generated_at": snapshot.generated_at,
        "record_count": snapshot.record_count,
        "snapshot_digest": snapshot.snapshot_digest,
        "schema_version": snapshot.schema_version,
        "contains_source_text": False,
        "contains_assertion_secrets": False,
        "journal_mutation_performed": False,
        "restore_performed": False,
        "deletion_performed": False,
    }
    if output_path is not None:
        result["output_path"] = output_path
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_set_signed_retirement_snapshot_cli",
        description=(
            "Export deterministic signed-retirement audit snapshots or verify an "
            "existing snapshot. No command restores or mutates journal state."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export")
    export.add_argument("--owner-id", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--limit", type=int, default=10_000)

    verify = commands.add_parser("verify")
    verify.add_argument("snapshot_path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "export":
            snapshot = export_signed_retirement_snapshot(
                owner_id=args.owner_id,
                journal=get_signed_publication_retirement_journal(),
                output_path=args.output,
                limit=args.limit,
            )
            _print(_summary(snapshot, output_path=args.output))
            return 0
        if args.command == "verify":
            snapshot = verify_signed_retirement_snapshot(args.snapshot_path)
            _print(_summary(snapshot))
            return 0
        raise ValueError("unsupported signed retirement snapshot command.")
    except (FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
