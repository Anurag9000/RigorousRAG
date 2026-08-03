"""Canonical query-only boundary for custody artifact status and listing."""

from __future__ import annotations

import os
import sys
from typing import Sequence

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_artifact_cli as _base,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_artifact_readonly import (
    ReadOnlyRestoreCustodyArtifactJournal,
)

_DEFAULT_PATH = "data/evidence_graph_set_signed_retirement_custody_artifacts.sqlite3"


def _read_only_journal() -> ReadOnlyRestoreCustodyArtifactJournal:
    return ReadOnlyRestoreCustodyArtifactJournal(
        os.getenv(
            "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_CUSTODY_ARTIFACT_DB_PATH",
            _DEFAULT_PATH,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _base.build_parser()
    args = parser.parse_args(argv)
    if args.command not in {"status", "list"}:
        return _base.main(argv)
    try:
        journal = _read_only_journal()
        if args.command == "status":
            payload = _base._attempt_summary(journal.get(args.artifact_id))
            payload.update(
                {
                    "journal_mutation_performed": False,
                    "artifact_mutation_performed": False,
                    "artifact_deletion_performed": False,
                }
            )
            _base._print(payload)
            return 0
        values = journal.list(
            owner_id=args.owner_id,
            state=args.state,
            limit=args.limit,
        )
        _base._print(
            {
                "owner_id": args.owner_id,
                "state": args.state,
                "count": len(values),
                "items": [_base._attempt_summary(value) for value in values],
                "journal_mutation_performed": False,
                "artifact_mutation_performed": False,
                "artifact_deletion_performed": False,
                "source_text_returned": False,
                "raw_paths_returned": False,
            }
        )
        return 0
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        _base._print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


__all__ = ["main"]
