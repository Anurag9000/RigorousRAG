"""Read-only CLI for expired weaker publication duplicate retirement preflight."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.evidence_graph_runtime import get_evidence_graph_store
from tools.evidence_graph_set_publish_runtime import (
    get_evidence_graph_set_publication_journal,
)
from tools.evidence_graph_set_runtime import get_evidence_graph_set_store
from tools.evidence_graph_set_signed_publication_runtime import (
    get_evidence_graph_set_signed_publication_journal,
)
from tools.evidence_graph_set_signed_retirement import (
    preflight_expired_signed_publication_duplicate_retirement,
)
from tools.sparse_runtime import get_generation_store


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_set_signed_retirement_cli",
        description=(
            "Preflight one expired authorization-only publication duplicate against "
            "a completed signed publication. This command is strictly read-only."
        ),
    )
    preflight = parser.add_subparsers(dest="command", required=True).add_parser(
        "preflight"
    )
    preflight.add_argument("operation_id")
    preflight.add_argument("--owner-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command != "preflight":
            raise ValueError("unsupported signed publication retirement command.")
        value = preflight_expired_signed_publication_duplicate_retirement(
            owner_id=args.owner_id,
            operation_id=args.operation_id,
            authorization_journal=get_evidence_graph_set_publication_journal(),
            signed_journal=get_evidence_graph_set_signed_publication_journal(),
            set_store=get_evidence_graph_set_store(),
            generations=get_generation_store(),
            graphs=get_evidence_graph_store(),
        )
        payload = asdict(value)
        payload.update(
            {
                "automatic_retirement_performed": False,
                "pointer_mutation_performed": False,
                "journal_mutation_performed": False,
                "source_text_returned": False,
            }
        )
        _print(payload)
        return 0 if value.eligible else 1
    except KeyError:
        _print({"error": "not_found"}, stream=sys.stderr)
        return 1
    except (OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
