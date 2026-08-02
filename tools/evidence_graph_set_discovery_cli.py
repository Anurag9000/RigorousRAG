"""Read-only CLI for authoritative evidence-graph-set discovery."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from tools.evidence_graph_set_discovery import list_evidence_graph_sets


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
        prog="python -m tools.evidence_graph_set_discovery_cli",
        description=(
            "List owner-scoped current reviewed evidence-graph sets without "
            "returning graph text or mutating storage."
        ),
    )
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--include-unavailable", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        values = list_evidence_graph_sets(
            owner_id=args.owner_id,
            limit=args.limit,
            include_unavailable=args.include_unavailable,
        )
        _print(
            {
                "count": len(values),
                "graph_sets": values,
                "mutation_performed": False,
                "source_text_returned": False,
                "semantic_inference_performed": False,
            }
        )
        return 0
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
