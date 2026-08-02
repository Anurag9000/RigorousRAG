"""Privacy-safe operator CLI for completed evidence-graph benchmark runs."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from tools.evidence_graph_rag_run_runtime import get_graph_rag_run_store


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _summary(value: Any) -> dict[str, Any]:
    return {
        "plan_fingerprint": value.plan_fingerprint,
        "benchmark_fingerprint": value.benchmark_fingerprint,
        "benchmark_id": value.benchmark_id,
        "run_id": value.run_id,
        "seed": value.seed,
        "case_count": value.case_count,
        "run_contract_digest": value.run_contract_digest,
        "run_report_digest": value.run_report.report_digest,
        "stored_run_digest": value.stored_run_digest,
        "completed_at": value.completed_at,
        "contains_raw_query": False,
        "contains_evidence_text": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_rag_run_cli",
        description="Inspect or exactly remove text-free completed GraphRAG benchmark runs.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status")
    status.add_argument("plan_fingerprint")
    remove = commands.add_parser("remove-plan")
    remove.add_argument("plan_fingerprint")
    remove.add_argument("--confirm-plan-fingerprint", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        store = get_graph_rag_run_store()
        if args.command == "status":
            values = store.list_plan(args.plan_fingerprint)
            if not values:
                _print({"error": "not_found"}, stream=sys.stderr)
                return 1
            _print(
                {
                    "plan_fingerprint": args.plan_fingerprint,
                    "run_count": len(values),
                    "runs": [_summary(value) for value in values],
                    "mutation_performed": False,
                }
            )
            return 0
        if args.command == "remove-plan":
            removed = store.remove_plan(
                args.plan_fingerprint,
                confirm_plan_fingerprint=args.confirm_plan_fingerprint,
            )
            _print(
                {
                    "plan_fingerprint": args.plan_fingerprint,
                    "removed": removed,
                    "query_or_evidence_text_removed": False,
                }
            )
            return 0 if removed else 1
        raise ValueError("unsupported run-store command.")
    except (OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
