"""Privacy-safe CLI for governed GraphRAG historical baselines."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

from tools.evidence_graph_rag_baseline import regression_report_from_mapping
from tools.evidence_graph_rag_baseline_runtime import get_graph_rag_baseline_store
from tools.evidence_graph_rag_benchmark_cli import _print, _read_object
from tools.evidence_graph_rag_regression import (
    GraphRAGRegressionPolicy,
    policy_from_mapping,
    report_from_mapping,
)


def _policy(path: str | None) -> GraphRAGRegressionPolicy:
    return (
        policy_from_mapping(_read_object(path))
        if path is not None
        else GraphRAGRegressionPolicy()
    )


def _summary(value: Any) -> dict[str, Any]:
    return {
        "benchmark_fingerprint": value.benchmark_fingerprint,
        "benchmark_id": value.benchmark_id,
        "policy_id": value.policy_id,
        "policy_digest": value.policy_digest,
        "baseline_digest": value.baseline_digest,
        "benchmark_report_digest": value.benchmark_report.report_digest,
        "previous_baseline_digest": value.previous_baseline_digest,
        "activation_regression_digest": value.activation_regression_digest,
        "activated_at": value.activated_at,
        "contains_raw_query": False,
        "contains_evidence_text": False,
        "runtime_policy_changed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_rag_baseline_cli",
        description=(
            "Initialize or replace append-only GraphRAG benchmark baselines. "
            "Replacement requires an eligible exact regression report."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser("initialize")
    initialize.add_argument("candidate_report")
    initialize.add_argument("--policy-file")
    initialize.add_argument("--expect-no-current", action="store_true", required=True)

    promote = commands.add_parser("promote")
    promote.add_argument("candidate_report")
    promote.add_argument("regression_report")
    promote.add_argument("--policy-file")
    promote.add_argument("--expected-current-baseline-digest", required=True)

    status = commands.add_parser("status")
    status.add_argument("benchmark_fingerprint")
    status.add_argument("--policy-id", required=True)

    history = commands.add_parser("history")
    history.add_argument("benchmark_fingerprint")
    history.add_argument("--policy-id", required=True)
    history.add_argument("--limit", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        store = get_graph_rag_baseline_store()
        if args.command == "initialize":
            if not args.expect_no_current:
                raise ValueError("initialization requires explicit no-current expectation.")
            candidate = report_from_mapping(_read_object(args.candidate_report))
            value = store.activate(
                candidate,
                _policy(args.policy_file),
                expected_current_baseline_digest=None,
            )
            payload = _summary(value)
            payload["baseline_pointer_changed"] = True
            _print(payload)
            return 0
        if args.command == "promote":
            candidate = report_from_mapping(_read_object(args.candidate_report))
            regression = regression_report_from_mapping(
                _read_object(args.regression_report)
            )
            value = store.activate(
                candidate,
                _policy(args.policy_file),
                expected_current_baseline_digest=(
                    args.expected_current_baseline_digest
                ),
                regression=regression,
            )
            payload = _summary(value)
            payload["baseline_pointer_changed"] = True
            _print(payload)
            return 0
        if args.command == "status":
            value = store.current(
                benchmark_fingerprint=args.benchmark_fingerprint,
                policy_id=args.policy_id,
            )
            if value is None:
                _print({"error": "not_found"}, stream=sys.stderr)
                return 1
            payload = _summary(value)
            payload["mutation_performed"] = False
            _print(payload)
            return 0
        if args.command == "history":
            values = store.history(
                benchmark_fingerprint=args.benchmark_fingerprint,
                policy_id=args.policy_id,
                limit=args.limit,
            )
            if not values:
                _print({"error": "not_found"}, stream=sys.stderr)
                return 1
            _print(
                {
                    "benchmark_fingerprint": args.benchmark_fingerprint,
                    "policy_id": args.policy_id,
                    "count": len(values),
                    "baselines": [_summary(value) for value in values],
                    "mutation_performed": False,
                }
            )
            return 0
        raise ValueError("unsupported baseline command.")
    except (OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
