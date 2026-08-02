"""Strict CLI for historical evidence-graph benchmark regression gates."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Sequence

from tools.evidence_graph_rag_benchmark_cli import _print, _read_object, _write_atomic
from tools.evidence_graph_rag_regression import (
    GraphRAGRegressionPolicy,
    evaluate_graph_rag_regression,
    policy_from_mapping,
    report_from_mapping,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_rag_regression_cli",
        description=(
            "Compare query-digest-only evidence-graph benchmark reports under "
            "versioned quality, paired non-inferiority and work-budget policy."
        ),
    )
    compare = parser.add_subparsers(dest="command", required=True).add_parser("compare")
    compare.add_argument("baseline_report")
    compare.add_argument("candidate_report")
    compare.add_argument("--policy-file")
    compare.add_argument("--output-file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command != "compare":
            raise ValueError("unsupported graph RAG regression command.")
        baseline = report_from_mapping(_read_object(args.baseline_report))
        candidate = report_from_mapping(_read_object(args.candidate_report))
        policy = (
            policy_from_mapping(_read_object(args.policy_file))
            if args.policy_file
            else GraphRAGRegressionPolicy()
        )
        report = evaluate_graph_rag_regression(baseline, candidate, policy)
        payload = asdict(report)
        payload["report_digest"] = report.report_digest
        payload["paired_interval_method"] = "normal_approximation_over_run_deltas"
        payload["contains_raw_query"] = False
        payload["contains_evidence_text"] = False
        payload["runtime_policy_changed"] = False
        if args.output_file:
            _write_atomic(args.output_file, payload)
        _print(payload)
        return 0 if report.decision == "eligible" else 1
    except (OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
