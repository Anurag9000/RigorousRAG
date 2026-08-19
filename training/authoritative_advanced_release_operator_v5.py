"""Installed production release operator v5.

V5 keeps the evaluator-bound v4 release chain and adds fail-early evaluator semantics at cohort
creation. Promotion-grade evaluators must use the exact result semantics implemented by the
streaming evidence path before a governed/retrieval cohort can be published.
"""
from __future__ import annotations

import argparse
import json
from typing import Sequence

from evaluation.strict_production_evaluator_contract import (
    verify_strict_production_evaluator_contract,
)
from training.authoritative_advanced_release_operator_v3 import (
    build_governed_production_cohort,
    build_production_evaluation_evidence,
    build_retrieval_production_cohort,
)
from training.strict_authoritative_advanced_release import (
    qualify_artifact_strictly,
    verify_promotion_strictly,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rigorousrag-authoritative-release",
        description="Strict evaluator-bound production release operator for advanced RAG artifacts",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    governed = sub.add_parser("cohort-governed")
    governed.add_argument("--benchmark-import-receipt", required=True)
    governed.add_argument("--leakage-receipt", required=True)
    governed.add_argument("--split", action="append", required=True)
    governed.add_argument("--evaluator-contract-receipt", required=True)
    governed.add_argument("--base-cohort-output", required=True)
    governed.add_argument("--output", required=True)

    retrieval = sub.add_parser("cohort-retrieval")
    retrieval.add_argument("--retrieval-benchmark-receipt", required=True)
    retrieval.add_argument("--evaluator-contract-receipt", required=True)
    retrieval.add_argument("--base-cohort-output", required=True)
    retrieval.add_argument("--output", required=True)

    evaluation = sub.add_parser("evaluation-evidence")
    evaluation.add_argument("--config", required=True)
    evaluation.add_argument("--checkpoint-digest", required=True)
    evaluation.add_argument("--evaluation-cohort", required=True)
    evaluation.add_argument("--result-receipt", action="append", required=True)
    evaluation.add_argument("--aggregation", choices=("mean", "median"), default="mean")
    evaluation.add_argument("--evaluation-receipt-output", required=True)
    evaluation.add_argument("--cohort-evidence-output", required=True)
    evaluation.add_argument("--evidence-output", required=True)

    qualify = sub.add_parser("qualify")
    qualify.add_argument("--artifact", required=True)
    qualify.add_argument("--evaluation-evidence", required=True)
    qualify.add_argument("--policy", required=True)
    qualify.add_argument("--output", required=True)

    verify = sub.add_parser("verify-promotion")
    verify.add_argument("--artifact", required=True)
    verify.add_argument("--evidence", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "cohort-governed":
        verify_strict_production_evaluator_contract(args.evaluator_contract_receipt)
        result = build_governed_production_cohort(
            args.benchmark_import_receipt,
            leakage_receipt_path=args.leakage_receipt,
            selected_splits=tuple(args.split),
            evaluator_contract_receipt_path=args.evaluator_contract_receipt,
            base_cohort_output=args.base_cohort_output,
            output_path=args.output,
        )
    elif args.command == "cohort-retrieval":
        verify_strict_production_evaluator_contract(args.evaluator_contract_receipt)
        result = build_retrieval_production_cohort(
            args.retrieval_benchmark_receipt,
            evaluator_contract_receipt_path=args.evaluator_contract_receipt,
            base_cohort_output=args.base_cohort_output,
            output_path=args.output,
        )
    elif args.command == "evaluation-evidence":
        result = build_production_evaluation_evidence(
            args.config,
            checkpoint_digest=args.checkpoint_digest,
            evaluator_bound_cohort_path=args.evaluation_cohort,
            result_receipt_paths=tuple(args.result_receipt),
            aggregation=args.aggregation,
            evaluation_receipt_output=args.evaluation_receipt_output,
            cohort_evidence_output=args.cohort_evidence_output,
            evidence_output=args.evidence_output,
        )
    elif args.command == "qualify":
        result = qualify_artifact_strictly(
            args.artifact,
            authoritative_evaluation_evidence_path=args.evaluation_evidence,
            policy_path=args.policy,
            promotion_output=args.output,
        )
    elif args.command == "verify-promotion":
        result = verify_promotion_strictly(args.artifact, args.evidence)
    else:  # pragma: no cover
        raise RuntimeError("unreachable command")
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
