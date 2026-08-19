"""Production release operator v2 with content-bound evaluator provenance.

This wrapper keeps the v1 release module's cohort/result/promotion authority but strengthens
cohort creation: operators supply an authoritative evaluator-contract receipt, not an opaque
64-hex digest.  The receipt is reconstructed from exact local config bytes before its contract
SHA enters the benchmark cohort.  Other commands delegate to the already cohort-bound v2
release implementation.
"""
from __future__ import annotations

import argparse
import json
from typing import Mapping, Sequence

from evaluation.authoritative_evaluator_contract import (
    verify_authoritative_evaluator_contract,
)
from training.authoritative_advanced_release_operator import (
    build_evaluation_evidence_from_config,
    build_governed_cohort,
    build_retrieval_cohort,
    qualify_artifact_authoritatively,
    verify_authoritative_promotion,
)


def _evaluator_sha(path: str) -> str:
    return verify_authoritative_evaluator_contract(path).contract_sha256


def build_governed_cohort_from_evaluator_receipt(
    benchmark_import_receipt_path: str,
    *,
    leakage_receipt_path: str,
    selected_splits: Sequence[str],
    evaluator_contract_receipt_path: str,
    output_path: str,
) -> Mapping[str, object]:
    contract = verify_authoritative_evaluator_contract(
        evaluator_contract_receipt_path
    )
    result = dict(
        build_governed_cohort(
            benchmark_import_receipt_path,
            leakage_receipt_path=leakage_receipt_path,
            selected_splits=tuple(selected_splits),
            base_evaluator_contract_sha256=contract.contract_sha256,
            output_path=output_path,
        )
    )
    result["evaluator_id"] = contract.evaluator_id
    result["evaluator_source_commit"] = contract.source_commit
    result["base_evaluator_contract_sha256"] = contract.contract_sha256
    return result


def build_retrieval_cohort_from_evaluator_receipt(
    retrieval_benchmark_receipt_path: str,
    *,
    evaluator_contract_receipt_path: str,
    output_path: str,
) -> Mapping[str, object]:
    contract = verify_authoritative_evaluator_contract(
        evaluator_contract_receipt_path
    )
    result = dict(
        build_retrieval_cohort(
            retrieval_benchmark_receipt_path,
            base_evaluator_contract_sha256=contract.contract_sha256,
            output_path=output_path,
        )
    )
    result["evaluator_id"] = contract.evaluator_id
    result["evaluator_source_commit"] = contract.source_commit
    result["base_evaluator_contract_sha256"] = contract.contract_sha256
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rigorousrag-authoritative-release",
        description="Authoritative advanced RAG release operator with content-bound evaluator provenance",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    governed = sub.add_parser("cohort-governed")
    governed.add_argument("--benchmark-import-receipt", required=True)
    governed.add_argument("--leakage-receipt", required=True)
    governed.add_argument("--split", action="append", required=True)
    governed.add_argument("--evaluator-contract-receipt", required=True)
    governed.add_argument("--output", required=True)

    retrieval = sub.add_parser("cohort-retrieval")
    retrieval.add_argument("--retrieval-benchmark-receipt", required=True)
    retrieval.add_argument("--evaluator-contract-receipt", required=True)
    retrieval.add_argument("--output", required=True)

    evaluation = sub.add_parser("evaluation-evidence")
    evaluation.add_argument("--config", required=True)
    evaluation.add_argument("--checkpoint-digest", required=True)
    evaluation.add_argument("--cohort-contract", required=True)
    evaluation.add_argument("--result-receipt", action="append", required=True)
    evaluation.add_argument("--aggregation", choices=("mean", "median"), default="mean")
    evaluation.add_argument("--evaluation-receipt-output", required=True)
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
        result = build_governed_cohort_from_evaluator_receipt(
            args.benchmark_import_receipt,
            leakage_receipt_path=args.leakage_receipt,
            selected_splits=tuple(args.split),
            evaluator_contract_receipt_path=args.evaluator_contract_receipt,
            output_path=args.output,
        )
    elif args.command == "cohort-retrieval":
        result = build_retrieval_cohort_from_evaluator_receipt(
            args.retrieval_benchmark_receipt,
            evaluator_contract_receipt_path=args.evaluator_contract_receipt,
            output_path=args.output,
        )
    elif args.command == "evaluation-evidence":
        result = build_evaluation_evidence_from_config(
            args.config,
            checkpoint_digest=args.checkpoint_digest,
            cohort_contract_path=args.cohort_contract,
            result_receipt_paths=tuple(args.result_receipt),
            aggregation=args.aggregation,
            evaluation_receipt_output=args.evaluation_receipt_output,
            evidence_output=args.evidence_output,
        )
    elif args.command == "qualify":
        result = qualify_artifact_authoritatively(
            args.artifact,
            authoritative_evaluation_evidence_path=args.evaluation_evidence,
            policy_path=args.policy,
            promotion_output=args.output,
        )
    elif args.command == "verify-promotion":
        result = verify_authoritative_promotion(args.artifact, args.evidence)
    else:  # pragma: no cover
        raise RuntimeError("unreachable command")
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_governed_cohort_from_evaluator_receipt",
    "build_retrieval_cohort_from_evaluator_receipt",
    "main",
]
