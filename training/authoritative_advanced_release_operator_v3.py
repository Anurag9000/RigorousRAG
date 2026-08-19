"""Production release operator v3: evaluator receipt -> bound cohort -> v3 evaluation -> promotion.

This is the installed production release surface. Historical v1/v2 operators remain importable
for reproducibility, but v3 persists evaluator provenance through the cohort and advanced-
evaluation envelopes consumed by promotion/runtime authority.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from evaluation.evaluator_bound_advanced_evaluation import (
    build_evaluator_bound_advanced_evaluation_evidence,
    verify_evaluator_bound_advanced_evaluation_evidence,
    write_evaluator_bound_advanced_evaluation_evidence,
)
from evaluation.evaluator_bound_evaluation_cohort import (
    build_evaluator_bound_evaluation_cohort,
    verify_evaluator_bound_evaluation_cohort,
    write_evaluator_bound_evaluation_cohort,
)
from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_operator import verify_checkpoint_from_config
from training.authoritative_advanced_release_operator import (
    qualify_artifact_authoritatively,
    verify_authoritative_promotion,
)
from training.authoritative_advanced_release_operator_v2 import (
    build_governed_cohort_from_evaluator_receipt,
    build_retrieval_cohort_from_evaluator_receipt,
)


def _bind_cohort(
    *,
    base_cohort_path: str | Path,
    evaluator_contract_receipt_path: str | Path,
    output_path: str | Path,
) -> Mapping[str, object]:
    binding = build_evaluator_bound_evaluation_cohort(
        base_cohort_path,
        evaluator_contract_receipt_path=evaluator_contract_receipt_path,
    )
    write_evaluator_bound_evaluation_cohort(output_path, binding)
    verified, cohort, evaluator = verify_evaluator_bound_evaluation_cohort(output_path)
    if verified.contract_sha256 != binding.contract_sha256:
        raise RuntimeError("evaluator-bound cohort changed during publication")
    return {
        "authority_kind": cohort.authority_kind,
        "benchmark_id": cohort.benchmark_id,
        "benchmark_manifest_sha256": cohort.benchmark_manifest_sha256,
        "benchmark_contract_sha256": cohort.benchmark_contract_sha256,
        "sample_count": cohort.sample_count,
        "sample_universe_sha256": cohort.sample_universe_sha256,
        "evaluator_id": evaluator.evaluator_id,
        "base_evaluator_contract_sha256": evaluator.contract_sha256,
        "base_cohort_contract_sha256": cohort.contract_sha256,
        "evaluator_bound_cohort_sha256": binding.contract_sha256,
        "output": str(
            safe_advanced_path(
                output_path,
                label="evaluator-bound evaluation cohort",
                must_exist=True,
                require_file=True,
            )
        ),
    }


def build_governed_production_cohort(
    benchmark_import_receipt_path: str | Path,
    *,
    leakage_receipt_path: str | Path,
    selected_splits: Sequence[str],
    evaluator_contract_receipt_path: str | Path,
    base_cohort_output: str | Path,
    output_path: str | Path,
) -> Mapping[str, object]:
    build_governed_cohort_from_evaluator_receipt(
        str(benchmark_import_receipt_path),
        leakage_receipt_path=str(leakage_receipt_path),
        selected_splits=tuple(selected_splits),
        evaluator_contract_receipt_path=str(evaluator_contract_receipt_path),
        output_path=str(base_cohort_output),
    )
    return _bind_cohort(
        base_cohort_path=base_cohort_output,
        evaluator_contract_receipt_path=evaluator_contract_receipt_path,
        output_path=output_path,
    )


def build_retrieval_production_cohort(
    retrieval_benchmark_receipt_path: str | Path,
    *,
    evaluator_contract_receipt_path: str | Path,
    base_cohort_output: str | Path,
    output_path: str | Path,
) -> Mapping[str, object]:
    build_retrieval_cohort_from_evaluator_receipt(
        str(retrieval_benchmark_receipt_path),
        evaluator_contract_receipt_path=str(evaluator_contract_receipt_path),
        output_path=str(base_cohort_output),
    )
    return _bind_cohort(
        base_cohort_path=base_cohort_output,
        evaluator_contract_receipt_path=evaluator_contract_receipt_path,
        output_path=output_path,
    )


def build_production_evaluation_evidence(
    config_path: str | Path,
    *,
    checkpoint_digest: str,
    evaluator_bound_cohort_path: str | Path,
    result_receipt_paths: Sequence[str | Path],
    aggregation: str,
    evaluation_receipt_output: str | Path,
    cohort_evidence_output: str | Path,
    evidence_output: str | Path,
) -> Mapping[str, object]:
    binding = verify_checkpoint_from_config(
        config_path,
        checkpoint_digest=checkpoint_digest,
    )
    evaluation, evidence = build_evaluator_bound_advanced_evaluation_evidence(
        binding,
        evaluator_bound_cohort_path=evaluator_bound_cohort_path,
        result_receipt_paths=result_receipt_paths,
        aggregation=aggregation,
        evaluation_receipt_path=evaluation_receipt_output,
        cohort_evidence_path=cohort_evidence_output,
    )
    write_evaluator_bound_advanced_evaluation_evidence(evidence_output, evidence)
    verified_evaluation, verified_evidence = verify_evaluator_bound_advanced_evaluation_evidence(
        evidence_output
    )
    if (
        verified_evaluation.receipt_sha256 != evaluation.receipt_sha256
        or verified_evidence.evidence_sha256 != evidence.evidence_sha256
    ):
        raise RuntimeError("evaluator-bound evaluation changed during publication")
    return {
        "evaluation_receipt_sha256": evaluation.receipt_sha256,
        "authoritative_evaluation_evidence_sha256": evidence.evidence_sha256,
        "evaluator_bound_cohort_sha256": evidence.evaluator_bound_cohort_sha256,
        "run_count": len(
            verify_evaluator_bound_advanced_evaluation_evidence(evidence_output)[0].runs
        ),
        "evaluation_receipt_output": str(
            safe_advanced_path(
                evaluation_receipt_output,
                label="advanced evaluation receipt",
                must_exist=True,
                require_file=True,
            )
        ),
        "cohort_evidence_output": str(
            safe_advanced_path(
                cohort_evidence_output,
                label="cohort-bound advanced evaluation evidence",
                must_exist=True,
                require_file=True,
            )
        ),
        "evidence_output": str(
            safe_advanced_path(
                evidence_output,
                label="evaluator-bound advanced evaluation evidence",
                must_exist=True,
                require_file=True,
            )
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rigorousrag-authoritative-release",
        description="Evaluator-bound production release operator for advanced RAG artifacts",
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
        result = build_governed_production_cohort(
            args.benchmark_import_receipt,
            leakage_receipt_path=args.leakage_receipt,
            selected_splits=tuple(args.split),
            evaluator_contract_receipt_path=args.evaluator_contract_receipt,
            base_cohort_output=args.base_cohort_output,
            output_path=args.output,
        )
    elif args.command == "cohort-retrieval":
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
    "build_governed_production_cohort",
    "build_production_evaluation_evidence",
    "build_retrieval_production_cohort",
    "main",
]
