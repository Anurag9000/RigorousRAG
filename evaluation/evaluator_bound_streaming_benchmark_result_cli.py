"""Evaluator-bound wrapper around the streaming promotion-grade result materializer."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Mapping, Sequence

from evaluation.authoritative_evaluation_cohort import verify_authoritative_evaluation_cohort
from evaluation.authoritative_evaluator_contract import verify_authoritative_evaluator_contract
from evaluation.authoritative_streaming_benchmark_result_cli import materialize_streaming_result
from evaluation.strict_authoritative_benchmark_result_verification import (
    verify_strict_authoritative_benchmark_result_receipt,
)
from training.advanced_path_authority import safe_advanced_path


def materialize_evaluator_bound_result(
    *,
    cohort_contract_path: str | Path,
    evaluator_contract_receipt_path: str | Path,
    result_input_path: str | Path,
    result_input_sha256: str,
    seed: int,
    repeat_index: int,
    output_dir: str | Path,
) -> Mapping[str, object]:
    cohort = verify_authoritative_evaluation_cohort(cohort_contract_path)
    evaluator = verify_authoritative_evaluator_contract(evaluator_contract_receipt_path)
    if evaluator.contract_sha256 != cohort.base_evaluator_contract_sha256:
        raise ValueError("evaluator contract receipt differs from evaluation cohort")
    output = safe_advanced_path(
        output_dir,
        label="authoritative benchmark result output",
        must_exist=False,
    )
    result = materialize_streaming_result(
        cohort_contract_path=cohort_contract_path,
        result_input_path=result_input_path,
        result_input_sha256=result_input_sha256,
        seed=seed,
        repeat_index=repeat_index,
        output_dir=output,
    )
    try:
        run, _ = verify_strict_authoritative_benchmark_result_receipt(
            output / "result_receipt.json"
        )
        declared = {metric.name for metric in evaluator.metrics}
        observed = set(run.metrics)
        if observed != declared:
            missing = sorted(declared - observed)
            extra = sorted(observed - declared)
            raise ValueError(
                "result aggregate metric schema differs from evaluator contract; "
                f"missing={missing[:50]} extra={extra[:50]}"
            )
        return {
            **dict(result),
            "evaluator_id": evaluator.evaluator_id,
            "base_evaluator_contract_sha256": evaluator.contract_sha256,
            "metric_names": sorted(observed),
        }
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rigorousrag-benchmark-result",
        description="Stream a local benchmark result into cohort/evaluator-bound v2 evidence",
    )
    parser.add_argument("--cohort-contract", required=True)
    parser.add_argument("--evaluator-contract-receipt", required=True)
    parser.add_argument("--result-input", required=True)
    parser.add_argument("--result-input-sha256", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--repeat-index", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = materialize_evaluator_bound_result(
        cohort_contract_path=args.cohort_contract,
        evaluator_contract_receipt_path=args.evaluator_contract_receipt,
        result_input_path=args.result_input,
        result_input_sha256=args.result_input_sha256,
        seed=args.seed,
        repeat_index=args.repeat_index,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "materialize_evaluator_bound_result"]
