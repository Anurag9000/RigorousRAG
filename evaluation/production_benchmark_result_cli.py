"""Production benchmark-result CLI consuming one persisted evaluator-bound cohort.

Production materialization fails before writing any result artifact unless the evaluator receipt
uses the exact semantics implemented by the streaming evidence path: one result row per
authorized sample, arithmetic-mean aggregation over the exact cohort, and row+aggregate
representation for every metric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from evaluation.evaluator_bound_evaluation_cohort import (
    verify_evaluator_bound_evaluation_cohort,
)
from evaluation.evaluator_bound_streaming_benchmark_result_cli import (
    materialize_evaluator_bound_result,
)
from evaluation.strict_production_evaluator_contract import (
    assert_strict_production_evaluator_contract,
)


def materialize_production_result(
    *,
    evaluator_bound_cohort_path: str | Path,
    result_input_path: str | Path,
    result_input_sha256: str,
    seed: int,
    repeat_index: int,
    output_dir: str | Path,
) -> Mapping[str, object]:
    binding, _, evaluator = verify_evaluator_bound_evaluation_cohort(
        evaluator_bound_cohort_path
    )
    assert_strict_production_evaluator_contract(evaluator)
    result = dict(
        materialize_evaluator_bound_result(
            cohort_contract_path=binding.cohort_contract_path,
            evaluator_contract_receipt_path=binding.evaluator_contract_receipt_path,
            result_input_path=result_input_path,
            result_input_sha256=result_input_sha256,
            seed=seed,
            repeat_index=repeat_index,
            output_dir=output_dir,
        )
    )
    result["evaluator_bound_cohort_sha256"] = binding.contract_sha256
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rigorousrag-benchmark-result",
        description="Stream a local benchmark result into strict evaluator/cohort-bound v2 result evidence",
    )
    parser.add_argument("--evaluation-cohort", required=True)
    parser.add_argument("--result-input", required=True)
    parser.add_argument("--result-input-sha256", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--repeat-index", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = materialize_production_result(
        evaluator_bound_cohort_path=args.evaluation_cohort,
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


__all__ = ["main", "materialize_production_result"]
