"""Production-only semantic restrictions for evaluator contracts.

The current promotion-grade result artifact represents exactly one row per authorized sample
and recomputes every exposed metric as an arithmetic mean over that exact cohort. Flexible
research evaluator contracts may describe other semantics, but production evidence must not
claim unsupported medians/percentiles or per-sample-only metrics.
"""
from __future__ import annotations

from pathlib import Path

from evaluation.authoritative_evaluator_contract import (
    AuthoritativeEvaluatorContract,
    verify_authoritative_evaluator_contract,
)

_PRODUCTION_SAMPLE_SEMANTICS = "one_result_row_per_authorized_sample"
_PRODUCTION_AGGREGATION_SEMANTICS = "arithmetic_mean_over_exact_cohort"


def assert_strict_production_evaluator_contract(
    evaluator: AuthoritativeEvaluatorContract,
) -> None:
    if not isinstance(evaluator, AuthoritativeEvaluatorContract):
        raise ValueError("evaluator must be AuthoritativeEvaluatorContract")
    if evaluator.sample_semantics != _PRODUCTION_SAMPLE_SEMANTICS:
        raise ValueError(
            "production evaluator sample_semantics must be "
            f"{_PRODUCTION_SAMPLE_SEMANTICS!r}"
        )
    if evaluator.aggregation_semantics != _PRODUCTION_AGGREGATION_SEMANTICS:
        raise ValueError(
            "production evaluator aggregation_semantics must be "
            f"{_PRODUCTION_AGGREGATION_SEMANTICS!r}"
        )
    per_sample_only = sorted(
        metric.name for metric in evaluator.metrics if metric.scope == "per_sample"
    )
    if per_sample_only:
        raise ValueError(
            "production result artifacts do not support per-sample-only evaluator metrics; "
            f"metrics={per_sample_only[:100]}"
        )


def verify_strict_production_evaluator_contract(
    path: str | Path,
) -> AuthoritativeEvaluatorContract:
    evaluator = verify_authoritative_evaluator_contract(path)
    assert_strict_production_evaluator_contract(evaluator)
    return evaluator


__all__ = [
    "assert_strict_production_evaluator_contract",
    "verify_strict_production_evaluator_contract",
]
