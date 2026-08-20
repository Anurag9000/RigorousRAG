"""Strict production policy coverage over authoritative advanced promotion evidence.

Production requires exact policy recomputation and machine-checkable evaluator semantics:
one result row per authorized sample, arithmetic-mean cohort aggregation, fully representable
metrics, and one correctly directed threshold for every directional evaluator metric.
Descriptive metrics cannot participate in qualification.
"""
from __future__ import annotations

from pathlib import Path

from evaluation.authoritative_advanced_evaluation_verification import (
    verify_authoritative_advanced_evaluation_evidence,
)
from evaluation.evaluator_bound_evaluation_cohort import (
    verify_evaluator_bound_evaluation_cohort,
)
from evaluation.strict_production_evaluator_contract import (
    assert_strict_production_evaluator_contract,
)
from training.advanced_rag_artifacts import AdvancedArtifactManifest, MetricQualificationPolicy
from training.authoritative_advanced_promotion import (
    AuthoritativeAdvancedPromotionEvidence,
    assert_authoritative_advanced_promotion,
    build_authoritative_advanced_promotion_evidence,
)


def _assert_policy_coverage(
    policy: MetricQualificationPolicy,
    *,
    evaluation_evidence_path: str | Path,
) -> None:
    if not isinstance(policy, MetricQualificationPolicy):
        raise ValueError("policy must be MetricQualificationPolicy")
    _, evidence = verify_authoritative_advanced_evaluation_evidence(evaluation_evidence_path)
    _, _, evaluator = verify_evaluator_bound_evaluation_cohort(
        evidence.evaluator_bound_cohort_path
    )
    assert_strict_production_evaluator_contract(evaluator)

    by_name = {metric.name: metric for metric in evaluator.metrics}
    declared = set(by_name)
    minimum = set(policy.minimum)
    maximum = set(policy.maximum)
    unknown = (minimum | maximum) - declared
    if unknown:
        raise ValueError(
            "production promotion policy references undeclared evaluator metrics: "
            + ",".join(sorted(unknown))
        )

    maximize = {name for name, metric in by_name.items() if metric.direction == "maximize"}
    minimize = {name for name, metric in by_name.items() if metric.direction == "minimize"}
    descriptive = declared - maximize - minimize
    if not maximize and not minimize:
        raise ValueError("production promotion requires at least one directional evaluator metric")

    missing_minimum = maximize - minimum
    missing_maximum = minimize - maximum
    wrong_minimum = minimum & (minimize | descriptive)
    wrong_maximum = maximum & (maximize | descriptive)
    if missing_minimum or missing_maximum or wrong_minimum or wrong_maximum:
        raise ValueError(
            "production promotion thresholds differ from evaluator metric directions; "
            f"missing_minimum={sorted(missing_minimum)} "
            f"missing_maximum={sorted(missing_maximum)} "
            f"wrong_minimum={sorted(wrong_minimum)} "
            f"wrong_maximum={sorted(wrong_maximum)}"
        )


def build_strict_authoritative_advanced_promotion_evidence(
    manifest: AdvancedArtifactManifest,
    *,
    authoritative_evaluation_evidence_path: str | Path,
    policy: MetricQualificationPolicy,
) -> AuthoritativeAdvancedPromotionEvidence:
    _assert_policy_coverage(
        policy,
        evaluation_evidence_path=authoritative_evaluation_evidence_path,
    )
    evidence = build_authoritative_advanced_promotion_evidence(
        manifest,
        authoritative_evaluation_evidence_path=authoritative_evaluation_evidence_path,
        policy=policy,
    )
    assert_strict_authoritative_advanced_promotion(manifest, evidence)
    return evidence


def assert_strict_authoritative_advanced_promotion(
    manifest: AdvancedArtifactManifest,
    evidence: AuthoritativeAdvancedPromotionEvidence,
) -> None:
    assert_authoritative_advanced_promotion(manifest, evidence)
    _assert_policy_coverage(
        evidence.policy,
        evaluation_evidence_path=evidence.authoritative_evaluation_evidence_path,
    )


__all__ = [
    "assert_strict_authoritative_advanced_promotion",
    "build_strict_authoritative_advanced_promotion_evidence",
]
