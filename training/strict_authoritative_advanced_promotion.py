"""Strict production policy coverage over authoritative advanced promotion evidence.

The primitive promotion object correctly self-verifies policy hashes and recomputes its
qualification decision. Production additionally requires the policy to cover the evaluator's
directional metric contract: maximize -> minimum threshold, minimize -> maximum threshold,
with no policy keys outside the declared evaluator metric schema.
"""
from __future__ import annotations

from pathlib import Path

from evaluation.authoritative_advanced_evaluation_verification import (
    verify_authoritative_advanced_evaluation_evidence,
)
from evaluation.evaluator_bound_evaluation_cohort import (
    verify_evaluator_bound_evaluation_cohort,
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
    _, evidence = verify_authoritative_advanced_evaluation_evidence(
        evaluation_evidence_path
    )
    _, _, evaluator = verify_evaluator_bound_evaluation_cohort(
        evidence.evaluator_bound_cohort_path
    )
    declared = {metric.name for metric in evaluator.metrics}
    minimum = set(policy.minimum)
    maximum = set(policy.maximum)
    unknown = (minimum | maximum) - declared
    if unknown:
        raise ValueError(
            "production promotion policy references undeclared evaluator metrics: "
            + ",".join(sorted(unknown))
        )
    maximize = {metric.name for metric in evaluator.metrics if metric.direction == "maximize"}
    minimize = {metric.name for metric in evaluator.metrics if metric.direction == "minimize"}
    if not maximize and not minimize:
        raise ValueError(
            "production promotion requires at least one directional evaluator metric"
        )
    missing_minimum = maximize - minimum
    missing_maximum = minimize - maximum
    if missing_minimum or missing_maximum:
        raise ValueError(
            "production promotion policy does not cover all directional evaluator metrics; "
            f"missing_minimum={sorted(missing_minimum)} "
            f"missing_maximum={sorted(missing_maximum)}"
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
