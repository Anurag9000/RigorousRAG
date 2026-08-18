"""Standard privacy-safe adapters from RigorousRAG uncertainty signals to active learning."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Sequence

from evaluation.active_learning import AcquisitionSignals, ActiveLearningCandidate
from evaluation.cross_profile_calibration_drift import CalibrationDriftDecision
from evaluation.semantic_support import ClaimEvidenceScore, SemanticLabel, SemanticProbabilities
from evaluation.structured_data_support import StructuredSupportScore


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _probability_entropy(probabilities: SemanticProbabilities) -> float:
    values = (probabilities.entailment, probabilities.neutral, probabilities.contradiction)
    entropy = -sum(value * math.log(value) for value in values if value > 0.0)
    return min(1.0, max(0.0, entropy / math.log(3.0)))


def _mean_probabilities(rows: Sequence[SemanticProbabilities]) -> tuple[float, float, float]:
    count = len(rows)
    return (
        sum(row.entailment for row in rows) / count,
        sum(row.neutral for row in rows) / count,
        sum(row.contradiction for row in rows) / count,
    )


def _kl(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(value * math.log(value / max(target, 1e-12)) for value, target in zip(left, right) if value > 0.0)


def _jensen_shannon_disagreement(rows: Sequence[SemanticProbabilities]) -> float:
    mean = _mean_probabilities(rows)
    divergences = []
    for row in rows:
        values = (row.entailment, row.neutral, row.contradiction)
        midpoint = tuple((left + right) / 2.0 for left, right in zip(values, mean))
        divergences.append(0.5 * _kl(values, midpoint) + 0.5 * _kl(mean, midpoint))
    # Jensen-Shannon divergence between two distributions is bounded by ln(2).
    return min(1.0, max(0.0, sum(divergences) / len(divergences) / math.log(2.0)))


def _claim_evidence_identity(score: ClaimEvidenceScore) -> tuple[str, str]:
    evidence = score.evidence
    digest = _digest(
        {
            "schema": "rigorousrag-semantic-evidence-anchor/v1",
            "evidence_id": evidence.evidence_id,
            "document_id": evidence.document_id,
            "generation_id": evidence.generation_id,
            "source_id": evidence.source_id,
            "page_number": evidence.page_number,
            "region_id": evidence.region_id,
            "artifact_path": evidence.artifact_path,
            "citation_id": evidence.citation_id,
        }
    )
    return score.claim.claim_sha256, digest


def semantic_ensemble_candidate(
    scores: Sequence[ClaimEvidenceScore],
    *,
    owner_id: str,
    task_id: str = "semantic_support",
    group_id: str,
    novelty: float = 0.0,
    expected_impact: float = 0.0,
    estimated_label_cost: float = 1.0,
    source_policy_sha256: str | None = None,
) -> ActiveLearningCandidate:
    rows = tuple(scores)
    if not rows or any(not isinstance(row, ClaimEvidenceScore) for row in rows):
        raise ValueError("scores must be a non-empty ClaimEvidenceScore collection")
    identities = {_claim_evidence_identity(row) for row in rows}
    if len(identities) != 1:
        raise ValueError("semantic ensemble rows must refer to the same claim/evidence identity")
    if len({row.model.model_sha256 for row in rows}) != len(rows):
        raise ValueError("semantic ensemble must not duplicate the same model identity")
    item_sha256, evidence_sha256 = next(iter(identities))
    probabilities = tuple(row.probabilities for row in rows)
    mean_values = _mean_probabilities(probabilities)
    mean = SemanticProbabilities(*mean_values)
    model_set_sha256 = _digest({"schema": "rigorousrag-semantic-ensemble-model-set/v1", "models": sorted(row.model.model_sha256 for row in rows)})
    return ActiveLearningCandidate(
        owner_id=owner_id,
        task_id=task_id,
        item_sha256=item_sha256,
        evidence_sha256s=(evidence_sha256,),
        group_id=group_id,
        signals=AcquisitionSignals(
            uncertainty=_probability_entropy(mean),
            disagreement=0.0 if len(rows) == 1 else _jensen_shannon_disagreement(probabilities),
            novelty=novelty,
            expected_impact=expected_impact,
            abstained=mean.predicted_label is SemanticLabel.NEUTRAL,
        ),
        estimated_label_cost=estimated_label_cost,
        source_policy_sha256=source_policy_sha256,
        source_model_sha256=model_set_sha256,
    )


_STRUCTURED_ABSTENTION_REASONS = frozenset(
    {
        "unit_incomparable",
        "categorical_x_order_not_assumed",
        "duplicate_x_coordinates",
        "insufficient_points_for_trend",
        "uncertainty_overlaps_claim_boundary",
        "uncertainty_crosses_equality_tolerance",
        "uncertainty_crosses_predicate_boundary",
        "series_has_mixed_trend",
    }
)


def structured_support_candidate(
    score: StructuredSupportScore,
    *,
    owner_id: str,
    task_id: str = "structured_support",
    group_id: str,
    novelty: float = 0.0,
    expected_impact: float = 0.0,
    estimated_label_cost: float = 1.0,
    source_policy_sha256: str | None = None,
) -> ActiveLearningCandidate:
    if not isinstance(score, StructuredSupportScore):
        raise ValueError("score must be StructuredSupportScore")
    neutral = score.label is SemanticLabel.NEUTRAL
    return ActiveLearningCandidate(
        owner_id=owner_id,
        task_id=task_id,
        item_sha256=score.claim_sha256,
        evidence_sha256s=(score.evidence_sha256,),
        group_id=group_id,
        signals=AcquisitionSignals(
            uncertainty=_probability_entropy(score.probabilities),
            disagreement=0.0,
            novelty=novelty,
            expected_impact=expected_impact,
            abstained=neutral or score.reason_code in _STRUCTURED_ABSTENTION_REASONS,
        ),
        estimated_label_cost=estimated_label_cost,
        source_policy_sha256=source_policy_sha256,
        source_model_sha256=score.model.model_sha256,
    )


def _bounded_drift(value: float | None) -> float:
    if value is None:
        return 0.0
    selected = max(0.0, float(value))
    return selected / (1.0 + selected)


def calibration_drift_candidate(
    decision: CalibrationDriftDecision,
    *,
    owner_id: str,
    task_id: str = "calibration_requalification",
    expected_impact: float = 1.0,
    estimated_label_cost: float = 1.0,
) -> ActiveLearningCandidate:
    if not isinstance(decision, CalibrationDriftDecision):
        raise ValueError("decision must be CalibrationDriftDecision")
    drift = max(
        _bounded_drift(decision.population_stability_index),
        _bounded_drift(decision.jensen_shannon_divergence),
        0.0 if decision.brier is None else min(1.0, decision.brier),
        0.0 if decision.ece is None else min(1.0, decision.ece),
        1.0 if "qualification_expired" in decision.reason_codes else 0.0,
    )
    evidence = (decision.reference_sha256, decision.decision_sha256)
    return ActiveLearningCandidate(
        owner_id=owner_id,
        task_id=task_id,
        item_sha256=decision.artifact_sha256,
        evidence_sha256s=evidence,
        group_id=decision.profile_id,
        signals=AcquisitionSignals(
            uncertainty=1.0 if decision.action == "requalify_rrf_only" else 0.0,
            disagreement=0.0,
            drift=drift,
            novelty=0.0,
            expected_impact=expected_impact,
            abstained=decision.action == "requalify_rrf_only",
        ),
        estimated_label_cost=estimated_label_cost,
        source_policy_sha256=decision.policy_sha256,
        source_model_sha256=decision.artifact_sha256,
    )


__all__ = [
    "calibration_drift_candidate",
    "semantic_ensemble_candidate",
    "structured_support_candidate",
]
