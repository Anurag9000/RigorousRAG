"""Active-learning acquisition adapter for table/chart authority decisions."""

from __future__ import annotations

from evaluation.active_learning import AcquisitionSignals, ActiveLearningCandidate
from scientific.structured_data_quality import StructuredDataAuthorityDecision


def structured_authority_candidate(
    decision: StructuredDataAuthorityDecision,
    *,
    owner_id: str,
    item_sha256: str,
    group_id: str,
    expected_impact: float = 1.0,
    novelty: float = 0.0,
    estimated_label_cost: float = 1.0,
    task_id: str = "structured_evidence_authority",
) -> ActiveLearningCandidate:
    if not isinstance(decision, StructuredDataAuthorityDecision):
        raise ValueError("decision must be StructuredDataAuthorityDecision")
    if decision.action == "authoritative":
        raise ValueError("authoritative structured evidence does not require review acquisition")
    uncertainty = 1.0 if decision.action == "blocked" else 0.75
    confidence_penalty = 1.0 if decision.confidence_floor is None else 1.0 - decision.confidence_floor
    interval_penalty = decision.relative_interval_width / (1.0 + decision.relative_interval_width)
    return ActiveLearningCandidate(
        owner_id=owner_id,
        task_id=task_id,
        item_sha256=item_sha256,
        evidence_sha256s=(decision.evidence_sha256, decision.decision_sha256),
        group_id=group_id,
        signals=AcquisitionSignals(
            uncertainty=max(uncertainty, confidence_penalty, interval_penalty),
            disagreement=0.0,
            drift=0.0,
            novelty=novelty,
            expected_impact=expected_impact,
            abstained=True,
        ),
        estimated_label_cost=estimated_label_cost,
        source_policy_sha256=decision.policy_sha256,
        source_model_sha256=None,
    )


__all__ = ["structured_authority_candidate"]
