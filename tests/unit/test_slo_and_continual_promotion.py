from __future__ import annotations

from tools.continual_promotion import (
    ContinualEvidence,
    ContinualPromotionPolicy,
    evaluate_continual_promotion,
)
from tools.feedback_promotion import (
    CandidateMetrics,
    PromotionPolicy,
    build_feedback_batch,
    evaluate_promotion,
)
from tools.feedback_store import ActiveLearningExample
from tools.service_slo import StageObservation
from tools.slo_alerts import BurnRatePolicy, evaluate_burn_rate


def _base_promotion():
    examples = [
        ActiveLearningExample(
            kind="answer_correct" if index < 18 else "answer_incorrect",
            subject_id=f"subject-{index}",
            weight=1.0,
            metadata={},
            query_sha256="a" * 64,
            evidence_sha256="b" * 64,
        )
        for index in range(20)
    ]
    batch = build_feedback_batch(owner_id="owner", examples=examples)
    return evaluate_promotion(
        batch=batch,
        baseline_version="v1",
        candidate_version="v2",
        baseline=CandidateMetrics(0.80, 100, 1.0),
        candidate=CandidateMetrics(0.83, 105, 1.02),
        policy=PromotionPolicy(
            min_examples=20,
            min_negative_weight_fraction=0.1,
            min_quality_gain=0.01,
        ),
    )


def test_multi_window_burn_rate_requires_both_windows_and_full_sample() -> None:
    observations = [
        StageObservation(str(index), "request", 10, index % 10 != 0) for index in range(100)
    ]
    report = evaluate_burn_rate(
        observations,
        BurnRatePolicy(
            availability_target=0.99,
            short_window_requests=20,
            long_window_requests=100,
            short_burn_threshold=5,
            long_burn_threshold=5,
        ),
    )
    assert report.short_burn_rate == 10.0
    assert report.long_burn_rate == 10.0
    assert report.alert

    partial = evaluate_burn_rate(observations[:10], BurnRatePolicy())
    assert not partial.alert


def test_continual_promotion_passes_when_all_safeguards_hold() -> None:
    decision = evaluate_continual_promotion(
        base=_base_promotion(),
        evidence=ContinualEvidence(
            drift_score=0.2,
            forgetting_delta=0.01,
            forward_transfer_delta=0.02,
            privacy_safe_replay=True,
            independent_rollback_ready=True,
            adapter_version="adapter-v2",
        ),
    )
    assert decision.eligible
    assert not decision.reason_codes


def test_continual_promotion_fails_closed_on_learning_and_safety_regressions() -> None:
    decision = evaluate_continual_promotion(
        base=_base_promotion(),
        evidence=ContinualEvidence(
            drift_score=1.5,
            forgetting_delta=0.05,
            forward_transfer_delta=-0.10,
            privacy_safe_replay=False,
            independent_rollback_ready=False,
            adapter_version="adapter-v2",
        ),
        policy=ContinualPromotionPolicy(
            max_drift_score=1.0,
            max_forgetting_delta=0.02,
            min_forward_transfer_delta=-0.02,
        ),
    )
    assert not decision.eligible
    assert set(decision.reason_codes) == {
        "drift_budget_exceeded",
        "forgetting_budget_exceeded",
        "forward_transfer_regression",
        "privacy_safe_replay_not_verified",
        "independent_rollback_not_ready",
    }
