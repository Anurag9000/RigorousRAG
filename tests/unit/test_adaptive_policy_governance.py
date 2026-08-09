from __future__ import annotations

from dataclasses import replace

import pytest

from tools.adaptive_policy_governance import (
    AdaptivePolicyGate,
    compare_adaptive_policies,
    evaluate_adaptive_policy_promotion,
    evaluate_adaptive_policy_rollback,
    jensen_shannon_divergence,
    route_distribution,
)
from tools.adaptive_route_experiments import (
    RouteExecution,
    RouteExperimentCase,
    run_route_benchmark,
)


def evidence(score: float):
    return [
        {
            "doc_id": f"doc-{index}",
            "source_id": f"source-{index}",
            "score": score,
            "page_number": index + 1,
            "generation_sequence": 1,
        }
        for index in range(5)
    ]


def base_report():
    cases = [
        RouteExperimentCase("a", "general question", scope="mixed"),
        RouteExperimentCase("b", "latest update", scope="public"),
        RouteExperimentCase(
            "c",
            "cite paper about retrieval",
            scope="public",
            domain="scholarly",
        ),
    ]
    adapters = {
        "corpus-hybrid": lambda _case: RouteExecution(evidence(0.9), 5, 30),
        "web": lambda _case: RouteExecution(evidence(0.9), 6, 40),
        "scholarly": lambda _case: RouteExecution(evidence(0.9), 7, 50),
    }
    return run_route_benchmark(cases, adapters=adapters)


def test_jsd_is_symmetric_bounded_and_zero_for_identical_distributions():
    left = {"dense": 0.5, "web": 0.5}
    right = {"dense": 0.1, "web": 0.9}
    assert jensen_shannon_divergence(left, left) == pytest.approx(0.0)
    assert jensen_shannon_divergence(left, right) == pytest.approx(
        jensen_shannon_divergence(right, left)
    )
    assert 0.0 < jensen_shannon_divergence(left, right) <= 1.0


def test_paired_policy_improvement_is_eligible_and_digest_is_deterministic():
    baseline = base_report()
    candidate = replace(
        baseline,
        selected_success_rate=min(1.0, baseline.selected_success_rate + 0.01),
        route_selection_accuracy=min(1.0, baseline.route_selection_accuracy + 0.01),
        mean_regret=max(0.0, baseline.mean_regret - 0.01),
        mean_selected_cost_units=baseline.mean_selected_cost_units - 1.0,
        mean_selected_latency_ms=max(0.0, baseline.mean_selected_latency_ms - 1.0),
    )
    comparison = compare_adaptive_policies(
        baseline_policy_id="baseline-v1",
        candidate_policy_id="candidate-v2",
        baseline=baseline,
        candidate=candidate,
    )
    decision = evaluate_adaptive_policy_promotion(comparison)
    assert decision.decision == "eligible"
    assert decision.reasons == ()
    assert decision.decision_digest == evaluate_adaptive_policy_promotion(
        comparison
    ).decision_digest
    assert len(decision.decision_digest) == 64


def test_regression_and_route_shift_hold_candidate_and_recommend_rollback():
    baseline = base_report()
    shifted_cases = tuple(
        replace(case, selected_route="web") for case in baseline.cases
    )
    candidate = replace(
        baseline,
        selected_success_rate=max(0.0, baseline.selected_success_rate - 0.2),
        route_selection_accuracy=max(0.0, baseline.route_selection_accuracy - 0.2),
        mean_regret=baseline.mean_regret + 0.2,
        mean_selected_cost_units=baseline.mean_selected_cost_units + 100.0,
        mean_selected_latency_ms=baseline.mean_selected_latency_ms + 500.0,
        cases=shifted_cases,
    )
    comparison = compare_adaptive_policies(
        baseline_policy_id="baseline-v1",
        candidate_policy_id="candidate-v2",
        baseline=baseline,
        candidate=candidate,
    )
    gate = AdaptivePolicyGate(max_route_shift_jsd=0.01)
    hold = evaluate_adaptive_policy_promotion(comparison, gate=gate)
    rollback = evaluate_adaptive_policy_rollback(comparison, gate=gate)
    assert hold.decision == "hold"
    assert rollback.decision == "rollback"
    assert "success_rate_regressed" in hold.reasons
    assert "regret_increased" in hold.reasons
    assert "cost_increased" in hold.reasons
    assert "latency_increased" in hold.reasons
    assert "route_distribution_shifted" in hold.reasons
    assert rollback.reasons == hold.reasons


def test_policy_comparison_requires_exact_paired_case_order():
    baseline = base_report()
    candidate = replace(
        baseline,
        cases=tuple(reversed(baseline.cases)),
    )
    with pytest.raises(ValueError, match="identical ordered"):
        compare_adaptive_policies(
            baseline_policy_id="baseline-v1",
            candidate_policy_id="candidate-v2",
            baseline=baseline,
            candidate=candidate,
        )


def test_route_distribution_contains_no_queries_or_evidence():
    report = base_report()
    distribution = route_distribution(report)
    assert set(distribution) == {
        "dense",
        "corpus-sparse",
        "corpus-hybrid",
        "web",
        "scholarly",
    }
    assert sum(distribution.values()) == pytest.approx(1.0)
    assert "general question" not in repr(distribution)
