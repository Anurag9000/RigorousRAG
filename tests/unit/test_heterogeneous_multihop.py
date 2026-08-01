from __future__ import annotations

import time
from fractions import Fraction

import pytest

from tools.heterogeneous_multihop import (
    DEFAULT_ROUTE_PROFILES,
    HeterogeneousHopBudget,
    HeterogeneousMultiHopBudget,
    RouteCostProfile,
    allocate_heterogeneous_budget,
    run_heterogeneous_multihop,
    select_subquestion_route,
)
from tools.query_decomposition import Subquestion, build_decomposition_plan


def comparison_plan():
    return build_decomposition_plan(
        "Compare system A and system B using public and uploaded evidence.",
        proposed_subquestions=[
            Subquestion("a", "Find uploaded evidence for system A."),
            Subquestion("b", "Find the latest public evidence for system B."),
            Subquestion(
                "compare",
                "Compare system A and system B.",
                depends_on=("a", "b"),
                relation="compare",
            ),
        ],
    )


def evidence(source: str, doc: str, score: float = 0.9):
    return {
        "source_id": source,
        "doc_id": doc,
        "text": f"Evidence from {source}",
        "score": score,
        "page_number": 1,
    }


def test_route_selection_covers_uploaded_public_scholarly_and_mixed_cases():
    assert select_subquestion_route(
        Subquestion("q1", 'Find "ABC-123".'), scope="uploaded"
    ) == "corpus-sparse"
    assert select_subquestion_route(
        Subquestion("q2", "Find the latest policy update."), scope="public"
    ) == "web"
    assert select_subquestion_route(
        Subquestion("q3", "Cite papers about the method."),
        scope="public",
        domain="scholarly",
    ) == "scholarly"
    assert select_subquestion_route(
        Subquestion("q4", "Compare A and B.", relation="compare"),
        scope="mixed",
    ) == "corpus-hybrid"


def test_budget_is_deterministic_and_respects_all_three_global_ceilings():
    plan = comparison_plan()
    overrides = {"a": "dense", "b": "web", "compare": "scholarly"}
    first = allocate_heterogeneous_budget(
        plan,
        route_overrides=overrides,
        top_k=6,
        total_cost_limit=100,
        total_latency_limit_ms=10_000,
        total_monetary_limit_microunits=2_000,
    )
    second = allocate_heterogeneous_budget(
        plan,
        route_overrides=overrides,
        top_k=6,
        total_cost_limit=100,
        total_latency_limit_ms=10_000,
        total_monetary_limit_microunits=2_000,
    )
    assert first == second
    assert first.allocated_cost_units <= first.total_cost_limit
    assert first.allocated_latency_ms <= first.total_latency_limit_ms
    assert (
        first.allocated_monetary_microunits
        <= first.total_monetary_limit_microunits
    )
    assert first.by_id()["a"].route == "dense"
    assert first.by_id()["b"].route == "web"
    assert first.by_id()["compare"].route == "scholarly"
    assert all(item.max_results >= 1 for item in first.allocations)


def test_budget_fails_when_minimum_route_resources_do_not_fit():
    plan = build_decomposition_plan("Latest public update")
    with pytest.raises(ValueError, match="minimum route allocation"):
        allocate_heterogeneous_budget(
            plan,
            scope="public",
            available_routes=("web",),
            total_cost_limit=100,
            total_latency_limit_ms=10_000,
            total_monetary_limit_microunits=0,
        )


def test_custom_profiles_and_exact_numeric_types_are_supported():
    plan = build_decomposition_plan("Question")
    custom = {
        "dense": RouteCostProfile("dense", 1, 1, 1, 1),
    }
    budget = allocate_heterogeneous_budget(
        plan,
        available_routes=("dense",),
        profiles=custom,
        top_k=3,
        total_cost_limit=4,
        total_latency_limit_ms=4,
        total_monetary_limit_microunits=0,
    )
    assert budget.allocations[0].max_results == 3
    with pytest.raises(ValueError, match="top_k"):
        allocate_heterogeneous_budget(plan, top_k=True)
    with pytest.raises(ValueError, match="top_k"):
        allocate_heterogeneous_budget(plan, top_k=Fraction(3, 2))


def test_execution_routes_each_hop_and_propagates_dependency_evidence():
    plan = comparison_plan()
    calls: list[tuple[str, str, int, int]] = []

    def adapter(route):
        def run(request):
            calls.append(
                (
                    route,
                    request.question.question_id,
                    len(request.dependencies),
                    request.budget.max_results,
                )
            )
            return [
                evidence(
                    f"{route}-{request.question.question_id}-{index}",
                    "shared",
                )
                for index in range(request.budget.max_results + 1)
            ]

        return run

    result = run_heterogeneous_multihop(
        plan,
        adapters={
            "dense": adapter("dense"),
            "web": adapter("web"),
            "scholarly": adapter("scholarly"),
        },
        route_overrides={"a": "dense", "b": "web", "compare": "scholarly"},
        top_k=2,
        total_cost_limit=100,
        total_latency_limit_ms=10_000,
        total_monetary_limit_microunits=2_000,
        max_workers=2,
    )
    assert result.routes_by_hop == (
        ("a", "dense"),
        ("b", "web"),
        ("compare", "scholarly"),
    )
    assert result.retrieval.abstain is False
    assert result.retrieval.terminal_evidence_count == 2
    compare_call = next(call for call in calls if call[1] == "compare")
    assert compare_call[2] == 4
    assert all(
        trace.accepted_evidence <= result.budget.by_id()[trace.hop_id].max_results
        for trace in result.retrieval.traces
    )


def test_adapter_errors_are_generic_and_preserve_other_parallel_hops():
    plan = comparison_plan()

    def dense(_request):
        raise RuntimeError("private provider detail")

    def web(request):
        return [evidence("web", request.question.question_id)]

    def hybrid(request):
        return [evidence("hybrid", request.question.question_id)]

    result = run_heterogeneous_multihop(
        plan,
        adapters={"dense": dense, "web": web, "corpus-hybrid": hybrid},
        route_overrides={"a": "dense", "b": "web", "compare": "corpus-hybrid"},
        total_cost_limit=100,
        total_latency_limit_ms=10_000,
        total_monetary_limit_microunits=1_000,
    )
    a_trace = next(trace for trace in result.retrieval.traces if trace.hop_id == "a")
    compare_trace = next(
        trace for trace in result.retrieval.traces if trace.hop_id == "compare"
    )
    assert a_trace.status == "error"
    assert a_trace.error_type == "RuntimeError"
    assert compare_trace.status == "skipped_missing_dependency_evidence"
    assert "private provider detail" not in repr(result)


def test_global_deadline_returns_without_waiting_for_slow_adapter():
    plan = build_decomposition_plan("Slow public question")

    def slow(_request):
        time.sleep(0.25)
        return [evidence("late", "doc")]

    started = time.monotonic()
    result = run_heterogeneous_multihop(
        plan,
        adapters={"web": slow},
        route_overrides={"q1": "web"},
        total_cost_limit=100,
        total_latency_limit_ms=10_000,
        total_monetary_limit_microunits=1_000,
        hop_timeout_seconds=0.01,
        global_timeout_seconds=0.02,
    )
    elapsed = time.monotonic() - started
    assert elapsed < 0.15
    assert result.retrieval.traces[0].status == "timeout"
    assert result.retrieval.abstain is True


def test_invalid_adapter_results_are_contained_as_route_errors():
    plan = build_decomposition_plan("Question")
    result = run_heterogeneous_multihop(
        plan,
        adapters={"dense": lambda _request: "not evidence"},
        route_overrides={"q1": "dense"},
        total_cost_limit=100,
        total_latency_limit_ms=10_000,
        total_monetary_limit_microunits=0,
    )
    assert result.retrieval.traces[0].status == "error"
    assert result.retrieval.traces[0].error_type == "RuntimeError"


def test_supplied_budget_must_match_plan_adapters_and_profile_estimates():
    plan = build_decomposition_plan("Question")
    profile = DEFAULT_ROUTE_PROFILES["dense"]
    cost, latency, money = profile.estimate(1)
    valid_hop = HeterogeneousHopBudget(
        "q1", "dense", 1, cost, latency, money, 1.0
    )
    valid = HeterogeneousMultiHopBudget(
        100, 10_000, 0, cost, latency, money, (valid_hop,)
    )
    result = run_heterogeneous_multihop(
        plan,
        adapters={"dense": lambda _request: []},
        budget=valid,
    )
    assert result.budget == valid

    forged_hop = HeterogeneousHopBudget(
        "q1", "dense", 1, cost + 1, latency, money, 1.0
    )
    forged = HeterogeneousMultiHopBudget(
        100, 10_000, 0, cost + 1, latency, money, (forged_hop,)
    )
    with pytest.raises(ValueError, match="cost profile"):
        run_heterogeneous_multihop(
            plan, adapters={"dense": lambda _request: []}, budget=forged
        )

    with pytest.raises(ValueError, match="question IDs"):
        other = HeterogeneousHopBudget(
            "other", "dense", 1, cost, latency, money, 1.0
        )
        run_heterogeneous_multihop(
            plan,
            adapters={"dense": lambda _request: []},
            budget=HeterogeneousMultiHopBudget(
                100, 10_000, 0, cost, latency, money, (other,)
            ),
        )


def test_budget_and_profile_records_reject_boolean_totals_and_bad_question_ids():
    with pytest.raises(ValueError, match="question_id"):
        HeterogeneousHopBudget(
            "bad\x7f", "dense", 1, 1, 1, 0, 1.0
        )
    valid_hop = HeterogeneousHopBudget(
        "q1", "dense", 1, 3, 240, 0, 1.0
    )
    with pytest.raises(ValueError, match="allocated_cost_units"):
        HeterogeneousMultiHopBudget(
            100, 10_000, 0, True, 240, 0, (valid_hop,)
        )
    with pytest.raises(ValueError, match="allocations must be a tuple"):
        HeterogeneousMultiHopBudget(
            100, 10_000, 0, 3, 240, 0, [valid_hop]
        )


def test_infinite_adapter_results_are_bounded_to_one_extra_inspection():
    plan = build_decomposition_plan("Question")
    consumed = 0

    def values():
        nonlocal consumed
        while True:
            consumed += 1
            yield evidence(f"source-{consumed}", "doc")

    result = run_heterogeneous_multihop(
        plan,
        adapters={"dense": lambda _request: values()},
        route_overrides={"q1": "dense"},
        top_k=3,
        total_cost_limit=100,
        total_latency_limit_ms=10_000,
        total_monetary_limit_microunits=0,
    )
    maximum = result.budget.allocations[0].max_results
    assert consumed == maximum + 1
    assert result.retrieval.terminal_evidence_count == maximum


def test_hostile_adapter_mapping_and_unavailable_override_fail_before_execution():
    plan = build_decomposition_plan("Question")

    class HostileMapping(dict):
        def items(self):
            raise RuntimeError("boom")

    with pytest.raises(ValueError, match="safely iterable"):
        run_heterogeneous_multihop(plan, adapters=HostileMapping())
    with pytest.raises(ValueError, match="not available"):
        allocate_heterogeneous_budget(
            plan,
            available_routes=("dense",),
            route_overrides={"q1": "web"},
            total_monetary_limit_microunits=1_000,
        )
