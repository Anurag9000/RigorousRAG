from __future__ import annotations

import pytest

from tools.adaptive_route_experiments import (
    RouteExecution,
    RouteExperimentCase,
    run_route_benchmark,
    select_route,
)


def evidence(prefix: str, count: int, score: float, *, source_kind: str):
    return [
        {
            "doc_id": f"{prefix}-doc-{index}",
            "source_id": f"{prefix}-source-{index}",
            "score": score,
            "page_number": index + 1,
            "generation_sequence": 1,
            "source_kind": source_kind,
        }
        for index in range(count)
    ]


def test_route_selection_covers_uploaded_public_temporal_and_scholarly_cases():
    assert select_route(
        RouteExperimentCase("u", 'Find "ABC-123"', scope="uploaded")
    ) == "corpus-sparse"
    assert select_route(
        RouteExperimentCase("t", "latest policy change", scope="public")
    ) == "web"
    assert select_route(
        RouteExperimentCase(
            "s", "cite papers about the method", scope="public", domain="scholarly"
        )
    ) == "scholarly"
    assert select_route(
        RouteExperimentCase("m", "compare two methods", scope="mixed")
    ) == "corpus-hybrid"


def test_route_benchmark_reports_router_oracle_regret_and_route_aggregates():
    cases = [
        RouteExperimentCase("temporal", "latest regulation", scope="public"),
        RouteExperimentCase(
            "paper", "cite papers about retrieval", scope="public", domain="scholarly"
        ),
    ]
    adapters = {
        "dense": lambda case: RouteExecution(
            evidence("dense", 1, 0.2, source_kind="dense"), 1, 3
        ),
        "corpus-sparse": lambda case: RouteExecution(
            evidence("sparse", 1, 0.3, source_kind="sparse"), 1, 2
        ),
        "corpus-hybrid": lambda case: RouteExecution(
            evidence("hybrid", 2, 0.5, source_kind="hybrid"), 2, 5
        ),
        "web": lambda case: RouteExecution(
            evidence(
                "web",
                5 if case.case_id == "temporal" else 1,
                0.95 if case.case_id == "temporal" else 0.2,
                source_kind="web",
            ),
            10,
            100,
        ),
        "scholarly": lambda case: RouteExecution(
            evidence(
                "scholarly",
                5 if case.case_id == "paper" else 1,
                0.95 if case.case_id == "paper" else 0.2,
                source_kind="scholarly",
            ),
            8,
            80,
        ),
    }
    report = run_route_benchmark(cases, adapters=adapters)
    assert report.case_count == 2
    assert report.selected_success_rate == 1.0
    assert report.oracle_success_rate == 1.0
    assert report.route_selection_accuracy == 1.0
    assert report.mean_regret == 0.0
    assert {row.route for row in report.per_route} == set(adapters)
    assert all(row.observations == 2 for row in report.per_route)


def test_route_benchmark_uses_relevance_ids_and_contains_adapter_failures():
    case = RouteExperimentCase(
        "known",
        "find the known study",
        relevant_ids=frozenset({"known-source"}),
    )

    def failing(_case):
        raise RuntimeError("private failure details")

    adapters = {
        "corpus-hybrid": lambda _case: RouteExecution(
            [{"source_id": "known-source", "score": 0.4}], cost_units=2
        ),
        "web": failing,
    }
    report = run_route_benchmark([case], adapters=adapters)
    hybrid = next(
        row for row in report.cases[0].observations if row.route == "corpus-hybrid"
    )
    web = next(row for row in report.cases[0].observations if row.route == "web")
    assert hybrid.relevant_hits == 1
    assert hybrid.success is True
    assert web.error_type == "RuntimeError"
    assert web.success is False
    assert "private failure" not in repr(report)


def test_route_benchmark_rejects_duplicate_cases_and_hostile_collections():
    case = RouteExperimentCase("same", "question")
    with pytest.raises(ValueError, match="unique"):
        run_route_benchmark([case, case], adapters={"dense": lambda _: []})
    report = run_route_benchmark([case], adapters={"dense": lambda _: "not evidence"})
    assert report.cases[0].observations[0].error_type == "RuntimeError"
    assert report.cases[0].observations[0].success is False


def test_route_selection_falls_back_when_preferred_route_is_unavailable():
    case = RouteExperimentCase("public", "latest update", scope="public")
    assert select_route(
        case, available_routes=("dense", "corpus-hybrid")
    ) == "corpus-hybrid"
