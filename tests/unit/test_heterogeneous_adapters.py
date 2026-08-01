from __future__ import annotations

import pytest

from tools.heterogeneous_adapters import build_production_route_adapters
from tools.heterogeneous_budget import allocate_heterogeneous_budget
from tools.heterogeneous_rag_tool import (
    HETEROGENEOUS_RAG_TOOL_DEF,
    heterogeneous_result_payload,
    search_research_heterogeneous,
)
from tools.heterogeneous_route_types import HeterogeneousRouteRequest
from tools.multihop_retrieval import HopEvidence
from tools.query_decomposition import Subquestion, build_decomposition_plan


def request(route: str, *, question=None, dependencies=(), results=3):
    question = question or Subquestion("q1", "Find evidence.")
    plan = build_decomposition_plan(
        question.text, proposed_subquestions=[question]
    )
    budget = allocate_heterogeneous_budget(
        plan,
        available_routes=(route,),
        route_overrides={question.question_id: route},
        top_k=results,
        total_cost_limit=1_000,
        total_latency_limit_ms=100_000,
        total_monetary_limit_microunits=100_000,
    ).allocations[0]
    return HeterogeneousRouteRequest(question, tuple(dependencies), budget)


def hop(text: str):
    return HopEvidence(
        "q0:source", "q0", "source", "doc", 1, text, 0.9,
        {"source_id": "source", "doc_id": "doc", "text": text},
    )


def test_public_routes_never_receive_dependency_evidence_text():
    captured = {}

    def web(query, allowed_domains, *, limit):
        captured["web"] = (query, allowed_domains, limit)
        return []

    def academic(query, year_from, year_to, *, limit):
        captured["academic"] = (query, year_from, year_to, limit)
        return []

    adapters = build_production_route_adapters(
        owner_id="alice",
        allowed_domains=["example.org"],
        web_search_fn=web,
        academic_search_fn=academic,
        uploaded_search=lambda *args, **kwargs: [],
        owner_normalizer=lambda value: value,
    )
    secret = "PRIVATE_UNIQUE_DEPENDENCY_TOKEN"
    web_request = request(
        "web",
        question=Subquestion(
            "q1", "Find the latest public result.",
            temporal_constraints=("2025",), relation="temporal",
        ),
        dependencies=(hop(secret),),
    )
    scholarly_request = request(
        "scholarly",
        question=Subquestion(
            "q2", "Find papers about the method.",
            temporal_constraints=("2020", "2024"),
        ),
        dependencies=(hop(secret),),
    )
    adapters["web"](web_request)
    adapters["scholarly"](scholarly_request)
    assert secret not in captured["web"][0]
    assert secret not in captured["academic"][0]
    assert captured["web"][1] == ["example.org"]
    assert captured["academic"][1:3] == (2020, 2024)


def test_uploaded_routes_use_exact_mode_and_bounded_dependency_terms():
    calls = []

    def uploaded(query, **kwargs):
        calls.append((query, kwargs))
        return []

    adapters = build_production_route_adapters(
        owner_id="alice",
        doc_id="doc-1",
        uploaded_search=uploaded,
        web_search_fn=lambda *args, **kwargs: [],
        academic_search_fn=lambda *args, **kwargs: [],
        owner_normalizer=lambda value: value,
    )
    dependency = hop("SpecificMarker SpecificMarker other supporting words")
    for route in ("dense", "corpus-sparse", "corpus-hybrid"):
        adapters[route](
            request(route, dependencies=(dependency,), results=3)
        )
    assert {kwargs["retrieval_mode"] for _, kwargs in calls} == {
        "dense", "corpus-sparse", "corpus-hybrid"
    }
    assert all(kwargs["owner_id"] == "alice" for _, kwargs in calls)
    assert all(kwargs["doc_id"] == "doc-1" for _, kwargs in calls)
    assert all(kwargs["n_results"] == 3 for _, kwargs in calls)
    assert all("specificmarker" in query for query, _ in calls)
    assert next(
        kwargs for _, kwargs in calls
        if kwargs["retrieval_mode"] == "corpus-hybrid"
    )["reranker"] == "heuristic"


def test_public_provider_result_ceiling_is_enforced_before_provider_call():
    calls = []
    adapters = build_production_route_adapters(
        owner_id="alice",
        web_search_fn=lambda *args, **kwargs: calls.append("web"),
        academic_search_fn=lambda *args, **kwargs: calls.append("academic"),
        uploaded_search=lambda *args, **kwargs: [],
        owner_normalizer=lambda value: value,
    )
    with pytest.raises(ValueError, match="at most 10"):
        adapters["web"](request("web", results=11))
    with pytest.raises(ValueError, match="at most 10"):
        adapters["scholarly"](request("scholarly", results=11))
    assert calls == []


def test_heterogeneous_public_tool_routes_and_sanitizes_payload():
    calls = []

    def adapter(route):
        def run(route_request):
            calls.append((route, route_request.question.question_id))
            return [{
                "source_id": f"{route}-source",
                "doc_id": f"{route}-doc",
                "text": "public evidence",
                "score": 0.9,
                "metadata": {
                    "file_path": "/private/source",
                    "token": "secret",
                    "public": "retained",
                },
            }]
        return run

    result = search_research_heterogeneous(
        "Find the latest policy update.",
        scope="public",
        top_k=2,
        total_cost_limit=100,
        total_latency_limit_ms=10_000,
        total_monetary_limit_microunits=2_000,
        _adapters={
            "web": adapter("web"),
            "scholarly": adapter("scholarly"),
        },
    )
    assert result.routes_by_hop == (("q1", "web"),)
    assert calls == [("web", "q1")]
    payload = heterogeneous_result_payload(result)
    rendered = repr(payload)
    assert "/private/source" not in rendered
    assert "secret" not in rendered
    assert "retained" in rendered
    assert payload["routes_by_hop"] == [["q1", "web"]]


def test_tool_schema_exposes_all_global_resource_and_deadline_controls():
    properties = HETEROGENEOUS_RAG_TOOL_DEF["function"]["parameters"]["properties"]
    assert properties["top_k"]["maximum"] == 10
    assert properties["total_cost_limit"]["maximum"] == 1_000_000
    assert properties["total_latency_limit_ms"]["maximum"] == 86_400_000
    assert properties["total_monetary_limit_microunits"]["maximum"] == 1_000_000_000
    assert properties["global_timeout_seconds"]["maximum"] == 3_600


def test_adapter_configuration_is_bounded_and_canonical():
    with pytest.raises(ValueError, match="allowed_domains"):
        build_production_route_adapters(
            owner_id="alice",
            allowed_domains="example.org",
            uploaded_search=lambda *args, **kwargs: [],
            web_search_fn=lambda *args, **kwargs: [],
            academic_search_fn=lambda *args, **kwargs: [],
            owner_normalizer=lambda value: value,
        )
    with pytest.raises(ValueError, match="diversity_lambda"):
        build_production_route_adapters(
            owner_id="alice",
            diversity_lambda=True,
            uploaded_search=lambda *args, **kwargs: [],
            web_search_fn=lambda *args, **kwargs: [],
            academic_search_fn=lambda *args, **kwargs: [],
            owner_normalizer=lambda value: value,
        )


def test_explicit_empty_adapter_mapping_fails_instead_of_enabling_providers():
    with pytest.raises(ValueError, match="adapters"):
        search_research_heterogeneous(
            "Question",
            _adapters={},
            total_monetary_limit_microunits=0,
        )
