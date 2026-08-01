from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest

import search_agent
from tools import adaptive_rag_tool, heterogeneous_rag_tool, multihop_rag_tool, rag_tool
from tools.models import Citation
from tools.multihop_retrieval import HopEvidence


def citation(source_id: str = "source") -> Citation:
    return Citation(
        label="[1]",
        title="Uploaded evidence",
        url="local://doc-1",
        source_type="uploaded_document",
        snippet="supporting text",
        quote="supporting text",
        source_id=source_id,
        doc_id="doc-1",
        chunk_id=f"chunk-{source_id}",
        page_number=1,
        metadata={"public": "retained"},
    )


def test_lazy_import_hook_installs_and_reinstalls_on_reload():
    assert rag_tool._rag_strategies_installed is True
    assert rag_tool._strategy_original_search_uploaded_docs is not rag_tool.search_uploaded_docs
    reloaded = importlib.reload(rag_tool)
    assert reloaded._rag_strategies_installed is True
    assert reloaded._strategy_original_search_uploaded_docs is not reloaded.search_uploaded_docs


def test_schema_exposes_explicit_bounded_strategy_controls():
    properties = rag_tool.RAG_SEARCH_TOOL_DEF["function"]["parameters"]["properties"]
    assert properties["strategy"]["enum"] == [
        "adaptive",
        "heterogeneous",
        "multihop",
        "single",
    ]
    assert properties["n_results"]["maximum"] == 50
    assert properties["max_attempts"]["maximum"] == 6
    assert properties["max_subquestions"]["maximum"] == 12
    assert properties["max_total_estimated_cost"]["maximum"] == 100_000
    assert properties["total_latency_limit_ms"]["maximum"] == 86_400_000
    assert properties["total_monetary_limit_microunits"]["maximum"] == 1_000_000_000


def test_adaptive_strategy_returns_only_authoritative_citations(monkeypatch):
    original = citation()
    monkeypatch.setattr(
        adaptive_rag_tool,
        "search_uploaded_docs_adaptive",
        lambda *args, **kwargs: SimpleNamespace(
            evidence=(original, {"not": "a citation"}),
            abstain=False,
            exhausted=True,
            estimated_cost=123,
            traces=(object(), object()),
        ),
    )
    result = rag_tool.search_uploaded_docs(
        "Question",
        owner_id="alice",
        strategy="adaptive",
        n_results=5,
        max_attempts=3,
        max_estimated_cost=300,
    )
    assert len(result) == 1
    assert result[0] is not original
    assert result[0].metadata["retrieval_strategy"] == "adaptive"
    assert result[0].metadata["adaptive_exhausted"] is True
    assert result[0].metadata["adaptive_estimated_cost"] == 123
    assert result[0].metadata["adaptive_attempt_count"] == 2
    assert result[0].metadata["public"] == "retained"


def test_adaptive_abstention_publishes_no_weak_citations(monkeypatch):
    monkeypatch.setattr(
        adaptive_rag_tool,
        "search_uploaded_docs_adaptive",
        lambda *args, **kwargs: SimpleNamespace(
            evidence=(citation(),),
            abstain=True,
            exhausted=True,
            estimated_cost=100,
            traces=(),
        ),
    )
    assert rag_tool.search_uploaded_docs("Question", strategy="adaptive") == []


def test_multihop_strategy_preserves_hop_lineage(monkeypatch):
    raw = citation("source-a")
    hop = HopEvidence(
        evidence_id="q1:source-a",
        hop_id="q1",
        source_id="source-a",
        doc_id="doc-1",
        page_number=1,
        text="supporting text",
        score=0.9,
        raw=raw,
    )
    result_value = SimpleNamespace(
        evidence=(hop,),
        abstain=False,
        retrieval=SimpleNamespace(plan_fingerprint="f" * 64),
        decomposition=SimpleNamespace(
            used_model=False,
            quality=SimpleNamespace(score=0.8),
            plan=SimpleNamespace(fingerprint="f" * 64),
        ),
        budget=SimpleNamespace(total_limit=200, allocated_cost=160),
        terminal_evidence_count=1,
    )
    monkeypatch.setattr(
        multihop_rag_tool,
        "search_uploaded_docs_multihop",
        lambda *args, **kwargs: result_value,
    )
    result = rag_tool.search_uploaded_docs(
        "Compare A and B",
        strategy="multihop",
        n_results=3,
        max_estimated_cost=100,
        max_total_estimated_cost=200,
    )
    assert len(result) == 1
    metadata = result[0].metadata
    assert metadata["retrieval_strategy"] == "multihop"
    assert metadata["multihop_hop_id"] == "q1"
    assert metadata["multihop_evidence_id"] == "q1:source-a"
    assert metadata["multihop_plan_fingerprint"] == "f" * 64
    assert metadata["multihop_budget_limit"] == 200
    assert metadata["multihop_allocated_budget"] == 160


def test_multihop_abstention_publishes_no_citations(monkeypatch):
    monkeypatch.setattr(
        multihop_rag_tool,
        "search_uploaded_docs_multihop",
        lambda *args, **kwargs: SimpleNamespace(abstain=True),
    )
    assert rag_tool.search_uploaded_docs("Question", strategy="multihop") == []


def test_heterogeneous_strategy_preserves_route_and_resource_lineage(monkeypatch):
    raw = citation("web-source")
    hop = HopEvidence(
        evidence_id="q1:web-source",
        hop_id="q1",
        source_id="web-source",
        doc_id="doc-1",
        page_number=1,
        text="supporting text",
        score=0.75,
        raw=raw,
    )
    monkeypatch.setattr(
        heterogeneous_rag_tool,
        "search_research_heterogeneous",
        lambda *args, **kwargs: SimpleNamespace(
            retrieval=SimpleNamespace(abstain=False, evidence=(hop,)),
            routes_by_hop=(("q1", "web"),),
            budget=SimpleNamespace(
                allocated_cost_units=12,
                allocated_latency_ms=345,
                allocated_monetary_microunits=67,
            ),
        ),
    )
    result = rag_tool.search_uploaded_docs(
        "Find the latest public evidence",
        strategy="heterogeneous",
        scope="public",
        n_results=3,
        total_cost_limit=100,
        total_latency_limit_ms=1_000,
        total_monetary_limit_microunits=100,
    )
    assert len(result) == 1
    metadata = result[0].metadata
    assert metadata["retrieval_strategy"] == "heterogeneous"
    assert metadata["heterogeneous_hop_id"] == "q1"
    assert metadata["heterogeneous_route"] == "web"
    assert metadata["heterogeneous_allocated_cost_units"] == 12
    assert metadata["heterogeneous_allocated_latency_ms"] == 345
    assert metadata["heterogeneous_allocated_monetary_microunits"] == 67


def test_non_single_strategies_reject_ambiguous_classic_controls():
    with pytest.raises(ValueError, match="Classic retrieval controls"):
        rag_tool.search_uploaded_docs(
            "Question",
            strategy="adaptive",
            retrieval_mode="hybrid",
        )
    with pytest.raises(ValueError, match="at most 10"):
        rag_tool.search_uploaded_docs(
            "Question",
            strategy="heterogeneous",
            n_results=11,
        )
    with pytest.raises(ValueError, match="year_from"):
        rag_tool.search_uploaded_docs(
            "Question",
            strategy="heterogeneous",
            year_from=2025,
            year_to=2020,
        )


def test_live_agent_schema_and_dispatch_use_same_authoritative_path(monkeypatch):
    schema = next(
        item
        for item in search_agent.TOOLS_SCHEMA
        if item["function"]["name"] == "search_uploaded_docs"
    )
    assert "strategy" in schema["function"]["parameters"]["properties"]
    captured = {}

    def fake_search(**kwargs):
        captured.update(kwargs)
        return [citation()]

    monkeypatch.setattr(search_agent, "search_uploaded_docs", fake_search)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    agent = search_agent.SearchAgent(owner_id="alice")
    call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="search_uploaded_docs",
            arguments=json.dumps(
                {
                    "query": "Question",
                    "strategy": "adaptive",
                    "n_results": 4,
                    "max_attempts": 3,
                }
            ),
        ),
    )
    execution = agent._execute_tool(call)
    assert execution.success is True
    assert len(execution.citations) == 1
    assert captured["owner_id"] == "alice"
    assert captured["strategy"] == "adaptive"
    assert captured["n_results"] == 4
    assert captured["max_attempts"] == 3
    assert "agent_client" in captured
    assert "expansion_model" in captured


def test_agent_schema_rejects_unknown_strategy_before_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(
        search_agent,
        "search_uploaded_docs",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    agent = search_agent.SearchAgent(owner_id="alice")
    call = SimpleNamespace(
        id="call-2",
        function=SimpleNamespace(
            name="search_uploaded_docs",
            arguments=json.dumps({"query": "Question", "strategy": "invalid"}),
        ),
    )
    execution = agent._execute_tool(call)
    assert execution.success is False
    assert execution.error_type == "ValueError"
    assert calls == []
