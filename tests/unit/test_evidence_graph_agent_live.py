from __future__ import annotations

import json
from types import SimpleNamespace

import search_agent
from tools import evidence_graph_agent_integration
from tools.models import Citation


def _citation() -> Citation:
    return Citation(
        label="[1]",
        title="Reviewed graph evidence",
        url="local://doc-1",
        source_type="uploaded_document",
        snippet="Supporting evidence.",
        quote="Supporting evidence.",
        source_id="e" * 64,
        doc_id="doc-1",
        chunk_id="n" * 64,
        page_number=1,
        metadata={
            "retrieval_strategy": "graph",
            "graph_set_id": "g" * 64,
        },
    )


def test_live_agent_schema_and_dispatch_use_canonical_graph_path(
    monkeypatch,
):
    schemas = [
        item
        for item in search_agent.TOOLS_SCHEMA
        if item["function"]["name"] == "search_evidence_graph"
    ]
    assert len(schemas) == 1
    parameters = schemas[0]["function"]["parameters"]
    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["query", "graph_set_key"]
    captured = {}

    def fake_search(**kwargs):
        captured.update(kwargs)
        return [_citation()]

    monkeypatch.setattr(
        evidence_graph_agent_integration,
        "search_evidence_graph",
        fake_search,
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    agent = search_agent.SearchAgent(owner_id="alice")
    call = SimpleNamespace(
        id="graph-call-1",
        function=SimpleNamespace(
            name="search_evidence_graph",
            arguments=json.dumps(
                {
                    "query": "Compare the reviewed results",
                    "graph_set_key": "review",
                    "node_types": ["claim"],
                    "cross_edge_types": ["supports"],
                    "max_citations": 7,
                }
            ),
        ),
    )

    execution = agent._execute_tool(call)

    assert execution.success is True
    assert execution.content == "Reviewed evidence-graph citations retrieved."
    assert execution.citations == [_citation()]
    assert captured == {
        "owner_id": "alice",
        "query": "Compare the reviewed results",
        "graph_set_key": "review",
        "node_types": ["claim"],
        "cross_edge_types": ["supports"],
        "max_citations": 7,
    }


def test_live_agent_rejects_unknown_graph_arguments_before_dispatch(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        evidence_graph_agent_integration,
        "search_evidence_graph",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    agent = search_agent.SearchAgent(owner_id="alice")
    call = SimpleNamespace(
        id="graph-call-2",
        function=SimpleNamespace(
            name="search_evidence_graph",
            arguments=json.dumps(
                {
                    "query": "Question",
                    "graph_set_key": "review",
                    "owner_id": "mallory",
                }
            ),
        ),
    )

    execution = agent._execute_tool(call)

    assert execution.success is False
    assert execution.error_type == "ValueError"
    assert calls == []


def test_live_agent_relabels_and_deduplicates_graph_citations(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    agent = search_agent.SearchAgent(owner_id="alice")
    registry = []
    seen = {}
    original = _citation()

    selected = agent._register_citations(
        [original, original.model_copy(deep=True)],
        registry,
        seen,
    )

    assert len(registry) == 1
    assert selected[0].label == "[1]"
    assert selected[0] is not original
    assert selected[0].metadata["retrieval_strategy"] == "graph"
