from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools import evidence_graph_rag_tool as tool
from tools.models import Citation


def test_schema_is_closed_and_uses_canonical_function_name():
    function = tool.GRAPH_RAG_SEARCH_TOOL_DEF["function"]
    assert function["name"] == "search_evidence_graph"
    assert function["parameters"]["additionalProperties"] is False
    assert function["parameters"]["required"] == ["query", "graph_set_key"]
    assert (
        function["parameters"]["properties"]["max_citations"]["maximum"]
        == 50
    )


def test_authoritative_selector_and_converter_receive_bounded_contract(
    monkeypatch,
):
    set_store = object()
    generations = object()
    graphs = object()
    captured = {}
    selection = SimpleNamespace(abstained=False)
    expected = [
        Citation(
            label="[1]",
            title="Result",
            url="local://doc-1",
            source_type="uploaded_document",
        )
    ]

    def select(**kwargs):
        captured["selection"] = kwargs
        return selection

    def convert(value, **kwargs):
        captured["conversion"] = (value, kwargs)
        return expected

    monkeypatch.setattr(tool, "select_current_graph_set_evidence", select)
    monkeypatch.setattr(tool, "graph_evidence_to_citations", convert)

    result = tool.search_evidence_graph(
        "  measured outcome  ",
        owner_id="alice",
        graph_set_key="review",
        node_types=("claim",),
        within_edge_types=("supports",),
        cross_edge_types=("cites",),
        allowed_origins=("lexical", "cross_document"),
        per_document_hits=5,
        max_lexical_seeds=12,
        max_within_per_seed=2,
        max_cross_depth=1,
        max_cross_per_seed=7,
        max_citations=9,
        set_store=set_store,
        generations=generations,
        graphs=graphs,
    )

    assert result == expected
    assert captured["selection"] == {
        "owner_id": "alice",
        "graph_set_key": "review",
        "query": "measured outcome",
        "set_store": set_store,
        "generations": generations,
        "graphs": graphs,
        "node_types": ("claim",),
        "within_edge_types": ("supports",),
        "cross_edge_types": ("cites",),
        "per_document_hits": 5,
        "max_lexical_seeds": 12,
        "max_within_per_seed": 2,
        "max_cross_depth": 1,
        "max_cross_per_seed": 7,
        "max_total_items": 9,
    }
    assert captured["conversion"] == (
        selection,
        {
            "max_citations": 9,
            "allowed_origins": ("cross_document", "lexical"),
        },
    )


def test_runtime_factories_are_used_only_without_injected_dependencies(
    monkeypatch,
):
    dependencies = {
        "set_store": object(),
        "generations": object(),
        "graphs": object(),
    }
    calls = []
    monkeypatch.setattr(
        tool,
        "get_evidence_graph_set_store",
        lambda: calls.append("set_store") or dependencies["set_store"],
    )
    monkeypatch.setattr(
        tool,
        "get_generation_store",
        lambda: calls.append("generations") or dependencies["generations"],
    )
    monkeypatch.setattr(
        tool,
        "get_evidence_graph_store",
        lambda: calls.append("graphs") or dependencies["graphs"],
    )
    monkeypatch.setattr(
        tool,
        "select_current_graph_set_evidence",
        lambda **kwargs: SimpleNamespace(abstained=True),
    )
    monkeypatch.setattr(
        tool,
        "graph_evidence_to_citations",
        lambda *args, **kwargs: [],
    )

    assert tool.search_evidence_graph("query", graph_set_key="review") == []
    assert calls == ["set_store", "generations", "graphs"]


def test_invalid_filters_and_budgets_fail_before_selection(monkeypatch):
    called = False

    def select(**kwargs):
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(tool, "select_current_graph_set_evidence", select)
    with pytest.raises(ValueError, match="node_types"):
        tool.search_evidence_graph(
            "query",
            graph_set_key="review",
            node_types=("invented",),
            set_store=object(),
            generations=object(),
            graphs=object(),
        )
    with pytest.raises(ValueError, match="max_citations"):
        tool.search_evidence_graph(
            "query",
            graph_set_key="review",
            max_citations=51,
            set_store=object(),
            generations=object(),
            graphs=object(),
        )
    assert called is False
