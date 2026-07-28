import math

import pytest

import search_agent
from search_agent import SearchAgent, ToolExecution


def test_runtime_schema_rejects_nonfinite_numbers():
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite number"):
            search_agent._validate_schema_value(
                value,
                {"type": "number"},
                "tool.metric",
            )


def test_tool_execution_bounds_provider_identifiers_and_duration():
    execution = ToolExecution(
        tool_call_id="c" * 1000,
        tool_name="n" * 1000,
        content="result",
        error_type="e" * 1000,
        duration=float("nan"),
    )

    assert len(execution.tool_call_id) == 200
    assert len(execution.tool_name) == 200
    assert len(execution.error_type) == 200
    assert execution.duration == 0.0
    assert math.isfinite(execution.duration)


def test_direct_agent_construction_validates_identity_model_and_timeouts():
    with pytest.raises(ValueError, match="model"):
        SearchAgent(model="m" * 201, owner_id="alice")
    with pytest.raises(ValueError, match="Owner identifiers"):
        SearchAgent(owner_id="../bob")
    with pytest.raises(ValueError, match="request_timeout"):
        SearchAgent(owner_id="alice", request_timeout=float("nan"))
    with pytest.raises(ValueError, match="tool_timeout"):
        SearchAgent(owner_id="alice", tool_timeout=float("inf"))


def test_valid_direct_agent_parameters_are_normalized(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    agent = SearchAgent(
        model=" model-name ",
        owner_id="alice",
        request_timeout=999,
        tool_timeout=999,
    )

    assert agent.model == "model-name"
    assert agent.owner_id == "alice"
    assert agent.tool_timeout == 300.0


def test_unmatched_handbook_lookup_returns_no_evidence(monkeypatch):
    agent = SearchAgent(owner_id="alice")
    monkeypatch.setattr(
        search_agent,
        "search_handbook",
        lambda **_kwargs: "No handbook passage matched the query.",
    )

    content, citations = agent._dispatch("search_handbook", {"query": "missing"})

    assert content == "No handbook evidence matched the query."
    assert citations == []


def test_matched_handbook_lookup_keeps_one_real_citation(monkeypatch):
    agent = SearchAgent(owner_id="alice")
    monkeypatch.setattr(
        search_agent,
        "search_handbook",
        lambda **_kwargs: "**handbook-1**\n\nPolicy evidence.",
    )

    content, citations = agent._dispatch("search_handbook", {"query": "policy"})

    assert "Policy evidence" in content
    assert len(citations) == 1
    assert citations[0].source_type == "handbook"
