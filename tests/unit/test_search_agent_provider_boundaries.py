import itertools
import math

import pytest

import search_agent
from search_agent import SearchAgent, ToolExecution
from tools.models import Citation


def test_runtime_schema_rejects_nonfinite_numbers():
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite number"):
            search_agent._validate_schema_value(
                value,
                {"type": "number"},
                "tool.metric",
            )


def test_tool_execution_bounds_provider_identifiers_duration_and_citations():
    citation = Citation(
        label="[1]",
        title="Evidence",
        url="local://evidence",
        source_type="uploaded_document",
    )

    def infinite_citations():
        while True:
            yield citation

    execution = ToolExecution(
        tool_call_id="c" * 1000,
        tool_name="n" * 1000,
        content="result",
        citations=infinite_citations(),
        error_type="e" * 1000,
        duration=float("nan"),
    )

    assert len(execution.tool_call_id) == 200
    assert len(execution.tool_name) == 200
    assert len(execution.error_type) == 200
    assert execution.duration == 0.0
    assert math.isfinite(execution.duration)
    assert len(execution.citations) == search_agent._MAX_EVIDENCE_SOURCES


def test_tool_execution_survives_hostile_string_objects():
    class Hostile:
        def __str__(self):
            raise RuntimeError("do not stringify")

    execution = ToolExecution(
        tool_call_id=Hostile(),
        tool_name=Hostile(),
        content=Hostile(),
    )

    assert execution.tool_call_id == "unknown"
    assert execution.tool_name == "unknown"
    assert execution.content == ""


def test_direct_agent_construction_validates_identity_model_and_timeouts():
    with pytest.raises(ValueError, match="model"):
        SearchAgent(model="m" * 201, owner_id="alice")
    with pytest.raises(ValueError, match="model"):
        SearchAgent(model=object(), owner_id="alice")
    with pytest.raises(ValueError, match="Owner identifiers"):
        SearchAgent(owner_id="../bob")
    with pytest.raises(ValueError, match="owner_id"):
        SearchAgent(owner_id=object())
    with pytest.raises(ValueError, match="request_timeout"):
        SearchAgent(owner_id="alice", request_timeout=float("nan"))
    with pytest.raises(ValueError, match="tool_timeout"):
        SearchAgent(owner_id="alice", tool_timeout=float("inf"))
    with pytest.raises(ValueError, match="max_turns"):
        SearchAgent(owner_id="alice", max_turns=0)
    with pytest.raises(ValueError, match="max_tool_calls"):
        SearchAgent(owner_id="alice", max_tool_calls=65)
    with pytest.raises(ValueError, match="max_response_tokens"):
        SearchAgent(owner_id="alice", max_response_tokens=127)


def test_provider_configuration_is_bounded_and_control_safe(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="4,096"):
        SearchAgent(owner_id="alice", api_key="x" * 4097)
    with pytest.raises(ValueError, match="control characters"):
        SearchAgent(owner_id="alice", base_url="https://example.test\r\nInjected: yes")
    monkeypatch.setenv("OPENAI_API_KEY", "x" * 4097)
    with pytest.raises(ValueError, match="4,096"):
        SearchAgent(owner_id="alice")


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


def test_bounded_tool_calls_never_materializes_infinite_provider_stream():
    def calls():
        for index in itertools.count():
            yield type(
                "Call",
                (),
                {
                    "id": f"call-{index}",
                    "function": type(
                        "Function",
                        (),
                        {"name": "search_handbook", "arguments": "{}"},
                    )(),
                },
            )()

    bounded, overflow = search_agent._bounded_tool_calls(calls(), 3)

    assert overflow is True
    assert len(bounded) == 3
    assert [call.id for call in bounded] == ["call-0", "call-1", "call-2"]


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
