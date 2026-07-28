import json
import time
from types import SimpleNamespace
from unittest.mock import patch

from search_agent import SearchAgent, ToolExecution


def tool_call(name, arguments, call_id="call-1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_single_tool_timeout_returns_without_waiting_for_worker(monkeypatch):
    agent = SearchAgent(owner_id="alice", tool_timeout=1.0)
    agent.tool_timeout = 0.05

    def slow_execution(call):
        time.sleep(0.4)
        return ToolExecution(
            tool_call_id=call.id,
            tool_name=call.function.name,
            content="late",
        )

    monkeypatch.setattr(agent, "_execute_tool", slow_execution)
    started = time.monotonic()
    results = agent._execute_tools([tool_call("fetch_page", '{"url":"https://example.com"}')])
    elapsed = time.monotonic() - started

    assert elapsed < 0.25
    assert results[0].success is False
    assert results[0].error_type == "TimeoutError"
    assert "timed out" in results[0].content.lower()


def test_runtime_schema_rejects_unknown_arguments_before_dispatch(monkeypatch):
    agent = SearchAgent(owner_id="alice")
    dispatched = []
    monkeypatch.setattr(agent, "_dispatch", lambda *args: dispatched.append(args))

    result = agent._execute_tool(
        tool_call(
            "search_uploaded_docs",
            json.dumps({"query": "evidence", "owner_id": "bob"}),
        )
    )

    assert result.success is False
    assert result.error_type == "ValueError"
    assert dispatched == []
    assert "owner_id" not in result.content
    assert "invalid" in result.content.lower()


def test_raw_tool_exception_is_not_exposed_to_model_context(monkeypatch):
    agent = SearchAgent(owner_id="alice")

    def fail(*_args, **_kwargs):
        raise RuntimeError("secret database path /private/alice/state.sqlite3")

    monkeypatch.setattr(agent, "_dispatch", fail)
    result = agent._execute_tool(
        tool_call("fetch_page", json.dumps({"url": "https://example.com"}))
    )

    assert result.success is False
    assert result.error_type == "RuntimeError"
    assert "secret" not in result.content
    assert "/private" not in result.content
    assert result.content == "Tool execution failed. Treat this tool result as unavailable."


def test_oversized_tool_arguments_are_rejected(monkeypatch):
    agent = SearchAgent(owner_id="alice")
    monkeypatch.setattr("search_agent._MAX_TOOL_ARGUMENT_CHARS", 20)
    result = agent._execute_tool(
        tool_call("search_handbook", json.dumps({"query": "x" * 100}))
    )
    assert result.success is False
    assert result.error_type == "ValueError"


def test_fallback_distinguishes_retrieval_outage_from_no_match(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with patch("search_agent.OpenAI", None), \
         patch("search_agent.search_uploaded_docs", side_effect=RuntimeError("vector down")), \
         patch("search_agent.search_internal", return_value=[]):
        answer = SearchAgent(owner_id="alice").run("question")

    assert not answer.citations
    assert "no matching local evidence" in answer.answer.lower()
    assert any("retrieval unavailable" in warning.lower() for warning in answer.warnings)
    assert all("vector down" not in warning for warning in answer.warnings)
