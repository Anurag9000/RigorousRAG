from __future__ import annotations

import sys
from types import ModuleType

from tools import evidence_graph_agent_import_hook as import_hook
from tools import evidence_graph_agent_integration as integration
from tools.models import Citation


def _module():
    module = ModuleType("fake_search_agent_legacy")
    module.TOOLS_SCHEMA = [
        {
            "type": "function",
            "function": {
                "name": "existing",
                "parameters": {"type": "object"},
            },
        }
    ]
    module._TOOL_PARAMETER_SCHEMAS = {
        "existing": {"type": "object"}
    }
    module.SYSTEM_PROMPT = "Base prompt."

    class SearchAgent:
        owner_id = "alice"

        def _dispatch(self, tool_name, arguments):
            return f"fallback:{tool_name}", []

    module.SearchAgent = SearchAgent
    return module


def test_install_adds_closed_schema_dispatch_and_prompt(monkeypatch):
    module = _module()
    citation = Citation(
        label="[1]",
        title="Graph result",
        url="local://doc-1",
        source_type="uploaded_document",
    )
    captured = {}

    def search(**kwargs):
        captured.update(kwargs)
        return [citation]

    monkeypatch.setattr(integration, "search_evidence_graph", search)
    result = integration.install_evidence_graph_agent_tool(module)

    assert result is module
    names = [value["function"]["name"] for value in module.TOOLS_SCHEMA]
    assert names == ["existing", "search_evidence_graph"]
    schema = module._TOOL_PARAMETER_SCHEMAS["search_evidence_graph"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["query", "graph_set_key"]
    agent = module.SearchAgent()
    content, citations = agent._dispatch(
        "search_evidence_graph",
        {"query": "result", "graph_set_key": "review"},
    )
    assert content == "Reviewed evidence-graph citations retrieved."
    assert citations == [citation]
    assert captured == {
        "owner_id": "alice",
        "query": "result",
        "graph_set_key": "review",
    }
    assert "Reviewed Evidence Graph" in module.SYSTEM_PROMPT


def test_install_is_idempotent_and_preserves_fallback(monkeypatch):
    module = _module()
    monkeypatch.setattr(
        integration,
        "search_evidence_graph",
        lambda **kwargs: [],
    )
    integration.install_evidence_graph_agent_tool(module)
    first_dispatch = module.SearchAgent._dispatch
    integration.install_evidence_graph_agent_tool(module)

    assert module.SearchAgent._dispatch is first_dispatch
    assert sum(
        value["function"]["name"] == "search_evidence_graph"
        for value in module.TOOLS_SCHEMA
    ) == 1
    assert module.SearchAgent()._dispatch("existing", {}) == (
        "fallback:existing",
        [],
    )


def test_install_rejects_incompatible_agent_modules():
    module = ModuleType("invalid_agent")
    module.TOOLS_SCHEMA = []
    module._TOOL_PARAMETER_SCHEMAS = {}

    try:
        integration.install_evidence_graph_agent_tool(module)
    except RuntimeError as exc:
        assert "dispatch" in str(exc)
    else:
        raise AssertionError("incompatible agent module was accepted")


def test_import_hook_registration_is_unique():
    before = sum(
        getattr(finder, import_hook._MARKER, False)
        for finder in sys.meta_path
    )
    import_hook.install_evidence_graph_agent_import_hook()
    after = sum(
        getattr(finder, import_hook._MARKER, False)
        for finder in sys.meta_path
    )
    assert before == after == 1
