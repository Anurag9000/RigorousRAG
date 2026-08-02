from __future__ import annotations

import json
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


def test_install_adds_closed_schemas_dispatch_and_prompt(monkeypatch):
    module = _module()
    citation = Citation(
        label="[1]",
        title="Graph result",
        url="local://doc-1",
        source_type="uploaded_document",
    )
    captured = {}

    def search(**kwargs):
        captured["search"] = kwargs
        return [citation]

    def listing(**kwargs):
        captured["list"] = kwargs
        return [{"graph_set_key": "review"}]

    monkeypatch.setattr(integration, "search_evidence_graph", search)
    monkeypatch.setattr(integration, "list_evidence_graph_sets", listing)
    result = integration.install_evidence_graph_agent_tool(module)

    assert result is module
    names = [value["function"]["name"] for value in module.TOOLS_SCHEMA]
    assert names == [
        "existing",
        "list_evidence_graph_sets",
        "search_evidence_graph",
    ]
    for name in ("list_evidence_graph_sets", "search_evidence_graph"):
        assert module._TOOL_PARAMETER_SCHEMAS[name]["additionalProperties"] is False
    agent = module.SearchAgent()
    content, citations = agent._dispatch(
        "list_evidence_graph_sets", {"limit": 3}
    )
    assert json.loads(content) == {
        "count": 1,
        "graph_sets": [{"graph_set_key": "review"}],
        "source_text_returned": False,
    }
    assert citations == []
    assert captured["list"] == {"owner_id": "alice", "limit": 3}

    content, citations = agent._dispatch(
        "search_evidence_graph",
        {"query": "result", "graph_set_key": "review"},
    )
    assert content == "Reviewed evidence-graph citations retrieved."
    assert citations == [citation]
    assert captured["search"] == {
        "owner_id": "alice",
        "query": "result",
        "graph_set_key": "review",
    }
    assert "Reviewed Evidence Graph Sets" in module.SYSTEM_PROMPT
    assert "Reviewed Evidence Graph (`search_evidence_graph`)" in module.SYSTEM_PROMPT


def test_install_is_idempotent_and_preserves_fallback(monkeypatch):
    module = _module()
    monkeypatch.setattr(
        integration,
        "search_evidence_graph",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        integration,
        "list_evidence_graph_sets",
        lambda **kwargs: [],
    )
    integration.install_evidence_graph_agent_tool(module)
    first_dispatch = module.SearchAgent._dispatch
    integration.install_evidence_graph_agent_tool(module)

    assert module.SearchAgent._dispatch is first_dispatch
    names = [value["function"]["name"] for value in module.TOOLS_SCHEMA]
    assert names.count("list_evidence_graph_sets") == 1
    assert names.count("search_evidence_graph") == 1
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
