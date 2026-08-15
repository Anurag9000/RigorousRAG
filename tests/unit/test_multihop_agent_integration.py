from __future__ import annotations

import json
from types import ModuleType, SimpleNamespace

from tools import multihop_agent_integration as integration
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
    module._TOOL_PARAMETER_SCHEMAS = {"existing": {"type": "object"}}
    module._MAX_EVIDENCE_SOURCES = 10
    module.SYSTEM_PROMPT = "Base prompt."

    class SearchAgent:
        owner_id = "alice"
        client = object()

        def _expansion_model(self):
            return "expansion-model"

        def _dispatch(self, tool_name, arguments):
            return f"fallback:{tool_name}", []

    module.SearchAgent = SearchAgent
    return module


def _citation(source_id: str = "chunk-1") -> Citation:
    return Citation(
        label="[1]",
        title="Uploaded evidence",
        url="local://doc-1",
        source_type="uploaded_document",
        snippet="Grounded evidence.",
        quote="Grounded evidence.",
        source_id=source_id,
        doc_id="doc-1",
        chunk_id=source_id,
        page_number=1,
    )


def _result(*, abstain: bool = False):
    citation = _citation()
    return SimpleNamespace(
        abstain=abstain,
        evidence=(SimpleNamespace(raw=citation), SimpleNamespace(raw=citation)),
    )


def test_install_adds_schema_owner_scoped_dispatch_and_prompt(monkeypatch):
    module = _module()
    captured = {}
    result = _result()

    def search(**kwargs):
        captured.update(kwargs)
        return result

    def payload(value):
        assert value is result
        return {
            "plan_fingerprint": "plan-1",
            "evidence": [
                {
                    "citation": {"label": "[99]", "snippet": "must not leak"},
                    "lineage": {
                        "evidence_id": "q1:chunk-1",
                        "hop_id": "q1",
                        "source_id": "chunk-1",
                    },
                }
            ],
            "abstain": False,
        }

    monkeypatch.setattr(integration, "search_uploaded_docs_multihop", search)
    monkeypatch.setattr(integration, "multihop_result_payload", payload)
    installed = integration.install_multihop_agent_tool(module)

    assert installed is module
    names = [item["function"]["name"] for item in module.TOOLS_SCHEMA]
    assert names == ["existing", "search_uploaded_docs_multihop"]
    schema = module._TOOL_PARAMETER_SCHEMAS["search_uploaded_docs_multihop"]
    assert schema["additionalProperties"] is False

    content, citations = module.SearchAgent()._dispatch(
        "search_uploaded_docs_multihop",
        {"query": "How does A depend on B?", "max_subquestions": 4},
    )
    body = json.loads(content)
    assert body["evidence"] == [
        {
            "evidence_id": "q1:chunk-1",
            "hop_id": "q1",
            "source_id": "chunk-1",
        }
    ]
    assert "citation" not in body["evidence"][0]
    assert body["citation_gate"] == {
        "status": "terminal_evidence_available",
        "authoritative_citation_count": 1,
        "instruction": "Use only the server-supplied citation objects outside this result payload.",
    }
    assert citations == [_citation()]
    assert captured == {
        "owner_id": "alice",
        "agent_client": module.SearchAgent.client,
        "query": "How does A depend on B?",
        "max_subquestions": 4,
        "expansion_model": "expansion-model",
    }
    assert "Multi-hop uploaded-document retrieval" in module.SYSTEM_PROMPT
    assert "if the tool abstains" in module.SYSTEM_PROMPT


def test_abstaining_chain_returns_no_authoritative_citations(monkeypatch):
    module = _module()
    result = _result(abstain=True)
    monkeypatch.setattr(
        integration,
        "search_uploaded_docs_multihop",
        lambda **kwargs: result,
    )
    monkeypatch.setattr(
        integration,
        "multihop_result_payload",
        lambda value: {"evidence": [], "abstain": True},
    )
    integration.install_multihop_agent_tool(module)

    content, citations = module.SearchAgent()._dispatch(
        "search_uploaded_docs_multihop", {"query": "unsupported chain"}
    )
    assert citations == []
    assert json.loads(content)["citation_gate"] == {
        "status": "abstain",
        "authoritative_citation_count": 0,
        "instruction": "Use only the server-supplied citation objects outside this result payload.",
    }


def test_install_is_idempotent_and_preserves_fallback(monkeypatch):
    module = _module()
    monkeypatch.setattr(
        integration,
        "search_uploaded_docs_multihop",
        lambda **kwargs: _result(),
    )
    integration.install_multihop_agent_tool(module)
    first_dispatch = module.SearchAgent._dispatch
    integration.install_multihop_agent_tool(module)

    assert module.SearchAgent._dispatch is first_dispatch
    names = [item["function"]["name"] for item in module.TOOLS_SCHEMA]
    assert names.count("search_uploaded_docs_multihop") == 1
    assert module.SearchAgent()._dispatch("existing", {}) == (
        "fallback:existing",
        [],
    )


def test_install_rejects_incompatible_agent_modules():
    module = ModuleType("invalid_agent")
    module.TOOLS_SCHEMA = []
    module._TOOL_PARAMETER_SCHEMAS = {}

    try:
        integration.install_multihop_agent_tool(module)
    except RuntimeError as exc:
        assert "dispatch" in str(exc)
    else:
        raise AssertionError("incompatible agent module was accepted")
