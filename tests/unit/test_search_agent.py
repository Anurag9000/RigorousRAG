from unittest.mock import patch

from search_agent import SearchAgent
from tools.models import Citation


def source(url="https://example.test/a", source_id="a"):
    return Citation(
        label="[temporary]",
        title="Evidence",
        url=url,
        source_type="web_page",
        snippet="supporting evidence",
        source_id=source_id,
    )


def test_evidence_registry_relabels_and_deduplicates_server_side():
    registry = []
    seen = {}
    selected = SearchAgent._register_citations([source()], registry, seen)
    repeated = SearchAgent._register_citations([source()], registry, seen)
    assert selected[0].label == "[1]"
    assert repeated[0].label == "[1]"
    assert len(registry) == 1


def test_only_answer_referenced_sources_are_returned():
    evidence = [source(source_id="a"), source("https://example.test/b", "b")]
    evidence[0].label = "[1]"
    evidence[1].label = "[2]"
    selected = SearchAgent._citations_used_by_answer("Supported [2].", evidence)
    assert [citation.label for citation in selected] == ["[2]"]


def test_uploaded_document_dispatch_always_passes_request_owner(monkeypatch):
    agent = SearchAgent(owner_id="alice")
    captured = {}

    def fake_search(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("search_agent.search_uploaded_docs", fake_search)
    content, citations = agent._dispatch("search_uploaded_docs", {"query": "q"})
    assert captured["owner_id"] == "alice"
    assert citations == []
    assert "retrieved" in content.lower()


def test_no_provider_uses_retrieval_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    citation = source()
    citation.label = "[1]"
    with patch("search_agent.OpenAI", None), \
         patch("search_agent.search_uploaded_docs", return_value=[citation]), \
         patch("search_agent.search_internal", return_value=[]):
        answer = SearchAgent(owner_id="alice").run("What is the evidence?")
    assert answer.citations
    assert answer.citations[0].label == "[1]"
    assert "without generative synthesis" in answer.answer


def test_final_json_parser_does_not_accept_model_supplied_citations():
    content = '```json\n{"answer":"Supported [1]","citations":[{"url":"invented"}]}\n```'
    assert SearchAgent._parse_final_text(content) == "Supported [1]"
