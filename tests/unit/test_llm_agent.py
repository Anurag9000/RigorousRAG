from unittest.mock import MagicMock, patch

from Searching import SearchHit
from llm_agent import ExtractiveFallback, LLMAgent


def hit(rank, url, title):
    return SearchHit(rank, url, title, "snippet", 0.9, 0.8, 0.1, 100)


def test_extractive_fallback_aligns_contexts_by_url_not_position():
    hits = [hit(1, "https://a.test", "A"), hit(2, "https://b.test", "B")]
    contexts = [
        {"url": "https://b.test", "text": "B evidence"},
        {"url": "https://a.test", "text": "A evidence"},
    ]
    result = ExtractiveFallback().summarise("q", hits, contexts)
    assert result.sources[0].startswith("[1] A")
    assert "[1] **A** — A evidence" in result.summary
    assert "[2] **B** — B evidence" in result.summary


def test_openai_summary_uses_aligned_source_list():
    hits = [hit(1, "https://a.test", "A")]
    contexts = [{"url": "https://a.test", "text": "A evidence"}]
    response = MagicMock()
    response.choices[0].message.content = "Supported [1]."
    with patch("llm_agent.OpenAI") as client_class:
        client_class.return_value.chat.completions.create.return_value = response
        agent = LLMAgent(api_key="test")
        result = agent.summarise("q", hits, contexts)
    assert result.summary == "Supported [1]."
    assert result.sources == ["[1] A — https://a.test"]


def test_provider_failure_returns_extractive_evidence():
    hits = [hit(1, "https://a.test", "A")]
    contexts = [{"url": "https://a.test", "text": "A evidence"}]
    agent = LLMAgent(api_key=None)
    agent.openai_client = MagicMock()
    agent.openai_client.chat.completions.create.side_effect = RuntimeError("failed")
    agent.ollama_client = None
    result = agent.summarise("q", hits, contexts)
    assert "retrieved evidence" in result.summary.lower()
    assert result.warning
