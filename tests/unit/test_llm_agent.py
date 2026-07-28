from unittest.mock import MagicMock, patch

import pytest

import llm_agent
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
    assert result.warning is None


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


def test_query_limit_fails_before_provider_calls():
    agent = LLMAgent(api_key=None)
    agent.openai_client = MagicMock()
    agent.ollama_client = MagicMock()

    with pytest.raises(ValueError, match="2,000"):
        agent.summarise(
            "q" * 2001,
            [hit(1, "https://a.test", "A")],
            [{"url": "https://a.test", "text": "evidence"}],
        )

    agent.openai_client.chat.completions.create.assert_not_called()
    agent.ollama_client.chat.assert_not_called()


def test_prompt_and_source_count_are_bounded():
    hits = [
        hit(index + 1, f"https://{index}.test", "T" * 1000)
        for index in range(50)
    ]
    contexts = [
        {"url": item.url, "text": "evidence " * 5000}
        for item in hits
    ]
    aligned = llm_agent._align_hits_and_contexts(hits, contexts)
    prompt = LLMAgent._build_prompt("question", aligned)
    sources = LLMAgent._source_list(aligned)

    assert len(aligned) == llm_agent._MAX_SOURCES
    assert len(prompt) <= llm_agent._MAX_PROMPT_CHARS
    assert len(sources) == llm_agent._MAX_SOURCES
    assert all(len(source) <= llm_agent._MAX_SOURCE_CHARS for source in sources)


def test_generated_summary_warns_for_missing_or_unsupported_markers():
    aligned = [
        (hit(1, "https://a.test", "A"), {"url": "https://a.test", "text": "A"}),
    ]

    missing = LLMAgent._generated_summary("No marker here.", aligned)
    unsupported = LLMAgent._generated_summary("Claim [2].", aligned)

    assert missing is not None and "no numeric citation markers" in missing.warning
    assert unsupported is not None and "[2]" in unsupported.warning


def test_generated_summary_and_sources_are_hard_bounded():
    aligned = [
        (hit(1, "https://a.test", "A"), {"url": "https://a.test", "text": "A"}),
    ]

    result = LLMAgent._generated_summary("x" * 100_000 + " [1]", aligned)

    assert result is not None
    assert len(result.summary) == llm_agent._MAX_SUMMARY_CHARS
    assert len(result.sources) == 1
