from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import llm_agent
from Searching import SearchHit
from llm_agent import CitationSummary, ExtractiveFallback, LLMAgent


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
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Supported [1].")
            )
        ]
    )
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


def test_generator_evidence_survives_provider_failure():
    def hits():
        yield hit(1, "https://a.test", "A")

    def contexts():
        yield {"url": "https://a.test", "text": "A evidence"}

    agent = LLMAgent(api_key=None)
    agent.openai_client = MagicMock()
    agent.openai_client.chat.completions.create.side_effect = RuntimeError("failed")
    agent.ollama_client = None

    result = agent.summarise("q", hits(), contexts())

    assert "A evidence" in result.summary
    assert result.sources == ["[1] A — https://a.test"]


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


def test_prompt_and_source_count_are_bounded_and_complete():
    hits = [
        hit(index + 1, f"https://{index}.test/" + "u" * 2000, "T" * 1000)
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
    assert "[20] Title:" in prompt
    assert "Cite every evidence-dependent statement" in prompt


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


def test_hit_and_context_iterables_are_bounded():
    def infinite_hits():
        index = 0
        while True:
            yield hit(index, f"https://{index}.test", str(index))
            index += 1

    with pytest.raises(ValueError, match="at most 1000"):
        llm_agent._align_hits_and_contexts(infinite_hits(), [])

    def infinite_sources():
        while True:
            yield "source"

    with pytest.raises(ValueError, match="at most 20"):
        CitationSummary("summary", infinite_sources())


def test_hostile_iterables_and_truthiness_fail_safely():
    class HostileSources:
        def __bool__(self):
            raise RuntimeError("truthiness must not be used")

        def __iter__(self):
            yield "source"

    class BrokenIterable:
        def __iter__(self):
            raise RuntimeError("private iterator detail")

    assert CitationSummary("summary", HostileSources()).sources == ["source"]
    with pytest.raises(ValueError, match="safely iterable"):
        llm_agent._align_hits_and_contexts(BrokenIterable(), [])


def test_malformed_timeout_falls_back_and_provider_values_are_bounded(monkeypatch):
    monkeypatch.setenv("LEGACY_LLM_TIMEOUT_SECONDS", "nan")
    monkeypatch.setattr(llm_agent, "OpenAI", None)
    monkeypatch.setattr(llm_agent, "ollama", None)

    agent = LLMAgent(api_key=None)
    assert agent.openai_client is None

    with pytest.raises(ValueError, match="4096"):
        LLMAgent(api_key="x" * 4097)
    with pytest.raises(ValueError, match="control characters"):
        LLMAgent(base_url="http://localhost\r\nInjected: yes")


def test_provider_and_model_values_require_strings(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setattr(llm_agent, "OpenAI", None)
    monkeypatch.setattr(llm_agent, "ollama", None)

    with pytest.raises(ValueError, match="api_key must be a string"):
        LLMAgent(api_key=object())
    with pytest.raises(ValueError, match="Model names must be strings"):
        LLMAgent(model=object())
    with pytest.raises(ValueError, match="Model names must be strings"):
        LLMAgent(ollama_model=object())


def test_non_string_provider_content_falls_back_to_extractive_evidence():
    hits = [hit(1, "https://a.test", "A")]
    contexts = [{"url": "https://a.test", "text": "A evidence"}]
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=object()))]
    )
    agent = LLMAgent(api_key=None)
    agent.openai_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response)
        )
    )
    agent.ollama_client = None

    result = agent.summarise("q", hits, contexts)

    assert "A evidence" in result.summary
    assert result.warning


def test_summary_masks_credentials_and_local_paths():
    result = CitationSummary(
        "See file:///private/secret and https://alice:password@example.test.",
        ["https://alice:password@example.test?api_key=secret"],
    )
    rendered = result.summary + " " + " ".join(result.sources)
    assert "password" not in rendered
    assert "api_key=secret" not in rendered
    assert "/private" not in rendered
