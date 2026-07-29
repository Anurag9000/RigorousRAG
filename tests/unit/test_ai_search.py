import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from Searching import SearchHit
from ai_search import format_summary, main, run_query
from llm_agent import CitationSummary


def test_format_summary_preserves_markdown_bounds_and_masks_private_metadata():
    markdown = "# Heading\n\n- item\n- item"
    assert format_summary(markdown) == markdown
    assert len(format_summary("x" * 30_000)) == 20_000
    assert "reliable summary" in format_summary(None).lower()
    masked = format_summary(
        "file:///private/secret and https://alice:password@example.test?api_key=secret"
    )
    assert "/private" not in masked
    assert "password" not in masked
    assert "api_key=secret" not in masked


def test_run_query_reports_no_results(capsys):
    engine = MagicMock()
    engine.search.return_value = []
    run_query(engine, MagicMock(), "q", 5)
    assert "No results" in capsys.readouterr().out


def test_run_query_prints_summary_and_sources(capsys):
    engine = MagicMock()
    engine.search.return_value = [
        SearchHit(
            1,
            "https://a.test",
            "A",
            "snippet",
            0.9,
            0.8,
            0.1,
            10,
        )
    ]
    engine.gather_context.return_value = [
        {"url": "https://a.test", "text": "evidence"}
    ]
    agent = MagicMock()
    agent.summarise.return_value = CitationSummary(
        "Supported [1].",
        ["[1] A — https://a.test"],
    )
    run_query(engine, agent, "q", 5)
    output = capsys.readouterr().out
    assert "Supported [1]." in output
    assert "[1] A" in output


def test_run_query_rejects_oversized_input_and_invalid_limit_before_search():
    engine = MagicMock()
    agent = MagicMock()

    with pytest.raises(ValueError, match="2,000"):
        run_query(engine, agent, "q" * 2001, 5)
    with pytest.raises(ValueError, match="integer"):
        run_query(engine, agent, "query", "bad")
    with pytest.raises(ValueError, match="between 1 and 20"):
        run_query(engine, agent, "query", 21)
    with pytest.raises(ValueError, match="control characters"):
        run_query(engine, agent, "query\x00hidden", 5)

    engine.search.assert_not_called()
    agent.summarise.assert_not_called()


def test_run_query_bounds_backend_iterables_and_validates_summary(capsys):
    valid = SearchHit(1, "https://a.test", "A", "snippet", 0.9, 0.8, 0.1, 10)

    def hits():
        yield valid
        while True:
            yield object()

    engine = MagicMock()
    engine.search.return_value = hits()
    engine.gather_context.return_value = [
        {"url": "https://a.test", "text": "evidence"}
    ]
    agent = MagicMock()
    agent.summarise.return_value = CitationSummary("Supported [1].", ["[1] A"])

    run_query(engine, agent, "q", 1)
    assert "Supported [1]." in capsys.readouterr().out

    engine.search.return_value = [valid]
    agent.summarise.return_value = object()
    with pytest.raises(RuntimeError, match="summarizer returned an invalid result"):
        run_query(engine, agent, "q", 1)


def test_run_query_masks_result_credentials_and_paths(capsys):
    engine = MagicMock()
    engine.search.return_value = [
        SearchHit(
            1,
            "https://alice:password@example.test?api_key=secret",
            "Report at /private/report.txt",
            "file:///private/source.txt",
            0.9,
            0.8,
            0.1,
            10,
        )
    ]
    engine.gather_context.return_value = [
        {
            "url": engine.search.return_value[0].url,
            "text": "evidence",
        }
    ]
    agent = MagicMock()
    agent.summarise.return_value = CitationSummary("Supported [1].", [])

    run_query(engine, agent, "q", 1)

    output = capsys.readouterr().out
    assert "password" not in output
    assert "api_key=secret" not in output
    assert "/private" not in output


def test_main_loads_persisted_index_without_rebuild_and_closes_engine():
    args = argparse.Namespace(
        query="q",
        max_pages=10,
        max_depth=1,
        delay=0,
        results=5,
        storage_dir="data",
        rebuild=False,
        model="m",
        api_key=None,
        base_url=None,
        ollama_model="o",
        ollama_host=None,
    )
    engine = MagicMock()
    engine.ready = True
    engine.index.documents = {"a": object()}
    engine_manager = MagicMock()
    engine_manager.__enter__.return_value = engine
    with patch("ai_search.parse_args", return_value=args), patch(
        "ai_search.AcademicSearchEngine",
        return_value=engine_manager,
    ), patch("ai_search.LLMAgent"), patch("ai_search.run_query") as run:
        assert main() == 0

    engine.build.assert_not_called()
    run.assert_called_once()
    engine_manager.__exit__.assert_called_once()


def test_main_returns_generic_initialization_failure(capsys):
    args = argparse.Namespace(
        query="q",
        max_pages=10,
        max_depth=1,
        delay=0,
        results=5,
        storage_dir="/private/secret",
        rebuild=False,
        model="m",
        api_key=None,
        base_url=None,
        ollama_model="o",
        ollama_host=None,
    )
    with patch("ai_search.parse_args", return_value=args), patch(
        "ai_search.AcademicSearchEngine",
        side_effect=RuntimeError("database failed at /private/secret"),
    ):
        assert main() == 1

    error = capsys.readouterr().err
    assert "could not be initialized" in error
    assert "/private" not in error


def test_main_returns_usage_error_for_invalid_result_limit(capsys):
    args = SimpleNamespace(results=0)
    with patch("ai_search.parse_args", return_value=args):
        assert main() == 2
    assert "between 1 and 20" in capsys.readouterr().err
