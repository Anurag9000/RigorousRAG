import argparse
from unittest.mock import MagicMock, patch

import pytest

from Searching import SearchHit
from ai_search import format_summary, main, run_query
from llm_agent import CitationSummary


def test_format_summary_preserves_markdown_and_bounds_output():
    markdown = "# Heading\n\n- item\n- item"
    assert format_summary(markdown) == markdown
    assert len(format_summary("x" * 30_000)) == 20_000
    assert "reliable summary" in format_summary(None).lower()


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

    engine.search.assert_not_called()
    agent.summarise.assert_not_called()


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
        main()

    engine.build.assert_not_called()
    run.assert_called_once()
    engine_manager.__exit__.assert_called_once()
