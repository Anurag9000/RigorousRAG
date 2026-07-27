import argparse
from unittest.mock import MagicMock, patch

from Searching import SearchHit
from ai_search import format_summary, main, run_query
from llm_agent import CitationSummary


def test_format_summary_preserves_markdown():
    markdown = "# Heading\n\n- item\n- item"
    assert format_summary(markdown) == markdown


def test_run_query_reports_no_results(capsys):
    engine = MagicMock()
    engine.search.return_value = []
    run_query(engine, MagicMock(), "q", 5)
    assert "No results" in capsys.readouterr().out


def test_run_query_prints_summary_and_sources(capsys):
    engine = MagicMock()
    engine.search.return_value = [SearchHit(1, "https://a.test", "A", "snippet", 0.9, 0.8, 0.1, 10)]
    engine.gather_context.return_value = [{"url": "https://a.test", "text": "evidence"}]
    agent = MagicMock()
    agent.summarise.return_value = CitationSummary("Supported [1].", ["[1] A — https://a.test"])
    run_query(engine, agent, "q", 5)
    output = capsys.readouterr().out
    assert "Supported [1]." in output
    assert "[1] A" in output


def test_main_loads_persisted_index_without_rebuild():
    args = argparse.Namespace(
        query="q", max_pages=10, max_depth=1, delay=0, results=5,
        storage_dir="data", rebuild=False, model="m", api_key=None,
        base_url=None, ollama_model="o", ollama_host=None,
    )
    with patch("ai_search.parse_args", return_value=args), \
         patch("ai_search.AcademicSearchEngine") as engine_class, \
         patch("ai_search.LLMAgent"), \
         patch("ai_search.run_query") as run:
        engine_class.return_value.ready = True
        engine_class.return_value.index.documents = {"a": object()}
        main()
    assert not engine_class.return_value.build.called
    assert run.called
