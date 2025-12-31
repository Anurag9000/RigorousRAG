import pytest
import argparse
from unittest.mock import MagicMock, patch
from ai_search import format_summary, run_query, parse_args, main
from llm_agent import CitationSummary
from Searching import SearchHit

class TestAISearch:
    def test_format_summary(self):
        long_string = "a " * 100
        formatted = format_summary(long_string)
        # Check if wrapping happened (width 100)
        assert "\n" in formatted or len(long_string) <= 100

    def test_run_query_no_hits(self):
        engine = MagicMock()
        engine.search.return_value = []
        agent = MagicMock()
        
        with patch('builtins.print') as mock_print:
            run_query(engine, agent, "query", 5)
            mock_print.assert_called_with("No results found.")

    def test_run_query_with_hits(self):
        engine = MagicMock()
        hit = SearchHit(1, "http://a.com", "Title A", "Snippet A", 0.9, 0.8, 0.1, 100)
        engine.search.return_value = [hit]
        engine.gather_context.return_value = [{"url": "http://a.com", "title": "Title A", "text": "Content A"}]
        
        agent = MagicMock()
        agent.summarise.return_value = CitationSummary(
            summary="AI Summary Result",
            sources=["[1] Title A — http://a.com"]
        )
        
        with patch('builtins.print') as mock_print:
            run_query(engine, agent, "test query", 5)
            
            # Check for key sections
            mock_print.assert_any_call("\n=== AI Summary ===")
            mock_print.assert_any_call("AI Summary Result")
            mock_print.assert_any_call("\n=== Sources ===")
            mock_print.assert_any_call("- [1] Title A — http://a.com")

    def test_parse_args(self):
        with patch('sys.argv', ['ai_search.py', '--query', 'hello', '--results', '5']):
            args = parse_args()
            assert args.query == 'hello'
            assert args.results == 5

    @patch('ai_search.AcademicSearchEngine')
    @patch('ai_search.LLMAgent')
    @patch('ai_search.parse_args')
    @patch('ai_search.run_query')
    def test_main_one_shot(self, mock_run_query, mock_args, mock_agent_class, mock_engine_class):
        # Mocking for one-shot query mode
        mock_args.return_value = argparse.Namespace(
            query="test query", results=5, max_pages=10, max_depth=1, delay=0,
            model="gpt-4", api_key=None, ollama_model="qwen", ollama_host=None
        )
        mock_engine = mock_engine_class.return_value
        mock_engine.build.return_value = 1
        
        with patch('builtins.print'):
            main()
            
        assert mock_run_query.called
        assert mock_engine.build.called

    @patch('ai_search.AcademicSearchEngine')
    @patch('ai_search.LLMAgent')
    @patch('ai_search.parse_args')
    @patch('ai_search.run_query')
    @patch('builtins.input')
    def test_main_interactive(self, mock_input, mock_run_query, mock_args, mock_agent_class, mock_engine_class):
        # Mocking for interactive mode
        mock_args.return_value = argparse.Namespace(
            query=None, results=5, max_pages=10, max_depth=1, delay=0,
            model="gpt-4", api_key=None, ollama_model="qwen", ollama_host=None
        )
        mock_input.side_effect = ["interactive query", ""] # Two inputs, second empty to exit
        mock_engine = mock_engine_class.return_value
        mock_engine.build.return_value = 1
        
        with patch('builtins.print'):
            main()
            
        assert mock_run_query.called
        assert mock_input.call_count == 2
