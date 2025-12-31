import pytest
import argparse
import sys
from unittest.mock import MagicMock, patch
from search_agent_cli import main, print_result
from tools.models import AgentAnswer, Citation

class TestSearchAgentCLI:
    def test_print_result(self, capsys):
        cit = Citation(label="[1]", title="T", url="u", source_type="web_page", snippet="very long snippet " * 20)
        ans = AgentAnswer(answer="Answer", citations=[cit])
        
        print_result(ans)
        captured = capsys.readouterr().out
        assert "Answer" in captured
        assert "[1] T (web_page)" in captured
        assert "Excerpt: very long snippet" in captured
        assert "..." in captured # Check truncation

    @patch('search_agent_cli.os.getenv')
    @patch('search_agent_cli.sys.exit')
    def test_main_no_api_key(self, mock_exit, mock_getenv):
        mock_getenv.return_value = None
        with patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(query="q", model="m")):
            with patch('builtins.print'):
                main()
        mock_exit.assert_called_with(1)

    @patch('search_agent_cli.SearchAgent')
    @patch('search_agent_cli.os.getenv')
    def test_main_one_shot(self, mock_getenv, mock_agent_class):
        mock_getenv.return_value = "sk-fake"
        mock_agent = mock_agent_class.return_value
        mock_agent.run.return_value = AgentAnswer(answer="Done", citations=[])
        
        with patch('sys.argv', ['search_agent_cli.py', '--query', 'test']):
            with patch('builtins.print'):
                main()
        
        assert mock_agent.run.called
        assert mock_agent_class.called

    @patch('search_agent_cli.SearchAgent')
    @patch('search_agent_cli.os.getenv')
    @patch('builtins.input')
    def test_main_interactive(self, mock_input, mock_getenv, mock_agent_class):
        mock_getenv.return_value = "sk-fake"
        mock_input.side_effect = ["hello", "exit"]
        mock_agent = mock_agent_class.return_value
        mock_agent.run.return_value = AgentAnswer(answer="Hi", citations=[])
        
        with patch('sys.argv', ['search_agent_cli.py']):
            with patch('builtins.print'):
                main()
        
        assert mock_agent.run.call_count == 1
        assert mock_input.call_count == 2
