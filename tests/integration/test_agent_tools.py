import pytest
import json
from unittest.mock import MagicMock, patch
from search_agent import SearchAgent

class TestAgentIntegration:

    @patch('search_agent.OpenAI')
    @patch('search_agent.log_tool_call')
    def test_end_to_end_agent_flow(self, mock_log, mock_openai):
        """
        Simulate a full run:
        User asks -> Agent calls Web Search -> Agent receives result -> Agent answers.
        """
        
        # 1. First Response: Tool Call "web_search"
        msg1 = MagicMock()
        tc = MagicMock()
        tc.function.name = "web_search"
        tc.function.arguments = '{"query": "latest news"}'
        msg1.tool_calls = [tc]
        msg1.content = None
        
        # 2. Second Response: Final Answer
        msg2 = MagicMock()
        msg2.tool_calls = None
        msg2.content = '{"answer": "Results found.", "citations": []}'
        
        mock_openai.return_value.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=msg1)]),
            MagicMock(choices=[MagicMock(message=msg2)])
        ]
        
        agent = SearchAgent(api_key="sk-test")
        
        # We need to mock the actual tools used in _handle_tool_call so we don't hit real APIs
        # But we want to test that the *handler* calls them properly.
        # Let's patch 'search_agent.web_search' (from imported module tools.web_search presumably, or where it is imported)
        
        # search_agent.py likely does `from tools.web_search import web_search`
        # We patch that reference.
        with patch('search_agent.web_search') as mock_web:
            mock_web.return_value = "Search Result Content"
            
            result = agent.run("What's new?")
            
            # Assertions
            assert mock_web.called
            # The agent should have passed the query
            assert mock_web.call_args[1]['query'] == 'latest news' 
            assert result.answer == "Results found."
