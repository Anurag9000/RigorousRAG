import pytest
from unittest.mock import MagicMock, patch
from search_agent import SearchAgent
from tools.models import AgentAnswer

class TestSearchAgent:
    
    @patch('search_agent.OpenAI')
    def test_run_basic(self, mock_openai):
        # Mock the chat completion response
        mock_msg = MagicMock()
        mock_msg.tool_calls = None
        mock_msg.content = '{"answer": "Test Answer", "citations": []}'
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_msg)]
        
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        agent = SearchAgent(api_key="dummy")
        result = agent.run("Questions?")
        
        assert result.answer == "Test Answer"
        assert len(result.citations) == 0

    @patch('search_agent.OpenAI')
    @patch('search_agent.log_tool_call')
    def test_tool_call_flow(self, mock_log, mock_openai):
        # Setup a sequence: Tool Call -> Final Answer
        
        # 1. Tool Call Response
        msg1 = MagicMock()
        tool_call = MagicMock()
        tool_call.function.name = "web_search"
        tool_call.function.arguments = '{"query": "foo"}'
        msg1.tool_calls = [tool_call]
        msg1.content = None
        
        # 2. Final Answer Response
        msg2 = MagicMock()
        msg2.tool_calls = None
        msg2.content = '{"answer": "Web Found", "citations": []}'
        
        mock_openai.return_value.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=msg1)]), # Turn 1
            MagicMock(choices=[MagicMock(message=msg2)])  # Turn 2
        ]
        
        agent = SearchAgent(api_key="dummy")
        # Mock the actual tool handler to avoid networking
        with patch.object(agent, '_handle_tool_call') as mock_handle:
            result = agent.run("Find foo")
            
            assert mock_handle.called
            assert result.answer == "Web Found"
