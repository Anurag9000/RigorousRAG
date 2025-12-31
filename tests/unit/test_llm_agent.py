import pytest
from unittest.mock import MagicMock, patch
from llm_agent import LLMAgent, ExtractiveFallback, CitationSummary
from Searching import SearchHit

class TestLLMAgent:
    @pytest.fixture
    def hits(self):
        return [
            SearchHit(1, "http://a.com", "Title A", "snippet", 0.9, 0.8, 0.1, 100),
            SearchHit(2, "http://b.com", "Title B", "snippet", 0.8, 0.7, 0.1, 100)
        ]

    @pytest.fixture
    def contexts(self):
        return [
            {"url": "http://a.com", "title": "Title A", "text": "Content A"},
            {"url": "http://b.com", "title": "Title B", "text": "Content B"}
        ]

    def test_extractive_fallback(self, hits, contexts):
        fallback = ExtractiveFallback()
        result = fallback.summarise("test query", hits, contexts)
        assert isinstance(result, CitationSummary)
        assert "Title A" in result.summary
        assert "[1]" in result.summary
        assert len(result.sources) == 2

    def test_extractive_fallback_empty(self):
        fallback = ExtractiveFallback()
        result = fallback.summarise("test", [], [])
        assert "No supporting documents" in result.summary

    @patch('llm_agent.OpenAI')
    @patch('llm_agent.ollama')
    def test_llm_agent_init(self, mock_ollama, mock_openai):
        agent = LLMAgent(api_key="sk-test")
        assert agent.api_key == "sk-test"
        assert mock_openai.called

    @patch('llm_agent.OpenAI')
    def test_summarise_with_openai_success(self, mock_openai_class, hits, contexts):
        mock_client = mock_openai_class.return_value
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "OpenAI Summary"
        mock_client.chat.completions.create.return_value = mock_response
        
        agent = LLMAgent(api_key="sk-test")
        agent.openai_client = mock_client
        
        result = agent.summarise("query", hits, contexts)
        assert result.summary == "OpenAI Summary"
        assert len(result.sources) == 2
        assert mock_client.chat.completions.create.called

    @patch('llm_agent.ollama')
    def test_summarise_with_ollama_success(self, mock_ollama, hits, contexts):
        # Setup agent with no openai client
        agent = LLMAgent(api_key=None)
        agent.openai_client = None
        
        # Mock ollama chat
        mock_ollama.chat.return_value = {
            "message": {"content": "Ollama Summary"}
        }
        agent.ollama_client = mock_ollama
        
        result = agent.summarise("query", hits, contexts)
        assert result.summary == "Ollama Summary"
        assert mock_ollama.chat.called

    def test_summarise_all_fail_fallback(self, hits, contexts):
        # Create agent with no clients
        agent = LLMAgent(api_key=None)
        agent.openai_client = None
        agent.ollama_client = None
        
        result = agent.summarise("query", hits, contexts)
        # Should use ExtractiveFallback
        assert "Title A" in result.summary
        assert "query" in result.summary

    def test_build_prompt(self, hits, contexts):
        agent = LLMAgent()
        prompt = agent._build_prompt("query X", hits, contexts)
        assert "query X" in prompt
        assert "Title A" in prompt
        assert "Content B" in prompt
        assert "Task:" in prompt

    @patch('llm_agent.OpenAI')
    def test_openai_api_error_fallthrough(self, mock_openai_class, hits, contexts):
        mock_client = mock_openai_class.return_value
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        agent = LLMAgent(api_key="sk-test")
        agent.openai_client = mock_client
        agent.ollama_client = None # Force fallback to extractive
        
        result = agent.summarise("query", hits, contexts)
        assert "Title A" in result.summary # Indicates extractive fallback
