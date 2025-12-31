import pytest
import json
from unittest.mock import MagicMock, patch
from search_agent import SearchAgent
from tools.models import AgentAnswer, Citation

class TestSearchAgentOrchestration:
    @pytest.fixture
    def mock_openai(self):
        with patch('search_agent.OpenAI') as mock:
            yield mock

    def test_agent_init(self, mock_openai):
        agent = SearchAgent(api_key="sk-test")
        assert agent.api_key == "sk-test"
        assert mock_openai.called

    @patch('search_agent.log_tool_call')
    @patch('search_agent.log_agent_run')
    @patch('search_agent.audit_hallucination', return_value="Verified")
    def test_agent_run_tool_flow(self, mock_audit, mock_log_run, mock_log_tool, mock_openai):
        mock_client = mock_openai.return_value
        agent = SearchAgent(api_key="sk-test")
        agent.client = mock_client
        
        # turn 1: Assistant calls search_internal
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function.name = "search_internal"
        mock_tool_call.function.arguments = json.dumps({"query": "AI"})
        
        mock_msg_1 = MagicMock()
        mock_msg_1.tool_calls = [mock_tool_call]
        mock_msg_1.content = None
        
        mock_response_1 = MagicMock()
        mock_response_1.choices[0].message = mock_msg_1
        
        # turn 2: Assistant provides final answer in JSON
        mock_msg_2 = MagicMock()
        mock_msg_2.tool_calls = None
        mock_msg_2.content = json.dumps({
            "answer": "Answer with [1]",
            "citations": [
                {"label": "[1]", "title": "T", "url": "u", "source_type": "internal_index", "snippet": "s"}
            ]
        })
        
        mock_response_2 = MagicMock()
        mock_response_2.choices[0].message = mock_msg_2
        
        mock_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]
        
        with patch('search_agent.search_internal', return_value=[Citation(label="[1]", title="T", url="u", source_type="internal_index", snippet="s")]):
            result = agent.run("What is AI?")
            
        assert isinstance(result, AgentAnswer)
        assert result.answer == "Answer with [1]"
        assert len(result.citations) == 1
        assert mock_log_tool.called
        assert mock_log_run.called

    def test_handle_tool_calls_all(self, mock_openai):
        agent = SearchAgent(api_key="sk-test")
        messages = []
        
        # List of tools to test
        tools_to_test = [
            ("web_search", {"query": "q"}, "tools.single_page.fetch_single_page"), # Wait, web_search is in tools.web_search
            ("search_handbook", {"query": "q"}, "tools.handbook.search_handbook"),
            ("search_internal", {"query": "q"}, "tools.internal_search.search_internal"),
            ("search_uploaded_docs", {"query": "q"}, "tools.rag_tool.search_uploaded_docs"),
            ("fetch_page", {"url": "u"}, "tools.single_page.fetch_single_page"),
            ("check_visual_entailment", {"claim_text": "c", "figure_id": "f", "doc_id": "d"}, "tools.integrity.check_visual_entailment"),
            ("extract_protocol", {"text": "t"}, "tools.integrity.extract_protocol"),
            ("run_scientific_debate", {"claim": "c", "context": "ctx"}, "tools.integrity.run_scientific_debate"),
            ("compare_papers", {"doc_ids": ["d1"], "query": "q"}, "tools.integrity.compare_papers"),
            ("generate_comparison_matrix", {"doc_ids": ["d1"], "metrics": ["m1"]}, "tools.integrity.generate_comparison_matrix"),
            ("detect_conflicts", {"topic": "t", "context": "c"}, "tools.integrity.detect_conflicts"),
            ("extract_limitations", {"doc_id": "d", "text": "t"}, "tools.integrity.extract_limitations"),
            ("export_to_bibtex", {"citations": []}, "tools.bib.export_to_bibtex"),
        ]
        
        for name, args, patch_target in tools_to_test:
            mock_call = MagicMock()
            mock_call.id = f"id_{name}"
            mock_call.function.name = name
            mock_call.function.arguments = json.dumps(args)
            
            # Special handling for tools that return lists vs strings
            with patch(patch_target, return_value=[] if "search" in name else "result"):
                 agent._handle_tool_call(mock_call, messages)
                 assert messages[-1]["role"] == "tool"
                 assert messages[-1]["tool_call_id"] == f"id_{name}"

    def test_agent_run_max_turns(self, mock_openai):
        mock_client = mock_openai.return_value
        agent = SearchAgent(api_key="sk-test")
        agent.client = mock_client
        
        # Constant tool call response
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call"
        mock_tool_call.function.name = "web_search"
        mock_tool_call.function.arguments = json.dumps({"query": "q"})
        
        mock_msg = MagicMock()
        mock_msg.tool_calls = [mock_tool_call]
        mock_msg.content = None
        
        mock_response = MagicMock()
        mock_response.choices[0].message = mock_msg
        
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('search_agent.web_search', return_value=[]):
            result = agent.run("query")
            
        assert "Maximum turn limit reached" in result.answer

    def test_agent_run_json_markdown_cleaning(self, mock_openai):
        mock_client = mock_openai.return_value
        agent = SearchAgent(api_key="sk-test")
        
        mock_msg = MagicMock()
        mock_msg.tool_calls = None
        mock_msg.content = "```json\n{\"answer\": \"Cleaned\", \"citations\": []}\n```"
        
        mock_response = MagicMock()
        mock_response.choices[0].message = mock_msg
        mock_client.chat.completions.create.return_value = mock_response
        
        result = agent.run("query")
        assert result.answer == "Cleaned"
