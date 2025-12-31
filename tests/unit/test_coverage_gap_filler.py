import pytest
import json
from unittest.mock import MagicMock, patch
from Crawler import AcademicCrawler, Page, is_trusted_domain, normalize_url
from tools.rag_tool import search_uploaded_docs
from tools.single_page import fetch_single_page
from Searching import AcademicSearchEngine, SearchHit

class TestCoverageGapFiller:
    
    # --- Crawler Gaps ---
    def test_crawler_helpers(self):
        # normalize_url gaps
        assert normalize_url("ftp://site.com") == ""
        assert normalize_url("http://") == ""
        # is_trusted_domain gaps
        assert is_trusted_domain("relative", ["a.com"]) is False

    def test_crawler_extract_title_variants(self):
        crawler = AcademicCrawler()
        from bs4 import BeautifulSoup
        
        # Test no title but H1
        soup1 = BeautifulSoup("<html><body><h1>Heading 1</h1></body></html>", "html.parser")
        assert crawler._extract_title(soup1) == "Heading 1"
        
        # Test nothing
        soup2 = BeautifulSoup("<html><body></body></html>", "html.parser")
        assert crawler._extract_title(soup2) == "Untitled"

    @patch('Crawler.robotparser.RobotFileParser')
    def test_crawler_robots_errors(self, mock_rp_class):
        crawler = AcademicCrawler()
        mock_rp = mock_rp_class.return_value
        
        # Set can_fetch to return a boolean
        mock_rp.can_fetch.return_value = True
        
        # Test exception in read()
        mock_rp.read.side_effect = Exception("Network Error")
        allowed = crawler._is_allowed_by_robots("http://err.com/p")
        assert allowed is True
        
        # Test exception in can_fetch()
        mock_rp.can_fetch.side_effect = Exception("Parsing Error")
        allowed_2 = crawler._is_allowed_by_robots("http://err.com/p2")
        assert allowed_2 is True

    def test_crawler_link_filtering(self):
        crawler = AcademicCrawler(allowed_domains=["a.com"])
        from bs4 import BeautifulSoup
        
        # Mock robots to allow everything for this test
        crawler._is_allowed_by_robots = MagicMock(return_value=True)
        
        html = """
        <a href="http://a.com/ok">OK</a>
        <a href="">Empty</a>
        <a href="http://a.com">Same as base</a>
        <a href="http://b.com/external">External</a>
        """
        soup = BeautifulSoup(html, "html.parser")
        links = crawler._extract_links("http://a.com", soup)
        
        assert "http://a.com/ok" in links
        assert "http://b.com/external" not in links # because of allowed_domains
        assert len(links) == 1

    # --- RAG Tool Gaps ---
    @patch('tools.rag_tool.get_rag_layer')
    def test_rag_tool_parent_context(self, mock_get_rag):
        mock_rag = mock_get_rag.return_value
        # Mock a chunk with parent_text
        chunk = MagicMock()
        chunk.text = "child text"
        chunk.metadata = {"parent_text": "large parent text", "filename": "test.pdf", "doc_id": "1"}
        mock_rag.query.return_value = [chunk]
        
        citations = search_uploaded_docs("query")
        assert "large parent text" in citations[0].snippet

    # --- Single Page Gaps ---
    @patch('tools.single_page.requests.get')
    def test_fetch_single_page_errors(self, mock_get):
        # Test bad status code
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        # raise_for_status should raise for 404
        mock_resp.raise_for_status.side_effect = Exception("404 Client Error")
        mock_get.return_value = mock_resp
        
        page = fetch_single_page("http://404.com")
        assert "404" in page.error
        
        # Test exception
        mock_get.side_effect = Exception("Connection Failed")
        page_2 = fetch_single_page("http://fail.com")
        assert "Connection Failed" in page_2.error

    # --- Searching Gaps ---
    @patch('builtins.input')
    def test_searching_interactive_interrupt(self, mock_input):
        engine = AcademicSearchEngine()
        # Test KeyboardInterrupt
        mock_input.side_effect = KeyboardInterrupt
        with patch('builtins.print'):
            engine.interactive_loop()
            
        # Test EOFError
        mock_input.side_effect = EOFError
        with patch('builtins.print'):
            engine.interactive_loop()

    # --- LLM Agent Gaps ---
    def test_llm_agent_no_clients(self):
        from llm_agent import LLMAgent
        # Patch the class-level imports within the instance or mock the client
        agent = LLMAgent(api_key=None)
        agent.client = None
        
        # Test summarise with no client
        # Requires query, hits, contexts
        res = agent.summarise("q", [], [])
        assert "no supporting documents" in res.summary.lower()

    def test_llm_agent_specific_fallbacks(self):
        from llm_agent import LLMAgent
        agent = LLMAgent()
        agent.client = MagicMock()
        
        # Test OpenAI failure -> fallback
        agent.client.chat.completions.create.side_effect = Exception("OpenAI Failed")
        hit = SearchHit(
            url="u", title="t", score=1.0, rank=1,
            snippet="", cosine=0.0, pagerank=0.0, length=0
        )
        res = agent.summarise("q", [hit], [{"text": "content"}])
        assert "content" in res.summary # Should use extractive fallback
        
        # Test Ollama explicitly
        agent.model = "ollama/llama3"
        with patch('llm_agent.ollama') as mock_ollama:
            mock_ollama.chat.side_effect = Exception("Ollama Failed")
            res2 = agent.summarise("q", [hit], [{"text": "item"}])
            assert "item" in res2.summary
