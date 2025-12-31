import pytest
import argparse
from unittest.mock import MagicMock, patch, seal
from Searching import AcademicSearchEngine, SearchHit, parse_args, main
from Crawler import Page
from storage import CrawlState

class TestSearchingExtended:
    @pytest.fixture
    def mock_storage(self):
        with patch('Searching.StorageManager') as mock:
            instance = mock.return_value
            instance.load_crawl_state.return_value = CrawlState.empty()
            instance.load_index.return_value = None
            instance.load_pagerank.return_value = {}
            yield instance

    @pytest.fixture
    def engine(self, mock_storage):
        with patch('Searching.AcademicCrawler'):
            return AcademicSearchEngine(seeds=["http://seed.com"], max_pages=10)

    def test_build(self, engine):
        # Setup mocks for build
        mock_page = Page("http://a.com", "Title", "Text", [], "text/html", 100)
        engine.crawler.crawl.return_value = CrawlState(
            pages={"http://a.com": mock_page},
            graph={"http://a.com": set()},
            visited={"http://a.com"},
            frontier=[]
        )
        
        with patch('Searching.compute_pagerank', return_value={"http://a.com": 1.0}):
            total = engine.build()
            
            assert total == 1
            assert engine.storage.save_crawl_state.called
            assert engine.storage.save_index.called
            assert engine.storage.save_pagerank.called
            assert engine.pagerank_scores["http://a.com"] == 1.0

    def test_search_no_tokens(self, engine):
        assert engine.search("") == []
        assert engine.search("   ") == []

    def test_search_missing_terms(self, engine):
        # Mock index with some data
        engine.index.idf = {"known": 1.0}
        engine.index.index = {"known": {"http://a.com": 0.5}}
        engine.index.doc_norms = {"http://a.com": 1.0}
        
        # Search for unknown term
        assert engine.search("unknown") == []

    def test_gather_context(self, engine):
        engine.pages = {
            "http://a.com": Page("http://a.com", "A", "Content A", [], "text/html", 9),
            "http://b.com": Page("http://b.com", "B", "Content B", [], "text/html", 9)
        }
        hits = [
            SearchHit(1, "http://a.com", "A", "snip", 0.9, 0.8, 0.1, 9),
            SearchHit(2, "http://b.com", "B", "snip", 0.7, 0.6, 0.1, 9)
        ]
        
        context = engine.gather_context(hits, max_chars=10)
        assert len(context) >= 1
        assert context[0]["url"] == "http://a.com"
        # Check truncation (max_chars // max(1, len(hits)) = 10 // 2 = 5)
        assert context[0]["text"] == "Conte"

    @patch('builtins.input')
    def test_interactive_loop_exit(self, mock_input, engine):
        mock_input.side_effect = [""]
        # Should exit immediately
        engine.interactive_loop()
        assert mock_input.called

    @patch('builtins.input')
    @patch('builtins.print')
    def test_interactive_loop_query(self, mock_print, mock_input, engine):
        mock_input.side_effect = ["test", ""]
        engine.search = MagicMock(return_value=[
            SearchHit(1, "http://a.com", "Title A", "Snippet A", 0.9, 0.8, 0.1, 100)
        ])
        engine.interactive_loop()
        # Verify output contains rank and title
        mock_print.assert_any_call("1. Title A")

    def test_parse_args(self):
        with patch('sys.argv', ['Searching.py', '--max-pages', '50', '--max-depth', '1']):
            args = parse_args()
            assert args.max_pages == 50
            assert args.max_depth == 1

    @patch('Searching.AcademicSearchEngine')
    @patch('Searching.parse_args')
    def test_main(self, mock_args, mock_engine_class):
        mock_args.return_value = argparse.Namespace(max_pages=10, max_depth=1, delay=0, results=5)
        mock_engine = mock_engine_class.return_value
        mock_engine.build.return_value = 10
        
        with patch('builtins.print'):
            main()
            
        assert mock_engine.build.called
        assert mock_engine.interactive_loop.called
