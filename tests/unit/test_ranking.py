import pytest
from unittest.mock import MagicMock, patch
from Searching import AcademicSearchEngine
from Crawler import Page
from Pagerank import compute_pagerank

class TestRanking:
    @pytest.fixture
    @patch('Searching.StorageManager')
    def engine(self, mock_storage_cls):
        # Setup mocks so init doesn't fail loading files
        mock_instance = mock_storage_cls.return_value
        mock_instance.load_crawl_state.return_value = MagicMock(pages={}, graph={}, visited=set(), frontier=[])
        mock_instance.load_index.return_value = None
        mock_instance.load_pagerank.return_value = {}
        
        return AcademicSearchEngine()

    def test_search_scoring(self, engine):
        # We need to populate the engine's index manually
        # Note: Indexer.py doesn't have add_document, it uses build()
        p1 = Page(url="u1", title="T1", text="alpha beta", links=[], content_type="text", content_length=10)
        
        # Populate manually or use helper if exists.
        # Engine has .pages and .index
        engine.pages = {"u1": p1}
        engine.index.build(engine.pages)
        
        # Mock pagerank checks in scoring if needed
        # search() method uses self.pagerank_scores
        engine.pagerank_scores = {"u1": 1.0}
        
        results = engine.search("alpha")
        assert len(results) > 0
        assert results[0].url == "u1"
        assert results[0].score > 0

    def test_pagerank_calculation(self):
        graph = {
            "A": ["B", "C"],
            "B": ["C"],
            "C": ["A"],
        }
        scores = compute_pagerank(graph, damping=0.85, iterations=10)
        assert abs(sum(scores.values()) - 1.0) < 0.001
        assert scores["C"] > scores["B"]
