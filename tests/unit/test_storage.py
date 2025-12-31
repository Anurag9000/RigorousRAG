import os
import json
import pytest
from storage import StorageManager, CrawlState
from Indexer import InvertedIndex

class TestStorage:
    @pytest.fixture
    def storage_dir(self, tmp_path):
        return tmp_path / "storage"

    @pytest.fixture
    def manager(self, storage_dir):
        return StorageManager(base_dir=str(storage_dir))

    def test_crawl_state_empty(self):
        state = CrawlState.empty()
        assert len(state.visited) == 0
        assert len(state.frontier) == 0
        assert len(state.pages) == 0
        assert len(state.graph) == 0

    def test_save_load_crawl_state(self, manager, storage_dir):
        from Crawler import Page
        p1 = Page(url="http://a.com", title="A", text="T", links=[], content_type="text/html", content_length=1)
        state = CrawlState(
            pages={"http://a.com": p1},
            graph={"http://a.com": {"http://b.com"}},
            visited={"http://a.com"},
            frontier=[("http://b.com", 1)]
        )
        manager.save_crawl_state(state)
        
        loaded = manager.load_crawl_state()
        assert "http://a.com" in loaded.pages
        assert loaded.pages["http://a.com"].title == "A"
        assert loaded.graph["http://a.com"] == {"http://b.com"}
        assert "http://a.com" in loaded.visited
        assert loaded.frontier == [("http://b.com", 1)]

    def test_load_nonexistent_crawl_state(self, manager):
        loaded = manager.load_crawl_state()
        assert len(loaded.visited) == 0
        assert len(loaded.frontier) == 0
        assert len(loaded.pages) == 0
        assert len(loaded.graph) == 0

    def test_save_load_index(self, manager):
        index = InvertedIndex()
        index.index = {"term": {"doc1": 0.5}}
        manager.save_index(index)
        
        loaded = manager.load_index()
        assert loaded.index == {"term": {"doc1": 0.5}}

    def test_load_nonexistent_index(self, manager):
        loaded = manager.load_index()
        assert loaded is None

    def test_save_load_pagerank(self, manager):
        pr = {"doc1": 0.1, "doc2": 0.9}
        manager.save_pagerank(pr)
        
        loaded = manager.load_pagerank()
        assert loaded == pr

    def test_load_nonexistent_pagerank(self, manager):
        loaded = manager.load_pagerank()
        assert len(loaded) == 0
