import pytest
from Indexer import InvertedIndex
from Crawler import Page

class TestInvertedIndex:

    @pytest.fixture
    def index(self):
        return InvertedIndex()

    def test_add_document(self, index):
        page = Page(url="http://example.com", title="Test Page",
                    text="Hello world world", links=[], content_type="text", content_length=100)
        
        # Need to put pages in a dict and call build() because add_document doesn't exist?
        # Checking Indexer.py: It has .build(pages). It does NOT have add_document.
        # So my previous test assuming add_document was wrong too.
        
        pages = {"http://example.com": page}
        index.build(pages)
        
        assert "hello" in index.index
        assert "http://example.com" in index.index["hello"]
        assert index.documents["http://example.com"] is not None

    def test_to_and_from_dict(self, index):
        page = Page(url="http://foo.com", title="Foo",
                    text="Bar baz", links=[], content_type="text", content_length=10)
        index.build({"http://foo.com": page})
        
        data = index.to_dict()
        assert "documents" in data
        assert "index" in data
        
        new_index = InvertedIndex.from_dict(data)
        assert "bar" in new_index.index
        assert "http://foo.com" in new_index.index["bar"]

    def test_idf_calculation(self, index):
        p1 = Page(url="1", title="A", text="term", links=[], content_type="text", content_length=10)
        p2 = Page(url="2", title="B", text="term", links=[], content_type="text", content_length=10)
        
        index.build({"1": p1, "2": p2})
        # Build calculates IDF
        assert "term" in index.idf
        assert index.idf["term"] > 0
