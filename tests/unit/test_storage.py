from pathlib import Path

from Crawler import Page
from Indexer import InvertedIndex
from storage import CrawlState, StorageManager


def test_round_trip_state_index_and_pagerank(tmp_path):
    manager = StorageManager(tmp_path)
    state = CrawlState(
        pages={"https://a.test/": Page("https://a.test/", "A", "text", [], "text/html", 4)},
        graph={"https://a.test/": set()},
        visited={"https://a.test/"},
        frontier=[("https://b.test/", 1)],
    )
    manager.save_crawl_state(state)
    loaded = manager.load_crawl_state()
    assert loaded.pages["https://a.test/"].title == "A"
    assert loaded.frontier == [("https://b.test/", 1)]

    index = InvertedIndex()
    index.build(state.pages)
    manager.save_index(index)
    assert manager.load_index().documents.keys() == index.documents.keys()

    manager.save_pagerank({"https://a.test/": 1.0})
    assert manager.load_pagerank() == {"https://a.test/": 1.0}


def test_corrupt_json_is_quarantined_instead_of_crashing(tmp_path):
    manager = StorageManager(tmp_path)
    manager.crawl_path.write_text("{not json", encoding="utf-8")
    assert manager.load_crawl_state() == CrawlState.empty()
    quarantined = list(Path(tmp_path).glob("crawl_state.json.corrupt-*"))
    assert len(quarantined) == 1
    assert not manager.crawl_path.exists()


def test_atomic_write_leaves_no_temporary_files(tmp_path):
    manager = StorageManager(tmp_path)
    manager.save_pagerank({"a": 1.0})
    assert manager.pagerank_path.exists()
    assert not list(Path(tmp_path).glob(".*.tmp"))
