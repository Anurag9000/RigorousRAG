import json
from pathlib import Path

import pytest

from Crawler import Page
from Indexer import InvertedIndex
from storage import CrawlState, StorageManager


def _state(url="https://a.test/", word="alpha"):
    page = Page(
        url,
        word.title(),
        f"{word} evidence " * 100,
        [],
        "text/html",
        1400,
    )
    return CrawlState(
        pages={url: page},
        graph={url: set()},
        visited={url},
        frontier=[],
    )


def _components(url="https://a.test/", word="alpha"):
    state = _state(url, word)
    index = InvertedIndex()
    index.build(state.pages)
    pagerank = {url: 1.0}
    return state, index, pagerank


def test_round_trip_legacy_state_index_and_pagerank(tmp_path):
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


def test_manifest_snapshot_round_trip_and_consistency(tmp_path):
    manager = StorageManager(tmp_path)
    state, index, pagerank = _components()

    generation = manager.save_snapshot(state, index, pagerank)
    loaded_state, loaded_index, loaded_pagerank = manager.load_snapshot()

    assert len(generation) == 32
    assert manager.snapshot_manifest_path.exists()
    manifest = json.loads(manager.snapshot_manifest_path.read_text(encoding="utf-8"))
    assert manifest["generation"] == generation
    assert set(manifest["files"]) == {"crawl", "index", "pagerank"}
    assert set(loaded_state.pages) == set(state.pages)
    assert loaded_index is not None
    assert set(loaded_index.documents) == set(index.documents)
    assert loaded_pagerank == pagerank


def test_tampered_snapshot_member_invalidates_entire_generation(tmp_path):
    manager = StorageManager(tmp_path)
    state, index, pagerank = _components()
    manager.save_snapshot(state, index, pagerank)
    manifest = json.loads(manager.snapshot_manifest_path.read_text(encoding="utf-8"))
    index_path = tmp_path / manifest["files"]["index"]["name"]
    index_path.write_bytes(index_path.read_bytes() + b"\n")

    loaded_state, loaded_index, loaded_pagerank = manager.load_snapshot()

    assert loaded_state == CrawlState.empty()
    assert loaded_index is None
    assert loaded_pagerank == {}
    assert not manager.snapshot_manifest_path.exists()
    assert list(tmp_path.glob("snapshot_manifest.json.corrupt-*"))


def test_interrupted_new_generation_keeps_previous_manifest_authoritative(
    tmp_path,
    monkeypatch,
):
    manager = StorageManager(tmp_path)
    first_state, first_index, first_pagerank = _components()
    first_generation = manager.save_snapshot(
        first_state,
        first_index,
        first_pagerank,
    )
    second_state, second_index, second_pagerank = _components(
        "https://b.test/",
        "beta",
    )
    original_write = manager._write_bytes

    def fail_new_index(path, encoded):
        if path.name.startswith("index.") and path.name != "index.json":
            raise OSError("simulated disk failure")
        return original_write(path, encoded)

    monkeypatch.setattr(manager, "_write_bytes", fail_new_index)
    with pytest.raises(OSError, match="simulated disk failure"):
        manager.save_snapshot(second_state, second_index, second_pagerank)

    manifest = json.loads(manager.snapshot_manifest_path.read_text(encoding="utf-8"))
    loaded_state, loaded_index, loaded_pagerank = manager.load_snapshot()
    assert manifest["generation"] == first_generation
    assert set(loaded_state.pages) == {"https://a.test/"}
    assert loaded_index is not None
    assert set(loaded_index.documents) == {"https://a.test/"}
    assert loaded_pagerank == first_pagerank


def test_snapshot_rejects_cross_component_key_mismatch(tmp_path):
    manager = StorageManager(tmp_path)
    state, index, _pagerank = _components()

    with pytest.raises(ValueError, match="PageRank keys"):
        manager.save_snapshot(state, index, {"https://other.test/": 1.0})


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
