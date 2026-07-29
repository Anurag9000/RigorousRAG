import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import Searching
from Crawler import Page
from Indexer import InvertedIndex
from Searching import AcademicSearchEngine, SearchHit
from storage import CrawlState, StorageManager


def make_engine(tmp_path):
    return AcademicSearchEngine(
        seeds=["https://seed.test"],
        max_pages=10,
        request_delay=0,
        storage_dir=str(tmp_path),
    )


def test_build_filters_graph_and_commits_one_snapshot(tmp_path):
    engine = make_engine(tmp_path)
    pages = {
        "https://a.test/": Page(
            "https://a.test/",
            "A",
            "alpha " * 200,
            ["https://missing.test/"],
            "text/html",
            1000,
        )
    }
    engine.crawler.crawl = MagicMock(
        return_value=CrawlState(
            pages=pages,
            graph={"https://a.test/": {"https://missing.test/"}},
            visited=set(pages),
            frontier=[],
        )
    )
    assert engine.build() == 1
    assert engine.state.graph == {"https://a.test/": set()}
    assert set(engine.pagerank_scores) == {"https://a.test/"}
    assert engine.storage.snapshot_manifest_path.exists()
    assert not engine.storage.crawl_path.exists()
    assert not engine.storage.index_path.exists()
    assert not engine.storage.pagerank_path.exists()

    reloaded = make_engine(tmp_path)
    assert set(reloaded.pages) == {"https://a.test/"}
    assert set(reloaded.index.documents) == {"https://a.test/"}
    assert reloaded.pagerank_scores == {"https://a.test/": 1.0}


def test_invalid_crawler_state_is_not_published(tmp_path):
    engine = make_engine(tmp_path)
    engine.crawler.crawl = MagicMock(return_value=object())

    with pytest.raises(RuntimeError, match="invalid state"):
        engine.build()

    assert not engine.storage.snapshot_manifest_path.exists()


def test_partial_legacy_generation_forces_rebuild(tmp_path):
    manager = StorageManager(tmp_path)
    url = "https://a.test/"
    state = CrawlState(
        pages={url: Page(url, "A", "alpha evidence", [], "text/html", 14)},
        graph={url: set()},
        visited={url},
        frontier=[],
    )
    index = InvertedIndex()
    index.build(state.pages)
    manager.save_crawl_state(state)
    manager.save_index(index)

    engine = make_engine(tmp_path)

    assert engine.ready is False
    assert engine.pages == {}
    assert engine.pagerank_scores == {}


def test_mismatched_legacy_generation_forces_rebuild(tmp_path):
    manager = StorageManager(tmp_path)
    url = "https://a.test/"
    state = CrawlState(
        pages={url: Page(url, "A", "alpha evidence", [], "text/html", 14)},
        graph={url: set()},
        visited={url},
        frontier=[],
    )
    index = InvertedIndex()
    index.build(state.pages)
    manager.save_crawl_state(state)
    manager.save_index(index)
    manager.save_pagerank({"https://other.test/": 1.0})

    engine = make_engine(tmp_path)

    assert engine.ready is False
    assert engine.pages == {}
    assert engine.pagerank_scores == {}


def test_query_snippet_centres_first_matching_term(tmp_path):
    engine = make_engine(tmp_path)
    text = "prefix " * 100 + "target evidence follows here " + "suffix " * 100
    engine.pages = {"u": Page("u", "Title", text, [], "text/html", len(text))}
    engine.index.build(engine.pages)
    hit = engine.search("target", limit=1)[0]
    assert "target evidence" in hit.snippet
    assert len(hit.snippet) < len(text)


def test_search_hit_masks_private_metadata_and_rejects_boolean_scores():
    hit = SearchHit(
        1,
        "https://alice:password@example.test?api_key=secret",
        "Report at /private/report.txt",
        "file:///private/source.txt",
        0.9,
        0.8,
        0.1,
        10,
    )
    rendered = hit.url + hit.title + hit.snippet
    assert "password" not in rendered
    assert "api_key=secret" not in rendered
    assert "/private" not in rendered

    with pytest.raises(ValueError, match="numeric"):
        SearchHit(1, "u", "t", "s", True, 0.8, 0.1, 10)


def test_gather_context_never_mislabels_missing_pages(tmp_path):
    engine = make_engine(tmp_path)
    engine.pages = {
        "present": Page("present", "Present", "correct text", [], "text/html", 12)
    }
    hits = [
        SearchHit(1, "missing", "Missing", "", 1, 1, 0, 0),
        SearchHit(2, "present", "Present", "", 1, 1, 0, 0),
    ]
    contexts = engine.gather_context(hits, max_chars=100)
    assert contexts == [
        {"url": "present", "title": "Present", "text": "correct text"}
    ]


def test_gather_context_rejects_hostile_iterators(tmp_path):
    class BrokenHits:
        def __iter__(self):
            raise RuntimeError("private iterator detail")

    engine = make_engine(tmp_path)
    with pytest.raises(ValueError, match="safely iterable"):
        engine.gather_context(BrokenHits())


def test_seed_collection_requires_valid_url_strings(tmp_path):
    for seeds in (
        [],
        [object()],
        ["file:///private/source"],
        ["https://alice:password@example.test"],
    ):
        with pytest.raises(ValueError):
            AcademicSearchEngine(
                seeds=seeds,
                request_delay=0,
                storage_dir=str(tmp_path / "state"),
            )

    class BrokenSeeds:
        def __iter__(self):
            raise RuntimeError("private iterator detail")

    with pytest.raises(ValueError, match="safely iterable"):
        AcademicSearchEngine(
            seeds=BrokenSeeds(),
            request_delay=0,
            storage_dir=str(tmp_path / "other"),
        )


def test_boolean_numeric_configuration_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="numeric"):
        AcademicSearchEngine(
            seeds=["https://seed.test"],
            lexical_weight=True,
            request_delay=0,
            storage_dir=str(tmp_path),
        )


def test_interactive_loop_exits_cleanly(tmp_path):
    engine = make_engine(tmp_path)
    with patch("builtins.input", return_value=""):
        engine.interactive_loop()


def test_interactive_loop_recovers_from_backend_failure(tmp_path, capsys):
    engine = make_engine(tmp_path)
    inputs = iter(["query", ""])
    engine.search = MagicMock(side_effect=RuntimeError("private backend detail"))
    with patch("builtins.input", side_effect=lambda _prompt: next(inputs)):
        engine.interactive_loop()

    output = capsys.readouterr().out
    assert "backend failed" in output
    assert "private backend detail" not in output


def test_main_returns_generic_initialization_failure(monkeypatch, capsys):
    args = argparse.Namespace(
        max_pages=10,
        max_depth=1,
        delay=0,
        results=10,
        storage_dir="/private/state",
        rebuild=False,
    )
    monkeypatch.setattr(Searching, "parse_args", lambda _argv=None: args)
    monkeypatch.setattr(
        Searching,
        "AcademicSearchEngine",
        MagicMock(side_effect=RuntimeError("failed at /private/state")),
    )

    assert Searching.main([]) == 1
    error = capsys.readouterr().err
    assert "could not be initialized" in error
    assert "/private" not in error


def test_main_returns_usage_error_for_invalid_configuration(monkeypatch, capsys):
    args = SimpleNamespace(
        max_pages=0,
        max_depth=1,
        delay=0,
        results=10,
        storage_dir="data",
        rebuild=False,
    )
    monkeypatch.setattr(Searching, "parse_args", lambda _argv=None: args)

    assert Searching.main([]) == 2
    assert "between" in capsys.readouterr().err
