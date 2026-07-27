from unittest.mock import MagicMock, patch

from Crawler import Page
from Searching import AcademicSearchEngine, SearchHit
from storage import CrawlState


def make_engine(tmp_path):
    return AcademicSearchEngine(
        seeds=["https://seed.test"],
        max_pages=10,
        request_delay=0,
        storage_dir=str(tmp_path),
    )


def test_build_filters_graph_to_fetched_pages(tmp_path):
    engine = make_engine(tmp_path)
    pages = {
        "https://a.test/": Page(
            "https://a.test/", "A", "alpha " * 200,
            ["https://missing.test/"], "text/html", 1000,
        )
    }
    engine.crawler.crawl = MagicMock(return_value=CrawlState(
        pages=pages,
        graph={"https://a.test/": {"https://missing.test/"}},
        visited=set(pages),
        frontier=[],
    ))
    assert engine.build() == 1
    assert engine.state.graph == {"https://a.test/": set()}
    assert set(engine.pagerank_scores) == {"https://a.test/"}


def test_query_snippet_centres_first_matching_term(tmp_path):
    engine = make_engine(tmp_path)
    text = "prefix " * 100 + "target evidence follows here " + "suffix " * 100
    engine.pages = {"u": Page("u", "Title", text, [], "text/html", len(text))}
    engine.index.build(engine.pages)
    hit = engine.search("target", limit=1)[0]
    assert "target evidence" in hit.snippet
    assert len(hit.snippet) < len(text)


def test_gather_context_never_mislabels_missing_pages(tmp_path):
    engine = make_engine(tmp_path)
    engine.pages = {"present": Page("present", "Present", "correct text", [], "text/html", 12)}
    hits = [
        SearchHit(1, "missing", "Missing", "", 1, 1, 0, 0),
        SearchHit(2, "present", "Present", "", 1, 1, 0, 0),
    ]
    contexts = engine.gather_context(hits, max_chars=100)
    assert contexts == [{"url": "present", "title": "Present", "text": "correct text"}]


def test_interactive_loop_exits_cleanly(tmp_path):
    engine = make_engine(tmp_path)
    with patch("builtins.input", return_value=""):
        engine.interactive_loop()
