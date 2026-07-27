import pytest

from Crawler import Page
from Pagerank import compute_pagerank
from Searching import AcademicSearchEngine


def test_pagerank_validates_parameters():
    with pytest.raises(ValueError):
        compute_pagerank({"a": {"b"}}, damping=1.0)
    with pytest.raises(ValueError):
        compute_pagerank({"a": {"b"}}, iterations=0)


def test_pagerank_converges_and_normalizes():
    scores = compute_pagerank({"a": {"b"}, "b": {"a"}, "sink": set()})
    assert sum(scores.values()) == pytest.approx(1.0)
    assert all(value >= 0 for value in scores.values())


def test_authority_prior_is_normalized_to_comparable_scale(tmp_path):
    engine = AcademicSearchEngine(
        seeds=["https://example.test"],
        request_delay=0,
        storage_dir=str(tmp_path),
        lexical_weight=0.5,
    )
    engine.pages = {
        "low": Page("low", "Alpha", "alpha evidence", [], "text/html", 10),
        "high": Page("high", "Alpha", "alpha evidence", [], "text/html", 10),
    }
    engine.index.build(engine.pages)
    engine.pagerank_scores = {"low": 0.001, "high": 0.1}
    hits = engine.search("alpha", limit=2)
    assert [hit.url for hit in hits] == ["high", "low"]
    assert hits[0].pagerank == pytest.approx(1.0)
    assert hits[1].pagerank == pytest.approx(0.01)


def test_search_limit_is_bounded(tmp_path):
    engine = AcademicSearchEngine(storage_dir=str(tmp_path), request_delay=0)
    engine.pages = {"a": Page("a", "Alpha", "alpha", [], "text/html", 10)}
    engine.index.build(engine.pages)
    assert len(engine.search("alpha", limit=10000)) == 1
