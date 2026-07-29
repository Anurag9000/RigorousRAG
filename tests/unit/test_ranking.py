import pytest

import Pagerank
from Crawler import Page
from Pagerank import compute_pagerank
from Searching import AcademicSearchEngine


def test_pagerank_validates_parameters():
    for kwargs in (
        {"damping": 1.0},
        {"damping": float("nan")},
        {"damping": True},
        {"iterations": 0},
        {"iterations": "bad"},
        {"tolerance": float("inf")},
        {"tolerance": -1.0},
        {"tolerance": False},
    ):
        with pytest.raises(ValueError):
            compute_pagerank({"a": {"b"}}, **kwargs)


def test_pagerank_rejects_invalid_and_unbounded_graph_values(monkeypatch):
    with pytest.raises(ValueError, match="mapping"):
        compute_pagerank([("a", ["b"])])
    with pytest.raises(ValueError, match="may not be strings"):
        compute_pagerank({"a": "b"})
    with pytest.raises(ValueError, match="identifiers must be strings"):
        compute_pagerank({"a": [1]})

    monkeypatch.setattr(Pagerank, "_MAX_EDGES_PER_NODE", 3)

    def infinite_targets():
        index = 0
        while True:
            yield f"target-{index}"
            index += 1

    with pytest.raises(ValueError, match="at most 3"):
        compute_pagerank({"a": infinite_targets()})


def test_hostile_graph_and_target_iterators_fail_safely():
    class BrokenMapping(dict):
        def items(self):
            raise RuntimeError("private graph detail")

    class BrokenTargets:
        def __iter__(self):
            raise RuntimeError("private target detail")

    with pytest.raises(ValueError, match="safely iterable mapping"):
        compute_pagerank(BrokenMapping())
    with pytest.raises(ValueError, match="safely iterable"):
        compute_pagerank({"a": BrokenTargets()})


def test_node_identifiers_reject_whitespace_controls_and_oversize():
    for graph in (
        {" a": {"b"}},
        {"a ": {"b"}},
        {"a\n": {"b"}},
        {"a": {"b\x00hidden"}},
        {"x" * (Pagerank._MAX_NODE_CHARS + 1): set()},
    ):
        with pytest.raises(ValueError):
            compute_pagerank(graph)


def test_duplicate_targets_are_deduplicated_before_scoring():
    scores = compute_pagerank({"a": ["b", "b", "b"], "b": ["a"]})
    assert sum(scores.values()) == pytest.approx(1.0)
    assert set(scores) == {"a", "b"}


def test_global_edge_and_node_limits_fail_closed(monkeypatch):
    monkeypatch.setattr(Pagerank, "_MAX_EDGES", 1)
    with pytest.raises(ValueError, match="at most 1 edges"):
        compute_pagerank({"a": ["b", "c"]})

    monkeypatch.setattr(Pagerank, "_MAX_NODES", 2)
    with pytest.raises(ValueError, match="at most 2 nodes"):
        compute_pagerank({"a": ["b", "c"]})


def test_pagerank_converges_and_normalizes():
    scores = compute_pagerank({"a": {"b"}, "b": {"a"}, "sink": set()})
    assert sum(scores.values()) == pytest.approx(1.0)
    assert all(value >= 0 for value in scores.values())
    assert list(scores) == sorted(scores)


def test_authority_prior_is_normalized_to_comparable_scale(tmp_path):
    with AcademicSearchEngine(
        seeds=["https://example.test"],
        request_delay=0,
        storage_dir=str(tmp_path),
        lexical_weight=0.5,
    ) as engine:
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


def test_search_rejects_invalid_limits_queries_and_nonfinite_weight(tmp_path):
    with AcademicSearchEngine(
        storage_dir=str(tmp_path),
        request_delay=0,
    ) as engine:
        engine.pages = {"a": Page("a", "Alpha", "alpha", [], "text/html", 10)}
        engine.index.build(engine.pages)
        engine.pagerank_scores = {"a": 1.0}
        with pytest.raises(ValueError, match="between 1 and 100"):
            engine.search("alpha", limit=10_000)
        with pytest.raises(ValueError, match="integer"):
            engine.search("alpha", limit="bad")
        with pytest.raises(ValueError, match="2,000"):
            engine.search("q" * 2001)

    with pytest.raises(ValueError, match="finite"):
        AcademicSearchEngine(
            storage_dir=str(tmp_path / "other"),
            request_delay=0,
            lexical_weight=float("nan"),
        )
