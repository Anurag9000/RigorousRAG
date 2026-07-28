import pytest

import Indexer
from Crawler import Page
from Indexer import InvertedIndex, tokenize


def test_tokenizer_preserves_scientific_identifiers_and_unicode():
    tokens = tokenize("IL-6 increased p53, H2O, GPT-4o and α-synuclein at 37.5 C.")
    assert "il-6" in tokens
    assert "p53" in tokens
    assert "h2o" in tokens
    assert "gpt-4o" in tokens
    assert "α-synuclein" in tokens
    assert "37.5" in tokens


def test_tokenizer_rejects_non_text_and_hard_bounds_input(monkeypatch):
    assert tokenize(None) == []
    monkeypatch.setattr(Indexer, "_MAX_TEXT_CHARS", 20)
    assert "tailtoken" not in tokenize("head " * 10 + "tailtoken")


def test_build_clears_stale_documents_and_postings():
    index = InvertedIndex()
    index.build({"old": Page("old", "Old", "obsolete token", [], "text/html", 10)})
    assert "obsolete" in index.index
    index.build({"new": Page("new", "New", "fresh evidence", [], "text/html", 10)})
    assert "old" not in index.documents
    assert "obsolete" not in index.index
    assert "new" in index.documents


def test_title_terms_receive_higher_weight():
    index = InvertedIndex()
    index.build({
        "title": Page("title", "Quantum", "other words", [], "text/html", 10),
        "body": Page("body", "Other", "quantum other words", [], "text/html", 10),
    })
    assert index.index["quantum"]["title"] > index.index["quantum"]["body"]


def test_serialized_index_round_trip_recomputes_norms():
    index = InvertedIndex()
    index.build({
        "u": Page("u", "Title", "finite scientific evidence", [], "text/html", 20)
    })
    payload = index.to_dict()
    payload["doc_norms"] = {"u": float("nan")}

    loaded = InvertedIndex.from_dict(payload)

    assert loaded.documents.keys() == index.documents.keys()
    assert loaded.doc_norms["u"] > 0
    assert loaded.doc_norms["u"] == pytest.approx(index.doc_norms["u"])


def test_serialized_index_rejects_unknown_schema_and_nonfinite_weights():
    with pytest.raises(ValueError, match="schema"):
        InvertedIndex.from_dict({"schema_version": 999})

    payload = {
        "schema_version": 2,
        "documents": {"u": {"title": "T", "snippet": "S", "length": 1}},
        "idf": {"term": 1.0},
        "index": {"term": {"u": float("nan")}},
        "doc_norms": {"u": 1.0},
    }
    with pytest.raises(ValueError, match="finite"):
        InvertedIndex.from_dict(payload)

    payload["index"]["term"]["u"] = 1.0
    payload["idf"]["term"] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        InvertedIndex.from_dict(payload)


def test_serialized_index_rejects_invalid_lengths_and_empty_vectors():
    invalid_length = {
        "schema_version": 2,
        "documents": {"u": {"title": "T", "snippet": "S", "length": "bad"}},
        "idf": {},
        "index": {},
    }
    with pytest.raises(ValueError, match="document length"):
        InvertedIndex.from_dict(invalid_length)

    no_vector = {
        "schema_version": 2,
        "documents": {"u": {"title": "T", "snippet": "S", "length": 1}},
        "idf": {},
        "index": {},
    }
    with pytest.raises(ValueError, match="no usable"):
        InvertedIndex.from_dict(no_vector)


def test_index_size_limits_apply_to_build_and_load(monkeypatch):
    monkeypatch.setattr(Indexer, "_MAX_DOCUMENTS", 1)
    with pytest.raises(ValueError, match="at most 1"):
        InvertedIndex().build({
            "a": Page("a", "A", "alpha", [], "text/html", 5),
            "b": Page("b", "B", "beta", [], "text/html", 4),
        })

    payload = {
        "schema_version": 2,
        "documents": {
            "a": {"title": "A", "snippet": "", "length": 1},
            "b": {"title": "B", "snippet": "", "length": 1},
        },
        "idf": {},
        "index": {},
    }
    with pytest.raises(ValueError, match="too many documents"):
        InvertedIndex.from_dict(payload)
