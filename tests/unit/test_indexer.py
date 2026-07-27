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


def test_serialized_index_rejects_unknown_schema():
    try:
        InvertedIndex.from_dict({"schema_version": 999})
    except ValueError as exc:
        assert "schema" in str(exc).lower()
    else:
        raise AssertionError("Unknown index schemas must be rejected.")
