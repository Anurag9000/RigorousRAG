from unittest.mock import patch

import pytest

from tools.integrity import compare_papers, generate_comparison_matrix


def test_compare_papers_rejects_infinite_document_iterable():
    def document_ids():
        index = 0
        while True:
            yield f"doc-{index}"
            index += 1

    with pytest.raises(ValueError, match="at most 10"):
        compare_papers(document_ids(), "accuracy", owner_id="alice")


def test_comparison_matrix_rejects_infinite_metric_iterable():
    def metrics():
        index = 0
        while True:
            yield f"metric-{index}"
            index += 1

    with pytest.raises(ValueError, match="at most 12"):
        generate_comparison_matrix(["doc-1"], metrics(), owner_id="alice")


def test_comparison_inputs_reject_strings_instead_of_iterating_characters():
    with pytest.raises(ValueError, match="array, not a string"):
        compare_papers("doc-1", "accuracy", owner_id="alice")
    with pytest.raises(ValueError, match="array, not a string"):
        generate_comparison_matrix(["doc-1"], "accuracy", owner_id="alice")


def test_comparison_item_and_query_lengths_are_enforced_before_retrieval():
    with patch(
        "tools.integrity._retrieve_document_evidence",
        side_effect=AssertionError("retrieval should not run"),
    ):
        with pytest.raises(ValueError, match="200"):
            compare_papers(["d" * 201, "doc-2"], "accuracy", owner_id="alice")
        with pytest.raises(ValueError, match="10,000"):
            compare_papers(["doc-1", "doc-2"], "q" * 10_001, owner_id="alice")
        with pytest.raises(ValueError, match="500"):
            generate_comparison_matrix(
                ["doc-1"],
                ["m" * 501],
                owner_id="alice",
            )


def test_valid_bounded_comparison_reaches_preserved_implementation(monkeypatch):
    captured = []

    def fake_compare(doc_ids, query, **kwargs):
        captured.append((doc_ids, query, kwargs))
        return "comparison"

    monkeypatch.setattr("tools.integrity_legacy.compare_papers", fake_compare)
    # The wrapper captures the original implementation at import, so patch the
    # preserved callable used by the boundary directly.
    monkeypatch.setattr("tools.integrity._original_compare_papers", fake_compare, raising=False)

    result = compare_papers(
        ["doc-1", "doc-1", "doc-2"],
        "accuracy",
        owner_id="alice",
        client=None,
    )

    assert result == "comparison"
    assert captured[0][0] == ["doc-1", "doc-2"]
    assert captured[0][1] == "accuracy"
    assert captured[0][2]["owner_id"] == "alice"
