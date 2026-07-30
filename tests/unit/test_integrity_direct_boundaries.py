from unittest.mock import patch

import pytest

from tools.integrity import (
    check_visual_entailment,
    compare_papers,
    detect_conflicts,
    extract_limitations,
    extract_protocol,
    generate_comparison_matrix,
    run_scientific_debate,
)


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


def test_comparison_inputs_reject_strings_and_non_string_items():
    with pytest.raises(ValueError, match="array, not a string"):
        compare_papers("doc-1", "accuracy", owner_id="alice")
    with pytest.raises(ValueError, match="array, not a string"):
        generate_comparison_matrix(["doc-1"], "accuracy", owner_id="alice")
    with pytest.raises(ValueError, match="must be a string"):
        compare_papers([object(), "doc-2"], "accuracy", owner_id="alice")
    with pytest.raises(ValueError, match="at least 2"):
        compare_papers(["doc-1", "doc-1"], "accuracy", owner_id="alice")


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


def test_all_scientific_text_and_identity_limits_are_enforced_before_work():
    with pytest.raises(ValueError, match="claim_text"):
        check_visual_entailment("", "Figure 1", "doc-1", owner_id="alice")
    with pytest.raises(ValueError, match="figure_id"):
        check_visual_entailment("claim", "f" * 201, "doc-1", owner_id="alice")
    with pytest.raises(ValueError, match="Owner identifiers"):
        check_visual_entailment(
            "claim",
            "Figure 1",
            "doc-1",
            owner_id="../other",
        )
    with pytest.raises(ValueError, match="model"):
        extract_protocol("methods", model="m" * 201)
    with pytest.raises(ValueError, match="10,000"):
        run_scientific_debate("c" * 10_001, "context")
    with pytest.raises(ValueError, match="35,000"):
        detect_conflicts("topic", "x" * 35_001)
    with pytest.raises(ValueError, match="doc_id"):
        extract_limitations("", owner_id="alice")


def test_empty_optional_contexts_preserve_conservative_fallbacks():
    assert "No methods text" in extract_protocol("")
    assert "No evidence context" in run_scientific_debate("claim", "")
    assert "No evidence context" in detect_conflicts("topic", "")
