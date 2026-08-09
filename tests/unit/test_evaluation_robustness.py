from __future__ import annotations

import pytest

from evaluation.robustness import (
    DocumentVersionHit,
    counterfactual_citation_report,
    long_context_position_report,
    metadata_poisoning_report,
    stale_duplicate_report,
)


def test_counterfactual_citation_report_separates_expected_decoy_and_unknown_sources():
    report = counterfactual_citation_report(
        expected_source_ids=["truth-a", "truth-b"],
        decoy_source_ids=["decoy-a"],
        cited_source_ids=["truth-a", "decoy-a", "unknown"],
    )
    assert report.cited_count == 3
    assert report.expected_count == 1
    assert report.decoy_count == 1
    assert report.expected_citation_fraction == pytest.approx(0.5)
    assert report.decoy_citation_rate == pytest.approx(1 / 3)
    assert report.unknown_citation_rate == pytest.approx(1 / 3)


def test_counterfactual_sets_must_not_overlap():
    with pytest.raises(ValueError, match="must not overlap"):
        counterfactual_citation_report(
            expected_source_ids=["same"],
            decoy_source_ids=["same"],
            cited_source_ids=[],
        )


def test_metadata_poisoning_report_measures_overlap_and_recall_drop():
    report = metadata_poisoning_report(
        clean_ranking=["r1", "r2", "noise"],
        perturbed_ranking=["noise", "poison", "r1"],
        relevant_ids=["r1", "r2"],
        k=3,
    )
    assert report.clean_recall_at_k == 1.0
    assert report.perturbed_recall_at_k == 0.5
    assert report.recall_drop == 0.5
    assert 0.0 < report.overlap_at_k < 1.0


def test_long_context_position_report_exposes_middle_degradation_and_slope():
    report = long_context_position_report(
        {
            0: [0.9, 0.8],
            50: [0.4, 0.5],
            100: [0.8, 0.7],
        }
    )
    assert report.positions == (0, 50, 100)
    assert report.edge_to_middle_gap > 0.0
    assert report.score_range > 0.0
    assert isinstance(report.monotonic_slope, float)


def test_stale_duplicate_report_separates_duplicate_and_version_risk():
    report = stale_duplicate_report(
        [
            DocumentVersionHit("hit-a-current", "a", 3, 3),
            DocumentVersionHit("hit-a-old", "a", 2, 3),
            DocumentVersionHit("hit-b", "b", 1, 1),
            DocumentVersionHit("hit-c-old", "c", 4, 5),
        ]
    )
    assert report.result_count == 4
    assert report.unique_logical_documents == 3
    assert report.duplicate_result_rate == pytest.approx(0.25)
    assert report.stale_result_rate == pytest.approx(0.5)
    assert report.stale_logical_document_rate == pytest.approx(2 / 3)


def test_document_versions_fail_closed_on_impossible_future_hit():
    with pytest.raises(ValueError, match="may not exceed"):
        DocumentVersionHit("hit", "doc", 4, 3)
