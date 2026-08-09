from __future__ import annotations

import pytest

from evaluation.expert_review import (
    ExpertReviewItem,
    expert_review_report,
    pairwise_cohens_kappa,
)


def test_review_report_measures_agreement_entropy_and_adjudication():
    items = [
        ExpertReviewItem(
            "a",
            {"r1": "supported", "r2": "supported", "r3": "supported"},
            adjudicated_label="supported",
            adjudication_confidence=0.95,
        ),
        ExpertReviewItem(
            "b",
            {"r1": "supported", "r2": "unsupported", "r3": "supported"},
            adjudicated_label="supported",
            adjudication_confidence=0.8,
        ),
        ExpertReviewItem(
            "c",
            {"r1": "unknown", "r2": "unknown", "r3": "supported"},
        ),
    ]
    report = expert_review_report(items)
    assert report.item_count == 3
    assert 0.0 < report.mean_agreement < 1.0
    assert report.mean_normalized_entropy > 0.0
    assert report.unanimous_fraction == pytest.approx(1 / 3)
    assert report.adjudicated_fraction == pytest.approx(2 / 3)
    assert report.adjudication_match_fraction == 1.0
    assert report.mean_adjudication_confidence == pytest.approx(0.875)


def test_pairwise_cohens_kappa_rewards_agreement():
    items = [
        ExpertReviewItem("a", {"r1": "yes", "r2": "yes"}),
        ExpertReviewItem("b", {"r1": "no", "r2": "no"}),
        ExpertReviewItem("c", {"r1": "yes", "r2": "yes"}),
        ExpertReviewItem("d", {"r1": "no", "r2": "yes"}),
    ]
    kappa = pairwise_cohens_kappa(items, reviewer_a="r1", reviewer_b="r2")
    assert 0.0 < kappa < 1.0
    assert pairwise_cohens_kappa(
        [
            ExpertReviewItem("a", {"r1": "yes", "r2": "yes"}),
            ExpertReviewItem("b", {"r1": "no", "r2": "no"}),
        ],
        reviewer_a="r1",
        reviewer_b="r2",
    ) == pytest.approx(1.0)


def test_adjudication_label_and_confidence_are_atomic():
    with pytest.raises(ValueError, match="together"):
        ExpertReviewItem(
            "a",
            {"r1": "yes", "r2": "no"},
            adjudicated_label="yes",
        )


def test_review_report_rejects_duplicate_item_ids():
    item = ExpertReviewItem("same", {"r1": "yes", "r2": "no"})
    with pytest.raises(ValueError, match="unique"):
        expert_review_report([item, item])
