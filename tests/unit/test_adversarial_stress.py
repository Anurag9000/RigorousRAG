import pytest

from evaluation.adversarial_stress import contradiction_exposure_report, ocr_stress_report


def test_ocr_stress_reports_exact_match_and_degradation():
    exact = ocr_stress_report(reference="Table 12 flow = 35.0 m3/s", observed="Table 12 flow = 35.0 m3/s")
    noisy = ocr_stress_report(reference="Table 12 flow = 35.0 m3/s", observed="Tab1e 12 flow = 350 m3/s")
    assert exact.exact_match is True
    assert exact.character_error_rate == 0.0
    assert exact.token_recall == 1.0
    assert noisy.exact_match is False
    assert noisy.character_error_rate > 0.0
    assert 0.0 <= noisy.token_recall < 1.0
    assert 0.0 <= noisy.token_precision <= 1.0


def test_ocr_stress_handles_empty_text_without_division_errors():
    empty = ocr_stress_report(reference="", observed="")
    assert empty.character_error_rate == 0.0
    assert empty.token_recall == 1.0
    assert empty.token_precision == 1.0


def test_contradiction_report_tracks_exposure_and_ordering():
    report = contradiction_exposure_report(
        ranking=["support-a", "neutral", "contradiction-a", "support-b"],
        support_ids=["support-a", "support-b"],
        contradiction_ids=["contradiction-a", "contradiction-b"],
        k=4,
    )
    assert report.support_recall_at_k == 1.0
    assert report.contradiction_exposure_rate == pytest.approx(0.25)
    assert report.first_contradiction_rank == 3
    assert report.support_before_contradiction is True


def test_contradiction_report_detects_contradiction_before_support():
    report = contradiction_exposure_report(
        ranking=["bad", "good"],
        support_ids=["good"],
        contradiction_ids=["bad"],
        k=2,
    )
    assert report.first_contradiction_rank == 1
    assert report.support_before_contradiction is False


def test_contradiction_report_rejects_ambiguous_labels_and_bad_k():
    with pytest.raises(ValueError, match="must not overlap"):
        contradiction_exposure_report(
            ranking=["x"], support_ids=["x"], contradiction_ids=["x"], k=1
        )
    with pytest.raises(ValueError):
        contradiction_exposure_report(
            ranking=[], support_ids=[], contradiction_ids=[], k=0
        )
