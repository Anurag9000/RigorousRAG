from __future__ import annotations

import hashlib

from evaluation.semantic_support import SemanticLabel
from evaluation.structured_data_support import NumericClaim, NumericOperator, Quantity, StructuredSupportScore, TableQuantityEvidence, evaluate_numeric_claim
from evaluation.structured_support_metrics import LabeledStructuredSupport, evaluate_structured_support, observations_from_structured_support, summarize_structured_support


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def evidence(value: float, *, unit: str = "m") -> TableQuantityEvidence:
    return TableQuantityEvidence("doc", "gen", "table", "cell", 0, 0, sha("cell"), sha("parser"), Quantity(value, unit))


def claim(value: float, *, unit: str = "m") -> NumericClaim:
    return NumericClaim("claim", sha(f"claim:{value}:{unit}"), NumericOperator.GT, unit, value=value)


def test_structured_labeled_examples_reuse_shared_semantic_metrics() -> None:
    entailed = evaluate_numeric_claim(claim(5.0), evidence(10.0))
    contradicted = evaluate_numeric_claim(claim(15.0), evidence(10.0))
    metrics = evaluate_structured_support(
        (
            LabeledStructuredSupport(entailed, SemanticLabel.ENTAILMENT, "table"),
            LabeledStructuredSupport(contradicted, SemanticLabel.CONTRADICTION, "table"),
        ),
        calibration_bins=2,
    )
    assert metrics.count == 2
    assert metrics.accuracy == 1.0
    assert metrics.brier == 0.0


def test_unit_incomparable_neutral_is_marked_as_abstention_by_default() -> None:
    score = evaluate_numeric_claim(claim(5.0, unit="s"), evidence(10.0, unit="m"))
    example = LabeledStructuredSupport(score, SemanticLabel.NEUTRAL, "table")
    assert example.effective_abstained
    metrics = evaluate_structured_support((example,))
    assert metrics.abstention_rate == 1.0


def test_unlabeled_structured_summary_exports_privacy_safe_quality_rates() -> None:
    scores = (
        evaluate_numeric_claim(claim(5.0), evidence(10.0)),
        evaluate_numeric_claim(claim(15.0), evidence(10.0)),
        evaluate_numeric_claim(claim(5.0, unit="s"), evidence(10.0, unit="m")),
    )
    summary = summarize_structured_support(scores, evidence_kind="table")
    assert summary.count == 3
    assert summary.entailment_count == 1
    assert summary.contradiction_count == 1
    assert summary.neutral_count == 1
    observations = observations_from_structured_support(summary)
    assert {item.name for item in observations} == {
        "structured_support.entailment_rate",
        "structured_support.neutral_rate",
        "structured_support.contradiction_rate",
        "structured_support.abstention_rate",
    }
    assert all(dict(item.tags)["variant"] == "table" for item in observations)
