"""Shared metrics/observability bridge for table/chart structured entailment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from evaluation.quality_observability import MetricObservation
from evaluation.semantic_support import LabeledSemanticExample, SemanticLabel, SemanticMetrics, evaluate_semantic_examples
from evaluation.structured_data_support import StructuredSupportScore

_DEFAULT_ABSTENTION_REASONS = frozenset(
    {
        "unit_incomparable",
        "categorical_x_order_not_assumed",
        "duplicate_x_coordinates",
        "insufficient_points_for_trend",
    }
)


@dataclass(frozen=True)
class LabeledStructuredSupport:
    score: StructuredSupportScore
    gold: SemanticLabel
    evidence_kind: str
    abstained: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.score, StructuredSupportScore):
            raise ValueError("score must be StructuredSupportScore")
        if not isinstance(self.gold, SemanticLabel):
            object.__setattr__(self, "gold", SemanticLabel(self.gold))
        if self.evidence_kind not in {"table", "chart", "aggregate"}:
            raise ValueError("evidence_kind must be table, chart, or aggregate")
        if self.abstained is not None and not isinstance(self.abstained, bool):
            raise ValueError("abstained must be boolean when set")

    @property
    def effective_abstained(self) -> bool:
        return self.score.reason_code in _DEFAULT_ABSTENTION_REASONS if self.abstained is None else self.abstained


def evaluate_structured_support(
    examples: Sequence[LabeledStructuredSupport],
    *,
    calibration_bins: int = 15,
) -> SemanticMetrics:
    rows = []
    for example in examples:
        if not isinstance(example, LabeledStructuredSupport):
            raise ValueError("examples must contain LabeledStructuredSupport values")
        rows.append(LabeledSemanticExample(example.gold, example.score.probabilities, example.effective_abstained))
    return evaluate_semantic_examples(tuple(rows), calibration_bins=calibration_bins)


@dataclass(frozen=True)
class StructuredSupportSummary:
    evidence_kind: str
    count: int
    entailment_count: int
    neutral_count: int
    contradiction_count: int
    abstention_count: int

    def __post_init__(self) -> None:
        if self.evidence_kind not in {"table", "chart", "aggregate"}:
            raise ValueError("evidence_kind is invalid")
        for name in ("count", "entailment_count", "neutral_count", "contradiction_count", "abstention_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.entailment_count + self.neutral_count + self.contradiction_count != self.count:
            raise ValueError("semantic label counts must sum to count")
        if self.abstention_count > self.count:
            raise ValueError("abstention_count may not exceed count")


def summarize_structured_support(
    scores: Sequence[StructuredSupportScore],
    *,
    evidence_kind: str,
    abstention_reason_codes: frozenset[str] = _DEFAULT_ABSTENTION_REASONS,
) -> StructuredSupportSummary:
    if evidence_kind not in {"table", "chart", "aggregate"}:
        raise ValueError("evidence_kind must be table, chart, or aggregate")
    if any(not isinstance(score, StructuredSupportScore) for score in scores):
        raise ValueError("scores must contain StructuredSupportScore values")
    labels = [score.label for score in scores]
    return StructuredSupportSummary(
        evidence_kind=evidence_kind,
        count=len(scores),
        entailment_count=sum(label is SemanticLabel.ENTAILMENT for label in labels),
        neutral_count=sum(label is SemanticLabel.NEUTRAL for label in labels),
        contradiction_count=sum(label is SemanticLabel.CONTRADICTION for label in labels),
        abstention_count=sum(score.reason_code in abstention_reason_codes for score in scores),
    )


def observations_from_structured_support(summary: StructuredSupportSummary) -> tuple[MetricObservation, ...]:
    if not isinstance(summary, StructuredSupportSummary):
        raise ValueError("summary must be StructuredSupportSummary")
    denominator = summary.count
    entailment = summary.entailment_count / denominator if denominator else 0.0
    neutral = summary.neutral_count / denominator if denominator else 0.0
    contradiction = summary.contradiction_count / denominator if denominator else 0.0
    abstention = summary.abstention_count / denominator if denominator else 0.0
    tags = (("metric_family", "structured_semantic_support"), ("variant", summary.evidence_kind))
    source = "evaluation.structured_support_metrics"
    return (
        MetricObservation("structured_support.entailment_rate", entailment, "neutral", "ratio", denominator, source, tags),
        MetricObservation("structured_support.neutral_rate", neutral, "neutral", "ratio", denominator, source, tags),
        MetricObservation("structured_support.contradiction_rate", contradiction, "lower", "ratio", denominator, source, tags),
        MetricObservation("structured_support.abstention_rate", abstention, "lower", "ratio", denominator, source, tags),
    )


__all__ = [
    "LabeledStructuredSupport",
    "StructuredSupportSummary",
    "evaluate_structured_support",
    "observations_from_structured_support",
    "summarize_structured_support",
]
