"""Privacy-safe expert-review agreement, disagreement and adjudication metrics."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_MAX_ITEMS = 100_000
_MAX_REVIEWERS = 100
_MAX_LABELS = 100


def _identifier(value: Any, label: str, maximum: int = 200) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    result = value.strip()
    if not result or len(result) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{label} is invalid.")
    return result


def _unit(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be between 0 and 1.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be between 0 and 1.") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return result


@dataclass(frozen=True)
class ExpertReviewItem:
    item_id: str
    labels_by_reviewer: Mapping[str, str]
    adjudicated_label: str | None = None
    adjudication_confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "item_id", _identifier(self.item_id, "item_id"))
        if not isinstance(self.labels_by_reviewer, Mapping) or not 2 <= len(self.labels_by_reviewer) <= _MAX_REVIEWERS:
            raise ValueError("labels_by_reviewer must contain 2-100 reviewers.")
        cleaned: dict[str, str] = {}
        for reviewer, label in self.labels_by_reviewer.items():
            selected_reviewer = _identifier(reviewer, "reviewer_id")
            selected_label = _identifier(label, "label")
            if selected_reviewer in cleaned:
                raise ValueError("reviewer IDs must be unique.")
            cleaned[selected_reviewer] = selected_label
        if len(set(cleaned.values())) > _MAX_LABELS:
            raise ValueError("item contains too many labels.")
        object.__setattr__(self, "labels_by_reviewer", cleaned)
        if self.adjudicated_label is not None:
            object.__setattr__(self, "adjudicated_label", _identifier(self.adjudicated_label, "adjudicated_label"))
        if self.adjudication_confidence is not None:
            object.__setattr__(
                self,
                "adjudication_confidence",
                _unit(self.adjudication_confidence, "adjudication_confidence"),
            )
        if (self.adjudicated_label is None) != (self.adjudication_confidence is None):
            raise ValueError("adjudication label and confidence must be provided together.")

    @property
    def agreement_fraction(self) -> float:
        counts = Counter(self.labels_by_reviewer.values())
        return max(counts.values()) / len(self.labels_by_reviewer)

    @property
    def label_entropy(self) -> float:
        counts = Counter(self.labels_by_reviewer.values())
        total = len(self.labels_by_reviewer)
        if len(counts) <= 1:
            return 0.0
        entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
        return entropy / math.log2(len(counts))


@dataclass(frozen=True)
class ExpertReviewReport:
    item_count: int
    mean_agreement: float
    mean_normalized_entropy: float
    unanimous_fraction: float
    adjudicated_fraction: float
    adjudication_match_fraction: float
    mean_adjudication_confidence: float


def expert_review_report(items: Sequence[ExpertReviewItem]) -> ExpertReviewReport:
    if isinstance(items, (str, bytes, bytearray)) or not items or len(items) > _MAX_ITEMS:
        raise ValueError("items must be a non-empty bounded sequence.")
    values = tuple(items)
    if any(not isinstance(item, ExpertReviewItem) for item in values):
        raise ValueError("every item must be ExpertReviewItem.")
    if len({item.item_id for item in values}) != len(values):
        raise ValueError("item IDs must be unique.")
    adjudicated = [item for item in values if item.adjudicated_label is not None]
    matches = 0
    for item in adjudicated:
        counts = Counter(item.labels_by_reviewer.values())
        majority_label = min(
            (label for label, count in counts.items() if count == max(counts.values())),
            default="",
        )
        matches += item.adjudicated_label == majority_label
    return ExpertReviewReport(
        item_count=len(values),
        mean_agreement=sum(item.agreement_fraction for item in values) / len(values),
        mean_normalized_entropy=sum(item.label_entropy for item in values) / len(values),
        unanimous_fraction=sum(item.agreement_fraction == 1.0 for item in values) / len(values),
        adjudicated_fraction=len(adjudicated) / len(values),
        adjudication_match_fraction=(matches / len(adjudicated) if adjudicated else 0.0),
        mean_adjudication_confidence=(
            sum(float(item.adjudication_confidence) for item in adjudicated) / len(adjudicated)
            if adjudicated
            else 0.0
        ),
    )


def pairwise_cohens_kappa(
    items: Sequence[ExpertReviewItem],
    *,
    reviewer_a: str,
    reviewer_b: str,
) -> float:
    """Compute Cohen's kappa over items labelled by both named reviewers."""

    first = _identifier(reviewer_a, "reviewer_a")
    second = _identifier(reviewer_b, "reviewer_b")
    if first == second:
        raise ValueError("reviewers must be distinct.")
    pairs: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, ExpertReviewItem):
            raise ValueError("every item must be ExpertReviewItem.")
        if first in item.labels_by_reviewer and second in item.labels_by_reviewer:
            pairs.append((item.labels_by_reviewer[first], item.labels_by_reviewer[second]))
    if not pairs:
        raise ValueError("reviewers have no overlapping labelled items.")
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    labels = set(left_counts) | set(right_counts)
    expected = sum(
        (left_counts[label] / len(pairs)) * (right_counts[label] / len(pairs))
        for label in labels
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


__all__ = [
    "ExpertReviewItem",
    "ExpertReviewReport",
    "expert_review_report",
    "pairwise_cohens_kappa",
]
