"""Privacy-preserving feedback lineage and candidate promotion gates.

The feedback store intentionally retains hashes rather than raw query/evidence text.  This
module builds immutable active-learning batch identities from those records and binds model
promotion decisions to a batch, candidate version, baseline version, and explicit policy.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from tools.feedback_store import ActiveLearningExample
from tools.security import normalize_owner_id

_POSITIVE_KINDS = frozenset({"answer_correct", "citation_valid", "abstention_good"})
_NEGATIVE_KINDS = frozenset({"answer_incorrect", "citation_invalid", "abstention_bad"})


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _finite(value: Any, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite.") from exc
    if not math.isfinite(parsed) or parsed < minimum:
        raise ValueError(f"{label} is invalid.")
    return parsed


def _identifier(value: str, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    text = value.strip()
    if not text or len(text) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError(f"{label} is invalid.")
    return text


@dataclass(frozen=True)
class FeedbackBatchManifest:
    owner_id: str
    batch_id: str
    example_count: int
    positive_weight: float
    negative_weight: float
    neutral_weight: float
    subject_count: int
    event_fingerprint: str


@dataclass(frozen=True)
class PromotionPolicy:
    min_examples: int = 20
    min_negative_weight_fraction: float = 0.05
    max_quality_regression: float = 0.0
    min_quality_gain: float = 0.0
    max_latency_ratio: float = 1.25
    max_cost_ratio: float = 1.25

    def __post_init__(self) -> None:
        if isinstance(self.min_examples, bool) or not isinstance(self.min_examples, int):
            raise ValueError("min_examples must be an integer.")
        if self.min_examples < 1:
            raise ValueError("min_examples must be positive.")
        for name in (
            "min_negative_weight_fraction",
            "max_quality_regression",
            "min_quality_gain",
            "max_latency_ratio",
            "max_cost_ratio",
        ):
            value = _finite(getattr(self, name), name)
            if name in {"min_negative_weight_fraction"} and value > 1.0:
                raise ValueError(f"{name} must not exceed 1.")
            if name in {"max_latency_ratio", "max_cost_ratio"} and value <= 0.0:
                raise ValueError(f"{name} must be positive.")


@dataclass(frozen=True)
class CandidateMetrics:
    quality: float
    p95_latency_ms: float
    estimated_cost: float

    def __post_init__(self) -> None:
        _finite(self.quality, "quality")
        _finite(self.p95_latency_ms, "p95_latency_ms")
        _finite(self.estimated_cost, "estimated_cost")


@dataclass(frozen=True)
class PromotionDecision:
    decision_id: str
    eligible: bool
    reason_codes: tuple[str, ...]
    owner_id: str
    batch_id: str
    baseline_version: str
    candidate_version: str
    baseline: CandidateMetrics
    candidate: CandidateMetrics
    quality_delta: float
    latency_ratio: float
    cost_ratio: float
    policy_fingerprint: str


def build_feedback_batch(
    *, owner_id: str, examples: Iterable[ActiveLearningExample]
) -> FeedbackBatchManifest:
    owner = normalize_owner_id(owner_id)
    normalized: list[dict[str, object]] = []
    positive = 0.0
    negative = 0.0
    neutral = 0.0
    subjects: set[str] = set()
    for item in examples:
        weight = _finite(item.weight, "weight", 0.000001)
        subject = _identifier(item.subject_id, "subject_id")
        subjects.add(subject)
        if item.kind in _POSITIVE_KINDS:
            positive += weight
        elif item.kind in _NEGATIVE_KINDS:
            negative += weight
        else:
            neutral += weight
        normalized.append(
            {
                "kind": item.kind,
                "subject_id": subject,
                "weight": weight,
                "metadata": dict(item.metadata),
                "query_sha256": item.query_sha256,
                "evidence_sha256": item.evidence_sha256,
            }
        )
    normalized.sort(key=lambda row: _sha256(row))
    fingerprint = _sha256(normalized)
    batch_id = _sha256({"owner_id": owner, "events": fingerprint, "count": len(normalized)})
    return FeedbackBatchManifest(
        owner_id=owner,
        batch_id=batch_id,
        example_count=len(normalized),
        positive_weight=positive,
        negative_weight=negative,
        neutral_weight=neutral,
        subject_count=len(subjects),
        event_fingerprint=fingerprint,
    )


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def evaluate_promotion(
    *,
    batch: FeedbackBatchManifest,
    baseline_version: str,
    candidate_version: str,
    baseline: CandidateMetrics,
    candidate: CandidateMetrics,
    policy: PromotionPolicy | None = None,
) -> PromotionDecision:
    selected = policy or PromotionPolicy()
    baseline_id = _identifier(baseline_version, "baseline_version")
    candidate_id = _identifier(candidate_version, "candidate_version")
    if baseline_id == candidate_id:
        raise ValueError("candidate_version must differ from baseline_version.")

    total_weight = batch.positive_weight + batch.negative_weight + batch.neutral_weight
    negative_fraction = batch.negative_weight / total_weight if total_weight else 0.0
    quality_delta = candidate.quality - baseline.quality
    latency_ratio = _ratio(candidate.p95_latency_ms, baseline.p95_latency_ms)
    cost_ratio = _ratio(candidate.estimated_cost, baseline.estimated_cost)
    reasons: list[str] = []
    if batch.example_count < selected.min_examples:
        reasons.append("insufficient_feedback_examples")
    if negative_fraction < selected.min_negative_weight_fraction:
        reasons.append("insufficient_negative_feedback_coverage")
    if quality_delta < -selected.max_quality_regression:
        reasons.append("quality_regression")
    if quality_delta < selected.min_quality_gain:
        reasons.append("quality_gain_below_policy")
    if latency_ratio > selected.max_latency_ratio:
        reasons.append("latency_budget_exceeded")
    if cost_ratio > selected.max_cost_ratio:
        reasons.append("cost_budget_exceeded")

    policy_fingerprint = _sha256(asdict(selected))
    payload = {
        "owner_id": batch.owner_id,
        "batch_id": batch.batch_id,
        "baseline_version": baseline_id,
        "candidate_version": candidate_id,
        "baseline": asdict(baseline),
        "candidate": asdict(candidate),
        "policy": policy_fingerprint,
        "reasons": reasons,
    }
    return PromotionDecision(
        decision_id=_sha256(payload),
        eligible=not reasons,
        reason_codes=tuple(reasons),
        owner_id=batch.owner_id,
        batch_id=batch.batch_id,
        baseline_version=baseline_id,
        candidate_version=candidate_id,
        baseline=baseline,
        candidate=candidate,
        quality_delta=quality_delta,
        latency_ratio=latency_ratio,
        cost_ratio=cost_ratio,
        policy_fingerprint=policy_fingerprint,
    )


def promotion_record(decision: PromotionDecision) -> Mapping[str, object]:
    """Return a canonical JSON-safe lineage record suitable for append-only journals."""

    return json.loads(_canonical_json(asdict(decision)).decode("utf-8"))


__all__ = [
    "CandidateMetrics",
    "FeedbackBatchManifest",
    "PromotionDecision",
    "PromotionPolicy",
    "build_feedback_batch",
    "evaluate_promotion",
    "promotion_record",
]
