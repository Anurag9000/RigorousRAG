"""Semantic entailment, contradiction, citation support and calibration evaluation.

This module defines provider-neutral contracts around an NLI/semantic-support scorer.
It never downloads or executes a model by itself.  A caller may inject a local or remote
model implementation, while the repository retains stable evaluation semantics:

* entailment / neutral / contradiction probabilities with immutable model identity;
* claim-to-evidence citation support records;
* coverage, abstention, contradiction, support and citation-support metrics;
* multiclass Brier score and expected calibration error; and
* contradiction-first promotion rules for evidence-grounded systems.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

_EPS = 1e-12
_MAX_RECORDS = 10_000_000


def _text(value: Any, label: str, maximum: int = 100_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if not result or len(result) > maximum or "\x00" in result:
        raise ValueError(f"{label} is empty or too long")
    return result


def _identifier(value: Any, label: str, maximum: int = 1_000) -> str:
    result = _text(value, label, maximum)
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{label} contains control characters")
    return result


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _probability(value: Any, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1")
    return result


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class SemanticLabel(str, Enum):
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


@dataclass(frozen=True)
class ModelIdentity:
    provider: str
    model_name: str
    model_version: str
    artifact_sha256: str | None = None
    calibration_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("provider", "model_name", "model_version"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name, 2_000))
        for name in ("artifact_sha256", "calibration_sha256"):
            value = getattr(self, name)
            if value is not None:
                digest = _identifier(value, name, 64).lower()
                if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                    raise ValueError(f"{name} must be a SHA-256 digest")
                object.__setattr__(self, name, digest)

    @property
    def identity_digest(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class SemanticProbabilities:
    entailment: float
    neutral: float
    contradiction: float

    def __post_init__(self) -> None:
        values = [_probability(getattr(self, name), name) for name in ("entailment", "neutral", "contradiction")]
        total = sum(values)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("semantic probabilities must sum to one")
        object.__setattr__(self, "entailment", values[0])
        object.__setattr__(self, "neutral", values[1])
        object.__setattr__(self, "contradiction", values[2])

    def for_label(self, label: SemanticLabel) -> float:
        selected = SemanticLabel(label)
        return {
            SemanticLabel.ENTAILMENT: self.entailment,
            SemanticLabel.NEUTRAL: self.neutral,
            SemanticLabel.CONTRADICTION: self.contradiction,
        }[selected]

    @property
    def predicted_label(self) -> SemanticLabel:
        candidates = (
            (self.entailment, SemanticLabel.ENTAILMENT),
            (self.neutral, SemanticLabel.NEUTRAL),
            (self.contradiction, SemanticLabel.CONTRADICTION),
        )
        return max(candidates, key=lambda item: item[0])[1]

    @property
    def confidence(self) -> float:
        return max(self.entailment, self.neutral, self.contradiction)


@dataclass(frozen=True)
class SemanticScore:
    premise: str
    hypothesis: str
    probabilities: SemanticProbabilities
    model: ModelIdentity
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "premise", _text(self.premise, "premise"))
        object.__setattr__(self, "hypothesis", _text(self.hypothesis, "hypothesis"))
        if not isinstance(self.probabilities, SemanticProbabilities):
            raise ValueError("probabilities must be SemanticProbabilities")
        if not isinstance(self.model, ModelIdentity):
            raise ValueError("model must be ModelIdentity")
        if not isinstance(self.metadata, Mapping) or len(self.metadata) > 1_000:
            raise ValueError("metadata must be a bounded mapping")
        cleaned = {
            _identifier(key, "metadata key", 300): _text(value, "metadata value", 20_000)
            for key, value in self.metadata.items()
        }
        object.__setattr__(self, "metadata", cleaned)


class SemanticScorer(Protocol):
    @property
    def model_identity(self) -> ModelIdentity: ...

    def score(self, premise: str, hypothesis: str) -> SemanticScore: ...


@dataclass(frozen=True)
class CitationAnchor:
    document_id: str
    generation_id: str
    chunk_id: str | None = None
    page: int | None = None
    block_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("document_id", "generation_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in ("chunk_id", "block_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _identifier(value, name))
        if self.page is not None and (isinstance(self.page, bool) or not isinstance(self.page, int) or self.page < 1):
            raise ValueError("page must be a positive integer")


@dataclass(frozen=True)
class ClaimEvidenceScore:
    claim_id: str
    claim_text: str
    evidence_text: str
    anchor: CitationAnchor
    probabilities: SemanticProbabilities
    model: ModelIdentity
    abstained: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _identifier(self.claim_id, "claim_id"))
        object.__setattr__(self, "claim_text", _text(self.claim_text, "claim_text"))
        object.__setattr__(self, "evidence_text", _text(self.evidence_text, "evidence_text"))
        if not isinstance(self.anchor, CitationAnchor):
            raise ValueError("anchor must be CitationAnchor")
        if not isinstance(self.probabilities, SemanticProbabilities):
            raise ValueError("probabilities must be SemanticProbabilities")
        if not isinstance(self.model, ModelIdentity):
            raise ValueError("model must be ModelIdentity")
        if not isinstance(self.abstained, bool):
            raise ValueError("abstained must be boolean")


@dataclass(frozen=True)
class LabeledSemanticExample:
    probabilities: SemanticProbabilities
    gold: SemanticLabel
    abstained: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.probabilities, SemanticProbabilities):
            raise ValueError("probabilities must be SemanticProbabilities")
        if not isinstance(self.gold, SemanticLabel):
            object.__setattr__(self, "gold", SemanticLabel(self.gold))
        if not isinstance(self.abstained, bool):
            raise ValueError("abstained must be boolean")


@dataclass(frozen=True)
class SemanticMetrics:
    count: int
    coverage: float
    accuracy_on_covered: float | None
    entailment_recall: float | None
    contradiction_recall: float | None
    contradiction_false_negative_rate: float | None
    multiclass_brier: float | None
    expected_calibration_error: float | None


@dataclass(frozen=True)
class CitationSupportMetrics:
    claim_count: int
    citation_count: int
    claim_coverage: float
    mean_best_entailment: float | None
    supported_claim_rate: float | None
    contradicted_claim_rate: float | None
    unsupported_claim_rate: float | None


def multiclass_brier_score(examples: Sequence[LabeledSemanticExample], *, include_abstained: bool = False) -> float | None:
    selected = [example for example in examples if include_abstained or not example.abstained]
    if not selected:
        return None
    total = 0.0
    labels = (SemanticLabel.ENTAILMENT, SemanticLabel.NEUTRAL, SemanticLabel.CONTRADICTION)
    for example in selected:
        total += sum(
            (example.probabilities.for_label(label) - (1.0 if example.gold == label else 0.0)) ** 2
            for label in labels
        ) / len(labels)
    return total / len(selected)


def expected_calibration_error(
    examples: Sequence[LabeledSemanticExample],
    *,
    bins: int = 15,
    include_abstained: bool = False,
) -> float | None:
    if isinstance(bins, bool) or not isinstance(bins, int) or not 2 <= bins <= 10_000:
        raise ValueError("bins must be between 2 and 10,000")
    selected = [example for example in examples if include_abstained or not example.abstained]
    if not selected:
        return None
    buckets: list[list[LabeledSemanticExample]] = [[] for _ in range(bins)]
    for example in selected:
        index = min(int(example.probabilities.confidence * bins), bins - 1)
        buckets[index].append(example)
    ece = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        accuracy = sum(example.probabilities.predicted_label == example.gold for example in bucket) / len(bucket)
        confidence = sum(example.probabilities.confidence for example in bucket) / len(bucket)
        ece += len(bucket) / len(selected) * abs(accuracy - confidence)
    return ece


def evaluate_semantic_examples(
    examples: Sequence[LabeledSemanticExample],
    *,
    calibration_bins: int = 15,
) -> SemanticMetrics:
    if len(examples) > _MAX_RECORDS:
        raise ValueError("too many semantic examples")
    count = len(examples)
    if count == 0:
        return SemanticMetrics(0, 0.0, None, None, None, None, None, None)
    covered = [example for example in examples if not example.abstained]
    coverage = len(covered) / count
    if not covered:
        return SemanticMetrics(count, coverage, None, None, None, None, None, None)
    accuracy = sum(example.probabilities.predicted_label == example.gold for example in covered) / len(covered)

    def recall(label: SemanticLabel) -> float | None:
        positives = [example for example in covered if example.gold == label]
        if not positives:
            return None
        return sum(example.probabilities.predicted_label == label for example in positives) / len(positives)

    entailment_recall = recall(SemanticLabel.ENTAILMENT)
    contradiction_recall = recall(SemanticLabel.CONTRADICTION)
    contradiction_fnr = None if contradiction_recall is None else 1.0 - contradiction_recall
    return SemanticMetrics(
        count=count,
        coverage=coverage,
        accuracy_on_covered=accuracy,
        entailment_recall=entailment_recall,
        contradiction_recall=contradiction_recall,
        contradiction_false_negative_rate=contradiction_fnr,
        multiclass_brier=multiclass_brier_score(covered),
        expected_calibration_error=expected_calibration_error(covered, bins=calibration_bins),
    )


def evaluate_citation_support(
    scores: Sequence[ClaimEvidenceScore],
    *,
    entailment_threshold: float = 0.70,
    contradiction_threshold: float = 0.70,
) -> CitationSupportMetrics:
    if len(scores) > _MAX_RECORDS:
        raise ValueError("too many citation-support scores")
    entailment_cutoff = _probability(entailment_threshold, "entailment_threshold")
    contradiction_cutoff = _probability(contradiction_threshold, "contradiction_threshold")
    if not scores:
        return CitationSupportMetrics(0, 0, 0.0, None, None, None, None)
    by_claim: dict[str, list[ClaimEvidenceScore]] = {}
    for score in scores:
        if not isinstance(score, ClaimEvidenceScore):
            raise ValueError("scores must contain ClaimEvidenceScore values")
        by_claim.setdefault(score.claim_id, []).append(score)
    claim_count = len(by_claim)
    covered_claims = [claim_scores for claim_scores in by_claim.values() if any(not score.abstained for score in claim_scores)]
    if not covered_claims:
        return CitationSupportMetrics(claim_count, len(scores), 0.0, None, None, None, None)
    best_entailments: list[float] = []
    supported = 0
    contradicted = 0
    unsupported = 0
    for claim_scores in covered_claims:
        active = [score for score in claim_scores if not score.abstained]
        best_entailment = max(score.probabilities.entailment for score in active)
        best_contradiction = max(score.probabilities.contradiction for score in active)
        best_entailments.append(best_entailment)
        if best_contradiction >= contradiction_cutoff and best_contradiction > best_entailment:
            contradicted += 1
        elif best_entailment >= entailment_cutoff:
            supported += 1
        else:
            unsupported += 1
    denominator = len(covered_claims)
    return CitationSupportMetrics(
        claim_count=claim_count,
        citation_count=len(scores),
        claim_coverage=denominator / claim_count,
        mean_best_entailment=sum(best_entailments) / denominator,
        supported_claim_rate=supported / denominator,
        contradicted_claim_rate=contradicted / denominator,
        unsupported_claim_rate=unsupported / denominator,
    )


@dataclass(frozen=True)
class SemanticPromotionRule:
    minimum_coverage: float = 0.95
    maximum_contradiction_fnr: float = 0.10
    maximum_brier: float = 0.20
    maximum_ece: float = 0.10
    minimum_citation_support: float = 0.90
    maximum_citation_contradiction: float = 0.02

    def __post_init__(self) -> None:
        for name in (
            "minimum_coverage",
            "maximum_contradiction_fnr",
            "maximum_brier",
            "maximum_ece",
            "minimum_citation_support",
            "maximum_citation_contradiction",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))


@dataclass(frozen=True)
class SemanticPromotionDecision:
    promoted: bool
    reasons: tuple[str, ...]


def decide_semantic_promotion(
    semantic: SemanticMetrics,
    citation: CitationSupportMetrics,
    *,
    rule: SemanticPromotionRule = SemanticPromotionRule(),
) -> SemanticPromotionDecision:
    reasons: list[str] = []
    if semantic.coverage < rule.minimum_coverage:
        reasons.append(f"semantic coverage {semantic.coverage:.4f} < {rule.minimum_coverage:.4f}")
    if semantic.contradiction_false_negative_rate is None:
        reasons.append("contradiction recall is undefined because no covered contradiction examples were supplied")
    elif semantic.contradiction_false_negative_rate > rule.maximum_contradiction_fnr:
        reasons.append(
            f"contradiction FNR {semantic.contradiction_false_negative_rate:.4f} > {rule.maximum_contradiction_fnr:.4f}"
        )
    if semantic.multiclass_brier is None or semantic.multiclass_brier > rule.maximum_brier:
        reasons.append("semantic Brier score is missing or above threshold")
    if semantic.expected_calibration_error is None or semantic.expected_calibration_error > rule.maximum_ece:
        reasons.append("semantic calibration error is missing or above threshold")
    if citation.supported_claim_rate is None or citation.supported_claim_rate < rule.minimum_citation_support:
        reasons.append("citation support rate is missing or below threshold")
    if citation.contradicted_claim_rate is None or citation.contradicted_claim_rate > rule.maximum_citation_contradiction:
        reasons.append("citation contradiction rate is missing or above threshold")
    return SemanticPromotionDecision(not reasons, tuple(reasons or ["all semantic support gates passed"]))


__all__ = [
    "CitationAnchor",
    "CitationSupportMetrics",
    "ClaimEvidenceScore",
    "LabeledSemanticExample",
    "ModelIdentity",
    "SemanticLabel",
    "SemanticMetrics",
    "SemanticProbabilities",
    "SemanticPromotionDecision",
    "SemanticPromotionRule",
    "SemanticScore",
    "SemanticScorer",
    "decide_semantic_promotion",
    "evaluate_citation_support",
    "evaluate_semantic_examples",
    "expected_calibration_error",
    "multiclass_brier_score",
]
