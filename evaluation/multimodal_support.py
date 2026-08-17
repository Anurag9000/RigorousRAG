"""Multimodal claim-to-evidence support, contradiction and calibration contracts.

Text NLI remains in :mod:`evaluation.semantic_support`.  This module extends the same
three-way semantics to visual/page-region evidence while preserving citation authority:
durable results bind immutable document/generation/page/region anchors and digests, not
raw image bytes.  Model execution is injected and no weights or datasets are loaded on
import.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from evaluation.semantic_support import ModelIdentity, SemanticLabel, SemanticProbabilities

_MAX_IMAGE_BYTES = 50_000_000
_MAX_SCORES = 10_000_000


def _text(value: Any, label: str, maximum: int = 100_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or "\x00" in selected:
        raise ValueError(f"{label} is invalid")
    return selected


def _identifier(value: Any, label: str, maximum: int = 1_000) -> str:
    selected = _text(value, label, maximum)
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} contains control characters")
    return selected


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _probability(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a probability")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a probability") from exc
    if not math.isfinite(selected) or not 0.0 <= selected <= 1.0:
        raise ValueError(f"{label} must be between zero and one")
    return selected


def image_digest(image_bytes: bytes) -> str:
    if not isinstance(image_bytes, bytes) or not 1 <= len(image_bytes) <= _MAX_IMAGE_BYTES:
        raise ValueError("image_bytes must be bounded non-empty bytes")
    return hashlib.sha256(image_bytes).hexdigest()


@dataclass(frozen=True)
class VisualEvidenceAnchor:
    document_id: str
    generation_id: str
    page: int
    region_id: str
    image_sha256: str
    block_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("document_id", "generation_id", "region_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if isinstance(self.page, bool) or not isinstance(self.page, int) or self.page < 1:
            raise ValueError("page must be a positive integer")
        object.__setattr__(self, "image_sha256", _sha(self.image_sha256, "image_sha256"))
        if self.block_id is not None:
            object.__setattr__(self, "block_id", _identifier(self.block_id, "block_id"))


@dataclass(frozen=True)
class MultimodalEvidence:
    anchor: VisualEvidenceAnchor
    image_bytes: bytes
    evidence_text: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.anchor, VisualEvidenceAnchor):
            raise ValueError("anchor must be VisualEvidenceAnchor")
        if image_digest(self.image_bytes) != self.anchor.image_sha256:
            raise ValueError("image bytes do not match authoritative visual evidence digest")
        if self.evidence_text is not None:
            object.__setattr__(self, "evidence_text", _text(self.evidence_text, "evidence_text", 500_000))


@dataclass(frozen=True)
class MultimodalSupportScore:
    claim_id: str
    claim_sha256: str
    anchor: VisualEvidenceAnchor
    probabilities: SemanticProbabilities
    model: ModelIdentity
    evidence_text_sha256: str | None = None
    abstained: bool = False
    abstention_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _identifier(self.claim_id, "claim_id"))
        object.__setattr__(self, "claim_sha256", _sha(self.claim_sha256, "claim_sha256"))
        if not isinstance(self.anchor, VisualEvidenceAnchor):
            raise ValueError("anchor must be VisualEvidenceAnchor")
        if not isinstance(self.probabilities, SemanticProbabilities):
            raise ValueError("probabilities must be SemanticProbabilities")
        if not isinstance(self.model, ModelIdentity):
            raise ValueError("model must be ModelIdentity")
        if self.evidence_text_sha256 is not None:
            object.__setattr__(self, "evidence_text_sha256", _sha(self.evidence_text_sha256, "evidence_text_sha256"))
        if not isinstance(self.abstained, bool):
            raise ValueError("abstained must be boolean")
        if self.abstention_reason is not None:
            object.__setattr__(self, "abstention_reason", _identifier(self.abstention_reason, "abstention_reason", 500))
        if self.abstained and self.abstention_reason is None:
            raise ValueError("abstained multimodal score requires abstention_reason")
        if not self.abstained and self.abstention_reason is not None:
            raise ValueError("covered multimodal score may not carry abstention_reason")


class MultimodalSemanticScorer(Protocol):
    @property
    def model_identity(self) -> ModelIdentity: ...

    def score(self, claim_id: str, claim_text: str, evidence: MultimodalEvidence) -> MultimodalSupportScore: ...


@dataclass(frozen=True)
class MultimodalSupportMetrics:
    claim_count: int
    evidence_count: int
    claim_coverage: float
    mean_best_entailment: float | None
    supported_claim_rate: float | None
    contradicted_claim_rate: float | None
    unsupported_claim_rate: float | None
    abstained_evidence_rate: float


def score_multimodal_evidence(
    scorer: MultimodalSemanticScorer,
    *,
    claim_id: str,
    claim_text: str,
    evidence: MultimodalEvidence,
) -> MultimodalSupportScore:
    """Invoke a scorer and revalidate all authority/model bindings."""

    selected_id = _identifier(claim_id, "claim_id")
    selected_claim = _text(claim_text, "claim_text")
    if not isinstance(evidence, MultimodalEvidence):
        raise ValueError("evidence must be MultimodalEvidence")
    result = scorer.score(selected_id, selected_claim, evidence)
    if not isinstance(result, MultimodalSupportScore):
        raise RuntimeError("multimodal scorer returned an invalid result")
    if result.claim_id != selected_id:
        raise RuntimeError("multimodal scorer changed claim identity")
    if result.claim_sha256 != hashlib.sha256(selected_claim.encode("utf-8")).hexdigest():
        raise RuntimeError("multimodal scorer changed claim content identity")
    if result.anchor != evidence.anchor:
        raise RuntimeError("multimodal scorer changed evidence authority anchor")
    if result.model.identity_digest != scorer.model_identity.identity_digest:
        raise RuntimeError("multimodal scorer result model identity differs from provider")
    expected_text_digest = None if evidence.evidence_text is None else hashlib.sha256(evidence.evidence_text.encode("utf-8")).hexdigest()
    if result.evidence_text_sha256 != expected_text_digest:
        raise RuntimeError("multimodal scorer changed auxiliary evidence-text identity")
    return result


def evaluate_multimodal_support(
    scores: Sequence[MultimodalSupportScore],
    *,
    entailment_threshold: float = 0.70,
    contradiction_threshold: float = 0.70,
) -> MultimodalSupportMetrics:
    """Aggregate visual citation support using contradiction-first claim semantics."""

    if len(scores) > _MAX_SCORES:
        raise ValueError("too many multimodal support scores")
    entailment_cutoff = _probability(entailment_threshold, "entailment_threshold")
    contradiction_cutoff = _probability(contradiction_threshold, "contradiction_threshold")
    if not scores:
        return MultimodalSupportMetrics(0, 0, 0.0, None, None, None, None, 0.0)
    by_claim: dict[str, list[MultimodalSupportScore]] = {}
    for score in scores:
        if not isinstance(score, MultimodalSupportScore):
            raise ValueError("scores must contain MultimodalSupportScore values")
        by_claim.setdefault(score.claim_id, []).append(score)
    covered_claims = [items for items in by_claim.values() if any(not item.abstained for item in items)]
    abstained_count = sum(item.abstained for item in scores)
    if not covered_claims:
        return MultimodalSupportMetrics(
            len(by_claim), len(scores), 0.0, None, None, None, None, abstained_count / len(scores)
        )
    supported = contradicted = unsupported = 0
    best_entailments: list[float] = []
    for items in covered_claims:
        active = [item for item in items if not item.abstained]
        best_entailment = max(item.probabilities.entailment for item in active)
        best_contradiction = max(item.probabilities.contradiction for item in active)
        best_entailments.append(best_entailment)
        if best_contradiction >= contradiction_cutoff and best_contradiction > best_entailment:
            contradicted += 1
        elif best_entailment >= entailment_cutoff:
            supported += 1
        else:
            unsupported += 1
    denominator = len(covered_claims)
    return MultimodalSupportMetrics(
        claim_count=len(by_claim),
        evidence_count=len(scores),
        claim_coverage=denominator / len(by_claim),
        mean_best_entailment=sum(best_entailments) / denominator,
        supported_claim_rate=supported / denominator,
        contradicted_claim_rate=contradicted / denominator,
        unsupported_claim_rate=unsupported / denominator,
        abstained_evidence_rate=abstained_count / len(scores),
    )


@dataclass(frozen=True)
class LabeledMultimodalExample:
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


__all__ = [
    "LabeledMultimodalExample",
    "MultimodalEvidence",
    "MultimodalSemanticScorer",
    "MultimodalSupportMetrics",
    "MultimodalSupportScore",
    "VisualEvidenceAnchor",
    "evaluate_multimodal_support",
    "image_digest",
    "score_multimodal_evidence",
]
