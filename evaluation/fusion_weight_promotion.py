"""Held-out promotion gates for learned heterogeneous-retriever fusion weights."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from training.cross_profile_fusion_fitting import FusionWeightExample, LearnedFusionWeightArtifact

_EPS = 1e-9


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 digest")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return selected


def _probability(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be in [0, 1]")
    selected = float(value)
    if not math.isfinite(selected) or not 0.0 <= selected <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return selected


def _logit(probability: float) -> float:
    p = min(max(probability, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _binary_log_loss(probability: float, relevant: bool) -> float:
    p = min(max(probability, _EPS), 1.0 - _EPS)
    return -math.log(p) if relevant else -math.log(1.0 - p)


@dataclass(frozen=True)
class FusionWeightPromotionPolicy:
    min_examples: int = 200
    min_positive_examples: int = 20
    min_negative_examples: int = 20
    max_log_loss: float = 0.70
    max_brier: float = 0.25
    min_log_loss_improvement: float = 0.0
    min_brier_improvement: float = 0.0
    max_single_profile_weight: float = 0.98

    def __post_init__(self) -> None:
        for name in ("min_examples", "min_positive_examples", "min_negative_examples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("max_log_loss", "max_brier", "min_log_loss_improvement", "min_brier_improvement"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "max_single_profile_weight", _probability(self.max_single_profile_weight, "max_single_profile_weight"))

    @property
    def policy_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-fusion-weight-promotion-policy/v1", **asdict(self)})


@dataclass(frozen=True)
class FusionWeightPromotionReceipt:
    learned_artifact_sha256: str
    evaluation_split_sha256: str
    evaluation_examples_sha256: str
    policy_sha256: str
    example_count: int
    positive_count: int
    negative_count: int
    learned_log_loss: float
    uniform_log_loss: float
    learned_brier: float
    uniform_brier: float
    eligible: bool
    reason_codes: tuple[str, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("learned_artifact_sha256", "evaluation_split_sha256", "evaluation_examples_sha256", "policy_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        for name in ("example_count", "positive_count", "negative_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.positive_count + self.negative_count != self.example_count:
            raise ValueError("class counts must sum to example_count")
        for name in ("learned_log_loss", "uniform_log_loss", "learned_brier", "uniform_brier"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if not isinstance(self.eligible, bool):
            raise ValueError("eligible must be boolean")
        reasons = tuple(sorted(set(self.reason_codes)))
        if self.eligible and reasons:
            raise ValueError("eligible receipt cannot contain failure reasons")
        if not self.eligible and not reasons:
            raise ValueError("ineligible receipt requires failure reasons")
        object.__setattr__(self, "reason_codes", reasons)
        expected = _digest(self._payload())
        provided = _sha(self.receipt_sha256, "receipt_sha256")
        if expected != provided:
            raise ValueError("receipt_sha256 does not match promotion receipt content")
        object.__setattr__(self, "receipt_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-fusion-weight-promotion/v1",
            "learned_artifact_sha256": self.learned_artifact_sha256,
            "evaluation_split_sha256": self.evaluation_split_sha256,
            "evaluation_examples_sha256": self.evaluation_examples_sha256,
            "policy_sha256": self.policy_sha256,
            "example_count": self.example_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "learned_log_loss": self.learned_log_loss,
            "uniform_log_loss": self.uniform_log_loss,
            "learned_brier": self.learned_brier,
            "uniform_brier": self.uniform_brier,
            "eligible": self.eligible,
            "reason_codes": self.reason_codes,
        }


def _examples_digest(examples: tuple[FusionWeightExample, ...], profile_ids: tuple[str, ...]) -> str:
    return _digest({
        "schema": "rigorousrag-fusion-weight-promotion-examples/v1",
        "profile_ids": profile_ids,
        "examples": [
            {
                "probabilities": tuple((profile, item.probabilities[profile]) for profile in profile_ids),
                "relevant": item.relevant,
                "weight": item.weight,
            }
            for item in examples
        ],
    })


def qualify_learned_fusion_weights(
    artifact: LearnedFusionWeightArtifact,
    examples: Iterable[FusionWeightExample],
    *,
    evaluation_split_sha256: str,
    policy: FusionWeightPromotionPolicy = FusionWeightPromotionPolicy(),
) -> FusionWeightPromotionReceipt:
    if not isinstance(artifact, LearnedFusionWeightArtifact):
        raise ValueError("artifact must be LearnedFusionWeightArtifact")
    values = tuple(examples)
    if not values or any(not isinstance(item, FusionWeightExample) for item in values):
        raise ValueError("examples must be a non-empty FusionWeightExample collection")
    profile_ids = tuple(profile for profile, _ in artifact.profile_weights)
    if any(set(item.probabilities) != set(profile_ids) for item in values):
        raise ValueError("evaluation examples must exactly cover learned profiles")
    uniform_weight = 1.0 / len(profile_ids)
    learned_loss = uniform_loss = learned_brier = uniform_brier = total_weight = 0.0
    positive_count = 0
    for item in values:
        if item.relevant:
            positive_count += 1
        learned_probability = artifact.probability(item.probabilities)
        uniform_probability = _sigmoid(sum(uniform_weight * _logit(item.probabilities[profile]) for profile in profile_ids))
        learned_loss += item.weight * _binary_log_loss(learned_probability, item.relevant)
        uniform_loss += item.weight * _binary_log_loss(uniform_probability, item.relevant)
        target = float(item.relevant)
        learned_brier += item.weight * (learned_probability - target) ** 2
        uniform_brier += item.weight * (uniform_probability - target) ** 2
        total_weight += item.weight
    learned_loss /= total_weight
    uniform_loss /= total_weight
    learned_brier /= total_weight
    uniform_brier /= total_weight
    negative_count = len(values) - positive_count
    reasons: list[str] = []
    if len(values) < policy.min_examples:
        reasons.append("insufficient_evaluation_examples")
    if positive_count < policy.min_positive_examples:
        reasons.append("insufficient_positive_examples")
    if negative_count < policy.min_negative_examples:
        reasons.append("insufficient_negative_examples")
    if learned_loss > policy.max_log_loss:
        reasons.append("log_loss_threshold_exceeded")
    if learned_brier > policy.max_brier:
        reasons.append("brier_threshold_exceeded")
    if uniform_loss - learned_loss < policy.min_log_loss_improvement:
        reasons.append("insufficient_log_loss_improvement")
    if uniform_brier - learned_brier < policy.min_brier_improvement:
        reasons.append("insufficient_brier_improvement")
    if max(weight for _, weight in artifact.profile_weights) > policy.max_single_profile_weight:
        reasons.append("profile_weight_collapse")
    split_digest = _sha(evaluation_split_sha256, "evaluation_split_sha256")
    payload = {
        "schema": "rigorousrag-fusion-weight-promotion/v1",
        "learned_artifact_sha256": artifact.artifact_sha256,
        "evaluation_split_sha256": split_digest,
        "evaluation_examples_sha256": _examples_digest(values, profile_ids),
        "policy_sha256": policy.policy_sha256,
        "example_count": len(values),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "learned_log_loss": learned_loss,
        "uniform_log_loss": uniform_loss,
        "learned_brier": learned_brier,
        "uniform_brier": uniform_brier,
        "eligible": not reasons,
        "reason_codes": tuple(sorted(set(reasons))),
    }
    return FusionWeightPromotionReceipt(**payload, receipt_sha256=_digest(payload))


__all__ = ["FusionWeightPromotionPolicy", "FusionWeightPromotionReceipt", "qualify_learned_fusion_weights"]
