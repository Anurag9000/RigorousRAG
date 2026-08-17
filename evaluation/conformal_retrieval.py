"""Finite-sample conformal calibration for retrieval support and abstention.

This module adds a distribution-free uncertainty layer around already-produced retrieval
scores. It does not train or invoke a retriever. Calibration examples supply a scalar
nonconformity score for the best known supporting candidate; the resulting finite-sample
quantile can be used to form support sets or force abstention when no candidate clears
the calibrated threshold.

Exchangeability is an assumption of the method, not something this source can guarantee;
the calibration manifest therefore binds domain/split/model identities so callers can
refuse invalid cross-domain reuse.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

_MAX_CALIBRATION = 10_000_000


def _identifier(value: Any, label: str, maximum: int = 2_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if not result or len(result) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{label} is invalid")
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
        raise ValueError(f"{label} must lie in [0,1]")
    return result


def _sha256(value: Any, label: str) -> str:
    digest = _identifier(value, label, 64).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ConformalCalibrationManifest:
    calibration_id: str
    dataset_manifest_digest: str
    split_digest: str
    retrieval_stack_digest: str
    scoring_contract_digest: str
    domain_id: str
    alpha: float
    example_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "calibration_id", _identifier(self.calibration_id, "calibration_id"))
        for name in ("dataset_manifest_digest", "split_digest", "retrieval_stack_digest", "scoring_contract_digest"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(self, "domain_id", _identifier(self.domain_id, "domain_id"))
        alpha = _probability(self.alpha, "alpha")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between zero and one")
        object.__setattr__(self, "alpha", alpha)
        if isinstance(self.example_count, bool) or not isinstance(self.example_count, int) or not 1 <= self.example_count <= _MAX_CALIBRATION:
            raise ValueError("example_count is invalid")

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))


@dataclass(frozen=True)
class ConformalThreshold:
    manifest: ConformalCalibrationManifest
    nonconformity_threshold: float
    finite_sample_rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, ConformalCalibrationManifest):
            raise ValueError("manifest must be ConformalCalibrationManifest")
        object.__setattr__(
            self,
            "nonconformity_threshold",
            _finite(self.nonconformity_threshold, "nonconformity_threshold"),
        )
        if isinstance(self.finite_sample_rank, bool) or not isinstance(self.finite_sample_rank, int):
            raise ValueError("finite_sample_rank must be an integer")
        if not 1 <= self.finite_sample_rank <= self.manifest.example_count:
            raise ValueError("finite_sample_rank lies outside the calibration sample")


@dataclass(frozen=True)
class RetrievalSupportCandidate:
    candidate_id: str
    support_score: float
    nonconformity_score: float
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _identifier(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "support_score", _finite(self.support_score, "support_score"))
        object.__setattr__(self, "nonconformity_score", _finite(self.nonconformity_score, "nonconformity_score"))
        if not isinstance(self.metadata, Mapping) or len(self.metadata) > 1_000:
            raise ValueError("metadata must be a bounded mapping")
        object.__setattr__(
            self,
            "metadata",
            {
                _identifier(key, "metadata key", 300): _identifier(value, "metadata value", 10_000)
                for key, value in self.metadata.items()
            },
        )


@dataclass(frozen=True)
class ConformalSupportDecision:
    accepted: tuple[RetrievalSupportCandidate, ...]
    abstain: bool
    threshold: float
    reason: str
    calibration_digest: str


def nonconformity_from_probability(support_probability: Any) -> float:
    """Map a calibrated support probability to the standard ``1 - p`` nonconformity score."""

    return 1.0 - _probability(support_probability, "support_probability")


def nonconformity_from_margin(best_support_score: Any, competing_score: Any) -> float:
    """A monotone margin nonconformity where larger competing margins are less conforming."""

    best = _finite(best_support_score, "best_support_score")
    competitor = _finite(competing_score, "competing_score")
    return competitor - best


def fit_split_conformal_threshold(
    nonconformity_scores: Sequence[Any],
    *,
    calibration_id: str,
    dataset_manifest_digest: str,
    split_digest: str,
    retrieval_stack_digest: str,
    scoring_contract_digest: str,
    domain_id: str,
    alpha: float = 0.10,
) -> ConformalThreshold:
    """Fit the finite-sample split-conformal upper quantile.

    The finite threshold uses rank ``ceil((n+1)*(1-alpha))``.  If this rank would be
    ``n+1``, the requested alpha is not supportable by a finite threshold from the
    supplied sample, so the function fails closed instead of clipping and overstating
    the nominal guarantee.
    """

    if not nonconformity_scores or len(nonconformity_scores) > _MAX_CALIBRATION:
        raise ValueError("nonconformity_scores must be non-empty and bounded")
    selected_alpha = _probability(alpha, "alpha")
    if not 0.0 < selected_alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    values = sorted(_finite(value, "nonconformity score") for value in nonconformity_scores)
    n = len(values)
    rank = int(math.ceil((n + 1) * (1.0 - selected_alpha)))
    if rank > n:
        minimum_alpha = 1.0 / (n + 1)
        raise ValueError(
            f"calibration sample of size {n} cannot provide a finite split-conformal threshold "
            f"for alpha={selected_alpha:.12g}; use alpha >= {minimum_alpha:.12g} or more calibration examples"
        )
    manifest = ConformalCalibrationManifest(
        calibration_id=calibration_id,
        dataset_manifest_digest=dataset_manifest_digest,
        split_digest=split_digest,
        retrieval_stack_digest=retrieval_stack_digest,
        scoring_contract_digest=scoring_contract_digest,
        domain_id=domain_id,
        alpha=selected_alpha,
        example_count=n,
    )
    return ConformalThreshold(manifest, values[rank - 1], rank)


def conformal_support_set(
    candidates: Sequence[RetrievalSupportCandidate],
    threshold: ConformalThreshold,
    *,
    expected_retrieval_stack_digest: str,
    expected_scoring_contract_digest: str,
    expected_domain_id: str,
    max_candidates: int = 100,
) -> ConformalSupportDecision:
    """Return candidates inside the calibrated support set or abstain when none qualify."""

    if not isinstance(threshold, ConformalThreshold):
        raise ValueError("threshold must be ConformalThreshold")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or not 1 <= max_candidates <= 1_000_000:
        raise ValueError("max_candidates is invalid")
    manifest = threshold.manifest
    if manifest.retrieval_stack_digest != _sha256(expected_retrieval_stack_digest, "expected_retrieval_stack_digest"):
        raise ValueError("retrieval stack differs from conformal calibration")
    if manifest.scoring_contract_digest != _sha256(expected_scoring_contract_digest, "expected_scoring_contract_digest"):
        raise ValueError("scoring contract differs from conformal calibration")
    if manifest.domain_id != _identifier(expected_domain_id, "expected_domain_id"):
        raise ValueError("domain differs from conformal calibration")
    if len(candidates) > 1_000_000 or any(not isinstance(value, RetrievalSupportCandidate) for value in candidates):
        raise ValueError("candidates must be bounded RetrievalSupportCandidate values")
    accepted = [
        candidate for candidate in candidates if candidate.nonconformity_score <= threshold.nonconformity_threshold
    ]
    accepted.sort(key=lambda candidate: (candidate.nonconformity_score, -candidate.support_score, candidate.candidate_id))
    selected = tuple(accepted[:max_candidates])
    return ConformalSupportDecision(
        accepted=selected,
        abstain=not selected,
        threshold=threshold.nonconformity_threshold,
        reason="no candidate lies inside calibrated support set" if not selected else "calibrated support set is non-empty",
        calibration_digest=manifest.digest,
    )


@dataclass(frozen=True)
class SelectiveRiskMetrics:
    total: int
    covered: int
    coverage: float
    error_rate_on_covered: float | None
    abstention_rate: float


def evaluate_selective_risk(correct_when_answered: Sequence[bool | None]) -> SelectiveRiskMetrics:
    """Measure coverage/error when ``None`` denotes an abstained example."""

    if len(correct_when_answered) > _MAX_CALIBRATION:
        raise ValueError("selective-risk sequence is too large")
    total = len(correct_when_answered)
    if total == 0:
        return SelectiveRiskMetrics(0, 0, 0.0, None, 0.0)
    for value in correct_when_answered:
        if value is not None and not isinstance(value, bool):
            raise ValueError("correct_when_answered values must be bool or None")
    answered = [value for value in correct_when_answered if value is not None]
    covered = len(answered)
    error = None if not answered else sum(not value for value in answered) / covered
    return SelectiveRiskMetrics(
        total=total,
        covered=covered,
        coverage=covered / total,
        error_rate_on_covered=error,
        abstention_rate=(total - covered) / total,
    )


__all__ = [
    "ConformalCalibrationManifest",
    "ConformalSupportDecision",
    "ConformalThreshold",
    "RetrievalSupportCandidate",
    "SelectiveRiskMetrics",
    "canonical_digest",
    "conformal_support_set",
    "evaluate_selective_risk",
    "fit_split_conformal_threshold",
    "nonconformity_from_margin",
    "nonconformity_from_probability",
]
