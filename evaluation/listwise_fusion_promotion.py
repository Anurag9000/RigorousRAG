"""Rank-aware promotion gates for learned ListNet cross-profile fusion weights."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from training.cross_profile_listwise_fusion import FusionRankingQuery, LearnedListwiseFusionArtifact, ranking_queries_sha256


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 digest")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return selected


def _bounded_probability(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be in [0, 1]")
    selected = float(value)
    if not math.isfinite(selected) or not 0.0 <= selected <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return selected


def _logit(value: float) -> float:
    p = min(max(float(value), 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def _ndcg(grades_and_scores: list[tuple[float, float]], k: int | None) -> float:
    if not grades_and_scores:
        return 0.0
    limit = len(grades_and_scores) if k is None else min(k, len(grades_and_scores))
    ranked = sorted(grades_and_scores, key=lambda row: (-row[1], -row[0]))[:limit]
    ideal = sorted((grade for grade, _ in grades_and_scores), reverse=True)[:limit]
    dcg = sum((2.0 ** grade - 1.0) / math.log2(index + 2.0) for index, (grade, _) in enumerate(ranked))
    idcg = sum((2.0 ** grade - 1.0) / math.log2(index + 2.0) for index, grade in enumerate(ideal))
    return dcg / idcg if idcg > 0.0 else 0.0


@dataclass(frozen=True)
class ListwiseFusionPromotionPolicy:
    min_queries: int = 50
    ndcg_k: int | None = 10
    min_mean_ndcg: float = 0.0
    min_mean_ndcg_improvement: float = 0.0
    max_query_regression_fraction: float = 0.25
    max_single_profile_weight: float = 0.98

    def __post_init__(self) -> None:
        if isinstance(self.min_queries, bool) or not isinstance(self.min_queries, int) or self.min_queries < 1:
            raise ValueError("min_queries must be positive")
        if self.ndcg_k is not None and (isinstance(self.ndcg_k, bool) or not isinstance(self.ndcg_k, int) or self.ndcg_k < 1):
            raise ValueError("ndcg_k must be positive when set")
        object.__setattr__(self, "min_mean_ndcg", _bounded_probability(self.min_mean_ndcg, "min_mean_ndcg"))
        improvement = float(self.min_mean_ndcg_improvement)
        if not math.isfinite(improvement) or improvement < 0.0:
            raise ValueError("min_mean_ndcg_improvement must be finite and non-negative")
        object.__setattr__(self, "min_mean_ndcg_improvement", improvement)
        object.__setattr__(self, "max_query_regression_fraction", _bounded_probability(self.max_query_regression_fraction, "max_query_regression_fraction"))
        object.__setattr__(self, "max_single_profile_weight", _bounded_probability(self.max_single_profile_weight, "max_single_profile_weight"))

    @property
    def policy_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-listwise-fusion-promotion-policy/v1", **asdict(self)})


@dataclass(frozen=True)
class ListwiseFusionPromotionReceipt:
    learned_artifact_sha256: str
    evaluation_split_sha256: str
    evaluation_queries_sha256: str
    policy_sha256: str
    query_count: int
    learned_mean_ndcg: float
    uniform_mean_ndcg: float
    regression_fraction: float
    eligible: bool
    reason_codes: tuple[str, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("learned_artifact_sha256", "evaluation_split_sha256", "evaluation_queries_sha256", "policy_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if isinstance(self.query_count, bool) or not isinstance(self.query_count, int) or self.query_count < 1:
            raise ValueError("query_count must be positive")
        object.__setattr__(self, "learned_mean_ndcg", _bounded_probability(self.learned_mean_ndcg, "learned_mean_ndcg"))
        object.__setattr__(self, "uniform_mean_ndcg", _bounded_probability(self.uniform_mean_ndcg, "uniform_mean_ndcg"))
        object.__setattr__(self, "regression_fraction", _bounded_probability(self.regression_fraction, "regression_fraction"))
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
            raise ValueError("receipt_sha256 does not match promotion receipt")
        object.__setattr__(self, "receipt_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-listwise-fusion-promotion/v1",
            "learned_artifact_sha256": self.learned_artifact_sha256,
            "evaluation_split_sha256": self.evaluation_split_sha256,
            "evaluation_queries_sha256": self.evaluation_queries_sha256,
            "policy_sha256": self.policy_sha256,
            "query_count": self.query_count,
            "learned_mean_ndcg": self.learned_mean_ndcg,
            "uniform_mean_ndcg": self.uniform_mean_ndcg,
            "regression_fraction": self.regression_fraction,
            "eligible": self.eligible,
            "reason_codes": self.reason_codes,
        }


def qualify_listwise_fusion_weights(
    artifact: LearnedListwiseFusionArtifact,
    queries: Iterable[FusionRankingQuery],
    *,
    evaluation_split_sha256: str,
    policy: ListwiseFusionPromotionPolicy = ListwiseFusionPromotionPolicy(),
) -> ListwiseFusionPromotionReceipt:
    if not isinstance(artifact, LearnedListwiseFusionArtifact):
        raise ValueError("artifact must be LearnedListwiseFusionArtifact")
    values = tuple(queries)
    if not values or any(not isinstance(query, FusionRankingQuery) for query in values):
        raise ValueError("queries must be a non-empty FusionRankingQuery collection")
    profile_ids = tuple(profile for profile, _ in artifact.profile_weights)
    if any(any(set(candidate.probabilities) != set(profile_ids) for candidate in query.candidates) for query in values):
        raise ValueError("evaluation queries must exactly cover learned profiles")
    uniform_weight = 1.0 / len(profile_ids)
    learned_rows: list[float] = []
    uniform_rows: list[float] = []
    regressions = 0
    for query in values:
        learned_pairs: list[tuple[float, float]] = []
        uniform_pairs: list[tuple[float, float]] = []
        for candidate in query.candidates:
            learned_score = artifact.score(candidate.probabilities)
            uniform_score = sum(uniform_weight * _logit(candidate.probabilities[profile]) for profile in profile_ids)
            learned_pairs.append((candidate.relevance_grade, learned_score))
            uniform_pairs.append((candidate.relevance_grade, uniform_score))
        learned_ndcg = _ndcg(learned_pairs, policy.ndcg_k)
        uniform_ndcg = _ndcg(uniform_pairs, policy.ndcg_k)
        learned_rows.append(learned_ndcg)
        uniform_rows.append(uniform_ndcg)
        if learned_ndcg + 1e-12 < uniform_ndcg:
            regressions += 1
    learned_mean = sum(learned_rows) / len(learned_rows)
    uniform_mean = sum(uniform_rows) / len(uniform_rows)
    regression_fraction = regressions / len(values)
    reasons: list[str] = []
    if len(values) < policy.min_queries:
        reasons.append("insufficient_evaluation_queries")
    if learned_mean < policy.min_mean_ndcg:
        reasons.append("mean_ndcg_below_threshold")
    if learned_mean - uniform_mean < policy.min_mean_ndcg_improvement:
        reasons.append("insufficient_ndcg_improvement")
    if regression_fraction > policy.max_query_regression_fraction:
        reasons.append("query_regression_fraction_exceeded")
    if max(weight for _, weight in artifact.profile_weights) > policy.max_single_profile_weight:
        reasons.append("profile_weight_collapse")
    payload = {
        "schema": "rigorousrag-listwise-fusion-promotion/v1",
        "learned_artifact_sha256": artifact.artifact_sha256,
        "evaluation_split_sha256": _sha(evaluation_split_sha256, "evaluation_split_sha256"),
        "evaluation_queries_sha256": ranking_queries_sha256(values, profile_ids),
        "policy_sha256": policy.policy_sha256,
        "query_count": len(values),
        "learned_mean_ndcg": learned_mean,
        "uniform_mean_ndcg": uniform_mean,
        "regression_fraction": regression_fraction,
        "eligible": not reasons,
        "reason_codes": tuple(sorted(set(reasons))),
    }
    return ListwiseFusionPromotionReceipt(**payload, receipt_sha256=_digest(payload))


__all__ = ["ListwiseFusionPromotionPolicy", "ListwiseFusionPromotionReceipt", "qualify_listwise_fusion_weights"]
