"""Promotion evidence for randomized retrieval-policy interleaving experiments."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from evaluation.retrieval_interleaving import (
    InterleavingAggregate,
    InterleavingImpression,
    InterleavingOutcome,
    InterleavingSpec,
    aggregate_interleaving_preferences,
    preference_from_outcome,
)


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


@dataclass(frozen=True)
class InterleavingPromotionPolicy:
    candidate_team: str = "b"
    min_impressions: int = 1000
    min_decisive: int = 200
    min_candidate_preference_rate: float = 0.52
    min_candidate_wilson_low: float = 0.50
    max_sign_test_p_value: float = 0.05
    max_tie_fraction: float = 0.80

    def __post_init__(self) -> None:
        if self.candidate_team not in {"a", "b"}:
            raise ValueError("candidate_team must be a or b")
        for name in ("min_impressions", "min_decisive"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("min_candidate_preference_rate", "min_candidate_wilson_low", "max_sign_test_p_value", "max_tie_fraction"):
            object.__setattr__(self, name, _probability(getattr(self, name), name))

    @property
    def policy_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-interleaving-promotion-policy/v1", **asdict(self)})


@dataclass(frozen=True)
class InterleavingEvidence:
    spec_sha256: str
    evidence_sha256: str
    impression_count: int
    outcome_count: int
    aggregate: InterleavingAggregate


@dataclass(frozen=True)
class InterleavingPromotionReceipt:
    spec_sha256: str
    policy_a_sha256: str
    policy_b_sha256: str
    candidate_team: str
    candidate_policy_sha256: str
    evidence_sha256: str
    promotion_policy_sha256: str
    impression_count: int
    decisive_count: int
    candidate_wins: int
    baseline_wins: int
    ties: int
    candidate_preference_rate: float
    candidate_wilson_low: float
    candidate_wilson_high: float
    sign_test_p_value: float
    eligible: bool
    reason_codes: tuple[str, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("spec_sha256", "policy_a_sha256", "policy_b_sha256", "candidate_policy_sha256", "evidence_sha256", "promotion_policy_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if self.candidate_team not in {"a", "b"}:
            raise ValueError("candidate_team must be a or b")
        for name in ("impression_count", "decisive_count", "candidate_wins", "baseline_wins", "ties"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.candidate_wins + self.baseline_wins != self.decisive_count or self.decisive_count + self.ties != self.impression_count:
            raise ValueError("interleaving promotion counts are inconsistent")
        for name in ("candidate_preference_rate", "candidate_wilson_low", "candidate_wilson_high", "sign_test_p_value"):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        if not self.candidate_wilson_low <= self.candidate_preference_rate <= self.candidate_wilson_high:
            raise ValueError("candidate preference rate must lie inside Wilson interval")
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
            raise ValueError("receipt_sha256 does not match interleaving promotion content")
        object.__setattr__(self, "receipt_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-interleaving-promotion/v1",
            "spec_sha256": self.spec_sha256,
            "policy_a_sha256": self.policy_a_sha256,
            "policy_b_sha256": self.policy_b_sha256,
            "candidate_team": self.candidate_team,
            "candidate_policy_sha256": self.candidate_policy_sha256,
            "evidence_sha256": self.evidence_sha256,
            "promotion_policy_sha256": self.promotion_policy_sha256,
            "impression_count": self.impression_count,
            "decisive_count": self.decisive_count,
            "candidate_wins": self.candidate_wins,
            "baseline_wins": self.baseline_wins,
            "ties": self.ties,
            "candidate_preference_rate": self.candidate_preference_rate,
            "candidate_wilson_low": self.candidate_wilson_low,
            "candidate_wilson_high": self.candidate_wilson_high,
            "sign_test_p_value": self.sign_test_p_value,
            "eligible": self.eligible,
            "reason_codes": self.reason_codes,
        }


def build_interleaving_evidence(
    spec: InterleavingSpec,
    impressions: Sequence[InterleavingImpression],
    outcomes: Sequence[InterleavingOutcome],
) -> InterleavingEvidence:
    if not isinstance(spec, InterleavingSpec):
        raise ValueError("spec must be InterleavingSpec")
    impression_rows = tuple(impressions)
    outcome_rows = tuple(outcomes)
    if not impression_rows or len(impression_rows) != len(outcome_rows):
        raise ValueError("interleaving evidence requires exactly one outcome per impression")
    if any(not isinstance(item, InterleavingImpression) for item in impression_rows) or any(not isinstance(item, InterleavingOutcome) for item in outcome_rows):
        raise ValueError("interleaving evidence contains invalid values")
    if any(item.spec_sha256 != spec.spec_sha256 for item in impression_rows):
        raise ValueError("impression does not belong to interleaving spec")
    if len({item.impression_sha256 for item in impression_rows}) != len(impression_rows):
        raise ValueError("duplicate impression identities are not allowed")
    by_impression: Mapping[str, InterleavingOutcome] = {item.impression_sha256: item for item in outcome_rows}
    if len(by_impression) != len(outcome_rows):
        raise ValueError("duplicate outcome identities are not allowed")
    preferences = []
    pairs = []
    for impression in sorted(impression_rows, key=lambda item: (item.query_sha256, item.impression_index, item.impression_sha256)):
        outcome = by_impression.get(impression.impression_sha256)
        if outcome is None:
            raise ValueError("missing outcome for impression")
        preferences.append(preference_from_outcome(impression, outcome))
        pairs.append((impression.impression_sha256, outcome.outcome_sha256))
    if set(by_impression) != {item.impression_sha256 for item in impression_rows}:
        raise ValueError("outcome references an impression outside this evidence set")
    aggregate = aggregate_interleaving_preferences(preferences)
    evidence_digest = _digest({
        "schema": "rigorousrag-interleaving-evidence/v1",
        "spec_sha256": spec.spec_sha256,
        "pairs": pairs,
        "aggregate_sha256": aggregate.aggregate_sha256,
    })
    return InterleavingEvidence(spec.spec_sha256, evidence_digest, len(impression_rows), len(outcome_rows), aggregate)


def qualify_interleaving_experiment(
    spec: InterleavingSpec,
    evidence: InterleavingEvidence,
    *,
    policy: InterleavingPromotionPolicy = InterleavingPromotionPolicy(),
) -> InterleavingPromotionReceipt:
    if evidence.spec_sha256 != spec.spec_sha256:
        raise ValueError("interleaving evidence does not belong to spec")
    aggregate = evidence.aggregate
    if policy.candidate_team == "a":
        candidate_wins, baseline_wins = aggregate.wins_a, aggregate.wins_b
        rate = aggregate.preference_rate_a
        low, high = aggregate.wilson_low, aggregate.wilson_high
        candidate_policy = spec.policy_a_sha256
    else:
        candidate_wins, baseline_wins = aggregate.wins_b, aggregate.wins_a
        rate = 1.0 - aggregate.preference_rate_a
        low, high = 1.0 - aggregate.wilson_high, 1.0 - aggregate.wilson_low
        candidate_policy = spec.policy_b_sha256
    tie_fraction = aggregate.ties / aggregate.impression_count
    reasons: list[str] = []
    if aggregate.impression_count < policy.min_impressions:
        reasons.append("insufficient_impressions")
    if aggregate.decisive_count < policy.min_decisive:
        reasons.append("insufficient_decisive_comparisons")
    if tie_fraction > policy.max_tie_fraction:
        reasons.append("tie_fraction_exceeded")
    if rate < policy.min_candidate_preference_rate:
        reasons.append("candidate_preference_below_threshold")
    if low < policy.min_candidate_wilson_low:
        reasons.append("candidate_confidence_bound_below_threshold")
    if aggregate.sign_test_p_value > policy.max_sign_test_p_value:
        reasons.append("sign_test_not_significant")
    payload = {
        "schema": "rigorousrag-interleaving-promotion/v1",
        "spec_sha256": spec.spec_sha256,
        "policy_a_sha256": spec.policy_a_sha256,
        "policy_b_sha256": spec.policy_b_sha256,
        "candidate_team": policy.candidate_team,
        "candidate_policy_sha256": candidate_policy,
        "evidence_sha256": evidence.evidence_sha256,
        "promotion_policy_sha256": policy.policy_sha256,
        "impression_count": aggregate.impression_count,
        "decisive_count": aggregate.decisive_count,
        "candidate_wins": candidate_wins,
        "baseline_wins": baseline_wins,
        "ties": aggregate.ties,
        "candidate_preference_rate": rate,
        "candidate_wilson_low": low,
        "candidate_wilson_high": high,
        "sign_test_p_value": aggregate.sign_test_p_value,
        "eligible": not reasons,
        "reason_codes": tuple(sorted(set(reasons))),
    }
    return InterleavingPromotionReceipt(**payload, receipt_sha256=_digest(payload))


__all__ = [
    "InterleavingEvidence",
    "InterleavingPromotionPolicy",
    "InterleavingPromotionReceipt",
    "build_interleaving_evidence",
    "qualify_interleaving_experiment",
]
