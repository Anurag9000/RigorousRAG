"""Transparent source-trust/applicability policy for evidence ranking and warnings.

Trust scores are policy features, never truth probabilities.  The engine separates source
integrity/status, methodological review, topical applicability and freshness so a highly
ranked source cannot silently become scientifically authoritative merely because of its
publisher or retrieval score.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

_STATUS = frozenset({"active", "corrected", "superseded", "retracted", "withdrawn", "unknown"})
_SOURCE_TYPES = frozenset({"primary_study", "systematic_review", "meta_analysis", "guideline", "dataset", "technical_report", "preprint", "conference", "web", "documentation", "model_output", "other"})


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _unit(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must lie in [0,1]")
    return parsed


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class SourceTrustFeatures:
    source_id: str
    source_type: str
    status: str = "unknown"
    provenance_integrity: float = 1.0
    methodological_quality: float = 0.5
    topical_applicability: float = 0.5
    freshness: float = 0.5
    independent_replication: float = 0.0
    reviewed: bool = False
    conflicts_of_interest_known: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id", 1000))
        source_type = _text(self.source_type, "source_type", 64).lower()
        if source_type not in _SOURCE_TYPES:
            raise ValueError("unsupported source_type")
        object.__setattr__(self, "source_type", source_type)
        status = _text(self.status, "status", 32).lower()
        if status not in _STATUS:
            raise ValueError("unsupported source status")
        object.__setattr__(self, "status", status)
        for name in ("provenance_integrity", "methodological_quality", "topical_applicability", "freshness", "independent_replication"):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        for name in ("reviewed", "conflicts_of_interest_known"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if len(self.notes) > 100:
            raise ValueError("notes exceed the item limit")
        object.__setattr__(self, "notes", tuple(_text(item, "note", 2000) for item in self.notes))


@dataclass(frozen=True)
class SourceTrustPolicy:
    version: str = "transparent-trust-v1"
    provenance_weight: float = 0.30
    methodology_weight: float = 0.25
    applicability_weight: float = 0.25
    freshness_weight: float = 0.10
    replication_weight: float = 0.10
    minimum_publish_score: float = 0.35
    require_review_for_causal_claims: bool = True
    block_statuses: tuple[str, ...] = ("retracted", "withdrawn", "superseded")

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _text(self.version, "version", 128))
        names = ("provenance_weight", "methodology_weight", "applicability_weight", "freshness_weight", "replication_weight")
        weights = [_unit(getattr(self, name), name) for name in names]
        if sum(weights) <= 0:
            raise ValueError("source trust policy weights must have positive sum")
        for name, value in zip(names, weights):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "minimum_publish_score", _unit(self.minimum_publish_score, "minimum_publish_score"))
        if not isinstance(self.require_review_for_causal_claims, bool):
            raise ValueError("require_review_for_causal_claims must be boolean")
        block = tuple(dict.fromkeys(_text(item, "blocked status", 32).lower() for item in self.block_statuses))
        if any(item not in _STATUS for item in block):
            raise ValueError("block_statuses contains an unsupported status")
        object.__setattr__(self, "block_statuses", block)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True)
class SourceTrustDecision:
    source_id: str
    score: float
    eligible_for_new_claims: bool
    reasons: tuple[str, ...]
    policy_sha256: str


def evaluate_source_trust(features: SourceTrustFeatures, policy: SourceTrustPolicy, *, causal_claim: bool = False) -> SourceTrustDecision:
    weights = {
        "provenance_integrity": policy.provenance_weight,
        "methodological_quality": policy.methodology_weight,
        "topical_applicability": policy.applicability_weight,
        "freshness": policy.freshness_weight,
        "independent_replication": policy.replication_weight,
    }
    total_weight = sum(weights.values())
    score = sum(getattr(features, name) * weight for name, weight in weights.items()) / total_weight
    reasons: list[str] = []
    eligible = score >= policy.minimum_publish_score
    if features.status in policy.block_statuses:
        eligible = False
        reasons.append(f"blocked_status:{features.status}")
    if features.provenance_integrity < 0.5:
        eligible = False
        reasons.append("weak_provenance_integrity")
    if causal_claim and policy.require_review_for_causal_claims and not features.reviewed:
        eligible = False
        reasons.append("causal_claim_requires_reviewed_source")
    if score < policy.minimum_publish_score:
        reasons.append("below_policy_score")
    if not features.conflicts_of_interest_known:
        reasons.append("conflict_of_interest_status_unknown")
    if not reasons:
        reasons.append("policy_eligible")
    return SourceTrustDecision(features.source_id, score, eligible, tuple(reasons), policy.fingerprint)


def rank_source_trust(features: Sequence[SourceTrustFeatures], policy: SourceTrustPolicy, *, causal_claim: bool = False) -> tuple[SourceTrustDecision, ...]:
    decisions = [evaluate_source_trust(item, policy, causal_claim=causal_claim) for item in features]
    decisions.sort(key=lambda item: (not item.eligible_for_new_claims, -item.score, item.source_id))
    return tuple(decisions)


__all__ = ["SourceTrustDecision", "SourceTrustFeatures", "SourceTrustPolicy", "evaluate_source_trust", "rank_source_trust"]
