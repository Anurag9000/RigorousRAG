"""Versioned adaptive retrieval-policy runtime with deterministic fallback.

A learned provider may select route, depth, top-k, expansion, reranking and stopping
behavior from bounded features. The runtime validates every action and falls back to a
conservative deterministic policy on provider failure. It performs no training itself.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence

_ALLOWED_ROUTES = frozenset({"dense", "sparse", "hybrid", "web", "scholarly", "graph", "multimodal", "multihop"})


def _text(value: Any, label: str, maximum: int = 500, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _unit(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must lie in [0,1]")
    return result


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class RetrievalPolicyFeatures:
    query_length: int
    lexical_specificity: float
    entity_count: int
    temporal_signal: float
    comparison_signal: float
    numerical_signal: float
    prior_evidence_count: int = 0
    prior_best_score: float = 0.0
    prior_document_diversity: float = 0.0
    remaining_budget_fraction: float = 1.0
    domain_scores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, minimum, maximum in (("query_length", 1, 100_000), ("entity_count", 0, 1000), ("prior_evidence_count", 0, 100_000)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{name} is invalid")
        for name in ("lexical_specificity", "temporal_signal", "comparison_signal", "numerical_signal", "prior_best_score", "prior_document_diversity", "remaining_budget_fraction"):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        if not isinstance(self.domain_scores, Mapping) or len(self.domain_scores) > 64:
            raise ValueError("domain_scores must be a bounded mapping")
        object.__setattr__(self, "domain_scores", {_text(str(k), "domain", 100): _unit(v, "domain score") for k, v in self.domain_scores.items()})


@dataclass(frozen=True)
class RetrievalPolicyAction:
    route: str
    top_k: int
    depth: int
    rerank: bool
    expand_query: bool
    continue_retrieval: bool
    abstain: bool
    confidence: float
    reason_code: str

    def __post_init__(self) -> None:
        route = _text(self.route, "route", 64).lower()
        if route not in _ALLOWED_ROUTES:
            raise ValueError("unsupported route")
        object.__setattr__(self, "route", route)
        for name, minimum, maximum in (("top_k", 1, 1000), ("depth", 1, 32)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{name} is invalid")
        for name in ("rerank", "expand_query", "continue_retrieval", "abstain"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        object.__setattr__(self, "confidence", _unit(self.confidence, "confidence"))
        object.__setattr__(self, "reason_code", _text(self.reason_code, "reason_code", 128))
        if self.abstain and self.continue_retrieval:
            raise ValueError("an action cannot both abstain and continue retrieval")


class RetrievalPolicyProvider(Protocol):
    @property
    def policy_id(self) -> str: ...
    @property
    def version(self) -> str: ...
    def decide(self, features: RetrievalPolicyFeatures) -> RetrievalPolicyAction: ...


@dataclass(frozen=True)
class DeterministicAdaptivePolicy:
    policy_id: str = "deterministic-adaptive"
    version: str = "1.0.0"

    def decide(self, features: RetrievalPolicyFeatures) -> RetrievalPolicyAction:
        if features.remaining_budget_fraction <= 0.05:
            if features.prior_evidence_count > 0 and features.prior_best_score >= 0.55:
                return RetrievalPolicyAction("hybrid", 10, 1, False, False, False, False, 0.8, "budget_stop_with_evidence")
            return RetrievalPolicyAction("hybrid", 5, 1, False, False, False, True, 0.9, "budget_exhausted")
        if features.prior_evidence_count and features.prior_best_score >= 0.8 and features.prior_document_diversity >= 0.5:
            return RetrievalPolicyAction("hybrid", 20, 1, True, False, False, False, 0.85, "sufficient_evidence")
        if features.comparison_signal >= 0.6 or features.entity_count >= 3:
            return RetrievalPolicyAction("multihop", 30, 4, True, True, True, False, 0.75, "multi_entity_or_comparison")
        if features.temporal_signal >= 0.6:
            return RetrievalPolicyAction("hybrid", 30, 2, True, True, True, False, 0.7, "temporal_query")
        if features.lexical_specificity >= 0.75:
            return RetrievalPolicyAction("sparse", 30, 1, True, False, True, False, 0.72, "lexically_specific")
        return RetrievalPolicyAction("hybrid", 40, 2, True, False, True, False, 0.68, "general_hybrid")


@dataclass(frozen=True)
class PolicyDecision:
    action: RetrievalPolicyAction
    policy_id: str
    version: str
    fallback_used: bool
    feature_sha256: str
    decision_sha256: str


def decide_policy(
    features: RetrievalPolicyFeatures,
    *,
    provider: RetrievalPolicyProvider | None = None,
    fallback: RetrievalPolicyProvider | None = None,
    allowed_routes: Sequence[str] = tuple(_ALLOWED_ROUTES),
) -> PolicyDecision:
    allowed = frozenset(_text(item, "allowed route", 64).lower() for item in allowed_routes)
    if not allowed or any(item not in _ALLOWED_ROUTES for item in allowed):
        raise ValueError("allowed_routes are invalid")
    fallback_policy = fallback or DeterministicAdaptivePolicy()
    chosen = provider or fallback_policy
    fallback_used = provider is None
    try:
        action = chosen.decide(features)
        if not isinstance(action, RetrievalPolicyAction) or action.route not in allowed:
            raise ValueError("policy action is invalid for this runtime")
        policy_id = _text(chosen.policy_id, "policy_id", 256)
        version = _text(chosen.version, "policy version", 100)
    except Exception:
        action = fallback_policy.decide(features)
        if action.route not in allowed:
            # Fail closed to the first locally allowed retrieval route, without enabling web.
            local_routes = [item for item in ("hybrid", "dense", "sparse", "graph", "multimodal") if item in allowed]
            if not local_routes:
                return PolicyDecision(RetrievalPolicyAction(next(iter(sorted(allowed))), 1, 1, False, False, False, True, 1.0, "no_safe_fallback_route"), "fail-closed", "1.0.0", True, hashlib.sha256(_canonical(asdict(features))).hexdigest(), hashlib.sha256(b"fail-closed").hexdigest())
            action = RetrievalPolicyAction(local_routes[0], 10, 1, False, False, False, True, 1.0, "provider_failure")
        policy_id = _text(fallback_policy.policy_id, "policy_id", 256)
        version = _text(fallback_policy.version, "policy version", 100)
        fallback_used = True
    feature_sha = hashlib.sha256(_canonical(asdict(features))).hexdigest()
    decision_sha = hashlib.sha256(_canonical({"features": feature_sha, "action": asdict(action), "policy_id": policy_id, "version": version, "fallback_used": fallback_used})).hexdigest()
    return PolicyDecision(action, policy_id, version, fallback_used, feature_sha, decision_sha)


__all__ = [
    "DeterministicAdaptivePolicy", "PolicyDecision", "RetrievalPolicyAction",
    "RetrievalPolicyFeatures", "RetrievalPolicyProvider", "decide_policy",
]
