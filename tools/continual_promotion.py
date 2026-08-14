"""Continual-learning safeguards layered onto feedback-driven promotion decisions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

from tools.feedback_promotion import PromotionDecision


def _finite(value: float, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite.")
    return parsed


@dataclass(frozen=True)
class ContinualEvidence:
    drift_score: float
    forgetting_delta: float
    forward_transfer_delta: float
    privacy_safe_replay: bool
    independent_rollback_ready: bool
    adapter_version: str

    def __post_init__(self) -> None:
        for name in ("drift_score", "forgetting_delta", "forward_transfer_delta"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.drift_score < 0.0:
            raise ValueError("drift_score must be non-negative.")
        if not isinstance(self.privacy_safe_replay, bool):
            raise ValueError("privacy_safe_replay must be boolean.")
        if not isinstance(self.independent_rollback_ready, bool):
            raise ValueError("independent_rollback_ready must be boolean.")
        version = str(self.adapter_version).strip()
        if not version or len(version) > 500:
            raise ValueError("adapter_version is invalid.")
        object.__setattr__(self, "adapter_version", version)


@dataclass(frozen=True)
class ContinualPromotionPolicy:
    max_drift_score: float = 1.0
    max_forgetting_delta: float = 0.02
    min_forward_transfer_delta: float = -0.02
    require_privacy_safe_replay: bool = True
    require_independent_rollback: bool = True

    def __post_init__(self) -> None:
        for name in ("max_drift_score", "max_forgetting_delta", "min_forward_transfer_delta"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.max_drift_score < 0.0 or self.max_forgetting_delta < 0.0:
            raise ValueError("maximum drift/forgetting thresholds must be non-negative.")


@dataclass(frozen=True)
class ContinualPromotionDecision:
    decision_id: str
    eligible: bool
    reason_codes: tuple[str, ...]
    base_promotion_decision_id: str
    adapter_version: str
    evidence: ContinualEvidence
    policy_fingerprint: str


def _sha256(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def evaluate_continual_promotion(
    *,
    base: PromotionDecision,
    evidence: ContinualEvidence,
    policy: ContinualPromotionPolicy | None = None,
) -> ContinualPromotionDecision:
    selected = policy or ContinualPromotionPolicy()
    reasons = list(base.reason_codes)
    if evidence.drift_score > selected.max_drift_score:
        reasons.append("drift_budget_exceeded")
    if evidence.forgetting_delta > selected.max_forgetting_delta:
        reasons.append("forgetting_budget_exceeded")
    if evidence.forward_transfer_delta < selected.min_forward_transfer_delta:
        reasons.append("forward_transfer_regression")
    if selected.require_privacy_safe_replay and not evidence.privacy_safe_replay:
        reasons.append("privacy_safe_replay_not_verified")
    if selected.require_independent_rollback and not evidence.independent_rollback_ready:
        reasons.append("independent_rollback_not_ready")
    policy_fingerprint = _sha256(asdict(selected))
    payload = {
        "base": base.decision_id,
        "adapter_version": evidence.adapter_version,
        "evidence": asdict(evidence),
        "policy": policy_fingerprint,
        "reasons": reasons,
    }
    return ContinualPromotionDecision(
        decision_id=_sha256(payload),
        eligible=base.eligible and not reasons,
        reason_codes=tuple(reasons),
        base_promotion_decision_id=base.decision_id,
        adapter_version=evidence.adapter_version,
        evidence=evidence,
        policy_fingerprint=policy_fingerprint,
    )


__all__ = [
    "ContinualEvidence",
    "ContinualPromotionDecision",
    "ContinualPromotionPolicy",
    "evaluate_continual_promotion",
]
