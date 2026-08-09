"""Deterministic uncertainty-aware stopping for retrieval/verification agents."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any


def _unit(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be between 0 and 1.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be between 0 and 1.") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return result


def _finite(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite.") from exc
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return result


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class AgentStoppingSignals:
    evidence_sufficiency: float
    answer_confidence: float
    agent_disagreement: float
    contradiction_risk: float
    uncertainty: float
    marginal_improvement: float
    remaining_budget_fraction: float
    external_currentness_required: bool = False
    current_evidence_fraction: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "evidence_sufficiency",
            "answer_confidence",
            "agent_disagreement",
            "contradiction_risk",
            "uncertainty",
            "remaining_budget_fraction",
            "current_evidence_fraction",
        ):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        object.__setattr__(
            self,
            "marginal_improvement",
            _finite(self.marginal_improvement, "marginal_improvement", -1.0, 1.0),
        )
        if not isinstance(self.external_currentness_required, bool):
            raise ValueError("external_currentness_required must be boolean.")


@dataclass(frozen=True)
class AgentStoppingPolicy:
    sufficiency_threshold: float = 0.80
    confidence_threshold: float = 0.75
    disagreement_threshold: float = 0.30
    contradiction_threshold: float = 0.30
    uncertainty_threshold: float = 0.35
    current_evidence_threshold: float = 0.80
    minimum_marginal_improvement: float = 0.01
    low_budget_threshold: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "sufficiency_threshold",
            "confidence_threshold",
            "disagreement_threshold",
            "contradiction_threshold",
            "uncertainty_threshold",
            "current_evidence_threshold",
            "low_budget_threshold",
        ):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        object.__setattr__(
            self,
            "minimum_marginal_improvement",
            _finite(
                self.minimum_marginal_improvement,
                "minimum_marginal_improvement",
                0.0,
                1.0,
            ),
        )


@dataclass(frozen=True)
class AgentStoppingDecision:
    action: str
    reasons: tuple[str, ...]
    confidence_to_stop: float
    decision_digest: str

    def __post_init__(self) -> None:
        if self.action not in {"stop", "continue", "escalate", "abstain"}:
            raise ValueError("stopping action is unsupported.")
        if not isinstance(self.reasons, tuple) or not self.reasons or any(
            not isinstance(reason, str) or not reason or len(reason) > 200
            for reason in self.reasons
        ):
            raise ValueError("stopping reasons are invalid.")
        object.__setattr__(
            self,
            "confidence_to_stop",
            _unit(self.confidence_to_stop, "confidence_to_stop"),
        )
        if (
            not isinstance(self.decision_digest, str)
            or len(self.decision_digest) != 64
            or any(ch not in "0123456789abcdef" for ch in self.decision_digest)
        ):
            raise ValueError("decision_digest must be SHA-256.")


def decide_agent_stopping(
    signals: AgentStoppingSignals,
    *,
    policy: AgentStoppingPolicy | None = None,
) -> AgentStoppingDecision:
    """Choose stop/continue/escalate/abstain from bounded evidence and budget signals."""

    if not isinstance(signals, AgentStoppingSignals):
        raise ValueError("signals must be AgentStoppingSignals.")
    selected = policy or AgentStoppingPolicy()
    if not isinstance(selected, AgentStoppingPolicy):
        raise ValueError("policy must be AgentStoppingPolicy.")

    conflicts: list[str] = []
    if signals.agent_disagreement > selected.disagreement_threshold:
        conflicts.append("agent_disagreement_high")
    if signals.contradiction_risk > selected.contradiction_threshold:
        conflicts.append("contradiction_risk_high")
    if signals.external_currentness_required and (
        signals.current_evidence_fraction < selected.current_evidence_threshold
    ):
        conflicts.append("current_evidence_insufficient")

    evidence_ready = signals.evidence_sufficiency >= selected.sufficiency_threshold
    answer_ready = signals.answer_confidence >= selected.confidence_threshold
    uncertainty_ok = signals.uncertainty <= selected.uncertainty_threshold
    plateaued = signals.marginal_improvement < selected.minimum_marginal_improvement
    budget_low = signals.remaining_budget_fraction <= selected.low_budget_threshold

    if conflicts and budget_low:
        action = "abstain"
        reasons = tuple(conflicts + ["budget_exhausted_with_unresolved_risk"])
    elif conflicts:
        action = "escalate"
        reasons = tuple(conflicts)
    elif evidence_ready and answer_ready and uncertainty_ok:
        action = "stop"
        reasons = ("evidence_and_confidence_sufficient",)
    elif budget_low and not (evidence_ready and answer_ready):
        action = "abstain"
        reasons = ("budget_exhausted_before_sufficiency",)
    elif plateaued and signals.evidence_sufficiency < selected.sufficiency_threshold:
        action = "escalate"
        reasons = ("retrieval_plateau_below_sufficiency",)
    else:
        action = "continue"
        reasons = ("additional_evidence_has_expected_value",)

    stop_confidence = min(
        signals.evidence_sufficiency,
        signals.answer_confidence,
        1.0 - signals.uncertainty,
        1.0 - signals.agent_disagreement,
        1.0 - signals.contradiction_risk,
        signals.current_evidence_fraction if signals.external_currentness_required else 1.0,
    )
    payload = {
        "signals": asdict(signals),
        "policy": asdict(selected),
        "action": action,
        "reasons": reasons,
        "confidence_to_stop": stop_confidence,
    }
    return AgentStoppingDecision(
        action=action,
        reasons=reasons,
        confidence_to_stop=stop_confidence,
        decision_digest=_digest(payload),
    )


__all__ = [
    "AgentStoppingDecision",
    "AgentStoppingPolicy",
    "AgentStoppingSignals",
    "decide_agent_stopping",
]
