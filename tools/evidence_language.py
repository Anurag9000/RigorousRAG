"""Deterministic evidence-strength constraints for publication wording.

This module does not generate scientific prose and does not infer evidence certainty. It
maps already-reviewed evidence state into an assertion-strength ceiling, required
qualifiers, prohibited phrase families, and abstention decisions. The result can be bound
into answer/report provenance so a generator cannot silently use stronger wording than
the reviewed evidence state permits.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from tools.evidence_certainty import EvidenceCertaintyAssessment

_STRENGTHS = ("abstain", "very_cautious", "cautious", "moderate", "assertive")
_CERTAINTY_CEILING = {
    "very_low": "very_cautious",
    "low": "cautious",
    "moderate": "moderate",
    "high": "assertive",
}
_SUPPORT_STATES = frozenset({"supported", "partially_supported", "unsupported", "unknown"})
_SOURCE_STATES = frozenset({"current", "corrected", "superseded", "withdrawn", "retracted", "unknown"})
_CLAIM_TYPES = frozenset({"descriptive", "associational", "causal", "treatment", "forecast", "quantitative"})

_CAUSAL_PATTERNS = (
    r"\bcauses?\b",
    r"\bcaused\b",
    r"\bcausal(?:ly)?\b",
    r"\bleads? to\b",
    r"\bresults? in\b",
    r"\bprevents?\b",
)
_TREATMENT_PATTERNS = (
    r"\bshould be treated\b",
    r"\bshould receive\b",
    r"\brecommended treatment\b",
    r"\bclinically indicated\b",
    r"\bstandard of care\b",
)
_CERTAINTY_PATTERNS = (
    r"\bproves?\b",
    r"\bdefinitive(?:ly)?\b",
    r"\bcertain(?:ly)?\b",
    r"\bconclusive(?:ly)?\b",
    r"\bguarantees?\b",
)


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _fraction(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} is invalid")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} is invalid") from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be between zero and one")
    return parsed


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def _strength_index(value: str) -> int:
    try:
        return _STRENGTHS.index(value)
    except ValueError as exc:
        raise ValueError("unsupported publication strength") from exc


def _minimum_strength(left: str, right: str) -> str:
    return _STRENGTHS[min(_strength_index(left), _strength_index(right))]


@dataclass(frozen=True)
class PublicationEvidenceState:
    claim_type: str
    support_state: str
    source_status: str = "current"
    reviewed_fraction: float = 0.0
    contradiction_count: int = 0
    causal_ready: bool = False
    treatment_ready: bool = False
    quantitative_consistent: bool | None = None
    certainty: EvidenceCertaintyAssessment | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        claim_type = _text(self.claim_type, "claim_type", 64).lower()
        if claim_type not in _CLAIM_TYPES:
            raise ValueError("unsupported claim_type")
        object.__setattr__(self, "claim_type", claim_type)
        support = _text(self.support_state, "support_state", 64).lower()
        if support not in _SUPPORT_STATES:
            raise ValueError("unsupported support_state")
        object.__setattr__(self, "support_state", support)
        status = _text(self.source_status, "source_status", 64).lower()
        if status not in _SOURCE_STATES:
            raise ValueError("unsupported source_status")
        object.__setattr__(self, "source_status", status)
        object.__setattr__(self, "reviewed_fraction", _fraction(self.reviewed_fraction, "reviewed_fraction"))
        if isinstance(self.contradiction_count, bool) or not isinstance(self.contradiction_count, int) or not 0 <= self.contradiction_count <= 1_000_000:
            raise ValueError("contradiction_count is invalid")
        if not isinstance(self.causal_ready, bool) or not isinstance(self.treatment_ready, bool):
            raise ValueError("causal_ready and treatment_ready must be booleans")
        if self.quantitative_consistent is not None and not isinstance(self.quantitative_consistent, bool):
            raise ValueError("quantitative_consistent must be boolean or null")
        if self.certainty is not None and not isinstance(self.certainty, EvidenceCertaintyAssessment):
            raise TypeError("certainty must be EvidenceCertaintyAssessment or null")
        if len(self.warnings) > 100:
            raise ValueError("too many warnings")
        object.__setattr__(self, "warnings", tuple(dict.fromkeys(_text(item, "warning", 2000) for item in self.warnings)))


@dataclass(frozen=True)
class LanguagePolicyDecision:
    allowed_strength: str
    abstain: bool
    reasons: tuple[str, ...]
    required_qualifiers: tuple[str, ...]
    prohibited_patterns: tuple[str, ...]
    certainty_level: str
    reviewed_fraction: float
    policy_fingerprint: str
    state_fingerprint: str

    def __post_init__(self) -> None:
        if self.allowed_strength not in _STRENGTHS:
            raise ValueError("allowed_strength is invalid")
        if not isinstance(self.abstain, bool):
            raise ValueError("abstain must be boolean")
        if self.abstain != (self.allowed_strength == "abstain"):
            raise ValueError("abstain must match allowed_strength")


@dataclass(frozen=True)
class ClaimLanguageCheck:
    compliant: bool
    violations: tuple[str, ...]
    decision_fingerprint: str
    claim_sha256: str


def evaluate_publication_language(
    state: PublicationEvidenceState,
    *,
    minimum_reviewed_fraction_for_assertive: float = 1.0,
    minimum_reviewed_fraction_for_moderate: float = 0.75,
) -> LanguagePolicyDecision:
    if not isinstance(state, PublicationEvidenceState):
        raise TypeError("state must be PublicationEvidenceState")
    assertive_review = _fraction(minimum_reviewed_fraction_for_assertive, "minimum_reviewed_fraction_for_assertive")
    moderate_review = _fraction(minimum_reviewed_fraction_for_moderate, "minimum_reviewed_fraction_for_moderate")
    if moderate_review > assertive_review:
        raise ValueError("moderate review threshold may not exceed assertive threshold")

    ceiling = "assertive"
    reasons: list[str] = []
    qualifiers: list[str] = []
    prohibited: list[str] = list(_CERTAINTY_PATTERNS)
    certainty_level = "unassessed"

    if state.source_status in {"retracted", "withdrawn"}:
        ceiling = "abstain"
        reasons.append(f"source_status:{state.source_status}")
    elif state.source_status in {"superseded", "corrected", "unknown"}:
        ceiling = _minimum_strength(ceiling, "very_cautious")
        reasons.append(f"source_status_requires_caution:{state.source_status}")
        qualifiers.append("source status may affect applicability")

    if state.support_state == "unsupported":
        ceiling = "abstain"
        reasons.append("claim_is_unsupported")
    elif state.support_state == "unknown":
        ceiling = _minimum_strength(ceiling, "very_cautious")
        reasons.append("support_status_unknown")
        qualifiers.append("evidence support has not been fully established")
    elif state.support_state == "partially_supported":
        ceiling = _minimum_strength(ceiling, "cautious")
        reasons.append("claim_is_only_partially_supported")
        qualifiers.append("available evidence only partially supports this statement")

    if state.contradiction_count:
        ceiling = _minimum_strength(ceiling, "cautious")
        reasons.append(f"contradictory_evidence:{state.contradiction_count}")
        qualifiers.append("contradictory evidence is present")

    if state.certainty is not None:
        certainty_level = state.certainty.final_level
        ceiling = _minimum_strength(ceiling, _CERTAINTY_CEILING[certainty_level])
        if state.certainty.unresolved_domains:
            ceiling = _minimum_strength(ceiling, "cautious")
            reasons.append("certainty_domains_unresolved")
            qualifiers.append("certainty assessment contains unresolved domains")
        if state.certainty.reviewed_fraction < state.reviewed_fraction:
            # The stricter reviewed fraction controls publication strength.
            reviewed_fraction = state.certainty.reviewed_fraction
        else:
            reviewed_fraction = state.reviewed_fraction
    else:
        reviewed_fraction = state.reviewed_fraction
        ceiling = _minimum_strength(ceiling, "cautious")
        reasons.append("certainty_not_assessed")
        qualifiers.append("evidence certainty has not been fully assessed")

    if reviewed_fraction < moderate_review:
        ceiling = _minimum_strength(ceiling, "cautious")
        reasons.append("review_coverage_below_moderate_threshold")
        qualifiers.append("human review is incomplete")
    elif reviewed_fraction < assertive_review:
        ceiling = _minimum_strength(ceiling, "moderate")
        reasons.append("review_coverage_below_assertive_threshold")

    if state.claim_type == "causal":
        prohibited.extend(_CAUSAL_PATTERNS)
        if not state.causal_ready:
            ceiling = "abstain"
            reasons.append("causal_assumptions_not_reviewed")
        else:
            prohibited = [item for item in prohibited if item not in _CAUSAL_PATTERNS]
    elif state.claim_type == "treatment":
        prohibited.extend(_TREATMENT_PATTERNS)
        prohibited.extend(_CAUSAL_PATTERNS)
        if not state.causal_ready or not state.treatment_ready:
            ceiling = "abstain"
            reasons.append("treatment_or_causal_review_not_ready")
        elif certainty_level not in {"moderate", "high"} or reviewed_fraction < assertive_review:
            ceiling = _minimum_strength(ceiling, "cautious")
            reasons.append("treatment_language_requires_high_review_coverage_and_certainty")
            qualifiers.append("treatment implications require expert interpretation")
        else:
            prohibited = [item for item in prohibited if item not in _TREATMENT_PATTERNS and item not in _CAUSAL_PATTERNS]
    elif state.claim_type == "quantitative" and state.quantitative_consistent is False:
        ceiling = "abstain"
        reasons.append("quantitative_consistency_failed")

    state_payload = {
        "claim_type": state.claim_type,
        "support_state": state.support_state,
        "source_status": state.source_status,
        "reviewed_fraction": state.reviewed_fraction,
        "contradiction_count": state.contradiction_count,
        "causal_ready": state.causal_ready,
        "treatment_ready": state.treatment_ready,
        "quantitative_consistent": state.quantitative_consistent,
        "certainty_fingerprint": None if state.certainty is None else state.certainty.fingerprint,
        "warnings": state.warnings,
    }
    state_fingerprint = hashlib.sha256(_canonical(state_payload)).hexdigest()
    policy_payload = {
        "schema": "rigorousrag.evidence-language/v1",
        "assertive_review": assertive_review,
        "moderate_review": moderate_review,
        "certainty_ceiling": _CERTAINTY_CEILING,
        "strengths": _STRENGTHS,
    }
    policy_fingerprint = hashlib.sha256(_canonical(policy_payload)).hexdigest()
    reasons.extend(f"warning:{item}" for item in state.warnings)
    qualifiers = list(dict.fromkeys(qualifiers))
    prohibited = list(dict.fromkeys(prohibited))
    return LanguagePolicyDecision(
        allowed_strength=ceiling,
        abstain=ceiling == "abstain",
        reasons=tuple(dict.fromkeys(reasons)),
        required_qualifiers=tuple(qualifiers),
        prohibited_patterns=tuple(prohibited),
        certainty_level=certainty_level,
        reviewed_fraction=reviewed_fraction,
        policy_fingerprint=policy_fingerprint,
        state_fingerprint=state_fingerprint,
    )


def check_claim_language(claim: str, decision: LanguagePolicyDecision) -> ClaimLanguageCheck:
    selected = _text(claim, "claim", 20_000)
    if not isinstance(decision, LanguagePolicyDecision):
        raise TypeError("decision must be LanguagePolicyDecision")
    violations: list[str] = []
    if decision.abstain:
        violations.append("publication_abstention_required")
    lowered = selected.lower()
    for pattern in decision.prohibited_patterns:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            violations.append(f"prohibited_phrase_pattern:{pattern}")
    return ClaimLanguageCheck(
        compliant=not violations,
        violations=tuple(dict.fromkeys(violations)),
        decision_fingerprint=hashlib.sha256(_canonical(asdict(decision))).hexdigest(),
        claim_sha256=hashlib.sha256(selected.encode("utf-8")).hexdigest(),
    )


__all__ = [
    "ClaimLanguageCheck",
    "LanguagePolicyDecision",
    "PublicationEvidenceState",
    "check_claim_language",
    "evaluate_publication_language",
]
