"""Retrieved-content trust contracts for indirect prompt-injection containment.

The primary security boundary is architectural: retrieved content is always untrusted data
with immutable provenance and never acquires system/developer/tool authority. Lightweight
injection signals are deliberately advisory risk/review signals, not an oracle that makes
content safe. Raw evidence text may exist transiently in memory but durable decisions bind
only digests, provenance, trust class and bounded reason codes.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

_TRUST_CLASSES = frozenset(
    {
        "reviewed_authoritative",
        "trusted_internal",
        "external_untrusted",
        "user_uploaded_untrusted",
        "generated_untrusted",
    }
)
_ACTIONS = frozenset({"allow_as_evidence", "allow_with_warning", "review", "quarantine"})
_MAX_CONTENT_BYTES = 20_000_000
_MAX_SIGNALS = 100
_HEX = frozenset("0123456789abcdef")

# Advisory only. These phrases are intentionally narrow and interpretable. The structural
# role boundary remains mandatory even when none of these signals fires.
_SIGNAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("instruction_override", re.compile(r"\b(ignore|disregard|override)\b.{0,80}\b(previous|prior|system|developer|instructions?)\b", re.I | re.S)),
    ("authority_impersonation", re.compile(r"\b(system|developer|assistant)\s*(message|prompt|instruction|role)\b", re.I)),
    ("tool_execution_request", re.compile(r"\b(call|invoke|execute|run|use)\b.{0,60}\b(tool|function|command|shell|terminal)\b", re.I | re.S)),
    ("secret_exfiltration_request", re.compile(r"\b(reveal|print|dump|send|exfiltrate)\b.{0,80}\b(secret|token|credential|api[- ]?key|system prompt)\b", re.I | re.S)),
    ("role_delimiter_like", re.compile(r"(^|\n)\s*(system|developer|assistant|tool)\s*[:>]", re.I)),
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: Any, label: str, maximum: int = 1000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 and ch not in "\t\n\r" for ch in selected) or "\x7f" in selected:
        raise ValueError(f"{label} is invalid")
    return selected


def _identifier(value: Any, label: str, maximum: int = 1000) -> str:
    selected = _text(value, label, maximum)
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} contains control characters")
    return selected


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def content_sha256(content: str | bytes) -> str:
    if isinstance(content, str):
        raw = content.encode("utf-8")
    elif isinstance(content, bytes):
        raw = content
    else:
        raise ValueError("content must be str or bytes")
    if not raw or len(raw) > _MAX_CONTENT_BYTES:
        raise ValueError("content must be non-empty and bounded")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class RetrievedEvidenceIdentity:
    evidence_id: str
    evidence_sha256: str
    document_id: str
    source_id: str
    generation_id: str
    provenance_sha256: str
    trust_class: str
    mime_type: str = "text/plain"

    def __post_init__(self) -> None:
        for name in ("evidence_id", "document_id", "source_id", "generation_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in ("evidence_sha256", "provenance_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        trust = _identifier(self.trust_class, "trust_class", 100).lower()
        if trust not in _TRUST_CLASSES:
            raise ValueError(f"trust_class must be one of {sorted(_TRUST_CLASSES)}")
        object.__setattr__(self, "trust_class", trust)
        object.__setattr__(self, "mime_type", _identifier(self.mime_type, "mime_type", 200).lower())

    @property
    def identity_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-retrieved-evidence-identity/v1", **asdict(self)})


@dataclass(frozen=True)
class RetrievedEvidenceMaterialization:
    identity: RetrievedEvidenceIdentity
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RetrievedEvidenceIdentity):
            raise ValueError("identity must be RetrievedEvidenceIdentity")
        if not isinstance(self.content, str):
            raise ValueError("content must be text")
        raw = self.content.encode("utf-8")
        if not raw or len(raw) > _MAX_CONTENT_BYTES:
            raise ValueError("content must be non-empty and bounded")
        if hashlib.sha256(raw).hexdigest() != self.identity.evidence_sha256:
            raise ValueError("retrieved content does not match authoritative evidence digest")


@dataclass(frozen=True)
class InjectionSignal:
    signal_type: str
    severity: float
    signal_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "signal_type", _identifier(self.signal_type, "signal_type", 200))
        if isinstance(self.severity, bool):
            raise ValueError("severity must be in [0,1]")
        severity = float(self.severity)
        if not 0.0 <= severity <= 1.0:
            raise ValueError("severity must be in [0,1]")
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "signal_sha256", _sha(self.signal_sha256, "signal_sha256"))


@dataclass(frozen=True)
class RetrievedContentTrustPolicy:
    quarantine_signal_count: int = 3
    review_signal_count: int = 1
    quarantine_severity: float = 0.90
    review_untrusted_with_signal: bool = True
    allow_reviewed_authoritative_with_warning: bool = True

    def __post_init__(self) -> None:
        for name in ("quarantine_signal_count", "review_signal_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.review_signal_count > self.quarantine_signal_count:
            raise ValueError("review_signal_count may not exceed quarantine_signal_count")
        severity = float(self.quarantine_severity)
        if not 0.0 <= severity <= 1.0:
            raise ValueError("quarantine_severity must be in [0,1]")
        object.__setattr__(self, "quarantine_severity", severity)
        for name in ("review_untrusted_with_signal", "allow_reviewed_authoritative_with_warning"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")

    @property
    def policy_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-retrieved-content-trust-policy/v1", **asdict(self)})


@dataclass(frozen=True)
class RetrievedContentTrustDecision:
    evidence_identity_sha256: str
    policy_sha256: str
    action: str
    signal_sha256s: tuple[str, ...]
    reason_codes: tuple[str, ...]
    decision_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_identity_sha256", _sha(self.evidence_identity_sha256, "evidence_identity_sha256"))
        object.__setattr__(self, "policy_sha256", _sha(self.policy_sha256, "policy_sha256"))
        if self.action not in _ACTIONS:
            raise ValueError(f"action must be one of {sorted(_ACTIONS)}")
        signals = tuple(sorted({_sha(value, "signal sha256") for value in self.signal_sha256s}))
        if len(signals) > _MAX_SIGNALS:
            raise ValueError("too many injection signals")
        object.__setattr__(self, "signal_sha256s", signals)
        reasons = tuple(sorted({_identifier(value, "reason code", 200) for value in self.reason_codes}))
        if self.action == "allow_as_evidence" and reasons:
            raise ValueError("unqualified allow decision may not contain reasons")
        if self.action != "allow_as_evidence" and not reasons:
            raise ValueError("qualified/review/quarantine decision requires reasons")
        object.__setattr__(self, "reason_codes", reasons)
        expected = _digest(self._payload())
        provided = _sha(self.decision_sha256, "decision_sha256")
        if expected != provided:
            raise ValueError("decision_sha256 does not match trust decision")
        object.__setattr__(self, "decision_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-retrieved-content-trust-decision/v1",
            "evidence_identity_sha256": self.evidence_identity_sha256,
            "policy_sha256": self.policy_sha256,
            "action": self.action,
            "signal_sha256s": self.signal_sha256s,
            "reason_codes": self.reason_codes,
        }


def inspect_injection_signals(materialization: RetrievedEvidenceMaterialization) -> tuple[InjectionSignal, ...]:
    """Return advisory content signals without changing evidence authority.

    Signal digests bind the evidence identity and signal type, not matched raw text.
    """

    if not isinstance(materialization, RetrievedEvidenceMaterialization):
        raise ValueError("materialization must be RetrievedEvidenceMaterialization")
    content = materialization.content
    rows: list[InjectionSignal] = []
    for signal_type, pattern in _SIGNAL_PATTERNS:
        matches = list(pattern.finditer(content))
        if not matches:
            continue
        severity = min(1.0, 0.55 + 0.15 * len(matches))
        payload = {
            "schema": "rigorousrag-injection-signal/v1",
            "evidence_identity_sha256": materialization.identity.identity_sha256,
            "signal_type": signal_type,
            "match_count": len(matches),
        }
        rows.append(InjectionSignal(signal_type, severity, _digest(payload)))
    return tuple(sorted(rows, key=lambda row: (row.signal_type, row.signal_sha256)))


def decide_retrieved_content_trust(
    materialization: RetrievedEvidenceMaterialization,
    *,
    policy: RetrievedContentTrustPolicy = RetrievedContentTrustPolicy(),
    additional_signals: Sequence[InjectionSignal] = (),
) -> RetrievedContentTrustDecision:
    """Evaluate trust without permitting callers to suppress native inspection signals."""

    if not isinstance(materialization, RetrievedEvidenceMaterialization):
        raise ValueError("materialization must be RetrievedEvidenceMaterialization")
    if not isinstance(policy, RetrievedContentTrustPolicy):
        raise ValueError("policy must be RetrievedContentTrustPolicy")
    extras = tuple(additional_signals)
    if any(not isinstance(value, InjectionSignal) for value in extras):
        raise ValueError("additional_signals contains invalid values")
    native = inspect_injection_signals(materialization)
    by_digest = {value.signal_sha256: value for value in native}
    for value in extras:
        existing = by_digest.get(value.signal_sha256)
        if existing is not None and existing != value:
            raise ValueError("conflicting signal identity")
        by_digest[value.signal_sha256] = value
    selected = tuple(sorted(by_digest.values(), key=lambda value: (value.signal_type, value.signal_sha256)))
    if len(selected) > _MAX_SIGNALS:
        raise ValueError("signals exceed the bounded signal limit")

    reasons: list[str] = []
    maximum = max((value.severity for value in selected), default=0.0)
    trust = materialization.identity.trust_class
    if len(selected) >= policy.quarantine_signal_count or maximum >= policy.quarantine_severity:
        action = "quarantine"
        reasons.append("advisory_injection_risk_above_quarantine_threshold")
    elif selected and trust in {"external_untrusted", "user_uploaded_untrusted", "generated_untrusted"} and policy.review_untrusted_with_signal:
        action = "review"
        reasons.append("untrusted_evidence_contains_instruction_like_content")
    elif selected and trust == "reviewed_authoritative" and policy.allow_reviewed_authoritative_with_warning:
        action = "allow_with_warning"
        reasons.append("reviewed_evidence_contains_instruction_like_content")
    elif len(selected) >= policy.review_signal_count:
        action = "review"
        reasons.append("instruction_like_content_requires_review")
    else:
        action = "allow_as_evidence"

    payload = {
        "schema": "rigorousrag-retrieved-content-trust-decision/v1",
        "evidence_identity_sha256": materialization.identity.identity_sha256,
        "policy_sha256": policy.policy_sha256,
        "action": action,
        "signal_sha256s": tuple(value.signal_sha256 for value in selected),
        "reason_codes": tuple(sorted(set(reasons))),
    }
    return RetrievedContentTrustDecision(**payload, decision_sha256=_digest(payload))


def safe_decision_summary(decision: RetrievedContentTrustDecision) -> Mapping[str, Any]:
    """Privacy-safe durable/logging representation; never includes raw retrieved text."""

    if not isinstance(decision, RetrievedContentTrustDecision):
        raise ValueError("decision must be RetrievedContentTrustDecision")
    return {
        "evidence_identity_sha256": decision.evidence_identity_sha256,
        "policy_sha256": decision.policy_sha256,
        "action": decision.action,
        "signal_count": len(decision.signal_sha256s),
        "reason_codes": decision.reason_codes,
        "decision_sha256": decision.decision_sha256,
    }


__all__ = [
    "InjectionSignal",
    "RetrievedContentTrustDecision",
    "RetrievedContentTrustPolicy",
    "RetrievedEvidenceIdentity",
    "RetrievedEvidenceMaterialization",
    "content_sha256",
    "decide_retrieved_content_trust",
    "inspect_injection_signals",
    "safe_decision_summary",
]
