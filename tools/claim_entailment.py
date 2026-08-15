"""Fail-closed claim-to-evidence semantic support gate.

The gate never creates citation authority.  It accepts only server-created ``Citation``
objects, decomposes/accepts atomic claims, calls a bounded provider through an explicit
protocol, and returns claim support decisions that downstream answer rendering can use
to remove unsupported statements.  Provider failure is neutral/unsupported, never
implicitly entailed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence

from tools.models import Citation

_MAX_CLAIMS = 128
_MAX_CITATIONS = 100
_MAX_TEXT = 20_000
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])" )
_ALLOWED_LABELS = frozenset({"entailment", "neutral", "contradiction"})


def _text(value: Any, label: str, maximum: int = _MAX_TEXT) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if not cleaned or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _probability(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a probability")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a probability") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1")
    return parsed


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def citation_identity(citation: Citation) -> str:
    if not isinstance(citation, Citation):
        raise TypeError("citation must be Citation")
    payload = {
        "source": citation.source_id or citation.url,
        "doc_id": citation.doc_id or "",
        "chunk_id": citation.chunk_id or "",
        "page": citation.page_number,
        "quote": citation.quote or citation.snippet or "",
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


@dataclass(frozen=True)
class AtomicClaim:
    claim_id: str
    text: str
    inferred: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _text(self.claim_id, "claim_id", 128))
        object.__setattr__(self, "text", _text(self.text, "claim text", 5000))
        if not isinstance(self.inferred, bool):
            raise ValueError("inferred must be boolean")


def segment_atomic_claims(answer: str, *, max_claims: int = 64) -> tuple[AtomicClaim, ...]:
    """Deterministic conservative sentence segmentation; learned splitters may plug in upstream."""
    text = _text(answer, "answer", 100_000)
    if isinstance(max_claims, bool) or not isinstance(max_claims, int) or not 1 <= max_claims <= _MAX_CLAIMS:
        raise ValueError("max_claims is invalid")
    raw = [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]
    claims: list[AtomicClaim] = []
    for index, part in enumerate(raw[:max_claims], start=1):
        digest = hashlib.sha256(part.encode("utf-8")).hexdigest()[:16]
        claims.append(AtomicClaim(f"claim-{index}-{digest}", part))
    return tuple(claims)


@dataclass(frozen=True)
class EntailmentScore:
    label: str
    confidence: float
    provider_version: str

    def __post_init__(self) -> None:
        label = _text(self.label, "label", 32).lower()
        if label not in _ALLOWED_LABELS:
            raise ValueError("unsupported entailment label")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "confidence", _probability(self.confidence, "confidence"))
        object.__setattr__(self, "provider_version", _text(self.provider_version, "provider_version", 128))


class EntailmentProvider(Protocol):
    def score(self, claim: str, evidence: str) -> EntailmentScore: ...


@dataclass(frozen=True)
class EvidenceAssessment:
    citation_id: str
    label: str
    confidence: float


@dataclass(frozen=True)
class ClaimAssessment:
    claim: AtomicClaim
    supported: bool
    contradicted: bool
    support_type: str
    support_confidence: float
    contradiction_confidence: float
    supporting_citation_ids: tuple[str, ...]
    contradicting_citation_ids: tuple[str, ...]
    evidence: tuple[EvidenceAssessment, ...]
    reason: str


@dataclass(frozen=True)
class ClaimGatePolicy:
    entailment_threshold: float = 0.75
    contradiction_threshold: float = 0.75
    minimum_supporting_sources: int = 1
    allow_inferential_support: bool = False
    maximum_evidence_per_claim: int = 12

    def __post_init__(self) -> None:
        object.__setattr__(self, "entailment_threshold", _probability(self.entailment_threshold, "entailment_threshold"))
        object.__setattr__(self, "contradiction_threshold", _probability(self.contradiction_threshold, "contradiction_threshold"))
        for name, low, high in (("minimum_supporting_sources", 1, 20), ("maximum_evidence_per_claim", 1, _MAX_CITATIONS)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} is invalid")
        if not isinstance(self.allow_inferential_support, bool):
            raise ValueError("allow_inferential_support must be boolean")


@dataclass(frozen=True)
class ClaimGateResult:
    assessments: tuple[ClaimAssessment, ...]
    supported_claim_ids: tuple[str, ...]
    rejected_claim_ids: tuple[str, ...]
    contradicted_claim_ids: tuple[str, ...]
    authoritative_citation_ids: tuple[str, ...]
    fingerprint: str

    @property
    def all_supported(self) -> bool:
        return not self.rejected_claim_ids


def _evidence_text(citation: Citation) -> str:
    return (citation.quote or citation.snippet or "").strip()


def assess_claims(
    claims: Sequence[AtomicClaim],
    citations: Sequence[Citation],
    provider: EntailmentProvider,
    *,
    policy: ClaimGatePolicy | None = None,
) -> ClaimGateResult:
    selected_policy = policy or ClaimGatePolicy()
    if not 1 <= len(claims) <= _MAX_CLAIMS:
        raise ValueError("claims must contain between 1 and 128 items")
    if len(citations) > _MAX_CITATIONS:
        raise ValueError("citations exceed the authoritative evidence limit")
    if any(not isinstance(claim, AtomicClaim) for claim in claims):
        raise TypeError("claims must contain AtomicClaim objects")
    if any(not isinstance(citation, Citation) for citation in citations):
        raise TypeError("citations must contain Citation objects")

    unique: list[tuple[str, Citation]] = []
    seen: set[str] = set()
    for citation in citations:
        identifier = citation_identity(citation)
        if identifier not in seen and _evidence_text(citation):
            seen.add(identifier)
            unique.append((identifier, citation))

    assessments: list[ClaimAssessment] = []
    authorized: set[str] = set()
    for claim in claims:
        scored: list[EvidenceAssessment] = []
        support_ids: list[str] = []
        contradiction_ids: list[str] = []
        support_confidence = 0.0
        contradiction_confidence = 0.0
        source_support: set[str] = set()
        for identifier, citation in unique[: selected_policy.maximum_evidence_per_claim]:
            evidence_text = _evidence_text(citation)
            try:
                result = provider.score(claim.text, evidence_text)
            except Exception:
                result = EntailmentScore("neutral", 0.0, "provider-failure")
            if not isinstance(result, EntailmentScore):
                result = EntailmentScore("neutral", 0.0, "invalid-provider-result")
            scored.append(EvidenceAssessment(identifier, result.label, result.confidence))
            if result.label == "entailment" and result.confidence >= selected_policy.entailment_threshold:
                support_ids.append(identifier)
                support_confidence = max(support_confidence, result.confidence)
                source_support.add(citation.source_id or citation.url)
            elif result.label == "contradiction" and result.confidence >= selected_policy.contradiction_threshold:
                contradiction_ids.append(identifier)
                contradiction_confidence = max(contradiction_confidence, result.confidence)

        enough_sources = len(source_support) >= selected_policy.minimum_supporting_sources
        inferential_allowed = (not claim.inferred) or selected_policy.allow_inferential_support
        contradicted = bool(contradiction_ids)
        supported = bool(support_ids) and enough_sources and inferential_allowed and not contradicted
        if contradicted:
            reason = "contradictory_authoritative_evidence"
        elif not support_ids:
            reason = "no_entailing_authoritative_evidence"
        elif not enough_sources:
            reason = "insufficient_independent_support"
        elif not inferential_allowed:
            reason = "inferential_support_disallowed"
        else:
            reason = "supported"
        support_type = "inferential" if claim.inferred else "direct"
        if supported:
            authorized.update(support_ids)
        assessments.append(
            ClaimAssessment(
                claim=claim,
                supported=supported,
                contradicted=contradicted,
                support_type=support_type,
                support_confidence=support_confidence,
                contradiction_confidence=contradiction_confidence,
                supporting_citation_ids=tuple(support_ids),
                contradicting_citation_ids=tuple(contradiction_ids),
                evidence=tuple(scored),
                reason=reason,
            )
        )

    supported_ids = tuple(item.claim.claim_id for item in assessments if item.supported)
    rejected_ids = tuple(item.claim.claim_id for item in assessments if not item.supported)
    contradicted_ids = tuple(item.claim.claim_id for item in assessments if item.contradicted)
    payload = {
        "policy": asdict(selected_policy),
        "assessments": [asdict(item) for item in assessments],
        "authoritative_citation_ids": sorted(authorized),
    }
    return ClaimGateResult(
        tuple(assessments),
        supported_ids,
        rejected_ids,
        contradicted_ids,
        tuple(sorted(authorized)),
        hashlib.sha256(_canonical(payload)).hexdigest(),
    )


def supported_answer_text(result: ClaimGateResult) -> str:
    """Render only supported atomic claims; unsupported/contradicted claims disappear."""
    if not isinstance(result, ClaimGateResult):
        raise TypeError("result must be ClaimGateResult")
    return " ".join(item.claim.text for item in result.assessments if item.supported).strip()


def contradiction_clusters(result: ClaimGateResult) -> Mapping[str, tuple[str, ...]]:
    """Group contradicted claims by the authoritative citations that contradict them."""
    clusters: dict[str, list[str]] = {}
    for assessment in result.assessments:
        for citation_id in assessment.contradicting_citation_ids:
            clusters.setdefault(citation_id, []).append(assessment.claim.claim_id)
    return {key: tuple(values) for key, values in sorted(clusters.items())}


__all__ = [
    "AtomicClaim",
    "ClaimAssessment",
    "ClaimGatePolicy",
    "ClaimGateResult",
    "EntailmentProvider",
    "EntailmentScore",
    "EvidenceAssessment",
    "assess_claims",
    "citation_identity",
    "contradiction_clusters",
    "segment_atomic_claims",
    "supported_answer_text",
]
