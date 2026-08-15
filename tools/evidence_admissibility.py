"""Policy-as-code for source and claim admissibility before answer publication.

The policy consumes server-owned citations plus optional reviewed ``SourceTrustFeatures``.
It does not infer truth from publisher identity. Unreviewed sources receive transparent
baseline features, while causal and treatment claims can require reviewed evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from tools.claim_entailment import AtomicClaim, segment_atomic_claims
from tools.models import Citation
from tools.source_trust import (
    SourceTrustDecision,
    SourceTrustFeatures,
    SourceTrustPolicy,
    evaluate_source_trust,
)

_MARKER_RE = re.compile(r"\[(\d+)\]")
_CAUSAL_RE = re.compile(
    r"\b(?:cause(?:s|d)?|causal|leads? to|result(?:s|ed) in|effect of|"
    r"increases? (?:the )?risk|reduces? (?:the )?risk|prevents?|drives?)\b",
    flags=re.IGNORECASE,
)
_TREATMENT_RE = re.compile(
    r"\b(?:treat(?:s|ed|ment)?|therapy|therapeutic|dose|dosing|drug|medication|"
    r"intervention|improves? survival|reduces? mortality|clinical benefit)\b",
    flags=re.IGNORECASE,
)

_SOURCE_TYPE_MAP = {
    "academic_index": "other",
    "uploaded_document": "other",
    "handbook": "documentation",
    "web_page": "web",
    "web_search": "web",
    "tool_output": "other",
    "unknown": "other",
}


class SourceTrustReader(Protocol):
    def latest(self, owner_id: str, source_id: str) -> Any | None: ...


@dataclass(frozen=True)
class EvidenceAdmissibilityPolicy:
    version: str = "rigorous-admissibility-v1"
    trust_policy: SourceTrustPolicy = field(default_factory=SourceTrustPolicy)
    require_reviewed_for_causal_claims: bool = True
    require_reviewed_for_treatment_claims: bool = True
    treatment_source_types: tuple[str, ...] = (
        "primary_study",
        "systematic_review",
        "meta_analysis",
        "guideline",
    )
    reject_claim_if_all_cited_sources_ineligible: bool = True
    preserve_uncited_non_evidentiary_text: bool = True
    minimum_independent_sources_for_causal_claim: int = 1
    minimum_independent_sources_for_treatment_claim: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.trust_policy, SourceTrustPolicy):
            raise TypeError("trust_policy must be SourceTrustPolicy")
        for name in (
            "require_reviewed_for_causal_claims",
            "require_reviewed_for_treatment_claims",
            "reject_claim_if_all_cited_sources_ineligible",
            "preserve_uncited_non_evidentiary_text",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in (
            "minimum_independent_sources_for_causal_claim",
            "minimum_independent_sources_for_treatment_claim",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 20:
                raise ValueError(f"{name} is invalid")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(
                asdict(self),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class CitationAdmissibility:
    label: str
    source_id: str
    eligible: bool
    decision: SourceTrustDecision
    trust_revision_id: str = ""
    reviewed: bool = False


@dataclass(frozen=True)
class ClaimAdmissibility:
    claim: AtomicClaim
    claim_kind: str
    admissible: bool
    cited_labels: tuple[str, ...]
    admitted_labels: tuple[str, ...]
    rejected_labels: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class AdmissibilityResult:
    answer: str
    citations: tuple[Citation, ...]
    citation_decisions: tuple[CitationAdmissibility, ...]
    claim_decisions: tuple[ClaimAdmissibility, ...]
    rejected_claim_ids: tuple[str, ...]
    rejected_citation_labels: tuple[str, ...]
    policy_sha256: str
    fingerprint: str


def _source_id(citation: Citation) -> str:
    return (citation.source_id or citation.url).strip()


def _baseline_features(citation: Citation) -> SourceTrustFeatures:
    source_id = _source_id(citation)
    source_type = _SOURCE_TYPE_MAP.get(citation.source_type, "other")
    # These are neutral policy baselines, not quality claims. Reviewed records can replace
    # every field explicitly. Public web evidence starts slightly more conservatively.
    methodology = 0.35 if source_type == "web" else 0.5
    provenance = 0.8 if source_type == "web" else 1.0
    return SourceTrustFeatures(
        source_id=source_id,
        source_type=source_type,
        status="unknown",
        provenance_integrity=provenance,
        methodological_quality=methodology,
        topical_applicability=0.5,
        freshness=0.5,
        independent_replication=0.0,
        reviewed=False,
        conflicts_of_interest_known=False,
        notes=("unreviewed_baseline",),
    )


def _reviewed_features(reader: SourceTrustReader | None, owner_id: str, citation: Citation) -> tuple[SourceTrustFeatures, str]:
    if reader is None:
        return _baseline_features(citation), ""
    try:
        revision = reader.latest(owner_id, _source_id(citation))
    except Exception:
        revision = None
    features = getattr(revision, "features", None) if revision is not None else None
    revision_id = getattr(revision, "revision_id", "") if revision is not None else ""
    if isinstance(features, SourceTrustFeatures):
        return features, str(revision_id or "")
    return _baseline_features(citation), ""


def _claim_kind(text: str) -> str:
    if _TREATMENT_RE.search(text):
        return "treatment"
    if _CAUSAL_RE.search(text):
        return "causal"
    return "general"


def _rewrite_claim_markers(text: str, admitted: set[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        label = f"[{match.group(1)}]"
        return label if label in admitted else ""

    return re.sub(r"\s+([,.;:!?])", r"\1", _MARKER_RE.sub(replace, text)).strip()


def evaluate_answer_admissibility(
    answer: str,
    citations: Sequence[Citation],
    *,
    owner_id: str,
    trust_reader: SourceTrustReader | None = None,
    policy: EvidenceAdmissibilityPolicy | None = None,
) -> AdmissibilityResult:
    selected_policy = policy or EvidenceAdmissibilityPolicy()
    if len(citations) > 100 or any(not isinstance(item, Citation) for item in citations):
        raise ValueError("citations are invalid")
    by_label = {item.label: item for item in citations}
    citation_decisions: dict[str, CitationAdmissibility] = {}

    for citation in citations:
        features, revision_id = _reviewed_features(trust_reader, owner_id, citation)
        decision = evaluate_source_trust(features, selected_policy.trust_policy, causal_claim=False)
        citation_decisions[citation.label] = CitationAdmissibility(
            label=citation.label,
            source_id=_source_id(citation),
            eligible=decision.eligible_for_new_claims,
            decision=decision,
            trust_revision_id=revision_id,
            reviewed=features.reviewed,
        )

    rendered_claims: list[str] = []
    claim_decisions: list[ClaimAdmissibility] = []
    admitted_labels_global: set[str] = set()
    rejected_claims: list[str] = []

    for claim in segment_atomic_claims(answer, max_claims=128):
        labels = tuple(dict.fromkeys(f"[{value}]" for value in _MARKER_RE.findall(claim.text)))
        kind = _claim_kind(claim.text)
        if not labels:
            admissible = selected_policy.preserve_uncited_non_evidentiary_text
            reason = "uncited_text_preserved" if admissible else "uncited_text_disallowed"
            if admissible:
                rendered_claims.append(claim.text)
            else:
                rejected_claims.append(claim.claim_id)
            claim_decisions.append(ClaimAdmissibility(claim, kind, admissible, (), (), (), reason))
            continue

        admitted: list[str] = []
        rejected: list[str] = []
        admitted_sources: set[str] = set()
        for label in labels:
            citation = by_label.get(label)
            base = citation_decisions.get(label)
            if citation is None or base is None or not base.eligible:
                rejected.append(label)
                continue
            features, revision_id = _reviewed_features(trust_reader, owner_id, citation)
            causal = kind == "causal"
            decision = evaluate_source_trust(
                features,
                selected_policy.trust_policy,
                causal_claim=causal,
            )
            eligible = decision.eligible_for_new_claims
            if kind == "causal" and selected_policy.require_reviewed_for_causal_claims and not features.reviewed:
                eligible = False
            if kind == "treatment":
                if selected_policy.require_reviewed_for_treatment_claims and not features.reviewed:
                    eligible = False
                if features.source_type not in selected_policy.treatment_source_types:
                    eligible = False
            if eligible:
                admitted.append(label)
                admitted_sources.add(features.source_id)
                citation_decisions[label] = CitationAdmissibility(
                    label,
                    features.source_id,
                    True,
                    decision,
                    revision_id,
                    features.reviewed,
                )
            else:
                rejected.append(label)

        minimum_sources = 1
        if kind == "causal":
            minimum_sources = selected_policy.minimum_independent_sources_for_causal_claim
        elif kind == "treatment":
            minimum_sources = selected_policy.minimum_independent_sources_for_treatment_claim
        enough_sources = len(admitted_sources) >= minimum_sources
        admissible = bool(admitted) and enough_sources
        if not admitted and selected_policy.reject_claim_if_all_cited_sources_ineligible:
            reason = "all_cited_sources_ineligible"
        elif not enough_sources:
            reason = "insufficient_independent_admissible_sources"
        else:
            reason = "admissible"
        if admissible:
            admitted_set = set(admitted)
            rewritten = _rewrite_claim_markers(claim.text, admitted_set)
            if rewritten:
                rendered_claims.append(rewritten)
            admitted_labels_global.update(admitted)
        else:
            rejected_claims.append(claim.claim_id)
        claim_decisions.append(
            ClaimAdmissibility(
                claim=claim,
                claim_kind=kind,
                admissible=admissible,
                cited_labels=labels,
                admitted_labels=tuple(admitted),
                rejected_labels=tuple(rejected),
                reason=reason,
            )
        )

    selected_citations = tuple(item for item in citations if item.label in admitted_labels_global)
    rejected_labels = tuple(sorted(set(by_label) - admitted_labels_global))
    payload = {
        "answer": rendered_claims,
        "citations": [item.label for item in selected_citations],
        "claim_decisions": [
            {
                "claim_id": item.claim.claim_id,
                "kind": item.claim_kind,
                "admissible": item.admissible,
                "admitted": item.admitted_labels,
                "rejected": item.rejected_labels,
                "reason": item.reason,
            }
            for item in claim_decisions
        ],
        "policy_sha256": selected_policy.fingerprint,
    }
    return AdmissibilityResult(
        answer=" ".join(rendered_claims).strip(),
        citations=selected_citations,
        citation_decisions=tuple(citation_decisions[label] for label in sorted(citation_decisions)),
        claim_decisions=tuple(claim_decisions),
        rejected_claim_ids=tuple(rejected_claims),
        rejected_citation_labels=rejected_labels,
        policy_sha256=selected_policy.fingerprint,
        fingerprint=hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        ).hexdigest(),
    )


__all__ = [
    "AdmissibilityResult",
    "CitationAdmissibility",
    "ClaimAdmissibility",
    "EvidenceAdmissibilityPolicy",
    "SourceTrustReader",
    "evaluate_answer_admissibility",
]
