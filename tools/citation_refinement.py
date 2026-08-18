"""Server-owned citation refinement for grounded long-form answers.

Generation-time citation output is useful but can be incomplete, redundant, or over-cite
weak evidence.  This module refines citation *sets* after generation while preserving the
answer text and the repository's server-owned citation authority.

The refiner never invents evidence, never accepts arbitrary model-authored citation ids,
and never rewrites a claim.  It operates only on an allowlisted evidence universe with
claim-specific support/contradiction scores supplied by governed evaluators.  The algorithm:

1. validates the generated claim/citation bindings against the allowed evidence universe;
2. rejects evidence whose contradiction score exceeds policy;
3. keeps useful already-cited evidence when possible;
4. greedily adds high-support evidence until claim support/independent-source requirements
   are met;
5. removes redundant/excess citations under a bounded per-claim cap;
6. marks unresolved claims for review or abstention instead of fabricating support; and
7. emits a content-addressed receipt containing only identities/scores, not raw answer text.

This is source-only logic.  It performs no retrieval, NLI inference, model call, network I/O
or answer generation at import or execution time.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

_HEX = frozenset("0123456789abcdef")
_MAX_CLAIMS = 100_000
_MAX_EVIDENCE = 1_000_000


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _sha256(value: Any, label: str) -> str:
    selected = _identifier(value, label, 64).lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return selected


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


def _unit(value: Any, label: str) -> float:
    selected = _finite(value, label)
    if not 0.0 <= selected <= 1.0:
        raise ValueError(f"{label} must lie in [0,1]")
    return selected


def _bounded_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer between {minimum} and {maximum}")
    return value


class ClaimRefinementStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"


class UnresolvedClaimAction(str, Enum):
    REVIEW = "review"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class CitationRefinementPolicy:
    minimum_evidence_support: float = 0.5
    minimum_claim_support: float = 0.75
    maximum_evidence_contradiction: float = 0.35
    maximum_claim_contradiction: float = 0.5
    minimum_independent_sources: int = 1
    maximum_citations_per_claim: int = 4
    diversity_bonus: float = 0.05
    original_citation_bonus: float = 0.02
    unresolved_action: UnresolvedClaimAction = UnresolvedClaimAction.REVIEW

    def __post_init__(self) -> None:
        for name in (
            "minimum_evidence_support",
            "minimum_claim_support",
            "maximum_evidence_contradiction",
            "maximum_claim_contradiction",
            "diversity_bonus",
            "original_citation_bonus",
        ):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        object.__setattr__(
            self,
            "minimum_independent_sources",
            _bounded_int(self.minimum_independent_sources, "minimum_independent_sources", 1, _MAX_EVIDENCE),
        )
        object.__setattr__(
            self,
            "maximum_citations_per_claim",
            _bounded_int(self.maximum_citations_per_claim, "maximum_citations_per_claim", 1, 1_000),
        )
        if not isinstance(self.unresolved_action, UnresolvedClaimAction):
            object.__setattr__(self, "unresolved_action", UnresolvedClaimAction(self.unresolved_action))

    @property
    def policy_sha256(self) -> str:
        payload = asdict(self)
        payload["unresolved_action"] = self.unresolved_action.value
        return _digest({"schema": "rigorousrag-citation-refinement-policy/v1", **payload})


@dataclass(frozen=True)
class ClaimBinding:
    claim_id: str
    claim_sha256: str
    original_citation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _identifier(self.claim_id, "claim_id", 240))
        object.__setattr__(self, "claim_sha256", _sha256(self.claim_sha256, "claim_sha256"))
        citations = tuple(_identifier(value, "citation id", 240) for value in self.original_citation_ids)
        if len(citations) > _MAX_EVIDENCE or len(set(citations)) != len(citations):
            raise ValueError("original citation ids must be unique and bounded")
        object.__setattr__(self, "original_citation_ids", citations)


@dataclass(frozen=True)
class ClaimEvidenceAssessment:
    claim_id: str
    evidence_id: str
    evidence_sha256: str
    source_group_sha256: str
    support_probability: float
    contradiction_probability: float
    evidence_quality: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _identifier(self.claim_id, "claim_id", 240))
        object.__setattr__(self, "evidence_id", _identifier(self.evidence_id, "evidence_id", 240))
        object.__setattr__(self, "evidence_sha256", _sha256(self.evidence_sha256, "evidence_sha256"))
        object.__setattr__(self, "source_group_sha256", _sha256(self.source_group_sha256, "source_group_sha256"))
        for name in ("support_probability", "contradiction_probability", "evidence_quality"):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        if self.support_probability + self.contradiction_probability > 1.0 + 1e-9:
            raise ValueError("support and contradiction probabilities may not sum above one")


@dataclass(frozen=True)
class RefinedClaimCitations:
    claim_id: str
    claim_sha256: str
    status: ClaimRefinementStatus
    original_citation_ids: tuple[str, ...]
    refined_citation_ids: tuple[str, ...]
    added_citation_ids: tuple[str, ...]
    removed_citation_ids: tuple[str, ...]
    independent_sources: int
    combined_support: float
    maximum_contradiction: float
    unresolved_action: UnresolvedClaimAction | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _identifier(self.claim_id, "claim_id", 240))
        object.__setattr__(self, "claim_sha256", _sha256(self.claim_sha256, "claim_sha256"))
        if not isinstance(self.status, ClaimRefinementStatus):
            object.__setattr__(self, "status", ClaimRefinementStatus(self.status))
        for name in ("original_citation_ids", "refined_citation_ids", "added_citation_ids", "removed_citation_ids"):
            selected = tuple(_identifier(value, "citation id", 240) for value in getattr(self, name))
            if len(set(selected)) != len(selected):
                raise ValueError(f"{name} must be unique")
            object.__setattr__(self, name, selected)
        if set(self.added_citation_ids) != set(self.refined_citation_ids) - set(self.original_citation_ids):
            raise ValueError("added citation identity mismatch")
        if set(self.removed_citation_ids) != set(self.original_citation_ids) - set(self.refined_citation_ids):
            raise ValueError("removed citation identity mismatch")
        object.__setattr__(self, "independent_sources", _bounded_int(self.independent_sources, "independent_sources", 0, _MAX_EVIDENCE))
        object.__setattr__(self, "combined_support", _unit(self.combined_support, "combined_support"))
        object.__setattr__(self, "maximum_contradiction", _unit(self.maximum_contradiction, "maximum_contradiction"))
        if self.unresolved_action is not None and not isinstance(self.unresolved_action, UnresolvedClaimAction):
            object.__setattr__(self, "unresolved_action", UnresolvedClaimAction(self.unresolved_action))
        if self.status in {ClaimRefinementStatus.SUPPORTED, ClaimRefinementStatus.PARTIALLY_SUPPORTED} and self.unresolved_action is not None:
            raise ValueError("supported claims may not carry an unresolved action")
        if self.status in {ClaimRefinementStatus.CONTRADICTED, ClaimRefinementStatus.UNRESOLVED} and self.unresolved_action is None:
            raise ValueError("contradicted/unresolved claims require an explicit action")


@dataclass(frozen=True)
class CitationRefinementReceipt:
    answer_sha256: str
    allowed_evidence_set_sha256: str
    policy_sha256: str
    claim_results: tuple[RefinedClaimCitations, ...]
    requires_review: bool
    requires_abstention: bool
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("answer_sha256", "allowed_evidence_set_sha256", "policy_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        results = tuple(self.claim_results)
        if not results or len(results) > _MAX_CLAIMS or any(not isinstance(result, RefinedClaimCitations) for result in results):
            raise ValueError("claim_results must be a non-empty bounded refined-claim sequence")
        if len({result.claim_id for result in results}) != len(results):
            raise ValueError("refinement receipt claim ids must be unique")
        object.__setattr__(self, "claim_results", results)
        if not isinstance(self.requires_review, bool) or not isinstance(self.requires_abstention, bool):
            raise ValueError("receipt review/abstention flags must be boolean")
        expected = _digest(self._payload())
        provided = _sha256(self.receipt_sha256, "receipt_sha256")
        if expected != provided:
            raise ValueError("citation refinement receipt digest mismatch")
        object.__setattr__(self, "receipt_sha256", provided)

    def _payload(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-citation-refinement-receipt/v1",
            "answer_sha256": self.answer_sha256,
            "allowed_evidence_set_sha256": self.allowed_evidence_set_sha256,
            "policy_sha256": self.policy_sha256,
            "claim_results": [
                {
                    **asdict(result),
                    "status": result.status.value,
                    "unresolved_action": None if result.unresolved_action is None else result.unresolved_action.value,
                }
                for result in self.claim_results
            ],
            "requires_review": self.requires_review,
            "requires_abstention": self.requires_abstention,
        }


def _combined_support(selected: Sequence[ClaimEvidenceAssessment]) -> float:
    """Noisy-OR support aggregation over independently scored evidence items."""

    remaining = 1.0
    for assessment in selected:
        effective = assessment.support_probability * assessment.evidence_quality
        remaining *= 1.0 - effective
    return min(1.0, max(0.0, 1.0 - remaining))


def _maximum_contradiction(selected: Sequence[ClaimEvidenceAssessment]) -> float:
    return 0.0 if not selected else max(item.contradiction_probability for item in selected)


def _score_candidate(
    assessment: ClaimEvidenceAssessment,
    *,
    original: set[str],
    selected_sources: set[str],
    policy: CitationRefinementPolicy,
) -> tuple[float, float, float, str]:
    diversity = policy.diversity_bonus if assessment.source_group_sha256 not in selected_sources else 0.0
    original_bonus = policy.original_citation_bonus if assessment.evidence_id in original else 0.0
    score = (
        assessment.support_probability * assessment.evidence_quality
        - assessment.contradiction_probability
        + diversity
        + original_bonus
    )
    return (-score, -assessment.support_probability, assessment.contradiction_probability, assessment.evidence_id)


def refine_citations(
    *,
    answer_sha256: str,
    allowed_evidence_set_sha256: str,
    claims: Sequence[ClaimBinding],
    assessments: Sequence[ClaimEvidenceAssessment],
    allowed_evidence_ids: Sequence[str],
    policy: CitationRefinementPolicy,
) -> CitationRefinementReceipt:
    """Refine citations deterministically without changing answer or evidence identity."""

    answer_digest = _sha256(answer_sha256, "answer_sha256")
    evidence_set_digest = _sha256(allowed_evidence_set_sha256, "allowed_evidence_set_sha256")
    selected_claims = tuple(claims)
    selected_assessments = tuple(assessments)
    if not selected_claims or len(selected_claims) > _MAX_CLAIMS or any(not isinstance(claim, ClaimBinding) for claim in selected_claims):
        raise ValueError("claims must be a non-empty bounded ClaimBinding sequence")
    if len({claim.claim_id for claim in selected_claims}) != len(selected_claims):
        raise ValueError("claim ids must be unique")
    if not selected_assessments or len(selected_assessments) > _MAX_EVIDENCE or any(not isinstance(item, ClaimEvidenceAssessment) for item in selected_assessments):
        raise ValueError("assessments must be a non-empty bounded assessment sequence")
    if not isinstance(policy, CitationRefinementPolicy):
        raise ValueError("policy must be CitationRefinementPolicy")
    allowed = {_identifier(value, "allowed evidence id", 240) for value in allowed_evidence_ids}
    if not allowed or len(allowed) > _MAX_EVIDENCE:
        raise ValueError("allowed_evidence_ids must be a non-empty bounded set")
    if len(allowed) != len(tuple(allowed_evidence_ids)):
        raise ValueError("allowed_evidence_ids must be unique")

    claim_ids = {claim.claim_id for claim in selected_claims}
    if any(assessment.claim_id not in claim_ids for assessment in selected_assessments):
        raise ValueError("assessment references an unknown claim")
    if any(assessment.evidence_id not in allowed for assessment in selected_assessments):
        raise ValueError("assessment references evidence outside the server-owned allowlist")
    if any(citation_id not in allowed for claim in selected_claims for citation_id in claim.original_citation_ids):
        raise ValueError("generated citation references evidence outside the server-owned allowlist")
    assessment_keys = [(item.claim_id, item.evidence_id) for item in selected_assessments]
    if len(set(assessment_keys)) != len(assessment_keys):
        raise ValueError("duplicate claim/evidence assessments are forbidden")

    by_claim: dict[str, list[ClaimEvidenceAssessment]] = {claim.claim_id: [] for claim in selected_claims}
    for assessment in selected_assessments:
        by_claim[assessment.claim_id].append(assessment)

    results: list[RefinedClaimCitations] = []
    for claim in selected_claims:
        original = set(claim.original_citation_ids)
        eligible = [
            assessment
            for assessment in by_claim[claim.claim_id]
            if assessment.support_probability >= policy.minimum_evidence_support
            and assessment.contradiction_probability <= policy.maximum_evidence_contradiction
        ]
        # Start from useful original citations, ordered by quality/support rather than model order.
        originals = [assessment for assessment in eligible if assessment.evidence_id in original]
        originals.sort(
            key=lambda assessment: (
                -(assessment.support_probability * assessment.evidence_quality),
                assessment.contradiction_probability,
                assessment.evidence_id,
            )
        )
        chosen = originals[: policy.maximum_citations_per_claim]
        chosen_ids = {item.evidence_id for item in chosen}
        chosen_sources = {item.source_group_sha256 for item in chosen}

        remaining = [assessment for assessment in eligible if assessment.evidence_id not in chosen_ids]
        while len(chosen) < policy.maximum_citations_per_claim:
            support = _combined_support(chosen)
            enough_sources = len(chosen_sources) >= policy.minimum_independent_sources
            if support >= policy.minimum_claim_support and enough_sources:
                break
            if not remaining:
                break
            remaining.sort(
                key=lambda assessment: _score_candidate(
                    assessment,
                    original=original,
                    selected_sources=chosen_sources,
                    policy=policy,
                )
            )
            candidate = remaining.pop(0)
            chosen.append(candidate)
            chosen_ids.add(candidate.evidence_id)
            chosen_sources.add(candidate.source_group_sha256)

        support = _combined_support(chosen)
        contradiction = _maximum_contradiction(chosen)
        all_assessments = by_claim[claim.claim_id]
        strongest_contradiction = 0.0 if not all_assessments else max(item.contradiction_probability for item in all_assessments)
        if strongest_contradiction >= policy.maximum_claim_contradiction:
            status = ClaimRefinementStatus.CONTRADICTED
            unresolved = policy.unresolved_action
        elif support >= policy.minimum_claim_support and len(chosen_sources) >= policy.minimum_independent_sources:
            status = ClaimRefinementStatus.SUPPORTED
            unresolved = None
        elif chosen:
            status = ClaimRefinementStatus.PARTIALLY_SUPPORTED
            unresolved = None
        else:
            status = ClaimRefinementStatus.UNRESOLVED
            unresolved = policy.unresolved_action

        refined = tuple(item.evidence_id for item in chosen)
        results.append(
            RefinedClaimCitations(
                claim_id=claim.claim_id,
                claim_sha256=claim.claim_sha256,
                status=status,
                original_citation_ids=claim.original_citation_ids,
                refined_citation_ids=refined,
                added_citation_ids=tuple(value for value in refined if value not in original),
                removed_citation_ids=tuple(value for value in claim.original_citation_ids if value not in set(refined)),
                independent_sources=len(chosen_sources),
                combined_support=support,
                maximum_contradiction=max(contradiction, strongest_contradiction if status == ClaimRefinementStatus.CONTRADICTED else contradiction),
                unresolved_action=unresolved,
            )
        )

    requires_review = any(
        result.status in {ClaimRefinementStatus.PARTIALLY_SUPPORTED, ClaimRefinementStatus.CONTRADICTED, ClaimRefinementStatus.UNRESOLVED}
        for result in results
    )
    requires_abstention = any(result.unresolved_action == UnresolvedClaimAction.ABSTAIN for result in results)
    payload = {
        "answer_sha256": answer_digest,
        "allowed_evidence_set_sha256": evidence_set_digest,
        "policy_sha256": policy.policy_sha256,
        "claim_results": tuple(results),
        "requires_review": requires_review,
        "requires_abstention": requires_abstention,
    }
    digest_payload = {
        "schema": "rigorousrag-citation-refinement-receipt/v1",
        "answer_sha256": answer_digest,
        "allowed_evidence_set_sha256": evidence_set_digest,
        "policy_sha256": policy.policy_sha256,
        "claim_results": [
            {
                **asdict(result),
                "status": result.status.value,
                "unresolved_action": None if result.unresolved_action is None else result.unresolved_action.value,
            }
            for result in results
        ],
        "requires_review": requires_review,
        "requires_abstention": requires_abstention,
    }
    return CitationRefinementReceipt(receipt_sha256=_digest(digest_payload), **payload)


__all__ = [
    "CitationRefinementPolicy",
    "CitationRefinementReceipt",
    "ClaimBinding",
    "ClaimEvidenceAssessment",
    "ClaimRefinementStatus",
    "RefinedClaimCitations",
    "UnresolvedClaimAction",
    "refine_citations",
]
