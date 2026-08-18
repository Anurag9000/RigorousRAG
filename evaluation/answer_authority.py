"""Contradiction-first authority checks for final RAG answer publication.

This module evaluates a draft using only hashes, claim identities, evidence identities and
semantic probability observations. Raw answer/evidence text is not persisted here. Every
factual claim must be grounded in evidence that was actually present in the verified
materialized context before a draft can become publishable.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from evaluation.semantic_support import SemanticProbabilities
from tools.evidence_context_materialization import MaterializedContext


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _text(value: Any, label: str, maximum: int = 1000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _probability(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be in [0,1]")
    selected = float(value)
    if not math.isfinite(selected) or not 0.0 <= selected <= 1.0:
        raise ValueError(f"{label} must be in [0,1]")
    return selected


@dataclass(frozen=True)
class DraftClaim:
    claim_id: str
    claim_sha256: str
    requires_evidence: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _text(self.claim_id, "claim_id"))
        object.__setattr__(self, "claim_sha256", _sha(self.claim_sha256, "claim_sha256"))
        if not isinstance(self.requires_evidence, bool):
            raise ValueError("requires_evidence must be boolean")


@dataclass(frozen=True)
class ClaimEvidenceAuthority:
    claim_id: str
    evidence_sha256: str
    probabilities: SemanticProbabilities
    authority_decision_sha256: str | None = None
    authoritative: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _text(self.claim_id, "claim_id"))
        object.__setattr__(self, "evidence_sha256", _sha(self.evidence_sha256, "evidence_sha256"))
        if not isinstance(self.probabilities, SemanticProbabilities):
            raise ValueError("probabilities must be SemanticProbabilities")
        if self.authority_decision_sha256 is not None:
            object.__setattr__(self, "authority_decision_sha256", _sha(self.authority_decision_sha256, "authority_decision_sha256"))
        if not isinstance(self.authoritative, bool):
            raise ValueError("authoritative must be boolean")

    @property
    def observation_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-answer-claim-evidence-authority/v1", **asdict(self)})


@dataclass(frozen=True)
class AnswerDraftManifest:
    request_sha256: str
    answer_sha256: str
    materialized_context_sha256: str
    generator_model_sha256: str
    generation_config_sha256: str
    prompt_template_sha256: str
    claims: tuple[DraftClaim, ...]

    def __post_init__(self) -> None:
        for name in ("request_sha256", "answer_sha256", "materialized_context_sha256", "generator_model_sha256", "generation_config_sha256", "prompt_template_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        claims = tuple(self.claims)
        if not claims or any(not isinstance(row, DraftClaim) for row in claims):
            raise ValueError("draft manifest requires non-empty DraftClaim values")
        if len({row.claim_id for row in claims}) != len(claims) or len({row.claim_sha256 for row in claims}) != len(claims):
            raise ValueError("draft claims must have unique ids and digests")
        object.__setattr__(self, "claims", claims)

    @property
    def manifest_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-answer-draft-manifest/v1", **asdict(self)})


@dataclass(frozen=True)
class AnswerAuthorityPolicy:
    min_entailment_probability: float = 0.70
    max_contradiction_probability: float = 0.20
    min_supported_claim_fraction: float = 1.0
    max_unverified_authority_fraction: float = 0.0
    review_on_unsupported: bool = False

    def __post_init__(self) -> None:
        for name in ("min_entailment_probability", "max_contradiction_probability", "min_supported_claim_fraction", "max_unverified_authority_fraction"):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        if not isinstance(self.review_on_unsupported, bool):
            raise ValueError("review_on_unsupported must be boolean")

    @property
    def policy_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-answer-authority-policy/v1", **asdict(self)})


@dataclass(frozen=True)
class ClaimAuthorityResult:
    claim_id: str
    claim_sha256: str
    status: str
    cited_evidence_sha256s: tuple[str, ...]
    max_entailment_probability: float
    max_contradiction_probability: float
    non_authoritative_evidence_count: int
    result_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _text(self.claim_id, "claim_id"))
        object.__setattr__(self, "claim_sha256", _sha(self.claim_sha256, "claim_sha256"))
        if self.status not in {"non_factual", "supported", "unsupported", "contradicted", "uncited", "context_mismatch", "authority_unverified"}:
            raise ValueError("claim authority status is invalid")
        evidence = tuple(sorted(_sha(value, "cited evidence sha256") for value in self.cited_evidence_sha256s))
        if len(set(evidence)) != len(evidence):
            raise ValueError("cited evidence identities must be unique")
        object.__setattr__(self, "cited_evidence_sha256s", evidence)
        object.__setattr__(self, "max_entailment_probability", _probability(self.max_entailment_probability, "max_entailment_probability"))
        object.__setattr__(self, "max_contradiction_probability", _probability(self.max_contradiction_probability, "max_contradiction_probability"))
        if isinstance(self.non_authoritative_evidence_count, bool) or not isinstance(self.non_authoritative_evidence_count, int) or self.non_authoritative_evidence_count < 0:
            raise ValueError("non_authoritative_evidence_count must be non-negative")
        expected = _digest(self._payload())
        provided = _sha(self.result_sha256, "result_sha256")
        if expected != provided:
            raise ValueError("result_sha256 does not match claim authority result")
        object.__setattr__(self, "result_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-claim-authority-result/v1",
            "claim_id": self.claim_id,
            "claim_sha256": self.claim_sha256,
            "status": self.status,
            "cited_evidence_sha256s": self.cited_evidence_sha256s,
            "max_entailment_probability": self.max_entailment_probability,
            "max_contradiction_probability": self.max_contradiction_probability,
            "non_authoritative_evidence_count": self.non_authoritative_evidence_count,
        }


@dataclass(frozen=True)
class AnswerAuthorityDecision:
    draft_manifest_sha256: str
    context_sha256: str
    policy_sha256: str
    action: str
    claim_results: tuple[ClaimAuthorityResult, ...]
    supported_claim_fraction: float
    contradicted_claim_count: int
    uncited_claim_count: int
    context_mismatch_count: int
    unverified_authority_fraction: float
    reason_codes: tuple[str, ...]
    decision_sha256: str

    def __post_init__(self) -> None:
        for name in ("draft_manifest_sha256", "context_sha256", "policy_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if self.action not in {"publish", "review_required", "abstain", "blocked"}:
            raise ValueError("answer authority action is invalid")
        rows = tuple(self.claim_results)
        if not rows:
            raise ValueError("answer authority decision requires claim results")
        object.__setattr__(self, "claim_results", rows)
        object.__setattr__(self, "supported_claim_fraction", _probability(self.supported_claim_fraction, "supported_claim_fraction"))
        object.__setattr__(self, "unverified_authority_fraction", _probability(self.unverified_authority_fraction, "unverified_authority_fraction"))
        for name in ("contradicted_claim_count", "uncited_claim_count", "context_mismatch_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        reasons = tuple(sorted(set(self.reason_codes)))
        if self.action == "publish" and reasons:
            raise ValueError("publish decision may not contain failure reasons")
        if self.action != "publish" and not reasons:
            raise ValueError("non-publish decision requires reason codes")
        object.__setattr__(self, "reason_codes", reasons)
        expected = _digest(self._payload())
        provided = _sha(self.decision_sha256, "decision_sha256")
        if expected != provided:
            raise ValueError("decision_sha256 does not match answer authority decision")
        object.__setattr__(self, "decision_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-answer-authority-decision/v1",
            "draft_manifest_sha256": self.draft_manifest_sha256,
            "context_sha256": self.context_sha256,
            "policy_sha256": self.policy_sha256,
            "action": self.action,
            "claim_results": [asdict(row) for row in self.claim_results],
            "supported_claim_fraction": self.supported_claim_fraction,
            "contradicted_claim_count": self.contradicted_claim_count,
            "uncited_claim_count": self.uncited_claim_count,
            "context_mismatch_count": self.context_mismatch_count,
            "unverified_authority_fraction": self.unverified_authority_fraction,
            "reason_codes": self.reason_codes,
        }


def evaluate_answer_authority(
    draft: AnswerDraftManifest,
    context: MaterializedContext,
    observations: Sequence[ClaimEvidenceAuthority],
    *,
    policy: AnswerAuthorityPolicy = AnswerAuthorityPolicy(),
) -> AnswerAuthorityDecision:
    if not isinstance(draft, AnswerDraftManifest) or not isinstance(context, MaterializedContext):
        raise ValueError("draft/context types are invalid")
    if draft.materialized_context_sha256 != context.context_sha256:
        raise ValueError("draft manifest does not bind the supplied materialized context")
    rows = tuple(observations)
    if any(not isinstance(row, ClaimEvidenceAuthority) for row in rows):
        raise ValueError("observations contains invalid values")
    claim_by_id = {claim.claim_id: claim for claim in draft.claims}
    if any(row.claim_id not in claim_by_id for row in rows):
        raise ValueError("support observation references a claim outside the draft manifest")
    context_evidence = {row.evidence_sha256 for row in context.evidence}
    by_claim: dict[str, list[ClaimEvidenceAuthority]] = {claim.claim_id: [] for claim in draft.claims}
    for row in rows:
        by_claim[row.claim_id].append(row)

    results = []
    for claim in draft.claims:
        evidence_rows = by_claim[claim.claim_id]
        if not claim.requires_evidence:
            status = "non_factual"
            cited = ()
            max_entailment = max_contradiction = 0.0
            non_authoritative = 0
        elif not evidence_rows:
            status = "uncited"
            cited = ()
            max_entailment = max_contradiction = 0.0
            non_authoritative = 0
        else:
            cited = tuple(sorted({row.evidence_sha256 for row in evidence_rows}))
            max_entailment = max(row.probabilities.entailment for row in evidence_rows)
            max_contradiction = max(row.probabilities.contradiction for row in evidence_rows)
            non_authoritative = sum(not row.authoritative for row in evidence_rows)
            if any(value not in context_evidence for value in cited):
                status = "context_mismatch"
            elif max_contradiction > policy.max_contradiction_probability:
                status = "contradicted"
            elif non_authoritative:
                status = "authority_unverified"
            elif max_entailment >= policy.min_entailment_probability:
                status = "supported"
            else:
                status = "unsupported"
        payload = {
            "schema": "rigorousrag-claim-authority-result/v1",
            "claim_id": claim.claim_id,
            "claim_sha256": claim.claim_sha256,
            "status": status,
            "cited_evidence_sha256s": cited,
            "max_entailment_probability": max_entailment,
            "max_contradiction_probability": max_contradiction,
            "non_authoritative_evidence_count": non_authoritative,
        }
        results.append(ClaimAuthorityResult(**payload, result_sha256=_digest(payload)))

    factual = [row for row, claim in zip(results, draft.claims) if claim.requires_evidence]
    supported = sum(row.status == "supported" for row in factual)
    supported_fraction = supported / len(factual) if factual else 1.0
    contradicted = sum(row.status == "contradicted" for row in factual)
    uncited = sum(row.status == "uncited" for row in factual)
    context_mismatch = sum(row.status == "context_mismatch" for row in factual)
    authority_unverified = sum(row.status == "authority_unverified" for row in factual)
    authority_fraction = authority_unverified / len(factual) if factual else 0.0

    reasons = []
    if context_mismatch:
        reasons.append("cited_evidence_outside_verified_context")
        action = "blocked"
    elif contradicted:
        reasons.append("contradicted_claim_present")
        action = "abstain"
    else:
        if uncited:
            reasons.append("uncited_factual_claim_present")
        if authority_fraction > policy.max_unverified_authority_fraction:
            reasons.append("unverified_evidence_authority_exceeded")
        if supported_fraction < policy.min_supported_claim_fraction:
            reasons.append("supported_claim_fraction_below_threshold")
        if reasons:
            action = "review_required" if policy.review_on_unsupported else "abstain"
        else:
            action = "publish"
    payload = {
        "schema": "rigorousrag-answer-authority-decision/v1",
        "draft_manifest_sha256": draft.manifest_sha256,
        "context_sha256": context.context_sha256,
        "policy_sha256": policy.policy_sha256,
        "action": action,
        "claim_results": [asdict(row) for row in results],
        "supported_claim_fraction": supported_fraction,
        "contradicted_claim_count": contradicted,
        "uncited_claim_count": uncited,
        "context_mismatch_count": context_mismatch,
        "unverified_authority_fraction": authority_fraction,
        "reason_codes": tuple(sorted(set(reasons))),
    }
    return AnswerAuthorityDecision(**payload, decision_sha256=_digest(payload))


__all__ = [
    "AnswerAuthorityDecision",
    "AnswerAuthorityPolicy",
    "AnswerDraftManifest",
    "ClaimAuthorityResult",
    "ClaimEvidenceAuthority",
    "DraftClaim",
    "evaluate_answer_authority",
]
