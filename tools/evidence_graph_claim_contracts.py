"""Closed, deterministic contracts for reviewed scientific claim extraction."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from tools.security import normalize_owner_id

CLAIM_TYPES = frozenset(
    {
        "finding",
        "hypothesis",
        "causal",
        "associational",
        "comparative",
        "methodological",
        "limitation",
        "null_result",
        "negative_result",
        "recommendation",
    }
)
CLAIM_MODALITIES = frozenset(
    {"asserted", "suggested", "conditional", "uncertain", "negated"}
)
PROPOSER_KINDS = frozenset({"human", "model", "rule"})
REVIEW_DECISIONS = frozenset({"approved", "rejected", "superseded"})

_MAX_METADATA_ITEMS = 64
_MAX_METADATA_BYTES = 100_000
_MAX_TEXT = 2_000
_MAX_EVIDENCE_SPAN = 100_000
_MAX_SCOPE_VALUES = 1_000
_MAX_REVIEWERS = 1_000


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    selected = value.strip()
    if (
        not selected
        or len(selected) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in selected)
    ):
        raise ValueError(f"{label} is invalid.")
    return selected


def _bounded_text(value: Any, label: str, maximum: int = _MAX_TEXT) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text.")
    selected = value.strip()
    if not selected or len(selected) > maximum or "\x00" in selected:
        raise ValueError(f"{label} is invalid or too long.")
    return selected


def _digest(value: Any, label: str) -> str:
    selected = _identifier(value, label, 64).lower()
    if len(selected) != 64 or any(character not in "0123456789abcdef" for character in selected):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return selected


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(selected) or selected < 0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return selected


def _confidence(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("confidence must be finite and between 0 and 1.")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("confidence must be finite and between 0 and 1.") from exc
    if not math.isfinite(selected) or not 0.0 <= selected <= 1.0:
        raise ValueError("confidence must be finite and between 0 and 1.")
    return selected


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a mapping.")
    result: dict[str, Any] = {}
    for index, (raw_key, item) in enumerate(value.items()):
        if index >= _MAX_METADATA_ITEMS:
            raise ValueError("metadata contains too many fields.")
        key = _identifier(raw_key, "metadata key", 200)
        if item is None or isinstance(item, (bool, int)):
            result[key] = item
        elif isinstance(item, float) and math.isfinite(item):
            result[key] = item
        elif isinstance(item, str) and len(item) <= 10_000 and "\x00" not in item:
            result[key] = item
        else:
            raise ValueError("metadata contains an unsupported value.")
    payload = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(payload) > _MAX_METADATA_BYTES:
        raise ValueError("metadata exceeds the serialized byte limit.")
    return result


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _scope_values(value: Any, label: str, *, owner_scope: bool = False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a bounded array.")
    if not 1 <= len(value) <= _MAX_SCOPE_VALUES:
        raise ValueError(f"{label} must contain 1-{_MAX_SCOPE_VALUES} entries.")
    selected: set[str] = set()
    for item in value:
        rendered = _identifier(item, label, 500)
        if rendered == "*":
            selected.add(rendered)
        elif owner_scope:
            selected.add(normalize_owner_id(rendered))
        else:
            selected.add(rendered)
    if "*" in selected and len(selected) != 1:
        raise ValueError(f"{label} wildcard may not be combined with explicit entries.")
    return tuple(sorted(selected))


def _allows(scope: tuple[str, ...], value: str) -> bool:
    return scope == ("*",) or value in scope


@dataclass(frozen=True)
class ClaimEvidenceLocator:
    section_index: int
    page_number: int | None
    char_start: int
    char_end: int
    evidence_sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "section_index",
            _integer(self.section_index, "section_index", 0, 999_999),
        )
        if self.page_number is not None:
            object.__setattr__(
                self,
                "page_number",
                _integer(self.page_number, "page_number", 1, 1_000_000),
            )
        start = _integer(self.char_start, "char_start", 0, 50_000_000)
        end = _integer(self.char_end, "char_end", 1, 50_000_000)
        if end <= start or end - start > _MAX_EVIDENCE_SPAN:
            raise ValueError("claim evidence span is empty or too large.")
        object.__setattr__(self, "char_start", start)
        object.__setattr__(self, "char_end", end)
        object.__setattr__(
            self,
            "evidence_sha256",
            _digest(self.evidence_sha256, "evidence_sha256"),
        )
        if self.schema_version != 1:
            raise ValueError("claim evidence locator schema is unsupported.")

    @property
    def locator_digest(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class ScientificClaimProposal:
    proposal_id: str
    owner_id: str
    doc_id: str
    generation: int
    content_sha256: str
    profile_fingerprint: str
    claim_key: str
    claim_text: str
    claim_type: str
    modality: str
    locator: ClaimEvidenceLocator
    proposer_kind: str
    proposer_id: str
    extractor_name: str | None
    extractor_version: str | None
    confidence: float
    supersedes_proposal_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    schema_version: int = 1

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        document = _identifier(self.doc_id, "doc_id", 200)
        generation = _integer(self.generation, "generation", 1, 2**63 - 1)
        content = _digest(self.content_sha256, "content_sha256")
        profile = _digest(self.profile_fingerprint, "profile_fingerprint")
        claim_key = _identifier(self.claim_key, "claim_key", 2_000)
        claim_text = _bounded_text(self.claim_text, "claim_text")
        claim_type = _identifier(self.claim_type, "claim_type", 50)
        if claim_type not in CLAIM_TYPES:
            raise ValueError("claim_type is unsupported.")
        modality = _identifier(self.modality, "modality", 50)
        if modality not in CLAIM_MODALITIES:
            raise ValueError("modality is unsupported.")
        if not isinstance(self.locator, ClaimEvidenceLocator):
            raise ValueError("locator must be ClaimEvidenceLocator.")
        proposer_kind = _identifier(self.proposer_kind, "proposer_kind", 20)
        if proposer_kind not in PROPOSER_KINDS:
            raise ValueError("proposer_kind is unsupported.")
        proposer_id = _identifier(self.proposer_id, "proposer_id", 200)
        extractor_name = None if self.extractor_name is None else _identifier(
            self.extractor_name, "extractor_name", 200
        )
        extractor_version = None if self.extractor_version is None else _identifier(
            self.extractor_version, "extractor_version", 200
        )
        if proposer_kind == "human" and (
            extractor_name is not None or extractor_version is not None
        ):
            raise ValueError("human proposals may not claim an extractor.")
        if proposer_kind in {"model", "rule"} and (
            extractor_name is None or extractor_version is None
        ):
            raise ValueError("model/rule proposals require extractor identity and version.")
        confidence = _confidence(self.confidence)
        supersedes = None if self.supersedes_proposal_id is None else _digest(
            self.supersedes_proposal_id, "supersedes_proposal_id"
        )
        metadata = _metadata(self.metadata)
        stable = {
            "scope": "rigorousrag-scientific-claim-proposal-v1",
            "owner_id": owner,
            "doc_id": document,
            "generation": generation,
            "content_sha256": content,
            "profile_fingerprint": profile,
            "claim_key": claim_key,
            "claim_text": claim_text,
            "claim_type": claim_type,
            "modality": modality,
            "locator_digest": self.locator.locator_digest,
            "proposer_kind": proposer_kind,
            "proposer_id": proposer_id,
            "extractor_name": extractor_name,
            "extractor_version": extractor_version,
            "confidence": confidence,
            "supersedes_proposal_id": supersedes,
            "metadata": metadata,
        }
        expected = _sha256(stable)
        if _digest(self.proposal_id, "proposal_id") != expected:
            raise ValueError("proposal_id does not match deterministic proposal identity.")
        object.__setattr__(self, "proposal_id", expected)
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "doc_id", document)
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "content_sha256", content)
        object.__setattr__(self, "profile_fingerprint", profile)
        object.__setattr__(self, "claim_key", claim_key)
        object.__setattr__(self, "claim_text", claim_text)
        object.__setattr__(self, "claim_type", claim_type)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "proposer_kind", proposer_kind)
        object.__setattr__(self, "proposer_id", proposer_id)
        object.__setattr__(self, "extractor_name", extractor_name)
        object.__setattr__(self, "extractor_version", extractor_version)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "supersedes_proposal_id", supersedes)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "created_at"))
        if self.schema_version != 1:
            raise ValueError("scientific claim proposal schema is unsupported.")

    @property
    def proposal_digest(self) -> str:
        payload = asdict(self)
        payload.pop("created_at", None)
        return _sha256(payload)

    @classmethod
    def create(cls, *, created_at: float | None = None, **kwargs: Any) -> "ScientificClaimProposal":
        locator = kwargs["locator"]
        if not isinstance(locator, ClaimEvidenceLocator):
            raise ValueError("locator must be ClaimEvidenceLocator.")
        owner = normalize_owner_id(kwargs["owner_id"])
        document = _identifier(kwargs["doc_id"], "doc_id", 200)
        generation = _integer(kwargs["generation"], "generation", 1, 2**63 - 1)
        content = _digest(kwargs["content_sha256"], "content_sha256")
        profile = _digest(kwargs["profile_fingerprint"], "profile_fingerprint")
        claim_key = _identifier(kwargs["claim_key"], "claim_key", 2_000)
        claim_text = _bounded_text(kwargs["claim_text"], "claim_text")
        claim_type = _identifier(kwargs["claim_type"], "claim_type", 50)
        modality = _identifier(kwargs["modality"], "modality", 50)
        proposer_kind = _identifier(kwargs["proposer_kind"], "proposer_kind", 20)
        proposer_id = _identifier(kwargs["proposer_id"], "proposer_id", 200)
        extractor_name = None if kwargs.get("extractor_name") is None else _identifier(
            kwargs["extractor_name"], "extractor_name", 200
        )
        extractor_version = None if kwargs.get("extractor_version") is None else _identifier(
            kwargs["extractor_version"], "extractor_version", 200
        )
        confidence = _confidence(kwargs["confidence"])
        supersedes = None if kwargs.get("supersedes_proposal_id") is None else _digest(
            kwargs["supersedes_proposal_id"], "supersedes_proposal_id"
        )
        metadata = _metadata(kwargs.get("metadata"))
        stable = {
            "scope": "rigorousrag-scientific-claim-proposal-v1",
            "owner_id": owner,
            "doc_id": document,
            "generation": generation,
            "content_sha256": content,
            "profile_fingerprint": profile,
            "claim_key": claim_key,
            "claim_text": claim_text,
            "claim_type": claim_type,
            "modality": modality,
            "locator_digest": locator.locator_digest,
            "proposer_kind": proposer_kind,
            "proposer_id": proposer_id,
            "extractor_name": extractor_name,
            "extractor_version": extractor_version,
            "confidence": confidence,
            "supersedes_proposal_id": supersedes,
            "metadata": metadata,
        }
        return cls(
            proposal_id=_sha256(stable),
            owner_id=owner,
            doc_id=document,
            generation=generation,
            content_sha256=content,
            profile_fingerprint=profile,
            claim_key=claim_key,
            claim_text=claim_text,
            claim_type=claim_type,
            modality=modality,
            locator=locator,
            proposer_kind=proposer_kind,
            proposer_id=proposer_id,
            extractor_name=extractor_name,
            extractor_version=extractor_version,
            confidence=confidence,
            supersedes_proposal_id=supersedes,
            metadata=metadata,
            created_at=time.time() if created_at is None else created_at,
        )


@dataclass(frozen=True)
class ClaimReviewDecision:
    decision_id: str
    proposal_id: str
    owner_id: str
    decision: str
    reviewer_id: str
    reason_code: str
    replacement_proposal_id: str | None
    decided_at: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        proposal = _digest(self.proposal_id, "proposal_id")
        owner = normalize_owner_id(self.owner_id)
        decision = _identifier(self.decision, "decision", 20)
        if decision not in REVIEW_DECISIONS:
            raise ValueError("claim review decision is unsupported.")
        reviewer = _identifier(self.reviewer_id, "reviewer_id", 200)
        reason = _identifier(self.reason_code, "reason_code", 200)
        replacement = None if self.replacement_proposal_id is None else _digest(
            self.replacement_proposal_id, "replacement_proposal_id"
        )
        if decision == "superseded" and replacement is None:
            raise ValueError("superseded decisions require a replacement proposal.")
        if decision != "superseded" and replacement is not None:
            raise ValueError("only superseded decisions may name a replacement proposal.")
        stable = {
            "scope": "rigorousrag-scientific-claim-review-decision-v1",
            "proposal_id": proposal,
            "owner_id": owner,
            "decision": decision,
            "reviewer_id": reviewer,
            "reason_code": reason,
            "replacement_proposal_id": replacement,
        }
        expected = _sha256(stable)
        if _digest(self.decision_id, "decision_id") != expected:
            raise ValueError("decision_id does not match deterministic decision identity.")
        object.__setattr__(self, "decision_id", expected)
        object.__setattr__(self, "proposal_id", proposal)
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "reviewer_id", reviewer)
        object.__setattr__(self, "reason_code", reason)
        object.__setattr__(self, "replacement_proposal_id", replacement)
        object.__setattr__(self, "decided_at", _timestamp(self.decided_at, "decided_at"))
        if self.schema_version != 1:
            raise ValueError("claim review decision schema is unsupported.")

    @classmethod
    def create(cls, *, decided_at: float | None = None, **kwargs: Any) -> "ClaimReviewDecision":
        proposal = _digest(kwargs["proposal_id"], "proposal_id")
        owner = normalize_owner_id(kwargs["owner_id"])
        decision = _identifier(kwargs["decision"], "decision", 20)
        reviewer = _identifier(kwargs["reviewer_id"], "reviewer_id", 200)
        reason = _identifier(kwargs["reason_code"], "reason_code", 200)
        replacement = None if kwargs.get("replacement_proposal_id") is None else _digest(
            kwargs["replacement_proposal_id"], "replacement_proposal_id"
        )
        stable = {
            "scope": "rigorousrag-scientific-claim-review-decision-v1",
            "proposal_id": proposal,
            "owner_id": owner,
            "decision": decision,
            "reviewer_id": reviewer,
            "reason_code": reason,
            "replacement_proposal_id": replacement,
        }
        return cls(
            decision_id=_sha256(stable),
            proposal_id=proposal,
            owner_id=owner,
            decision=decision,
            reviewer_id=reviewer,
            reason_code=reason,
            replacement_proposal_id=replacement,
            decided_at=time.time() if decided_at is None else decided_at,
        )


@dataclass(frozen=True)
class ClaimReviewerGrant:
    reviewer_id: str
    owners: tuple[str, ...]
    doc_ids: tuple[str, ...]
    decisions: tuple[str, ...]
    expires_at: float | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "reviewer_id", _identifier(self.reviewer_id, "reviewer_id", 200))
        object.__setattr__(self, "owners", _scope_values(self.owners, "owners", owner_scope=True))
        object.__setattr__(self, "doc_ids", _scope_values(self.doc_ids, "doc_ids"))
        decisions = _scope_values(self.decisions, "decisions")
        if decisions == ("*",) or any(value not in REVIEW_DECISIONS for value in decisions):
            raise ValueError("decisions contains an unsupported value.")
        object.__setattr__(self, "decisions", decisions)
        if self.expires_at is not None:
            object.__setattr__(self, "expires_at", _timestamp(self.expires_at, "expires_at"))
        if self.schema_version != 1:
            raise ValueError("claim reviewer grant schema is unsupported.")

    @property
    def grant_digest(self) -> str:
        return _sha256(asdict(self))

    def permits(self, *, owner_id: str, doc_id: str, decision: str, now: float) -> bool:
        return bool(
            _allows(self.owners, owner_id)
            and _allows(self.doc_ids, doc_id)
            and decision in self.decisions
            and (self.expires_at is None or now <= self.expires_at)
        )


@dataclass(frozen=True)
class ClaimReviewPolicy:
    reviewers: tuple[ClaimReviewerGrant, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.reviewers, tuple)
            or not 1 <= len(self.reviewers) <= _MAX_REVIEWERS
            or any(not isinstance(value, ClaimReviewerGrant) for value in self.reviewers)
        ):
            raise ValueError("reviewers must be a bounded non-empty tuple of grants.")
        ordered = tuple(sorted(self.reviewers, key=lambda value: value.reviewer_id))
        if len({value.reviewer_id for value in ordered}) != len(ordered):
            raise ValueError("reviewer IDs must be unique.")
        object.__setattr__(self, "reviewers", ordered)
        if self.schema_version != 1:
            raise ValueError("claim review policy schema is unsupported.")

    @property
    def policy_digest(self) -> str:
        return _sha256(asdict(self))

    def grant_for(self, reviewer_id: str) -> ClaimReviewerGrant:
        selected = _identifier(reviewer_id, "reviewer_id", 200)
        for grant in self.reviewers:
            if grant.reviewer_id == selected:
                return grant
        raise PermissionError("reviewer is not authorized by claim policy.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ClaimReviewPolicy":
        if not isinstance(value, Mapping) or set(value) != {"schema_version", "reviewers"}:
            raise ValueError("claim review policy schema is invalid.")
        if value["schema_version"] != 1:
            raise ValueError("claim review policy schema is unsupported.")
        raw_reviewers = value["reviewers"]
        if (
            isinstance(raw_reviewers, (str, bytes, bytearray))
            or not isinstance(raw_reviewers, Sequence)
            or not 1 <= len(raw_reviewers) <= _MAX_REVIEWERS
        ):
            raise ValueError("reviewers must be a bounded non-empty array.")
        allowed = {"reviewer_id", "owners", "doc_ids", "decisions", "expires_at"}
        required = allowed - {"expires_at"}
        grants: list[ClaimReviewerGrant] = []
        for raw in raw_reviewers:
            if not isinstance(raw, Mapping) or not required <= set(raw) <= allowed:
                raise ValueError("claim reviewer grant schema is invalid.")
            grants.append(
                ClaimReviewerGrant(
                    reviewer_id=raw["reviewer_id"],
                    owners=_scope_values(raw["owners"], "owners", owner_scope=True),
                    doc_ids=_scope_values(raw["doc_ids"], "doc_ids"),
                    decisions=_scope_values(raw["decisions"], "decisions"),
                    expires_at=raw.get("expires_at"),
                )
            )
        return cls(reviewers=tuple(grants))


@dataclass(frozen=True)
class ClaimReviewAuthorization:
    proposal_id: str
    decision_id: str
    owner_id: str
    doc_id: str
    generation: int
    decision: str
    reviewer_id: str
    policy_digest: str
    grant_digest: str
    authorization_digest: str
    authorized_at: float
    separation_of_duties_enforced: bool = True
    replacement_scope_validated: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "proposal_id",
            "decision_id",
            "policy_digest",
            "grant_digest",
            "authorization_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "doc_id", _identifier(self.doc_id, "doc_id", 200))
        object.__setattr__(self, "generation", _integer(self.generation, "generation", 1, 2**63 - 1))
        decision = _identifier(self.decision, "decision", 20)
        if decision not in REVIEW_DECISIONS:
            raise ValueError("authorization decision is unsupported.")
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "reviewer_id", _identifier(self.reviewer_id, "reviewer_id", 200))
        object.__setattr__(self, "authorized_at", _timestamp(self.authorized_at, "authorized_at"))
        if self.separation_of_duties_enforced is not True:
            raise ValueError("separation_of_duties_enforced must remain true.")
        if not isinstance(self.replacement_scope_validated, bool):
            raise ValueError("replacement_scope_validated must be boolean.")
        stable = {
            "scope": "rigorousrag-scientific-claim-review-authorization-v1",
            "proposal_id": self.proposal_id,
            "decision_id": self.decision_id,
            "owner_id": self.owner_id,
            "doc_id": self.doc_id,
            "generation": self.generation,
            "decision": self.decision,
            "reviewer_id": self.reviewer_id,
            "policy_digest": self.policy_digest,
            "grant_digest": self.grant_digest,
            "separation_of_duties_enforced": True,
            "replacement_scope_validated": self.replacement_scope_validated,
        }
        if self.authorization_digest != _sha256(stable):
            raise ValueError("authorization_digest differs from governed claim scope.")
        if self.schema_version != 1:
            raise ValueError("claim review authorization schema is unsupported.")

    @classmethod
    def create(cls, *, authorized_at: float | None = None, **kwargs: Any) -> "ClaimReviewAuthorization":
        stable = {
            "scope": "rigorousrag-scientific-claim-review-authorization-v1",
            "proposal_id": _digest(kwargs["proposal_id"], "proposal_id"),
            "decision_id": _digest(kwargs["decision_id"], "decision_id"),
            "owner_id": normalize_owner_id(kwargs["owner_id"]),
            "doc_id": _identifier(kwargs["doc_id"], "doc_id", 200),
            "generation": _integer(kwargs["generation"], "generation", 1, 2**63 - 1),
            "decision": _identifier(kwargs["decision"], "decision", 20),
            "reviewer_id": _identifier(kwargs["reviewer_id"], "reviewer_id", 200),
            "policy_digest": _digest(kwargs["policy_digest"], "policy_digest"),
            "grant_digest": _digest(kwargs["grant_digest"], "grant_digest"),
            "separation_of_duties_enforced": True,
            "replacement_scope_validated": bool(kwargs.get("replacement_scope_validated", False)),
        }
        return cls(
            **{key: value for key, value in stable.items() if key != "scope"},
            authorization_digest=_sha256(stable),
            authorized_at=time.time() if authorized_at is None else authorized_at,
        )


__all__ = [
    "CLAIM_MODALITIES",
    "CLAIM_TYPES",
    "PROPOSER_KINDS",
    "REVIEW_DECISIONS",
    "ClaimEvidenceLocator",
    "ClaimReviewAuthorization",
    "ClaimReviewDecision",
    "ClaimReviewPolicy",
    "ClaimReviewerGrant",
    "ScientificClaimProposal",
    "_digest",
    "_identifier",
    "_integer",
    "_metadata",
    "_sha256",
    "_timestamp",
]
