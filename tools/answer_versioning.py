"""Evidence-aware answer versioning and claim/citation diffs.

Versions are immutable content-addressed records of answer claims, authoritative citation
IDs, corpus/model/policy identities and source-status events. Diffs expose *why* an answer
changed without reconstructing private evidence or treating text similarity as truth.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


def _text(value: Any, label: str, maximum: int = 20000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _sha(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return ""
    digest = _text(value, label, 64).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class VersionedClaim:
    claim_id: str
    text: str
    citation_ids: tuple[str, ...]
    support_status: str
    entailment_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _text(self.claim_id, "claim_id", 256))
        object.__setattr__(self, "text", _text(self.text, "claim text", 5000))
        if len(self.citation_ids) > 1000:
            raise ValueError("citation_ids exceed the item limit")
        object.__setattr__(self, "citation_ids", tuple(dict.fromkeys(_text(item, "citation_id", 256) for item in self.citation_ids)))
        status = _text(self.support_status, "support_status", 32).lower()
        if status not in {"supported", "unsupported", "contradicted", "mixed", "unreviewed"}:
            raise ValueError("unsupported support status")
        object.__setattr__(self, "support_status", status)
        object.__setattr__(self, "entailment_sha256", _sha(self.entailment_sha256, "entailment_sha256", allow_empty=True))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True)
class AnswerVersion:
    answer_id: str
    version_id: str
    answer_text_sha256: str
    claims: tuple[VersionedClaim, ...]
    citation_ids: tuple[str, ...]
    corpus_generation_sha256: str
    model_sha256: str = ""
    policy_sha256: str = ""
    graph_sha256: str = ""
    source_status_sha256: str = ""
    parent_version_id: str = ""
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        for name in ("answer_id", "version_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name, 256))
        object.__setattr__(self, "answer_text_sha256", _sha(self.answer_text_sha256, "answer_text_sha256"))
        if len(self.claims) > 10_000 or len({claim.claim_id for claim in self.claims}) != len(self.claims):
            raise ValueError("claims are invalid or contain duplicate IDs")
        if len(self.citation_ids) > 100_000:
            raise ValueError("citation_ids exceed the item limit")
        object.__setattr__(self, "citation_ids", tuple(dict.fromkeys(_text(item, "citation_id", 256) for item in self.citation_ids)))
        object.__setattr__(self, "corpus_generation_sha256", _sha(self.corpus_generation_sha256, "corpus_generation_sha256"))
        for name in ("model_sha256", "policy_sha256", "graph_sha256", "source_status_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name, allow_empty=True))
        object.__setattr__(self, "parent_version_id", _text(self.parent_version_id, "parent_version_id", 256, allow_empty=True))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True)
class ClaimDelta:
    claim_id: str
    change: str
    before_sha256: str = ""
    after_sha256: str = ""
    citation_added: tuple[str, ...] = ()
    citation_removed: tuple[str, ...] = ()
    support_before: str = ""
    support_after: str = ""


@dataclass(frozen=True)
class AnswerDiff:
    answer_id: str
    before_version_id: str
    after_version_id: str
    claim_deltas: tuple[ClaimDelta, ...]
    citation_added: tuple[str, ...]
    citation_removed: tuple[str, ...]
    changed_inputs: tuple[str, ...]
    fingerprint: str


def diff_answers(before: AnswerVersion, after: AnswerVersion) -> AnswerDiff:
    if before.answer_id != after.answer_id:
        raise ValueError("answer versions belong to different answer IDs")
    before_claims = {claim.claim_id: claim for claim in before.claims}
    after_claims = {claim.claim_id: claim for claim in after.claims}
    deltas: list[ClaimDelta] = []
    for claim_id in sorted(set(before_claims) | set(after_claims)):
        left, right = before_claims.get(claim_id), after_claims.get(claim_id)
        if left is None and right is not None:
            deltas.append(ClaimDelta(claim_id, "added", "", right.fingerprint, tuple(right.citation_ids), (), "", right.support_status))
            continue
        if right is None and left is not None:
            deltas.append(ClaimDelta(claim_id, "removed", left.fingerprint, "", (), tuple(left.citation_ids), left.support_status, ""))
            continue
        assert left is not None and right is not None
        if left.fingerprint == right.fingerprint:
            continue
        added = tuple(sorted(set(right.citation_ids) - set(left.citation_ids)))
        removed = tuple(sorted(set(left.citation_ids) - set(right.citation_ids)))
        deltas.append(ClaimDelta(claim_id, "changed", left.fingerprint, right.fingerprint, added, removed, left.support_status, right.support_status))
    citations_before, citations_after = set(before.citation_ids), set(after.citation_ids)
    changed_inputs: list[str] = []
    for name in ("corpus_generation_sha256", "model_sha256", "policy_sha256", "graph_sha256", "source_status_sha256"):
        if getattr(before, name) != getattr(after, name):
            changed_inputs.append(name)
    payload = {
        "answer_id": before.answer_id,
        "before": before.version_id,
        "after": after.version_id,
        "deltas": [asdict(item) for item in deltas],
        "citation_added": sorted(citations_after - citations_before),
        "citation_removed": sorted(citations_before - citations_after),
        "changed_inputs": changed_inputs,
    }
    return AnswerDiff(
        before.answer_id,
        before.version_id,
        after.version_id,
        tuple(deltas),
        tuple(sorted(citations_after - citations_before)),
        tuple(sorted(citations_before - citations_after)),
        tuple(changed_inputs),
        hashlib.sha256(_canonical(payload)).hexdigest(),
    )


__all__ = ["AnswerDiff", "AnswerVersion", "ClaimDelta", "VersionedClaim", "diff_answers"]
