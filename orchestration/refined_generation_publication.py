"""Fenced publication authority for citation-refined grounded generation.

This module prevents post-generation citation refinement from becoming an optional side
channel around RigorousRAG's serving authority.  A publishable envelope must bind:

* the exact authoritative generation receipt and runtime stack/fence;
* the SHA-256 of the immutable answer text;
* the exact server-owned evidence-id universe;
* claim/evidence semantic assessments;
* the citation-refinement policy and resulting receipt; and
* a no-review/no-abstention decision before durable publication.

The SQLite ledger shares the runtime-stack authority database and re-checks the current
stack/fence under ``BEGIN IMMEDIATE`` before recording the refined publication.  It stores
only digests and counters, never answer/evidence text.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from orchestration.authoritative_generation import AuthoritativeGenerationResult
from orchestration.runtime_stack_authority import SQLiteRuntimeStackAuthorityStore
from tools.citation_refinement import (
    CitationRefinementPolicy,
    CitationRefinementReceipt,
    ClaimBinding,
    ClaimEvidenceAssessment,
    refine_citations,
)

_HEX = frozenset("0123456789abcdef")
_MAX_EVIDENCE = 1_000_000


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _time(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite/non-negative")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite/non-negative") from exc
    if not math.isfinite(selected) or selected < 0.0:
        raise ValueError(f"{label} must be finite/non-negative")
    return selected


def evidence_universe_sha256(evidence_ids: Sequence[str]) -> str:
    selected = tuple(_text(value, "evidence id", 1_000) for value in evidence_ids)
    if not selected or len(selected) > _MAX_EVIDENCE or len(set(selected)) != len(selected):
        raise ValueError("evidence ids must be unique, non-empty and bounded")
    return _digest({"schema": "rigorousrag-evidence-id-universe/v1", "evidence_ids": sorted(selected)})


def _scope_sha256(owner_id: str, service_id: str, domain_id: str) -> str:
    return _digest(
        {
            "schema": "rigorousrag-serving-scope/v1",
            "owner_id": _text(owner_id, "owner_id"),
            "service_id": _text(service_id, "service_id"),
            "domain_id": _text(domain_id, "domain_id"),
        }
    )


@dataclass(frozen=True)
class RefinedGenerationEnvelope:
    base_result: AuthoritativeGenerationResult
    evidence_universe_sha256: str
    refinement_receipt: CitationRefinementReceipt
    publication_allowed: bool
    envelope_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.base_result, AuthoritativeGenerationResult):
            raise ValueError("base_result must be AuthoritativeGenerationResult")
        if self.base_result.receipt.action != "published" or self.base_result.grounded_output is None:
            raise ValueError("refinement requires a published authoritative generation result")
        object.__setattr__(self, "evidence_universe_sha256", _sha(self.evidence_universe_sha256, "evidence_universe_sha256"))
        if not isinstance(self.refinement_receipt, CitationRefinementReceipt):
            raise ValueError("refinement_receipt must be CitationRefinementReceipt")
        if self.refinement_receipt.allowed_evidence_set_sha256 != self.evidence_universe_sha256:
            raise ValueError("refinement receipt belongs to another evidence universe")
        if not isinstance(self.publication_allowed, bool):
            raise ValueError("publication_allowed must be boolean")
        expected_allowed = not self.refinement_receipt.requires_review and not self.refinement_receipt.requires_abstention
        if self.publication_allowed != expected_allowed:
            raise ValueError("publication_allowed differs from fail-closed refinement status")
        provided = _sha(self.envelope_sha256, "envelope_sha256")
        if provided != _digest(self._payload()):
            raise ValueError("refined-generation envelope digest mismatch")
        object.__setattr__(self, "envelope_sha256", provided)

    def _payload(self) -> Mapping[str, Any]:
        grounded = self.base_result.grounded_output
        assert grounded is not None
        return {
            "schema": "rigorousrag-refined-generation-envelope/v1",
            "serving_receipt_sha256": self.base_result.receipt.receipt_sha256,
            "grounded_output_sha256": grounded.grounded_output_sha256,
            "evidence_universe_sha256": self.evidence_universe_sha256,
            "refinement_receipt_sha256": self.refinement_receipt.receipt_sha256,
            "publication_allowed": self.publication_allowed,
        }


def build_refined_generation_envelope(
    result: AuthoritativeGenerationResult,
    *,
    evidence_ids: Sequence[str],
    claims: Sequence[ClaimBinding],
    assessments: Sequence[ClaimEvidenceAssessment],
    policy: CitationRefinementPolicy,
) -> RefinedGenerationEnvelope:
    if not isinstance(result, AuthoritativeGenerationResult):
        raise ValueError("result must be AuthoritativeGenerationResult")
    if result.receipt.action != "published" or result.grounded_output is None:
        raise ValueError("only published authoritative generation may be refined")
    grounded = result.grounded_output
    allowed = tuple(_text(value, "evidence id", 1_000) for value in evidence_ids)
    universe = evidence_universe_sha256(allowed)
    allowed_set = set(allowed)
    if any(value not in allowed_set for value in grounded.citation_ids):
        raise ValueError("grounded result contains a citation outside the supplied evidence universe")
    claim_rows = tuple(claims)
    if any(value not in set(grounded.citation_ids) for claim in claim_rows for value in claim.original_citation_ids):
        raise ValueError("claim binding attributes a citation the grounded result never emitted")
    answer_sha = hashlib.sha256(grounded.answer.encode("utf-8")).hexdigest()
    receipt = refine_citations(
        answer_sha256=answer_sha,
        allowed_evidence_set_sha256=universe,
        claims=claim_rows,
        assessments=assessments,
        allowed_evidence_ids=allowed,
        policy=policy,
    )
    payload = {
        "schema": "rigorousrag-refined-generation-envelope/v1",
        "serving_receipt_sha256": result.receipt.receipt_sha256,
        "grounded_output_sha256": grounded.grounded_output_sha256,
        "evidence_universe_sha256": universe,
        "refinement_receipt_sha256": receipt.receipt_sha256,
        "publication_allowed": not receipt.requires_review and not receipt.requires_abstention,
    }
    return RefinedGenerationEnvelope(
        base_result=result,
        evidence_universe_sha256=universe,
        refinement_receipt=receipt,
        publication_allowed=payload["publication_allowed"],
        envelope_sha256=_digest(payload),
    )


@dataclass(frozen=True)
class RefinedGenerationPublicationReceipt:
    request_sha256: str
    scope_sha256: str
    serving_receipt_sha256: str
    grounded_output_sha256: str
    refinement_receipt_sha256: str
    envelope_sha256: str
    stack_sha256: str
    fencing_token: int
    authority_revision: int
    published_at: float
    publication_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "request_sha256", "scope_sha256", "serving_receipt_sha256", "grounded_output_sha256",
            "refinement_receipt_sha256", "envelope_sha256", "stack_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        object.__setattr__(self, "fencing_token", _positive_int(self.fencing_token, "fencing_token"))
        object.__setattr__(self, "authority_revision", _positive_int(self.authority_revision, "authority_revision"))
        object.__setattr__(self, "published_at", _time(self.published_at, "published_at"))
        provided = _sha(self.publication_sha256, "publication_sha256")
        if provided != _digest(self._payload()):
            raise ValueError("refined publication digest mismatch")
        object.__setattr__(self, "publication_sha256", provided)

    def _payload(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-refined-generation-publication/v1",
            "request_sha256": self.request_sha256,
            "scope_sha256": self.scope_sha256,
            "serving_receipt_sha256": self.serving_receipt_sha256,
            "grounded_output_sha256": self.grounded_output_sha256,
            "refinement_receipt_sha256": self.refinement_receipt_sha256,
            "envelope_sha256": self.envelope_sha256,
            "stack_sha256": self.stack_sha256,
            "fencing_token": self.fencing_token,
            "authority_revision": self.authority_revision,
            "published_at": self.published_at,
        }


class SQLiteRefinedGenerationPublicationLedger:
    """Atomic digest-only publication ledger sharing runtime-stack serialization."""

    def __init__(self, runtime_authority: SQLiteRuntimeStackAuthorityStore) -> None:
        if not isinstance(runtime_authority, SQLiteRuntimeStackAuthorityStore):
            raise ValueError("runtime_authority must be SQLiteRuntimeStackAuthorityStore")
        self.runtime_authority = runtime_authority
        self.path = runtime_authority.path
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS refined_generation_publication_ledger (
                    owner_id TEXT NOT NULL, service_id TEXT NOT NULL, domain_id TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL, scope_sha256 TEXT NOT NULL,
                    serving_receipt_sha256 TEXT NOT NULL, grounded_output_sha256 TEXT NOT NULL,
                    refinement_receipt_sha256 TEXT NOT NULL, envelope_sha256 TEXT NOT NULL,
                    stack_sha256 TEXT NOT NULL, fencing_token INTEGER NOT NULL,
                    authority_revision INTEGER NOT NULL, published_at REAL NOT NULL,
                    publication_sha256 TEXT NOT NULL,
                    PRIMARY KEY(owner_id,service_id,domain_id,request_sha256),
                    UNIQUE(publication_sha256)
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _decode(row: sqlite3.Row) -> RefinedGenerationPublicationReceipt:
        return RefinedGenerationPublicationReceipt(
            request_sha256=row["request_sha256"], scope_sha256=row["scope_sha256"],
            serving_receipt_sha256=row["serving_receipt_sha256"], grounded_output_sha256=row["grounded_output_sha256"],
            refinement_receipt_sha256=row["refinement_receipt_sha256"], envelope_sha256=row["envelope_sha256"],
            stack_sha256=row["stack_sha256"], fencing_token=int(row["fencing_token"]),
            authority_revision=int(row["authority_revision"]), published_at=float(row["published_at"]),
            publication_sha256=row["publication_sha256"],
        )

    def publish(
        self,
        envelope: RefinedGenerationEnvelope,
        *,
        owner_id: str,
        service_id: str,
        domain_id: str,
        request_sha256: str,
        now: float,
    ) -> RefinedGenerationPublicationReceipt:
        if not isinstance(envelope, RefinedGenerationEnvelope) or not envelope.publication_allowed:
            raise ValueError("only a fully supported refined generation envelope may be published")
        result = envelope.base_result
        grounded = result.grounded_output
        assert grounded is not None
        owner, service, domain = (_text(owner_id, "owner_id"), _text(service_id, "service_id"), _text(domain_id, "domain_id"))
        request = _sha(request_sha256, "request_sha256")
        timestamp = _time(now, "now")
        scope = _scope_sha256(owner, service, domain)
        if result.receipt.scope_sha256 != scope:
            raise ValueError("generation result belongs to a different publication scope")

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """SELECT * FROM refined_generation_publication_ledger
                   WHERE owner_id=? AND service_id=? AND domain_id=? AND request_sha256=?""",
                (owner, service, domain, request),
            ).fetchone()
            if existing is not None:
                decoded = self._decode(existing)
                if decoded.envelope_sha256 != envelope.envelope_sha256:
                    raise RuntimeError("request identity already has a different refined publication")
                return decoded

            authority = connection.execute(
                """SELECT stack_sha256,authority_revision,fencing_token FROM runtime_stack_authority
                   WHERE owner_id=? AND service_id=? AND domain_id=?""",
                (owner, service, domain),
            ).fetchone()
            if authority is None:
                raise RuntimeError("cannot publish before runtime-stack authority is established")
            current_stack = _sha(authority["stack_sha256"], "persisted stack_sha256")
            current_revision = _positive_int(int(authority["authority_revision"]), "persisted authority_revision")
            current_fence = _positive_int(int(authority["fencing_token"]), "persisted fencing_token")
            if current_stack != result.receipt.stack_sha256 or current_fence != result.receipt.fencing_token:
                raise RuntimeError("refined generation result is stale under current runtime authority")

            payload = {
                "schema": "rigorousrag-refined-generation-publication/v1",
                "request_sha256": request,
                "scope_sha256": scope,
                "serving_receipt_sha256": result.receipt.receipt_sha256,
                "grounded_output_sha256": grounded.grounded_output_sha256,
                "refinement_receipt_sha256": envelope.refinement_receipt.receipt_sha256,
                "envelope_sha256": envelope.envelope_sha256,
                "stack_sha256": current_stack,
                "fencing_token": current_fence,
                "authority_revision": current_revision,
                "published_at": timestamp,
            }
            receipt = RefinedGenerationPublicationReceipt(
                **{key: value for key, value in payload.items() if key != "schema"},
                publication_sha256=_digest(payload),
            )
            connection.execute(
                """INSERT INTO refined_generation_publication_ledger(
                    owner_id,service_id,domain_id,request_sha256,scope_sha256,
                    serving_receipt_sha256,grounded_output_sha256,refinement_receipt_sha256,
                    envelope_sha256,stack_sha256,fencing_token,authority_revision,published_at,
                    publication_sha256
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    owner, service, domain, request, scope, receipt.serving_receipt_sha256,
                    receipt.grounded_output_sha256, receipt.refinement_receipt_sha256,
                    receipt.envelope_sha256, receipt.stack_sha256, receipt.fencing_token,
                    receipt.authority_revision, receipt.published_at, receipt.publication_sha256,
                ),
            )
            return receipt


__all__ = [
    "RefinedGenerationEnvelope", "RefinedGenerationPublicationReceipt",
    "SQLiteRefinedGenerationPublicationLedger", "build_refined_generation_envelope",
    "evidence_universe_sha256",
]
