"""Cryptographic attestations for resolved human-review decisions.

Attestations bind the immutable decision-relevant fields of a resolved ReviewRecord while
keeping raw queries and arbitrary review metadata out of the signed manifest. The review
queue remains the authority for the decision; the signature proves which authority state
was reviewed and signed, not that the decision is substantively correct.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from tools.manifest_attestation import (
    ManifestAttestation,
    ManifestSigner,
    ManifestVerifier,
    attest_manifest,
    verify_attestation,
)
from tools.review_store import ReviewRecord


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_payload(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(dict(value))).hexdigest()


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or "\x00" in selected:
        raise ValueError(f"{label} is invalid")
    return selected


def review_manifest(record: ReviewRecord) -> Mapping[str, Any]:
    if not isinstance(record, ReviewRecord):
        raise TypeError("record must be ReviewRecord")
    if record.state != "resolved" or not record.reviewer_id or not record.resolution:
        raise ValueError("only resolved, reviewer-bound review records may be attested")
    metadata = dict(record.metadata or {})
    reasons = tuple(str(item) for item in record.reasons)
    return {
        "schema": "rigorousrag.review-decision/v1",
        "owner_id": _text(record.owner_id, "owner_id", 256),
        "request_id": _text(record.request_id, "request_id", 500),
        "reviewer_id": _text(record.reviewer_id, "reviewer_id", 500),
        "resolution": _text(record.resolution, "resolution", 500),
        "lease_token": int(record.lease_token),
        "query_sha256": record.query_sha256 or "",
        "reasons_sha256": hashlib.sha256(_canonical(reasons)).hexdigest(),
        "metadata_sha256": _sha_payload(metadata),
        "created_at": float(record.created_at),
        "updated_at": float(record.updated_at),
    }


@dataclass(frozen=True)
class ReviewDecisionAttestation:
    request_id: str
    owner_id: str
    review_manifest_sha256: str
    attestation: ManifestAttestation

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _text(self.request_id, "request_id", 500))
        object.__setattr__(self, "owner_id", _text(self.owner_id, "owner_id", 256))
        digest = _text(self.review_manifest_sha256, "review_manifest_sha256", 64).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("review_manifest_sha256 must be SHA-256")
        object.__setattr__(self, "review_manifest_sha256", digest)
        if not isinstance(self.attestation, ManifestAttestation):
            raise TypeError("attestation must be ManifestAttestation")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


def attest_review_record(record: ReviewRecord, signer: ManifestSigner) -> ReviewDecisionAttestation:
    manifest = review_manifest(record)
    digest = hashlib.sha256(_canonical(manifest)).hexdigest()
    attestation = attest_manifest(
        f"review:{record.owner_id}:{record.request_id}:{record.lease_token}",
        manifest,
        signer,
    )
    return ReviewDecisionAttestation(
        request_id=record.request_id,
        owner_id=record.owner_id,
        review_manifest_sha256=digest,
        attestation=attestation,
    )


def verify_review_attestation(
    signed: ReviewDecisionAttestation,
    record: ReviewRecord,
    verifier: ManifestVerifier,
) -> bool:
    if not isinstance(signed, ReviewDecisionAttestation) or not isinstance(record, ReviewRecord):
        return False
    if signed.owner_id != record.owner_id or signed.request_id != record.request_id:
        return False
    try:
        manifest = review_manifest(record)
    except (TypeError, ValueError):
        return False
    digest = hashlib.sha256(_canonical(manifest)).hexdigest()
    if digest != signed.review_manifest_sha256:
        return False
    return verify_attestation(signed.attestation, manifest, verifier)


__all__ = [
    "ReviewDecisionAttestation",
    "attest_review_record",
    "review_manifest",
    "verify_review_attestation",
]
