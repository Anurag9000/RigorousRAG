"""Verified, privacy-bounded identity for a signed human-review decision.

A persisted attestation is not automatically trusted. ``verify_review_decision`` requires
that the current review is still the exact resolved state captured by the attestation and
that the signature verifies under the supplied verifier. Downstream artifacts should bind
this DTO's fingerprints rather than raw review text or an unverified attestation row.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from tools.manifest_attestation import ManifestVerifier, canonical_manifest_bytes
from tools.review_attestation import review_manifest, verify_review_attestation
from tools.review_attestation_store import StoredReviewAttestation
from tools.review_store import ReviewRecord


def _text(value: Any, label: str, maximum: int = 1000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or "\x00" in selected:
        raise ValueError(f"{label} is invalid")
    return selected


def _sha(value: str, label: str) -> str:
    selected = _text(value, label, 64).lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


@dataclass(frozen=True)
class VerifiedReviewDecision:
    owner_id: str
    request_id: str
    reviewer_id: str
    resolution: str
    lease_token: int
    review_manifest_sha256: str
    attestation_id: str
    key_id: str
    algorithm: str
    signed_at: float
    review_updated_at: float
    verification_fingerprint: str

    def __post_init__(self) -> None:
        for field, maximum in (
            ("owner_id", 256),
            ("request_id", 500),
            ("reviewer_id", 500),
            ("resolution", 500),
            ("key_id", 500),
            ("algorithm", 100),
        ):
            object.__setattr__(self, field, _text(getattr(self, field), field, maximum))
        if isinstance(self.lease_token, bool) or not isinstance(self.lease_token, int) or self.lease_token < 1:
            raise ValueError("lease_token must be positive")
        for field in ("review_manifest_sha256", "attestation_id", "verification_fingerprint"):
            object.__setattr__(self, field, _sha(getattr(self, field), field))
        for field in ("signed_at", "review_updated_at"):
            value = getattr(self, field)
            if isinstance(value, bool):
                raise ValueError(f"{field} is invalid")
            try:
                parsed = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{field} is invalid") from exc
            if parsed < 0 or parsed != parsed or parsed in (float("inf"), float("-inf")):
                raise ValueError(f"{field} is invalid")
            object.__setattr__(self, field, parsed)

    @property
    def provenance_fingerprint(self) -> str:
        payload = asdict(self)
        payload.pop("verification_fingerprint", None)
        return hashlib.sha256(_canonical(payload)).hexdigest()


def verify_review_decision(
    record: ReviewRecord,
    stored: StoredReviewAttestation,
    verifier: ManifestVerifier,
) -> VerifiedReviewDecision:
    if not isinstance(record, ReviewRecord):
        raise TypeError("record must be ReviewRecord")
    if not isinstance(stored, StoredReviewAttestation):
        raise TypeError("stored must be StoredReviewAttestation")
    if record.state != "resolved" or not record.reviewer_id or not record.resolution:
        raise ValueError("review is not a resolved reviewer-bound decision")
    if stored.owner_id != record.owner_id or stored.request_id != record.request_id:
        raise ValueError("review and attestation identities do not match")
    if stored.lease_token != record.lease_token:
        raise ValueError("attestation was produced for a different review lease epoch")
    current_manifest = review_manifest(record)
    current_digest = hashlib.sha256(canonical_manifest_bytes(current_manifest)).hexdigest()
    captured_digest = hashlib.sha256(canonical_manifest_bytes(stored.captured_manifest)).hexdigest()
    if current_digest != captured_digest or current_digest != stored.signed.review_manifest_sha256:
        raise ValueError("attestation does not capture the current resolved review state")
    if not verify_review_attestation(stored.signed, record, verifier):
        raise ValueError("review attestation signature does not verify")
    verification_payload = {
        "schema": "rigorousrag.verified-review/v1",
        "owner_id": record.owner_id,
        "request_id": record.request_id,
        "reviewer_id": record.reviewer_id,
        "resolution": record.resolution,
        "lease_token": record.lease_token,
        "review_manifest_sha256": current_digest,
        "attestation_id": stored.attestation_id,
        "key_id": stored.signed.attestation.key_id,
        "algorithm": stored.signed.attestation.algorithm,
        "signed_at": stored.signed.attestation.signed_at,
        "review_updated_at": record.updated_at,
    }
    verification_fingerprint = hashlib.sha256(_canonical(verification_payload)).hexdigest()
    return VerifiedReviewDecision(
        owner_id=record.owner_id,
        request_id=record.request_id,
        reviewer_id=record.reviewer_id,
        resolution=record.resolution,
        lease_token=record.lease_token,
        review_manifest_sha256=current_digest,
        attestation_id=stored.attestation_id,
        key_id=stored.signed.attestation.key_id,
        algorithm=stored.signed.attestation.algorithm,
        signed_at=stored.signed.attestation.signed_at,
        review_updated_at=record.updated_at,
        verification_fingerprint=verification_fingerprint,
    )


__all__ = ["VerifiedReviewDecision", "verify_review_decision"]
