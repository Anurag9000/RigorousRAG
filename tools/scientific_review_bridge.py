"""Bridge fenced human-review records into scientific evidence review stamps.

The bridge prevents callers from manufacturing an ``EvidenceReviewStamp`` by supplying an
arbitrary reviewer name. Reviewer identity, resolution state and review timestamp come
from the durable resolved ``ReviewRecord``. Optional cryptographic attestation validity is
supplied by a separate verifier, never trusted from record metadata alone.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from tools.review_store import ReviewRecord
from tools.scientific_evidence_pipeline import EvidenceReviewStamp


def scientific_review_metadata(
    *,
    subject_kind: str,
    subject_id: str,
    subject_fingerprint: str,
    evidence_ids: Sequence[str],
    rationale_sha256: str,
    attestation_id: str = "",
    extra: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    payload: dict[str, Any] = dict(extra or {})
    reserved = {
        "scientific_review_subject_kind",
        "scientific_review_subject_id",
        "scientific_review_subject_fingerprint",
        "scientific_review_evidence_ids",
        "scientific_review_rationale_sha256",
        "scientific_review_attestation_id",
    }
    if reserved.intersection(payload):
        raise ValueError("extra metadata may not override scientific review bindings")
    payload.update(
        {
            "scientific_review_subject_kind": subject_kind,
            "scientific_review_subject_id": subject_id,
            "scientific_review_subject_fingerprint": subject_fingerprint,
            "scientific_review_evidence_ids": list(evidence_ids),
            "scientific_review_rationale_sha256": rationale_sha256,
            "scientific_review_attestation_id": attestation_id,
        }
    )
    return payload


def review_stamp_from_resolved_record(
    record: ReviewRecord,
    *,
    verified_attestation_ids: Sequence[str] = (),
) -> EvidenceReviewStamp:
    if not isinstance(record, ReviewRecord):
        raise TypeError("record must be ReviewRecord")
    if record.state != "resolved" or record.reviewer_id is None or not record.resolution:
        raise ValueError("scientific review stamp requires a resolved fenced review record")
    status = record.resolution.strip().lower()
    if status not in {"accepted", "rejected", "needs_revision"}:
        raise ValueError("scientific review resolution is unsupported")
    metadata = record.metadata
    if not isinstance(metadata, Mapping):
        raise ValueError("scientific review metadata is missing")
    evidence_raw = metadata.get("scientific_review_evidence_ids", ())
    if not isinstance(evidence_raw, (list, tuple)):
        raise ValueError("scientific review evidence_ids are invalid")
    attestation_id = str(metadata.get("scientific_review_attestation_id") or "").strip()
    verified = set(str(item).strip() for item in verified_attestation_ids if str(item).strip())
    return EvidenceReviewStamp(
        subject_kind=str(metadata.get("scientific_review_subject_kind") or ""),
        subject_id=str(metadata.get("scientific_review_subject_id") or ""),
        subject_fingerprint=str(metadata.get("scientific_review_subject_fingerprint") or ""),
        status=status,
        reviewer_id=record.reviewer_id,
        evidence_ids=tuple(str(item) for item in evidence_raw),
        rationale_sha256=str(metadata.get("scientific_review_rationale_sha256") or ""),
        reviewed_at=record.updated_at,
        attestation_id=attestation_id,
        attestation_verified=bool(attestation_id and attestation_id in verified),
    )


__all__ = ["review_stamp_from_resolved_record", "scientific_review_metadata"]
