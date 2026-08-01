"""Validated immutable prerequisites for a future migration cutover operation."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

from tools.migration_cutover_preflight import CutoverPreflight
from tools.migration_promotion import PromotionReport
from tools.migration_rollback_artifact import EncryptedRollbackManifest
from tools.migration_rollback_staging import StagingVerification
from tools.migration_types import digest, exact_integer, identifier, timestamp
from tools.security import normalize_owner_id

_STATES = {"planned", "running", "ready", "failed", "cancelled"}


def _sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CutoverPreparation:
    task_id: str
    owner_id: str
    doc_id: str
    source_sequence: int
    source_profile_fingerprint: str
    target_profile_fingerprint: str
    source_content_sha256: str
    validation_digest: str
    promotion_report_digest: str
    benchmark_fingerprint: str
    preflight_digest: str
    rollback_identity_digest: str
    rollback_artifact_digest: str
    rollback_key_id: str
    staging_verification_digest: str
    target_artifact_digest: str
    vector_snapshot_digest: str
    sparse_snapshot_digest: str
    source_vector_rows: int
    source_sparse_generation: int
    source_sparse_fields: int
    target_vector_rows: int
    target_sparse_rows: int
    prepared_at: float
    contract_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", identifier(self.task_id, "task_id", 64))
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "doc_id", identifier(self.doc_id, "doc_id", 200))
        object.__setattr__(
            self,
            "source_sequence",
            exact_integer(self.source_sequence, "source_sequence", 1, 2**63 - 1),
        )
        for name in (
            "source_profile_fingerprint",
            "target_profile_fingerprint",
            "source_content_sha256",
            "validation_digest",
            "promotion_report_digest",
            "benchmark_fingerprint",
            "preflight_digest",
            "rollback_identity_digest",
            "rollback_artifact_digest",
            "staging_verification_digest",
            "target_artifact_digest",
            "vector_snapshot_digest",
            "sparse_snapshot_digest",
        ):
            object.__setattr__(self, name, digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "rollback_key_id",
            identifier(self.rollback_key_id, "rollback_key_id", 128),
        )
        for name in (
            "source_vector_rows",
            "source_sparse_fields",
            "target_vector_rows",
            "target_sparse_rows",
        ):
            object.__setattr__(
                self,
                name,
                exact_integer(getattr(self, name), name, 1, 100_000_000),
            )
        object.__setattr__(
            self,
            "source_sparse_generation",
            exact_integer(
                self.source_sparse_generation,
                "source_sparse_generation",
                1,
                2**63 - 1,
            ),
        )
        object.__setattr__(self, "prepared_at", timestamp(self.prepared_at, "prepared_at"))
        if self.contract_version != 1:
            raise ValueError("cutover preparation contract is unsupported.")

    @property
    def operation_id(self) -> str:
        stable = asdict(self)
        stable.pop("prepared_at", None)
        return _sha256(stable)


@dataclass(frozen=True)
class CutoverOperation:
    operation_id: str
    preparation: CutoverPreparation
    state: str
    attempt: int
    created_at: float
    updated_at: float
    lease_owner: str | None = None
    lease_expires_at: float | None = None
    failure_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation_id",
            digest(self.operation_id, "operation_id"),
        )
        if not isinstance(self.preparation, CutoverPreparation):
            raise ValueError("preparation must be CutoverPreparation.")
        if self.operation_id != self.preparation.operation_id:
            raise ValueError("cutover operation ID does not match preparation.")
        if self.state not in _STATES:
            raise ValueError("cutover operation state is invalid.")
        object.__setattr__(
            self,
            "attempt",
            exact_integer(self.attempt, "attempt", 0, 1_000_000),
        )
        object.__setattr__(self, "created_at", timestamp(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", timestamp(self.updated_at, "updated_at"))
        if self.lease_owner is not None:
            object.__setattr__(
                self,
                "lease_owner",
                identifier(self.lease_owner, "lease_owner", 128),
            )
        if self.lease_expires_at is not None:
            object.__setattr__(
                self,
                "lease_expires_at",
                timestamp(self.lease_expires_at, "lease_expires_at"),
            )
        if self.failure_type is not None:
            object.__setattr__(
                self,
                "failure_type",
                identifier(self.failure_type, "failure_type", 200),
            )
        if self.state == "running" and (
            self.lease_owner is None or self.lease_expires_at is None
        ):
            raise ValueError("running cutover preparation requires a lease.")
        if self.state != "running" and (
            self.lease_owner is not None or self.lease_expires_at is not None
        ):
            raise ValueError("inactive cutover preparation may not retain a lease.")


def build_cutover_preparation(
    *,
    task: Any,
    preflight: CutoverPreflight,
    promotion: PromotionReport,
    rollback_manifest: EncryptedRollbackManifest,
    staging: StagingVerification,
    generation: Any,
    now: float | None = None,
) -> CutoverPreparation:
    if getattr(task, "state", None) != "validated":
        raise ValueError("cutover preparation requires a validated migration task.")
    if not isinstance(preflight, CutoverPreflight):
        raise ValueError("preflight must be CutoverPreflight.")
    if not isinstance(promotion, PromotionReport):
        raise ValueError("promotion must be PromotionReport.")
    if promotion.decision != "eligible" or promotion.policy_id != "paired-promotion-v1":
        raise RuntimeError("cutover preparation requires eligible paired promotion.")
    if not isinstance(rollback_manifest, EncryptedRollbackManifest):
        raise ValueError("rollback_manifest must be EncryptedRollbackManifest.")
    if not isinstance(staging, StagingVerification):
        raise ValueError("staging must be StagingVerification.")

    task_id = identifier(getattr(task, "task_id", None), "task_id", 64)
    owner = normalize_owner_id(getattr(task, "owner_id", None))
    doc_id = identifier(getattr(task, "doc_id", None), "doc_id", 200)
    source_sequence = exact_integer(
        getattr(task, "source_sequence", None),
        "source_sequence",
        1,
        2**63 - 1,
    )
    source_profile = digest(
        getattr(task, "source_profile_fingerprint", None),
        "source_profile_fingerprint",
    )
    target_profile = digest(
        getattr(task, "target_profile_fingerprint", None),
        "target_profile_fingerprint",
    )
    validation = digest(
        getattr(task, "validation_digest", None),
        "validation_digest",
    )

    identities = (
        (preflight.task_id, task_id),
        (preflight.owner_id, owner),
        (preflight.doc_id, doc_id),
        (preflight.source_sequence, source_sequence),
        (preflight.source_profile_fingerprint, source_profile),
        (preflight.target_profile_fingerprint, target_profile),
        (preflight.validation_digest, validation),
        (promotion.task_id, task_id),
        (promotion.owner_id, owner),
        (promotion.doc_id, doc_id),
        (promotion.source_sequence, source_sequence),
        (promotion.source_profile_fingerprint, source_profile),
        (promotion.target_profile_fingerprint, target_profile),
        (promotion.validation_digest, validation),
        (promotion.report_digest, preflight.promotion_report_digest),
        (rollback_manifest.task_id, task_id),
        (rollback_manifest.owner_id, owner),
        (rollback_manifest.doc_id, doc_id),
        (rollback_manifest.preflight_digest, preflight.preflight_digest),
        (
            rollback_manifest.rollback_identity_digest,
            preflight.rollback_identity_digest,
        ),
        (rollback_manifest.source_sequence, source_sequence),
        (rollback_manifest.source_profile_fingerprint, source_profile),
        (
            rollback_manifest.source_content_sha256,
            preflight.source_content_sha256,
        ),
        (
            rollback_manifest.vector_snapshot_digest,
            preflight.vector_snapshot_digest,
        ),
        (
            rollback_manifest.sparse_snapshot_digest,
            preflight.sparse_snapshot_digest,
        ),
        (staging.task_id, task_id),
        (staging.preflight_digest, preflight.preflight_digest),
        (
            staging.rollback_identity_digest,
            preflight.rollback_identity_digest,
        ),
        (staging.source_sequence, source_sequence),
        (staging.source_profile_fingerprint, source_profile),
        (staging.source_content_sha256, preflight.source_content_sha256),
        (staging.vector_snapshot_digest, preflight.vector_snapshot_digest),
        (staging.sparse_snapshot_digest, preflight.sparse_snapshot_digest),
        (staging.vector_rows, preflight.source_vector_rows),
        (staging.sparse_generation, preflight.source_sparse_generation),
        (staging.sparse_fields, preflight.source_sparse_fields),
    )
    if any(actual != expected for actual, expected in identities):
        raise RuntimeError("cutover preparation prerequisites are inconsistent.")

    if generation is None or getattr(generation, "state", None) not in {
        "active",
        "restored",
    }:
        raise RuntimeError("current source generation is unavailable.")
    if (
        getattr(generation, "sequence", None) != source_sequence
        or getattr(generation, "profile_fingerprint", None) != source_profile
        or getattr(generation, "content_sha256", None)
        != preflight.source_content_sha256
        or getattr(generation, "vector_rows", None) != preflight.source_vector_rows
        or getattr(generation, "sparse_generation", None)
        != preflight.source_sparse_generation
    ):
        raise RuntimeError("current source generation changed after preflight.")

    return CutoverPreparation(
        task_id=task_id,
        owner_id=owner,
        doc_id=doc_id,
        source_sequence=source_sequence,
        source_profile_fingerprint=source_profile,
        target_profile_fingerprint=target_profile,
        source_content_sha256=preflight.source_content_sha256,
        validation_digest=validation,
        promotion_report_digest=promotion.report_digest,
        benchmark_fingerprint=promotion.benchmark_fingerprint,
        preflight_digest=preflight.preflight_digest,
        rollback_identity_digest=preflight.rollback_identity_digest,
        rollback_artifact_digest=rollback_manifest.artifact_digest,
        rollback_key_id=rollback_manifest.key_id,
        staging_verification_digest=staging.verification_digest,
        target_artifact_digest=preflight.target_artifact_digest,
        vector_snapshot_digest=preflight.vector_snapshot_digest,
        sparse_snapshot_digest=preflight.sparse_snapshot_digest,
        source_vector_rows=preflight.source_vector_rows,
        source_sparse_generation=preflight.source_sparse_generation,
        source_sparse_fields=preflight.source_sparse_fields,
        target_vector_rows=preflight.target_vector_rows,
        target_sparse_rows=preflight.target_sparse_rows,
        prepared_at=time.time() if now is None else timestamp(now, "prepared_at"),
    )


__all__ = [
    "CutoverOperation",
    "CutoverPreparation",
    "build_cutover_preparation",
]
