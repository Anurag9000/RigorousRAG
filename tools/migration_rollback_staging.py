"""Isolated in-memory staging verification for reconstructed rollback snapshots."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

from tools.generation_store import GenerationRecord
from tools.migration_cutover_preflight import (
    CutoverPreflight,
    _sparse_identity,
    _vector_identity,
)
from tools.migration_rollback_reconstruction import ReconstructedRollback
from tools.migration_types import digest, exact_integer, identifier, timestamp
from tools.sparse_types import SparseDocumentSnapshot, SparseFieldSnapshot
from tools.vector_generation import VectorGenerationSnapshot

_MAX_STAGED = 10_000


def _sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _copy_vector(value: VectorGenerationSnapshot) -> VectorGenerationSnapshot:
    return VectorGenerationSnapshot(
        owner_id=value.owner_id,
        doc_id=value.doc_id,
        ids=tuple(value.ids),
        documents=tuple(value.documents),
        metadatas=tuple(dict(item) for item in value.metadatas),
    )


def _copy_sparse(value: SparseDocumentSnapshot) -> SparseDocumentSnapshot:
    return SparseDocumentSnapshot(
        owner_id=value.owner_id,
        doc_id=value.doc_id,
        generation=value.generation,
        profile_fingerprint=value.profile_fingerprint,
        metadata=dict(value.metadata),
        fields=tuple(
            SparseFieldSnapshot(
                field_id=field.field_id,
                field_type=field.field_type,
                text=field.text,
                position=field.position,
                token_count=field.token_count,
                page_number=field.page_number,
                section=field.section,
                metadata=dict(field.metadata),
            )
            for field in value.fields
        ),
        schema_version=value.schema_version,
    )


def _copy_generation(value: GenerationRecord) -> GenerationRecord:
    return GenerationRecord(
        owner_id=value.owner_id,
        doc_id=value.doc_id,
        sequence=value.sequence,
        state=value.state,
        content_sha256=value.content_sha256,
        profile_fingerprint=value.profile_fingerprint,
        vector_rows=value.vector_rows,
        sparse_generation=value.sparse_generation,
        committed_at=value.committed_at,
        metadata=dict(value.metadata),
    )


def _copy_rollback(value: ReconstructedRollback) -> ReconstructedRollback:
    if not isinstance(value, ReconstructedRollback):
        raise ValueError("rollback must be ReconstructedRollback.")
    return ReconstructedRollback(
        vector=_copy_vector(value.vector),
        sparse=_copy_sparse(value.sparse),
        generation=_copy_generation(value.generation),
    )


def staging_identity(preflight: CutoverPreflight) -> str:
    if not isinstance(preflight, CutoverPreflight):
        raise ValueError("preflight must be CutoverPreflight.")
    return _sha256(
        {
            "contract": "rigorousrag-isolated-rollback-staging-v1",
            "task_id": preflight.task_id,
            "preflight_digest": preflight.preflight_digest,
            "rollback_identity_digest": preflight.rollback_identity_digest,
        }
    )


class InMemoryRollbackStagingStore:
    """Process-local non-authoritative snapshot store used only for verification."""

    def __init__(self, *, maximum_entries: int = 100) -> None:
        self.maximum_entries = exact_integer(
            maximum_entries,
            "maximum_entries",
            1,
            _MAX_STAGED,
        )
        self._lock = threading.RLock()
        self._entries: dict[str, ReconstructedRollback] = {}

    def stage(
        self,
        staging_id: str,
        rollback: ReconstructedRollback,
    ) -> ReconstructedRollback:
        key = digest(staging_id, "staging_id")
        copied = _copy_rollback(rollback)
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                if existing != copied:
                    raise RuntimeError("staging identity already contains different snapshots.")
                return _copy_rollback(existing)
            if len(self._entries) >= self.maximum_entries:
                raise RuntimeError("isolated staging store reached its entry limit.")
            self._entries[key] = copied
            return _copy_rollback(copied)

    def snapshot(self, staging_id: str) -> ReconstructedRollback:
        key = digest(staging_id, "staging_id")
        with self._lock:
            try:
                value = self._entries[key]
            except KeyError as exc:
                raise KeyError(key) from exc
            return _copy_rollback(value)

    def remove(self, staging_id: str) -> bool:
        key = digest(staging_id, "staging_id")
        with self._lock:
            return self._entries.pop(key, None) is not None

    def count(self) -> int:
        with self._lock:
            return len(self._entries)


@dataclass(frozen=True)
class StagingVerification:
    task_id: str
    preflight_digest: str
    rollback_identity_digest: str
    staging_id: str
    vector_snapshot_digest: str
    sparse_snapshot_digest: str
    source_sequence: int
    source_profile_fingerprint: str
    source_content_sha256: str
    vector_rows: int
    sparse_generation: int
    sparse_fields: int
    verified_at: float
    contract_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", identifier(self.task_id, "task_id", 64))
        for name in (
            "preflight_digest",
            "rollback_identity_digest",
            "staging_id",
            "vector_snapshot_digest",
            "sparse_snapshot_digest",
            "source_profile_fingerprint",
            "source_content_sha256",
        ):
            object.__setattr__(self, name, digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "source_sequence",
            exact_integer(self.source_sequence, "source_sequence", 1, 2**63 - 1),
        )
        object.__setattr__(
            self,
            "vector_rows",
            exact_integer(self.vector_rows, "vector_rows", 1, 100_000_000),
        )
        object.__setattr__(
            self,
            "sparse_generation",
            exact_integer(
                self.sparse_generation,
                "sparse_generation",
                1,
                2**63 - 1,
            ),
        )
        object.__setattr__(
            self,
            "sparse_fields",
            exact_integer(self.sparse_fields, "sparse_fields", 1, 100_000_000),
        )
        object.__setattr__(self, "verified_at", timestamp(self.verified_at, "verified_at"))
        if self.contract_version != 1:
            raise ValueError("staging verification contract is unsupported.")

    @property
    def verification_digest(self) -> str:
        stable = asdict(self)
        stable.pop("verified_at", None)
        return _sha256(stable)


def verify_in_isolated_staging(
    preflight: CutoverPreflight,
    rollback: ReconstructedRollback,
    *,
    store: InMemoryRollbackStagingStore | None = None,
    now: float | None = None,
) -> StagingVerification:
    if not isinstance(preflight, CutoverPreflight):
        raise ValueError("preflight must be CutoverPreflight.")
    if not isinstance(rollback, ReconstructedRollback):
        raise ValueError("rollback must be ReconstructedRollback.")
    staging = store or InMemoryRollbackStagingStore(maximum_entries=1)
    identity = staging_identity(preflight)
    staging.stage(identity, rollback)
    restored = staging.snapshot(identity)

    vector_digest, vector_count = _vector_identity(
        restored.vector,
        preflight.owner_id,
        preflight.doc_id,
    )
    sparse_digest, sparse_count = _sparse_identity(
        restored.sparse,
        preflight.owner_id,
        preflight.doc_id,
        preflight.source_profile_fingerprint,
        preflight.source_sparse_generation,
    )
    generation = restored.generation
    if (
        vector_digest != preflight.vector_snapshot_digest
        or sparse_digest != preflight.sparse_snapshot_digest
        or vector_count != preflight.source_vector_rows
        or sparse_count != preflight.source_sparse_fields
        or generation.sequence != preflight.source_sequence
        or generation.profile_fingerprint != preflight.source_profile_fingerprint
        or generation.content_sha256 != preflight.source_content_sha256
        or generation.vector_rows != vector_count
        or generation.sparse_generation != restored.sparse.generation
    ):
        raise RuntimeError("isolated staging snapshot does not match the preflight.")

    return StagingVerification(
        task_id=preflight.task_id,
        preflight_digest=preflight.preflight_digest,
        rollback_identity_digest=preflight.rollback_identity_digest,
        staging_id=identity,
        vector_snapshot_digest=vector_digest,
        sparse_snapshot_digest=sparse_digest,
        source_sequence=generation.sequence,
        source_profile_fingerprint=generation.profile_fingerprint,
        source_content_sha256=generation.content_sha256,
        vector_rows=vector_count,
        sparse_generation=restored.sparse.generation,
        sparse_fields=sparse_count,
        verified_at=time.time() if now is None else timestamp(now, "verified_at"),
    )


__all__ = [
    "InMemoryRollbackStagingStore",
    "StagingVerification",
    "staging_identity",
    "verify_in_isolated_staging",
]
