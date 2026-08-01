"""Typed in-memory reconstruction of validated encrypted rollback payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from tools.generation_store import GenerationRecord
from tools.migration_cutover_preflight import (
    CutoverPreflight,
    _sparse_identity,
    _vector_identity,
)
from tools.migration_rollback_artifact import validate_rollback_payload
from tools.sparse_types import SparseDocumentSnapshot, SparseFieldSnapshot
from tools.vector_generation import VectorGenerationSnapshot


@dataclass(frozen=True)
class ReconstructedRollback:
    vector: VectorGenerationSnapshot
    sparse: SparseDocumentSnapshot
    generation: GenerationRecord

    def __post_init__(self) -> None:
        if not isinstance(self.vector, VectorGenerationSnapshot):
            raise ValueError("vector must be VectorGenerationSnapshot.")
        if not isinstance(self.sparse, SparseDocumentSnapshot):
            raise ValueError("sparse must be SparseDocumentSnapshot.")
        if not isinstance(self.generation, GenerationRecord):
            raise ValueError("generation must be GenerationRecord.")
        if (
            self.vector.owner_id != self.sparse.owner_id
            or self.vector.owner_id != self.generation.owner_id
            or self.vector.doc_id != self.sparse.doc_id
            or self.vector.doc_id != self.generation.doc_id
        ):
            raise ValueError("reconstructed rollback scope is inconsistent.")
        if len(self.vector.ids) != self.generation.vector_rows:
            raise ValueError("reconstructed vector row count is inconsistent.")
        if self.sparse.generation != self.generation.sparse_generation:
            raise ValueError("reconstructed sparse generation is inconsistent.")
        if self.sparse.profile_fingerprint != self.generation.profile_fingerprint:
            raise ValueError("reconstructed sparse profile is inconsistent.")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be an object.")
    return value


def reconstruct_rollback_snapshots(
    preflight: CutoverPreflight,
    payload: Mapping[str, Any],
) -> ReconstructedRollback:
    """Rebuild public immutable snapshot types without mutating any backend."""

    if not isinstance(preflight, CutoverPreflight):
        raise ValueError("preflight must be CutoverPreflight.")
    normalized = validate_rollback_payload(preflight, payload)

    vector_rows = normalized["vector_rows"]
    vector = VectorGenerationSnapshot(
        owner_id=preflight.owner_id,
        doc_id=preflight.doc_id,
        ids=tuple(row["id"] for row in vector_rows),
        documents=tuple(row["document"] for row in vector_rows),
        metadatas=tuple(dict(row["metadata"]) for row in vector_rows),
    )

    raw_sparse = _mapping(normalized["sparse_snapshot"], "sparse_snapshot")
    raw_fields = raw_sparse["fields"]
    fields = tuple(
        SparseFieldSnapshot(
            field_id=field["field_id"],
            field_type=field["field_type"],
            text=field["text"],
            position=field["position"],
            token_count=field["token_count"],
            page_number=field["page_number"],
            section=field["section"],
            metadata=dict(field["metadata"]),
        )
        for field in raw_fields
    )
    sparse = SparseDocumentSnapshot(
        owner_id=raw_sparse["owner_id"],
        doc_id=raw_sparse["doc_id"],
        generation=raw_sparse["generation"],
        profile_fingerprint=raw_sparse["profile_fingerprint"],
        metadata=dict(raw_sparse["metadata"]),
        fields=fields,
        schema_version=raw_sparse["schema_version"],
    )

    raw_generation = _mapping(normalized["generation"], "generation")
    generation = GenerationRecord(
        owner_id=raw_generation["owner_id"],
        doc_id=raw_generation["doc_id"],
        sequence=raw_generation["sequence"],
        state=raw_generation["state"],
        content_sha256=raw_generation["content_sha256"],
        profile_fingerprint=raw_generation["profile_fingerprint"],
        vector_rows=raw_generation["vector_rows"],
        sparse_generation=raw_generation["sparse_generation"],
        committed_at=raw_generation["committed_at"],
        metadata=dict(raw_generation["metadata"]),
    )

    result = ReconstructedRollback(vector, sparse, generation)
    vector_digest, vector_count = _vector_identity(
        result.vector,
        preflight.owner_id,
        preflight.doc_id,
    )
    sparse_digest, sparse_count = _sparse_identity(
        result.sparse,
        preflight.owner_id,
        preflight.doc_id,
        preflight.source_profile_fingerprint,
        preflight.source_sparse_generation,
    )
    if (
        vector_digest != preflight.vector_snapshot_digest
        or sparse_digest != preflight.sparse_snapshot_digest
        or vector_count != preflight.source_vector_rows
        or sparse_count != preflight.source_sparse_fields
    ):
        raise RuntimeError("typed rollback reconstruction does not match preflight.")
    if (
        result.generation.sequence != preflight.source_sequence
        or result.generation.content_sha256 != preflight.source_content_sha256
        or result.generation.profile_fingerprint
        != preflight.source_profile_fingerprint
    ):
        raise RuntimeError("typed rollback generation does not match preflight.")
    return result


__all__ = ["ReconstructedRollback", "reconstruct_rollback_snapshots"]
