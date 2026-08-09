"""Concrete single-host cutover adapter over RigorousRAG authoritative stores.

This adapter intentionally supports only target vector dimensionality equal to the
currently authoritative collection dimensionality. Dimension-changing migrations need
blue/green physical collections and a collection-pointer cutover; they fail before any
visibility mutation here.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.generation_store import GenerationRecord, GenerationStore
from tools.index_coordinator import (
    CrossStoreSnapshot,
    DocumentGenerationManifest,
    IndexCoordinator,
    _document_lock,
)
from tools.migration_cutover_control import CutoverOperation
from tools.migration_cutover_preflight import _sparse_identity, _vector_identity
from tools.migration_cutover_saga import BackendStateIdentity, TargetPublication
from tools.migration_shadow_store import MigrationShadowStore, ShadowArtifactManifest
from tools.sparse_index import SparseField
from tools.vector_generation import delete_vector_generation

_MAX_ROWS = 100_000
_MAX_DIMENSIONS = 1_000_000
_MAX_FILE_BYTES = 512 * 1024 * 1024
_BATCH_SIZE = 128
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _redirecting(value: os.stat_result) -> bool:
    return stat.S_ISLNK(value.st_mode) or bool(
        int(getattr(value, "st_file_attributes", 0)) & _REPARSE
    )


def _read_regular(path: Path) -> bytes:
    try:
        info = path.lstat()
    except OSError as exc:
        raise RuntimeError("shadow artifact member is unavailable.") from exc
    if _redirecting(info) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError("shadow artifact member is not a regular file.")
    if not 0 < info.st_size <= _MAX_FILE_BYTES:
        raise RuntimeError("shadow artifact member exceeds its size limit.")
    try:
        with path.open("rb") as handle:
            payload = handle.read(_MAX_FILE_BYTES + 1)
    except OSError as exc:
        raise RuntimeError("shadow artifact member could not be read.") from exc
    if len(payload) != info.st_size or len(payload) > _MAX_FILE_BYTES:
        raise RuntimeError("shadow artifact member changed during read.")
    return payload


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _strict_json(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise RuntimeError(f"{label} is invalid JSON.") from exc


def _finite_vector(value: Any) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise RuntimeError("shadow vector embedding is invalid.")
    try:
        raw = list(itertools.islice(iter(value), _MAX_DIMENSIONS + 1))
    except Exception as exc:
        raise RuntimeError("shadow vector embedding is invalid.") from exc
    if not raw or len(raw) > _MAX_DIMENSIONS:
        raise RuntimeError("shadow vector embedding dimension is invalid.")
    result: list[float] = []
    for item in raw:
        if isinstance(item, bool):
            raise RuntimeError("shadow vector embedding is non-finite.")
        try:
            numeric = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("shadow vector embedding is non-finite.") from exc
        if not math.isfinite(numeric):
            raise RuntimeError("shadow vector embedding is non-finite.")
        result.append(numeric)
    return tuple(result)


def _scope_filter(owner_id: str, doc_id: str) -> dict[str, Any]:
    return {
        "$and": [
            {"owner_id": {"$eq": owner_id}},
            {"doc_id": {"$eq": doc_id}},
        ]
    }


@dataclass(frozen=True)
class _VectorRow:
    row_id: str
    text: str
    metadata: Mapping[str, Any]
    embedding: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.row_id, str)
            or not self.row_id.strip()
            or len(self.row_id) > 500
            or any(ord(character) < 32 or ord(character) == 127 for character in self.row_id)
        ):
            raise ValueError("vector row_id is invalid.")
        if not isinstance(self.text, str) or "\x00" in self.text:
            raise ValueError("vector row text is invalid.")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("vector row metadata must be a mapping.")
        object.__setattr__(self, "embedding", _finite_vector(self.embedding))


@dataclass(frozen=True)
class _LoadedTarget:
    manifest: ShadowArtifactManifest
    vectors: tuple[_VectorRow, ...]
    sparse_fields: tuple[SparseField, ...]


@dataclass(frozen=True)
class _VectorRollback:
    ids: tuple[str, ...]
    documents: tuple[str, ...]
    metadatas: tuple[Mapping[str, Any], ...]
    embeddings: tuple[tuple[float, ...], ...]

    @property
    def dimension(self) -> int:
        return 0 if not self.embeddings else len(self.embeddings[0])


class LocalCutoverBackendAdapter:
    """Concrete cutover adapter for one local authoritative vector+sparse store pair."""

    def __init__(
        self,
        *,
        index: IndexCoordinator,
        generations: GenerationStore,
        shadow: MigrationShadowStore,
    ) -> None:
        if not isinstance(index, IndexCoordinator):
            raise ValueError("index must be IndexCoordinator.")
        if not isinstance(generations, GenerationStore):
            raise ValueError("generations must be GenerationStore.")
        if not isinstance(shadow, MigrationShadowStore):
            raise ValueError("shadow must be MigrationShadowStore.")
        collection = getattr(index.rag, "collection", None)
        if not all(
            callable(getattr(collection, name, None))
            for name in ("get", "delete", "upsert")
        ):
            raise ValueError("vector collection must expose get/delete/upsert.")
        self.index = index
        self.generations = generations
        self.shadow = shadow
        self._operation_id: str | None = None
        self._source_generation: GenerationRecord | None = None
        self._source_stores: CrossStoreSnapshot | None = None
        self._source_vectors: _VectorRollback | None = None
        self._target: _LoadedTarget | None = None

    def _bind(self, operation: CutoverOperation) -> None:
        if not isinstance(operation, CutoverOperation):
            raise ValueError("operation must be CutoverOperation.")
        if self._operation_id is None:
            self._operation_id = operation.operation_id
        elif self._operation_id != operation.operation_id:
            raise RuntimeError("cutover adapter instance is already bound to another operation.")

    def exclusive_lock(self, operation: CutoverOperation) -> Any:
        self._bind(operation)
        preparation = operation.preparation
        return _document_lock(preparation.owner_id, preparation.doc_id)

    def _collection_get(
        self,
        operation: CutoverOperation,
        *,
        include_embeddings: bool,
    ) -> Mapping[str, Any]:
        preparation = operation.preparation
        include = ["documents", "metadatas"]
        if include_embeddings:
            include.append("embeddings")
        try:
            result = self.index.rag.collection.get(
                where=_scope_filter(preparation.owner_id, preparation.doc_id),
                include=include,
                limit=_MAX_ROWS + 1,
            )
        except Exception as exc:
            raise RuntimeError("vector collection inspection failed.") from exc
        if not isinstance(result, Mapping):
            raise RuntimeError("vector collection inspection returned invalid data.")
        return result

    def _capture_source_vectors(
        self,
        operation: CutoverOperation,
        snapshot: CrossStoreSnapshot,
    ) -> _VectorRollback:
        result = self._collection_get(operation, include_embeddings=True)
        ids = result.get("ids")
        documents = result.get("documents")
        metadatas = result.get("metadatas")
        embeddings = result.get("embeddings")
        if not all(isinstance(value, list) for value in (ids, documents, metadatas)):
            raise RuntimeError("vector rollback arrays are invalid.")
        try:
            embedding_rows = list(embeddings)
        except Exception as exc:
            raise RuntimeError("vector rollback embeddings are unavailable.") from exc
        if not len(ids) == len(documents) == len(metadatas) == len(embedding_rows):
            raise RuntimeError("vector rollback arrays are inconsistent.")
        if len(ids) != snapshot.vector.row_count or len(ids) > _MAX_ROWS:
            raise RuntimeError("vector rollback row count changed.")
        raw_by_id: dict[str, tuple[str, Mapping[str, Any], tuple[float, ...]]] = {}
        for raw_id, text, metadata, embedding in zip(
            ids,
            documents,
            metadatas,
            embedding_rows,
            strict=True,
        ):
            if not isinstance(raw_id, str) or raw_id in raw_by_id:
                raise RuntimeError("vector rollback identifiers are invalid.")
            if not isinstance(text, str) or not isinstance(metadata, Mapping):
                raise RuntimeError("vector rollback rows are invalid.")
            raw_by_id[raw_id] = (text, dict(metadata), _finite_vector(embedding))
        ordered_documents: list[str] = []
        ordered_metadatas: list[Mapping[str, Any]] = []
        ordered_embeddings: list[tuple[float, ...]] = []
        dimension: int | None = None
        for row_id, expected_text, expected_metadata in zip(
            snapshot.vector.ids,
            snapshot.vector.documents,
            snapshot.vector.metadatas,
            strict=True,
        ):
            row = raw_by_id.get(row_id)
            if row is None or row[0] != expected_text or dict(row[1]) != dict(expected_metadata):
                raise RuntimeError("vector rollback identity changed during capture.")
            current_dimension = len(row[2])
            if dimension is None:
                dimension = current_dimension
            elif current_dimension != dimension:
                raise RuntimeError("source vector dimensionality is inconsistent.")
            ordered_documents.append(row[0])
            ordered_metadatas.append(row[1])
            ordered_embeddings.append(row[2])
        if not ordered_embeddings:
            raise RuntimeError("source vector rollback is empty.")
        return _VectorRollback(
            ids=snapshot.vector.ids,
            documents=tuple(ordered_documents),
            metadatas=tuple(ordered_metadatas),
            embeddings=tuple(ordered_embeddings),
        )

    def _identity(self, operation: CutoverOperation) -> tuple[BackendStateIdentity, CrossStoreSnapshot, GenerationRecord]:
        preparation = operation.preparation
        generation = self.generations.current(
            owner_id=preparation.owner_id,
            doc_id=preparation.doc_id,
        )
        if generation is None or generation.state not in {"active", "restored"}:
            raise RuntimeError("authoritative generation is unavailable.")
        stores = self.index.snapshot(
            owner_id=preparation.owner_id,
            doc_id=preparation.doc_id,
        )
        vector_digest, vector_rows = _vector_identity(
            stores.vector,
            preparation.owner_id,
            preparation.doc_id,
        )
        sparse_digest, sparse_fields = _sparse_identity(
            stores.sparse,
            preparation.owner_id,
            preparation.doc_id,
            generation.profile_fingerprint,
            generation.sparse_generation,
        )
        return (
            BackendStateIdentity(
                source_sequence=generation.sequence,
                profile_fingerprint=generation.profile_fingerprint,
                content_sha256=generation.content_sha256,
                vector_snapshot_digest=vector_digest,
                sparse_snapshot_digest=sparse_digest,
                vector_rows=vector_rows,
                sparse_generation=generation.sparse_generation,
                sparse_fields=sparse_fields,
            ),
            stores,
            generation,
        )

    def current_identity(self, operation: CutoverOperation) -> BackendStateIdentity:
        self._bind(operation)
        identity, stores, generation = self._identity(operation)
        expected = BackendStateIdentity.from_preparation(operation.preparation)
        if identity == expected and self._source_generation is None:
            self._source_vectors = self._capture_source_vectors(operation, stores)
            self._source_stores = stores
            self._source_generation = generation
        return identity

    def _load_target(self, operation: CutoverOperation) -> _LoadedTarget:
        preparation = operation.preparation
        try:
            manifest = self.shadow.validate(preparation.task_id)
        except Exception as exc:
            raise RuntimeError("validated migration shadow is unavailable.") from exc
        if (
            manifest.owner_id != preparation.owner_id
            or manifest.doc_id != preparation.doc_id
            or manifest.source_sequence != preparation.source_sequence
            or manifest.source_profile_fingerprint != preparation.source_profile_fingerprint
            or manifest.target_profile_fingerprint != preparation.target_profile_fingerprint
            or manifest.content_sha256 != preparation.source_content_sha256
            or manifest.validation_digest != preparation.validation_digest
            or manifest.vector_count != preparation.target_vector_rows
            or manifest.sparse_count != preparation.target_sparse_rows
        ):
            raise RuntimeError("migration shadow identity differs from cutover preparation.")
        target_digest = _sha256_json(
            {
                "validation_digest": manifest.validation_digest,
                "target_profile_fingerprint": manifest.target_profile_fingerprint,
                "content_sha256": manifest.content_sha256,
                "vector_sha256": manifest.vector_sha256,
                "sparse_sha256": manifest.sparse_sha256,
                "vector_count": manifest.vector_count,
                "sparse_count": manifest.sparse_count,
            }
        )
        if target_digest != preparation.target_artifact_digest:
            raise RuntimeError("migration shadow artifact digest changed.")

        directory = Path(self.shadow.root) / preparation.task_id
        vector_payload = _read_regular(directory / "vectors.json")
        sparse_payload = _read_regular(directory / "sparse.json")
        if (
            len(vector_payload) != manifest.vector_bytes
            or _sha256_bytes(vector_payload) != manifest.vector_sha256
            or len(sparse_payload) != manifest.sparse_bytes
            or _sha256_bytes(sparse_payload) != manifest.sparse_sha256
        ):
            raise RuntimeError("migration shadow payload changed after validation.")
        vector_raw = _strict_json(vector_payload, "shadow vector artifact")
        sparse_raw = _strict_json(sparse_payload, "shadow sparse artifact")
        if (
            not isinstance(vector_raw, list)
            or not isinstance(sparse_raw, list)
            or len(vector_raw) != manifest.vector_count
            or len(sparse_raw) != manifest.sparse_count
            or len(vector_raw) > _MAX_ROWS
            or len(sparse_raw) > _MAX_ROWS
        ):
            raise RuntimeError("migration shadow row counts are invalid.")

        vectors: list[_VectorRow] = []
        seen: set[str] = set()
        dimension: int | None = None
        for raw in vector_raw:
            if not isinstance(raw, Mapping) or set(raw) != {"row_id", "text", "embedding", "metadata"}:
                raise RuntimeError("shadow vector row schema is invalid.")
            try:
                row = _VectorRow(
                    row_id=raw["row_id"],
                    text=raw["text"],
                    metadata=raw["metadata"],
                    embedding=raw["embedding"],
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError("shadow vector row is invalid.") from exc
            if row.row_id in seen:
                raise RuntimeError("shadow vector row IDs must be unique.")
            seen.add(row.row_id)
            if (
                row.metadata.get("owner_id") != preparation.owner_id
                or row.metadata.get("doc_id") != preparation.doc_id
                or row.metadata.get("target_profile_fingerprint")
                != preparation.target_profile_fingerprint
                or row.metadata.get("content_sha256") != preparation.source_content_sha256
            ):
                raise RuntimeError("shadow vector metadata escaped cutover identity.")
            current_dimension = len(row.embedding)
            if dimension is None:
                dimension = current_dimension
            elif current_dimension != dimension:
                raise RuntimeError("target vector dimensionality is inconsistent.")
            vectors.append(row)

        fields: list[SparseField] = []
        for raw in sparse_raw:
            if not isinstance(raw, Mapping):
                raise RuntimeError("shadow sparse row schema is invalid.")
            try:
                fields.append(SparseField(**dict(raw)))
            except (TypeError, ValueError) as exc:
                raise RuntimeError("shadow sparse row is invalid.") from exc
        target = _LoadedTarget(manifest, tuple(vectors), tuple(fields))
        source = self._source_vectors
        if source is None:
            raise RuntimeError("source rollback embeddings were not captured.")
        if source.dimension != len(target.vectors[0].embedding):
            raise RuntimeError(
                "target vector dimensionality requires blue-green collection cutover."
            )
        return target

    def write_hidden_target(self, operation: CutoverOperation) -> TargetPublication:
        self._bind(operation)
        self._target = self._load_target(operation)
        return TargetPublication.expected(operation.preparation)

    def validate_hidden_target(
        self,
        operation: CutoverOperation,
        publication: TargetPublication,
    ) -> TargetPublication:
        self._bind(operation)
        expected = TargetPublication.expected(operation.preparation)
        if publication != expected:
            raise RuntimeError("hidden publication identity changed.")
        target = self._load_target(operation)
        if self._target is None or target.manifest.validation_digest != self._target.manifest.validation_digest:
            raise RuntimeError("hidden target changed between write and validation.")
        self._target = target
        return expected

    def _install_vectors(self, operation: CutoverOperation, target: _LoadedTarget) -> None:
        preparation = operation.preparation
        delete_vector_generation(
            self.index.rag,
            owner_id=preparation.owner_id,
            doc_id=preparation.doc_id,
        )
        upsert = self.index.rag.collection.upsert
        for start in range(0, len(target.vectors), _BATCH_SIZE):
            rows = target.vectors[start : start + _BATCH_SIZE]
            metadatas: list[dict[str, Any]] = []
            for row in rows:
                metadata = dict(row.metadata)
                metadata.update(
                    {
                        "owner_id": preparation.owner_id,
                        "doc_id": preparation.doc_id,
                        "content_sha256": preparation.source_content_sha256,
                        "embedding_profile_fingerprint": preparation.target_profile_fingerprint,
                        "migration_operation_id": operation.operation_id,
                        "migration_target_artifact_digest": preparation.target_artifact_digest,
                    }
                )
                metadatas.append(metadata)
            try:
                upsert(
                    ids=[row.row_id for row in rows],
                    documents=[row.text for row in rows],
                    metadatas=metadatas,
                    embeddings=[list(row.embedding) for row in rows],
                )
            except Exception as exc:
                raise RuntimeError("target vector publication failed.") from exc

    def _restore_source(self, operation: CutoverOperation) -> None:
        preparation = operation.preparation
        stores = self._source_stores
        generation = self._source_generation
        vectors = self._source_vectors
        if stores is None or generation is None or vectors is None:
            raise RuntimeError("source rollback state was not captured.")
        delete_vector_generation(
            self.index.rag,
            owner_id=preparation.owner_id,
            doc_id=preparation.doc_id,
        )
        try:
            for start in range(0, len(vectors.ids), _BATCH_SIZE):
                stop = min(start + _BATCH_SIZE, len(vectors.ids))
                self.index.rag.collection.upsert(
                    ids=list(vectors.ids[start:stop]),
                    documents=list(vectors.documents[start:stop]),
                    metadatas=[dict(value) for value in vectors.metadatas[start:stop]],
                    embeddings=[list(value) for value in vectors.embeddings[start:stop]],
                )
            self.index.sparse.restore_document(
                owner_id=preparation.owner_id,
                doc_id=preparation.doc_id,
                snapshot=stores.sparse,
            )
            current = self.generations.current(
                owner_id=preparation.owner_id,
                doc_id=preparation.doc_id,
            )
            if current is None:
                raise RuntimeError("generation pointer disappeared during rollback.")
            if current.sequence != generation.sequence:
                self.generations.restore_current(
                    generation,
                    owner_id=preparation.owner_id,
                    doc_id=preparation.doc_id,
                    expected_sequence=current.sequence,
                    reason="migration_cutover_rollback",
                )
        except Exception as exc:
            raise RuntimeError("source rollback restoration failed.") from exc

    def commit_visibility(
        self,
        operation: CutoverOperation,
        publication: TargetPublication,
    ) -> None:
        self._bind(operation)
        if publication != TargetPublication.expected(operation.preparation):
            raise RuntimeError("target publication identity changed before commit.")
        if self._target is None:
            raise RuntimeError("hidden target was not validated.")
        preparation = operation.preparation
        current, _, _ = self._identity(operation)
        if current != BackendStateIdentity.from_preparation(preparation):
            raise RuntimeError("source identity changed immediately before visibility commit.")
        try:
            self._install_vectors(operation, self._target)
            sparse_generation = self.index.sparse.replace_document(
                owner_id=preparation.owner_id,
                doc_id=preparation.doc_id,
                fields=self._target.sparse_fields,
                profile_fingerprint=preparation.target_profile_fingerprint,
                metadata={
                    "owner_id": preparation.owner_id,
                    "doc_id": preparation.doc_id,
                    "content_sha256": preparation.source_content_sha256,
                    "vector_rows": preparation.target_vector_rows,
                    "migration_operation_id": operation.operation_id,
                    "migration_target_artifact_digest": preparation.target_artifact_digest,
                },
                expected_generation=preparation.source_sparse_generation,
            )
            manifest = DocumentGenerationManifest(
                owner_id=preparation.owner_id,
                doc_id=preparation.doc_id,
                content_sha256=preparation.source_content_sha256,
                profile_fingerprint=preparation.target_profile_fingerprint,
                vector_rows=preparation.target_vector_rows,
                sparse_generation=sparse_generation,
            )
            self.generations.record_active(
                manifest,
                expected_sequence=preparation.source_sequence,
                metadata={
                    "migration_operation_id": operation.operation_id,
                    "migration_target_artifact_digest": preparation.target_artifact_digest,
                    "migration_source_sequence": preparation.source_sequence,
                },
            )
        except Exception as exc:
            try:
                self._restore_source(operation)
            except Exception as rollback_exc:
                raise RuntimeError("visibility commit failed and local compensation failed.") from rollback_exc
            raise RuntimeError("visibility commit failed; source state restored.") from exc

    def _visible_vectors(self, operation: CutoverOperation) -> tuple[_VectorRow, ...]:
        result = self._collection_get(operation, include_embeddings=True)
        ids = result.get("ids")
        documents = result.get("documents")
        metadatas = result.get("metadatas")
        embeddings = result.get("embeddings")
        if not all(isinstance(value, list) for value in (ids, documents, metadatas)):
            raise RuntimeError("visible vector arrays are invalid.")
        try:
            embedding_rows = list(embeddings)
        except Exception as exc:
            raise RuntimeError("visible embeddings are unavailable.") from exc
        if not len(ids) == len(documents) == len(metadatas) == len(embedding_rows):
            raise RuntimeError("visible vector arrays are inconsistent.")
        rows: list[_VectorRow] = []
        for row_id, text, metadata, embedding in zip(
            ids,
            documents,
            metadatas,
            embedding_rows,
            strict=True,
        ):
            try:
                rows.append(_VectorRow(row_id, text, metadata, _finite_vector(embedding)))
            except (TypeError, ValueError) as exc:
                raise RuntimeError("visible vector row is invalid.") from exc
        rows.sort(key=lambda row: row.row_id)
        return tuple(rows)

    @staticmethod
    def _embeddings_close(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
        return len(left) == len(right) and all(
            math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7)
            for a, b in zip(left, right, strict=True)
        )

    def validate_visible_target(
        self,
        operation: CutoverOperation,
        publication: TargetPublication,
    ) -> None:
        self._bind(operation)
        if publication != TargetPublication.expected(operation.preparation) or self._target is None:
            raise RuntimeError("visible target publication identity is unavailable.")
        preparation = operation.preparation
        generation = self.generations.current(
            owner_id=preparation.owner_id,
            doc_id=preparation.doc_id,
        )
        sparse = self.index.sparse.snapshot_document(
            owner_id=preparation.owner_id,
            doc_id=preparation.doc_id,
        )
        if (
            generation is None
            or generation.state not in {"active", "restored"}
            or generation.sequence != preparation.source_sequence + 1
            or generation.profile_fingerprint != preparation.target_profile_fingerprint
            or generation.content_sha256 != preparation.source_content_sha256
            or generation.vector_rows != preparation.target_vector_rows
            or sparse is None
            or sparse.generation != generation.sparse_generation
            or sparse.profile_fingerprint != preparation.target_profile_fingerprint
            or len(sparse.fields) != preparation.target_sparse_rows
        ):
            raise RuntimeError("visible generation or sparse target does not match preparation.")
        actual_vectors = self._visible_vectors(operation)
        expected_vectors = tuple(sorted(self._target.vectors, key=lambda row: row.row_id))
        if len(actual_vectors) != len(expected_vectors):
            raise RuntimeError("visible vector row count does not match preparation.")
        for actual, expected in zip(actual_vectors, expected_vectors, strict=True):
            if (
                actual.row_id != expected.row_id
                or actual.text != expected.text
                or actual.metadata.get("owner_id") != preparation.owner_id
                or actual.metadata.get("doc_id") != preparation.doc_id
                or actual.metadata.get("content_sha256") != preparation.source_content_sha256
                or actual.metadata.get("embedding_profile_fingerprint")
                != preparation.target_profile_fingerprint
                or not self._embeddings_close(actual.embedding, expected.embedding)
            ):
                raise RuntimeError("visible vector target differs from validated shadow.")
        actual_sparse = tuple(
            (
                field.field_id,
                field.field_type,
                field.text,
                field.position,
                field.page_number,
                field.section,
                dict(field.metadata),
            )
            for field in sparse.fields
        )
        expected_sparse = tuple(
            (
                field.field_id,
                field.field_type,
                field.text,
                field.position,
                field.page_number,
                field.section,
                dict(field.metadata),
            )
            for field in self._target.sparse_fields
        )
        if actual_sparse != expected_sparse:
            raise RuntimeError("visible sparse target differs from validated shadow.")

    def discard_hidden_target(
        self,
        operation: CutoverOperation,
        publication: TargetPublication,
    ) -> None:
        self._bind(operation)
        if publication != TargetPublication.expected(operation.preparation):
            raise RuntimeError("hidden target discard identity changed.")
        self.shadow.remove(operation.preparation.task_id)
        self._target = None

    def restore_rollback(self, operation: CutoverOperation) -> None:
        self._bind(operation)
        self._restore_source(operation)

    def validate_rollback(self, operation: CutoverOperation) -> None:
        self._bind(operation)
        preparation = operation.preparation
        generation = self.generations.current(
            owner_id=preparation.owner_id,
            doc_id=preparation.doc_id,
        )
        if (
            generation is None
            or generation.state not in {"active", "restored"}
            or generation.profile_fingerprint != preparation.source_profile_fingerprint
            or generation.content_sha256 != preparation.source_content_sha256
            or generation.vector_rows != preparation.source_vector_rows
            or generation.sparse_generation != preparation.source_sparse_generation
        ):
            raise RuntimeError("rollback generation does not match prepared source identity.")
        stores = self.index.snapshot(
            owner_id=preparation.owner_id,
            doc_id=preparation.doc_id,
        )
        vector_digest, vector_rows = _vector_identity(
            stores.vector,
            preparation.owner_id,
            preparation.doc_id,
        )
        sparse_digest, sparse_fields = _sparse_identity(
            stores.sparse,
            preparation.owner_id,
            preparation.doc_id,
            preparation.source_profile_fingerprint,
            preparation.source_sparse_generation,
        )
        if (
            vector_digest != preparation.vector_snapshot_digest
            or sparse_digest != preparation.sparse_snapshot_digest
            or vector_rows != preparation.source_vector_rows
            or sparse_fields != preparation.source_sparse_fields
        ):
            raise RuntimeError("rollback store snapshots do not match prepared source identity.")
        source_vectors = self._source_vectors
        if source_vectors is None:
            raise RuntimeError("source rollback embeddings are unavailable.")
        actual = self._visible_vectors(operation)
        expected = sorted(
            zip(
                source_vectors.ids,
                source_vectors.documents,
                source_vectors.metadatas,
                source_vectors.embeddings,
                strict=True,
            ),
            key=lambda row: row[0],
        )
        if len(actual) != len(expected):
            raise RuntimeError("rollback vector embeddings are incomplete.")
        for row, expected_row in zip(actual, expected, strict=True):
            if (
                row.row_id != expected_row[0]
                or row.text != expected_row[1]
                or dict(row.metadata) != dict(expected_row[2])
                or not self._embeddings_close(row.embedding, expected_row[3])
            ):
                raise RuntimeError("rollback vector embeddings differ from captured source.")


__all__ = ["LocalCutoverBackendAdapter"]
