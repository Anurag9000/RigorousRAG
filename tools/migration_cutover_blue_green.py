"""Blue/green dimension-changing cutover adapter over routed physical collections."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, runtime_checkable

from tools.generation_store import GenerationRecord, GenerationStore
from tools.index_coordinator import DocumentGenerationManifest, _document_lock
from tools.migration_cutover_control import CutoverOperation
from tools.migration_cutover_local import (
    _LoadedTarget,
    _VectorRow,
    _finite_vector,
    _read_regular,
    _scope_filter,
    _sha256_bytes,
    _sha256_json,
    _strict_json,
)
from tools.migration_cutover_preflight import _sparse_identity, _vector_identity
from tools.migration_cutover_saga import BackendStateIdentity, TargetPublication
from tools.migration_shadow_store import MigrationShadowStore
from tools.sparse_index import SparseField
from tools.vector_collection_registry import (
    PhysicalVectorCollection,
    VectorCollectionRegistry,
    VectorRouteRevision,
)
from tools.vector_generation import (
    VectorGenerationSnapshot,
    capture_vector_generation,
    delete_vector_generation,
)

_MAX_ROWS = 100_000
_BATCH_SIZE = 128


@runtime_checkable
class PhysicalCollectionProvider(Protocol):
    def collection(self, spec: PhysicalVectorCollection) -> Any: ...


class ChromaPhysicalCollectionProvider:
    """Concrete provider for deterministic physical Chroma collections."""

    def __init__(self, persist_directory: str | Path) -> None:
        candidate = Path(persist_directory)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        candidate.mkdir(parents=True, exist_ok=True)
        self.persist_directory = candidate.absolute()
        try:
            import chromadb

            self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        except Exception as exc:
            raise RuntimeError("physical Chroma collection provider initialization failed.") from exc

    def collection(self, spec: PhysicalVectorCollection) -> Any:
        if not isinstance(spec, PhysicalVectorCollection) or spec.state != "ready":
            raise ValueError("physical collection spec must be ready.")
        try:
            return self.client.get_or_create_collection(name=spec.collection_name)
        except Exception as exc:
            raise RuntimeError("physical Chroma collection acquisition failed.") from exc


def _rollback_operation_id(operation: CutoverOperation, phase: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "contract": "rigorousrag-blue-green-route-recovery-v1",
                "operation_id": operation.operation_id,
                "phase": phase,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _embeddings_close(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7)
        for a, b in zip(left, right, strict=True)
    )


class BlueGreenCutoverBackendAdapter:
    """Publish a validated target into another physical collection, then route CAS."""

    def __init__(
        self,
        *,
        registry: VectorCollectionRegistry,
        provider: PhysicalCollectionProvider,
        sparse: Any,
        generations: GenerationStore,
        shadow: MigrationShadowStore,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(registry, VectorCollectionRegistry):
            raise ValueError("registry must be VectorCollectionRegistry.")
        if not callable(getattr(provider, "collection", None)):
            raise ValueError("provider must expose collection().")
        if not all(
            callable(getattr(sparse, name, None))
            for name in ("snapshot_document", "replace_document", "restore_document")
        ):
            raise ValueError("sparse backend does not expose required lifecycle methods.")
        if not isinstance(generations, GenerationStore):
            raise ValueError("generations must be GenerationStore.")
        if not isinstance(shadow, MigrationShadowStore):
            raise ValueError("shadow must be MigrationShadowStore.")
        if not callable(clock):
            raise ValueError("clock must be callable.")
        self.registry = registry
        self.provider = provider
        self.sparse = sparse
        self.generations = generations
        self.shadow = shadow
        self.clock = clock
        self._operation_id: str | None = None
        self._source_route: VectorRouteRevision | None = None
        self._source_spec: PhysicalVectorCollection | None = None
        self._source_vector: VectorGenerationSnapshot | None = None
        self._source_sparse: Any = None
        self._source_generation: GenerationRecord | None = None
        self._target: _LoadedTarget | None = None
        self._target_spec: PhysicalVectorCollection | None = None
        self._target_route: VectorRouteRevision | None = None
        self._target_generation: GenerationRecord | None = None

    def _now(self) -> float:
        value = float(self.clock())
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError("cutover clock returned an invalid timestamp.")
        return value

    def _bind(self, operation: CutoverOperation) -> None:
        if not isinstance(operation, CutoverOperation):
            raise ValueError("operation must be CutoverOperation.")
        if self._operation_id is None:
            self._operation_id = operation.operation_id
        elif self._operation_id != operation.operation_id:
            raise RuntimeError("blue-green adapter is already bound to another operation.")

    def exclusive_lock(self, operation: CutoverOperation) -> Any:
        self._bind(operation)
        preparation = operation.preparation
        return _document_lock(preparation.owner_id, preparation.doc_id)

    def _source_state(
        self,
        operation: CutoverOperation,
    ) -> tuple[
        BackendStateIdentity,
        VectorRouteRevision,
        PhysicalVectorCollection,
        VectorGenerationSnapshot,
        Any,
        GenerationRecord,
    ]:
        preparation = operation.preparation
        generation = self.generations.current(
            owner_id=preparation.owner_id,
            doc_id=preparation.doc_id,
        )
        route = self.registry.current_route(preparation.owner_id, preparation.doc_id)
        if generation is None or generation.state not in {"active", "restored"} or route is None:
            raise RuntimeError("authoritative generation or physical route is unavailable.")
        if (
            route.generation_sequence != generation.sequence
            or route.profile_fingerprint != generation.profile_fingerprint
        ):
            raise RuntimeError("physical vector route disagrees with authoritative generation.")
        spec = self.registry.get_collection(route.collection_id)
        if (
            spec is None
            or spec.state != "ready"
            or spec.profile_fingerprint != route.profile_fingerprint
        ):
            raise RuntimeError("source physical vector collection is unavailable.")
        collection = self.provider.collection(spec)
        vector = capture_vector_generation(
            SimpleNamespace(collection=collection),
            owner_id=preparation.owner_id,
            doc_id=preparation.doc_id,
        )
        sparse = self.sparse.snapshot_document(
            owner_id=preparation.owner_id,
            doc_id=preparation.doc_id,
        )
        vector_digest, vector_rows = _vector_identity(
            vector,
            preparation.owner_id,
            preparation.doc_id,
        )
        sparse_digest, sparse_fields = _sparse_identity(
            sparse,
            preparation.owner_id,
            preparation.doc_id,
            generation.profile_fingerprint,
            generation.sparse_generation,
        )
        identity = BackendStateIdentity(
            source_sequence=generation.sequence,
            profile_fingerprint=generation.profile_fingerprint,
            content_sha256=generation.content_sha256,
            vector_snapshot_digest=vector_digest,
            sparse_snapshot_digest=sparse_digest,
            vector_rows=vector_rows,
            sparse_generation=generation.sparse_generation,
            sparse_fields=sparse_fields,
        )
        return identity, route, spec, vector, sparse, generation

    def current_identity(self, operation: CutoverOperation) -> BackendStateIdentity:
        self._bind(operation)
        state = self._source_state(operation)
        identity = state[0]
        if identity == BackendStateIdentity.from_preparation(operation.preparation):
            if self._source_generation is None:
                (
                    _,
                    self._source_route,
                    self._source_spec,
                    self._source_vector,
                    self._source_sparse,
                    self._source_generation,
                ) = state
        return identity

    def _load_target(self, operation: CutoverOperation) -> _LoadedTarget:
        preparation = operation.preparation
        manifest = self.shadow.validate(preparation.task_id)
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
        artifact_digest = _sha256_json(
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
        if artifact_digest != preparation.target_artifact_digest:
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
            elif dimension != current_dimension:
                raise RuntimeError("target vector dimensionality is inconsistent.")
            vectors.append(row)
        target_spec = self.registry.collection_for_profile(preparation.target_profile_fingerprint)
        if target_spec is None or target_spec.state != "ready":
            raise RuntimeError("target physical vector collection is not registered and ready.")
        if dimension is None or target_spec.dimensions != dimension:
            raise RuntimeError("target physical collection dimensionality does not match shadow vectors.")
        if self._source_spec is not None and target_spec.collection_id == self._source_spec.collection_id:
            raise RuntimeError("blue-green target must use a different physical collection.")
        fields: list[SparseField] = []
        for raw in sparse_raw:
            if not isinstance(raw, Mapping):
                raise RuntimeError("shadow sparse row schema is invalid.")
            try:
                fields.append(SparseField(**dict(raw)))
            except (TypeError, ValueError) as exc:
                raise RuntimeError("shadow sparse row is invalid.") from exc
        self._target_spec = target_spec
        return _LoadedTarget(manifest, tuple(vectors), tuple(fields))

    def _collection_rows(
        self,
        collection: Any,
        operation: CutoverOperation,
    ) -> tuple[_VectorRow, ...]:
        preparation = operation.preparation
        getter = getattr(collection, "get", None)
        if not callable(getter):
            raise RuntimeError("physical vector collection does not expose get().")
        try:
            result = getter(
                where=_scope_filter(preparation.owner_id, preparation.doc_id),
                include=["documents", "metadatas", "embeddings"],
                limit=_MAX_ROWS + 1,
            )
        except Exception as exc:
            raise RuntimeError("physical vector collection inspection failed.") from exc
        if not isinstance(result, Mapping):
            raise RuntimeError("physical vector collection inspection returned invalid data.")
        ids = result.get("ids")
        documents = result.get("documents")
        metadatas = result.get("metadatas")
        embeddings = result.get("embeddings")
        if not all(isinstance(value, list) for value in (ids, documents, metadatas)):
            raise RuntimeError("physical vector collection arrays are invalid.")
        try:
            embedding_rows = list(embeddings)
        except Exception as exc:
            raise RuntimeError("physical vector collection embeddings are unavailable.") from exc
        if (
            len(ids) > _MAX_ROWS
            or not len(ids) == len(documents) == len(metadatas) == len(embedding_rows)
        ):
            raise RuntimeError("physical vector collection arrays are inconsistent.")
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
                raise RuntimeError("physical vector row is invalid.") from exc
        rows.sort(key=lambda row: row.row_id)
        return tuple(rows)

    def _delete_target_document(self, operation: CutoverOperation) -> None:
        if self._target_spec is None:
            return
        collection = self.provider.collection(self._target_spec)
        delete_vector_generation(
            SimpleNamespace(collection=collection),
            owner_id=operation.preparation.owner_id,
            doc_id=operation.preparation.doc_id,
        )

    def write_hidden_target(self, operation: CutoverOperation) -> TargetPublication:
        self._bind(operation)
        target = self._load_target(operation)
        if self._target_spec is None:
            raise RuntimeError("target physical collection was not resolved.")
        collection = self.provider.collection(self._target_spec)
        self._delete_target_document(operation)
        try:
            upsert = getattr(collection, "upsert", None)
            if not callable(upsert):
                raise RuntimeError("target physical collection does not expose upsert().")
            for start in range(0, len(target.vectors), _BATCH_SIZE):
                rows = target.vectors[start : start + _BATCH_SIZE]
                metadatas: list[dict[str, Any]] = []
                for row in rows:
                    metadata = dict(row.metadata)
                    metadata.update(
                        {
                            "owner_id": operation.preparation.owner_id,
                            "doc_id": operation.preparation.doc_id,
                            "content_sha256": operation.preparation.source_content_sha256,
                            "embedding_profile_fingerprint": operation.preparation.target_profile_fingerprint,
                            "migration_operation_id": operation.operation_id,
                            "migration_target_artifact_digest": operation.preparation.target_artifact_digest,
                        }
                    )
                    metadatas.append(metadata)
                upsert(
                    ids=[row.row_id for row in rows],
                    documents=[row.text for row in rows],
                    metadatas=metadatas,
                    embeddings=[list(row.embedding) for row in rows],
                )
        except Exception as exc:
            try:
                self._delete_target_document(operation)
            except Exception:
                pass
            raise RuntimeError("blue-green target publication failed.") from exc
        self._target = target
        return TargetPublication.expected(operation.preparation)

    def _validate_target_vectors(self, operation: CutoverOperation) -> None:
        if self._target is None or self._target_spec is None:
            raise RuntimeError("validated target state is unavailable.")
        actual = self._collection_rows(self.provider.collection(self._target_spec), operation)
        expected = tuple(sorted(self._target.vectors, key=lambda row: row.row_id))
        if len(actual) != len(expected):
            raise RuntimeError("blue-green target vector count differs from validated shadow.")
        preparation = operation.preparation
        for current, planned in zip(actual, expected, strict=True):
            if (
                current.row_id != planned.row_id
                or current.text != planned.text
                or current.metadata.get("owner_id") != preparation.owner_id
                or current.metadata.get("doc_id") != preparation.doc_id
                or current.metadata.get("content_sha256") != preparation.source_content_sha256
                or current.metadata.get("embedding_profile_fingerprint")
                != preparation.target_profile_fingerprint
                or not _embeddings_close(current.embedding, planned.embedding)
            ):
                raise RuntimeError("blue-green target vectors differ from validated shadow.")

    def validate_hidden_target(
        self,
        operation: CutoverOperation,
        publication: TargetPublication,
    ) -> TargetPublication:
        self._bind(operation)
        expected = TargetPublication.expected(operation.preparation)
        if publication != expected:
            raise RuntimeError("hidden blue-green publication identity changed.")
        self._validate_target_vectors(operation)
        return expected

    def _restore_previsibility(self, operation: CutoverOperation) -> None:
        preparation = operation.preparation
        if self._source_generation is None or self._source_route is None or self._source_sparse is None:
            raise RuntimeError("source recovery state was not captured.")
        self.sparse.restore_document(
            owner_id=preparation.owner_id,
            doc_id=preparation.doc_id,
            snapshot=self._source_sparse,
        )
        current_generation = self.generations.current(
            owner_id=preparation.owner_id,
            doc_id=preparation.doc_id,
        )
        if current_generation is None:
            raise RuntimeError("generation pointer disappeared during compensation.")
        restored_generation = current_generation
        if current_generation.sequence != self._source_generation.sequence:
            restored_generation = self.generations.restore_current(
                self._source_generation,
                owner_id=preparation.owner_id,
                doc_id=preparation.doc_id,
                expected_sequence=current_generation.sequence,
                reason="blue_green_previsibility_compensation",
            )
        current_route = self.registry.current_route(preparation.owner_id, preparation.doc_id)
        if current_route is None:
            raise RuntimeError("vector route disappeared during compensation.")
        if current_route.collection_id != self._source_route.collection_id:
            self.registry.transition_route(
                owner_id=preparation.owner_id,
                doc_id=preparation.doc_id,
                expected_revision=current_route.revision,
                expected_collection_id=current_route.collection_id,
                expected_profile_fingerprint=current_route.profile_fingerprint,
                expected_generation_sequence=current_route.generation_sequence,
                target_collection_id=self._source_route.collection_id,
                target_generation_sequence=restored_generation.sequence,
                operation_id=_rollback_operation_id(operation, "previsibility-switch"),
                action="rollback",
                now=self._now(),
            )
        elif current_route.generation_sequence != restored_generation.sequence:
            self.registry.transition_route(
                owner_id=preparation.owner_id,
                doc_id=preparation.doc_id,
                expected_revision=current_route.revision,
                expected_collection_id=current_route.collection_id,
                expected_profile_fingerprint=current_route.profile_fingerprint,
                expected_generation_sequence=current_route.generation_sequence,
                target_collection_id=current_route.collection_id,
                target_generation_sequence=restored_generation.sequence,
                operation_id=_rollback_operation_id(operation, "previsibility-generation"),
                action="generation_advance",
                now=self._now(),
            )

    def commit_visibility(
        self,
        operation: CutoverOperation,
        publication: TargetPublication,
    ) -> None:
        self._bind(operation)
        preparation = operation.preparation
        if publication != TargetPublication.expected(preparation):
            raise RuntimeError("target publication identity changed before visibility commit.")
        if self._target is None or self._target_spec is None or self._source_route is None:
            raise RuntimeError("blue-green source or target state was not captured.")
        if self.current_identity(operation) != BackendStateIdentity.from_preparation(preparation):
            raise RuntimeError("source identity changed immediately before blue-green cutover.")
        try:
            sparse_generation = self.sparse.replace_document(
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
            target_generation = self.generations.record_active(
                DocumentGenerationManifest(
                    owner_id=preparation.owner_id,
                    doc_id=preparation.doc_id,
                    content_sha256=preparation.source_content_sha256,
                    profile_fingerprint=preparation.target_profile_fingerprint,
                    vector_rows=preparation.target_vector_rows,
                    sparse_generation=sparse_generation,
                ),
                expected_sequence=preparation.source_sequence,
                metadata={
                    "migration_operation_id": operation.operation_id,
                    "migration_target_artifact_digest": preparation.target_artifact_digest,
                    "blue_green_collection_id": self._target_spec.collection_id,
                },
            )
            target_route = self.registry.transition_route(
                owner_id=preparation.owner_id,
                doc_id=preparation.doc_id,
                expected_revision=self._source_route.revision,
                expected_collection_id=self._source_route.collection_id,
                expected_profile_fingerprint=self._source_route.profile_fingerprint,
                expected_generation_sequence=self._source_route.generation_sequence,
                target_collection_id=self._target_spec.collection_id,
                target_generation_sequence=target_generation.sequence,
                operation_id=operation.operation_id,
                action="switch",
                now=self._now(),
            )
            self._target_generation = target_generation
            self._target_route = target_route
        except Exception as exc:
            try:
                self._restore_previsibility(operation)
            except Exception as recovery_exc:
                raise RuntimeError(
                    "blue-green visibility commit failed and compensation failed."
                ) from recovery_exc
            raise RuntimeError("blue-green visibility commit failed; source state restored.") from exc

    def validate_visible_target(
        self,
        operation: CutoverOperation,
        publication: TargetPublication,
    ) -> None:
        self._bind(operation)
        preparation = operation.preparation
        if publication != TargetPublication.expected(preparation):
            raise RuntimeError("visible blue-green publication identity changed.")
        self._validate_target_vectors(operation)
        generation = self.generations.current(
            owner_id=preparation.owner_id,
            doc_id=preparation.doc_id,
        )
        route = self.registry.current_route(preparation.owner_id, preparation.doc_id)
        sparse = self.sparse.snapshot_document(
            owner_id=preparation.owner_id,
            doc_id=preparation.doc_id,
        )
        if (
            generation is None
            or route is None
            or self._target_spec is None
            or generation.sequence != route.generation_sequence
            or generation.profile_fingerprint != preparation.target_profile_fingerprint
            or route.profile_fingerprint != preparation.target_profile_fingerprint
            or route.collection_id != self._target_spec.collection_id
            or generation.content_sha256 != preparation.source_content_sha256
            or generation.vector_rows != preparation.target_vector_rows
            or sparse is None
            or sparse.generation != generation.sparse_generation
            or sparse.profile_fingerprint != preparation.target_profile_fingerprint
            or len(sparse.fields) != preparation.target_sparse_rows
        ):
            raise RuntimeError("visible blue-green generation/route/sparse state is inconsistent.")

    def discard_hidden_target(
        self,
        operation: CutoverOperation,
        publication: TargetPublication,
    ) -> None:
        self._bind(operation)
        if publication != TargetPublication.expected(operation.preparation):
            raise RuntimeError("hidden target discard identity changed.")
        self._delete_target_document(operation)
        self._target = None

    def validate_aborted_source(self, operation: CutoverOperation) -> None:
        """Validate semantic source restoration even if append-only sequences advanced."""

        self._bind(operation)
        preparation = operation.preparation
        generation = self.generations.current(
            owner_id=preparation.owner_id,
            doc_id=preparation.doc_id,
        )
        route = self.registry.current_route(preparation.owner_id, preparation.doc_id)
        if generation is None or route is None or self._source_spec is None:
            raise RuntimeError("aborted source generation or route is unavailable.")
        if (
            generation.profile_fingerprint != preparation.source_profile_fingerprint
            or generation.content_sha256 != preparation.source_content_sha256
            or generation.vector_rows != preparation.source_vector_rows
            or generation.sparse_generation != preparation.source_sparse_generation
            or route.collection_id != self._source_spec.collection_id
            or route.profile_fingerprint != preparation.source_profile_fingerprint
            or route.generation_sequence != generation.sequence
        ):
            raise RuntimeError("aborted source semantics were not restored.")
        vector = capture_vector_generation(
            SimpleNamespace(collection=self.provider.collection(self._source_spec)),
            owner_id=preparation.owner_id,
            doc_id=preparation.doc_id,
        )
        sparse = self.sparse.snapshot_document(
            owner_id=preparation.owner_id,
            doc_id=preparation.doc_id,
        )
        vector_digest, vector_rows = _vector_identity(vector, preparation.owner_id, preparation.doc_id)
        sparse_digest, sparse_fields = _sparse_identity(
            sparse,
            preparation.owner_id,
            preparation.doc_id,
            preparation.source_profile_fingerprint,
            preparation.source_sparse_generation,
        )
        if (
            vector_digest != preparation.vector_snapshot_digest
            or vector_rows != preparation.source_vector_rows
            or sparse_digest != preparation.sparse_snapshot_digest
            or sparse_fields != preparation.source_sparse_fields
        ):
            raise RuntimeError("aborted source snapshots differ from preparation.")

    def restore_rollback(self, operation: CutoverOperation) -> None:
        self._bind(operation)
        preparation = operation.preparation
        if self._source_generation is None or self._source_route is None or self._source_sparse is None:
            raise RuntimeError("source rollback state was not captured.")
        self.sparse.restore_document(
            owner_id=preparation.owner_id,
            doc_id=preparation.doc_id,
            snapshot=self._source_sparse,
        )
        current_generation = self.generations.current(
            owner_id=preparation.owner_id,
            doc_id=preparation.doc_id,
        )
        current_route = self.registry.current_route(preparation.owner_id, preparation.doc_id)
        if current_generation is None or current_route is None:
            raise RuntimeError("visible target state disappeared before rollback.")
        restored = self.generations.restore_current(
            self._source_generation,
            owner_id=preparation.owner_id,
            doc_id=preparation.doc_id,
            expected_sequence=current_generation.sequence,
            reason="blue_green_postvisibility_rollback",
        )
        self.registry.transition_route(
            owner_id=preparation.owner_id,
            doc_id=preparation.doc_id,
            expected_revision=current_route.revision,
            expected_collection_id=current_route.collection_id,
            expected_profile_fingerprint=current_route.profile_fingerprint,
            expected_generation_sequence=current_route.generation_sequence,
            target_collection_id=self._source_route.collection_id,
            target_generation_sequence=restored.sequence,
            operation_id=_rollback_operation_id(operation, "postvisibility"),
            action="rollback",
            now=self._now(),
        )

    def validate_rollback(self, operation: CutoverOperation) -> None:
        self.validate_aborted_source(operation)


__all__ = [
    "BlueGreenCutoverBackendAdapter",
    "ChromaPhysicalCollectionProvider",
    "PhysicalCollectionProvider",
]
