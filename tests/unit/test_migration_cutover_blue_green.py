from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from tools.embedding_models import EmbeddingProfile
from tools.generation_store import GenerationStore
from tools.index_coordinator import DocumentGenerationManifest
from tools.migration_cutover_blue_green import BlueGreenCutoverBackendAdapter
from tools.migration_cutover_control import CutoverOperation, CutoverPreparation
from tools.migration_cutover_preflight import _sparse_identity, _vector_identity
from tools.migration_cutover_saga import execute_cutover_saga
from tools.migration_shadow_store import MigrationShadowStore, ShadowBuild
from tools.sparse_index import SparseDocumentSnapshot, SparseField, SparseFieldSnapshot
from tools.vector_collection_registry import VectorCollectionRegistry
from tools.vector_generation import capture_vector_generation


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def profile(alias: str, model: str, dimensions: int) -> EmbeddingProfile:
    return EmbeddingProfile(
        alias=alias,
        model_name=model,
        dimensions=dimensions,
        max_sequence_tokens=512,
    )


class FakeCollection:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.rows = {}

    def get(self, *, where=None, include=None, limit=None, offset=0):
        selected = list(self.rows.items())[offset : None if limit is None else offset + limit]
        result = {
            "ids": [row_id for row_id, _ in selected],
            "documents": [row[0] for _, row in selected],
            "metadatas": [dict(row[1]) for _, row in selected],
        }
        if include and "embeddings" in include:
            result["embeddings"] = [list(row[2]) for _, row in selected]
        return result

    def delete(self, *, where=None):
        self.rows.clear()

    def upsert(self, *, ids, documents, metadatas, embeddings=None):
        if embeddings is None:
            raise ValueError("precomputed embeddings are required")
        if not len(ids) == len(documents) == len(metadatas) == len(embeddings):
            raise ValueError("inconsistent arrays")
        for row_id, text, metadata, embedding in zip(
            ids, documents, metadatas, embeddings, strict=True
        ):
            vector = [float(value) for value in embedding]
            if len(vector) != self.dimension:
                raise ValueError("dimension mismatch")
            self.rows[row_id] = (text, dict(metadata), vector)


class FakeProvider:
    def __init__(self):
        self.collections = {}

    def collection(self, spec):
        collection = self.collections.get(spec.collection_id)
        if collection is None:
            collection = FakeCollection(spec.dimensions)
            self.collections[spec.collection_id] = collection
        return collection


class FakeSparse:
    def __init__(self, snapshot):
        self.current = snapshot

    def snapshot_document(self, *, owner_id, doc_id):
        if self.current is None:
            return None
        assert self.current.owner_id == owner_id and self.current.doc_id == doc_id
        return self.current

    def replace_document(
        self,
        *,
        owner_id,
        doc_id,
        fields,
        profile_fingerprint,
        metadata,
        expected_generation=None,
    ):
        current_generation = 0 if self.current is None else self.current.generation
        if expected_generation is not None and current_generation != expected_generation:
            raise RuntimeError("sparse generation changed")
        values = tuple(fields)
        generation = current_generation + 1
        self.current = SparseDocumentSnapshot(
            owner_id=owner_id,
            doc_id=doc_id,
            generation=generation,
            profile_fingerprint=profile_fingerprint,
            metadata=dict(metadata),
            fields=tuple(
                SparseFieldSnapshot(
                    field_id=field.field_id,
                    field_type=field.field_type,
                    text=field.text,
                    position=field.position,
                    token_count=max(len(field.text.split()), 1),
                    page_number=field.page_number,
                    section=field.section,
                    metadata=dict(field.metadata),
                )
                for field in values
            ),
        )
        return generation

    def restore_document(self, *, owner_id, doc_id, snapshot):
        if snapshot is not None:
            assert snapshot.owner_id == owner_id and snapshot.doc_id == doc_id
        self.current = snapshot


class Fixture:
    owner = "alice"
    doc_id = "doc-1"
    content = digest("content")

    def __init__(self, tmp_path):
        self.source_profile = profile("source-v1", "model/source", 2)
        self.target_profile = profile("target-v2", "model/target", 3)
        self.registry = VectorCollectionRegistry(tmp_path / "routes.sqlite3")
        self.source_spec = self.registry.register_collection(self.source_profile, now=1.0)
        self.target_spec = self.registry.register_collection(self.target_profile, now=1.0)
        self.provider = FakeProvider()
        source_collection = self.provider.collection(self.source_spec)
        source_collection.upsert(
            ids=["source-row"],
            documents=["source evidence"],
            metadatas=[
                {
                    "owner_id": self.owner,
                    "doc_id": self.doc_id,
                    "content_sha256": self.content,
                    "embedding_profile_fingerprint": self.source_profile.fingerprint,
                }
            ],
            embeddings=[[0.2, 0.8]],
        )
        self.source_sparse = SparseDocumentSnapshot(
            owner_id=self.owner,
            doc_id=self.doc_id,
            generation=1,
            profile_fingerprint=self.source_profile.fingerprint,
            metadata={"owner_id": self.owner, "doc_id": self.doc_id},
            fields=(
                SparseFieldSnapshot(
                    field_id="source-field",
                    field_type="body",
                    text="source evidence",
                    position=0,
                    token_count=2,
                    page_number=1,
                    section="Source",
                    metadata={"kind": "source"},
                ),
            ),
        )
        self.sparse = FakeSparse(self.source_sparse)
        self.generations = GenerationStore(tmp_path / "generations.sqlite3")
        source_generation = self.generations.record_active(
            DocumentGenerationManifest(
                owner_id=self.owner,
                doc_id=self.doc_id,
                content_sha256=self.content,
                profile_fingerprint=self.source_profile.fingerprint,
                vector_rows=1,
                sparse_generation=1,
            ),
            expected_sequence=0,
            committed_at=1.0,
        )
        self.source_route = self.registry.bootstrap_route(
            owner_id=self.owner,
            doc_id=self.doc_id,
            collection_id=self.source_spec.collection_id,
            generation_sequence=source_generation.sequence,
            now=1.0,
        )
        self.shadow = MigrationShadowStore(tmp_path / "shadow")
        self.task = SimpleNamespace(
            task_id="task-1",
            owner_id=self.owner,
            doc_id=self.doc_id,
            source_sequence=1,
            source_profile_fingerprint=self.source_profile.fingerprint,
            target_profile_name=self.target_profile.alias,
            target_profile_fingerprint=self.target_profile.fingerprint,
        )
        target_field = SparseField(
            field_id="target-field",
            field_type="body",
            text="target evidence",
            position=0,
            page_number=1,
            section="Target",
            metadata={"kind": "target"},
        )
        self.manifest = self.shadow.write(
            task=self.task,
            build=ShadowBuild(
                content_sha256=self.content,
                parser_fingerprint=digest("parser"),
                vector_rows=(
                    {
                        "row_id": "target-row",
                        "text": "target evidence",
                        "embedding": [0.9, 0.1, 0.0],
                        "metadata": {
                            "owner_id": self.owner,
                            "doc_id": self.doc_id,
                            "source_sequence": 1,
                            "target_profile_name": self.target_profile.alias,
                            "target_profile_fingerprint": self.target_profile.fingerprint,
                            "content_sha256": self.content,
                            "field_id": "target-field",
                            "field_type": "body",
                            "field_position": 0,
                            "page_number": 1,
                            "section": "Target",
                        },
                    },
                ),
                sparse_rows=(asdict(target_field),),
            ),
            now=2.0,
        )
        vector_snapshot = capture_vector_generation(
            SimpleNamespace(collection=source_collection),
            owner_id=self.owner,
            doc_id=self.doc_id,
        )
        vector_digest, vector_rows = _vector_identity(
            vector_snapshot, self.owner, self.doc_id
        )
        sparse_digest, sparse_fields = _sparse_identity(
            self.source_sparse,
            self.owner,
            self.doc_id,
            self.source_profile.fingerprint,
            1,
        )
        target_artifact_digest = json_digest(
            {
                "validation_digest": self.manifest.validation_digest,
                "target_profile_fingerprint": self.target_profile.fingerprint,
                "content_sha256": self.content,
                "vector_sha256": self.manifest.vector_sha256,
                "sparse_sha256": self.manifest.sparse_sha256,
                "vector_count": 1,
                "sparse_count": 1,
            }
        )
        self.preparation = CutoverPreparation(
            task_id=self.task.task_id,
            owner_id=self.owner,
            doc_id=self.doc_id,
            source_sequence=1,
            source_profile_fingerprint=self.source_profile.fingerprint,
            target_profile_fingerprint=self.target_profile.fingerprint,
            source_content_sha256=self.content,
            validation_digest=self.manifest.validation_digest,
            promotion_report_digest=digest("promotion"),
            benchmark_fingerprint=digest("benchmark"),
            preflight_digest=digest("preflight"),
            rollback_identity_digest=digest("rollback-identity"),
            rollback_artifact_digest=digest("rollback-artifact"),
            rollback_key_id="key-1",
            staging_verification_digest=digest("staging"),
            target_artifact_digest=target_artifact_digest,
            vector_snapshot_digest=vector_digest,
            sparse_snapshot_digest=sparse_digest,
            source_vector_rows=vector_rows,
            source_sparse_generation=1,
            source_sparse_fields=sparse_fields,
            target_vector_rows=1,
            target_sparse_rows=1,
            prepared_at=3.0,
        )
        self.operation = CutoverOperation(
            operation_id=self.preparation.operation_id,
            preparation=self.preparation,
            state="ready",
            attempt=1,
            created_at=3.0,
            updated_at=3.0,
        )

    def adapter(self):
        return BlueGreenCutoverBackendAdapter(
            registry=self.registry,
            provider=self.provider,
            sparse=self.sparse,
            generations=self.generations,
            shadow=self.shadow,
            clock=lambda: 10.0,
        )


def test_dimension_changing_blue_green_cutover_switches_route_without_mutating_source(tmp_path):
    fixture = Fixture(tmp_path)
    source_before = dict(fixture.provider.collection(fixture.source_spec).rows)
    result = execute_cutover_saga(fixture.operation, fixture.adapter())
    assert result.outcome == "published"
    current = fixture.generations.current(owner_id=fixture.owner, doc_id=fixture.doc_id)
    route = fixture.registry.current_route(fixture.owner, fixture.doc_id)
    assert current.sequence == 2
    assert current.profile_fingerprint == fixture.target_profile.fingerprint
    assert route.generation_sequence == current.sequence
    assert route.collection_id == fixture.target_spec.collection_id
    assert route.action == "switch"
    assert fixture.provider.collection(fixture.source_spec).rows == source_before
    target = fixture.provider.collection(fixture.target_spec)
    assert target.rows["target-row"][2] == pytest.approx([0.9, 0.1, 0.0])
    assert fixture.sparse.current.profile_fingerprint == fixture.target_profile.fingerprint


def test_postvisibility_fault_rolls_route_generation_and_sparse_back_append_only(tmp_path):
    fixture = Fixture(tmp_path)

    def fault(phase):
        if phase == "visibility_committed":
            raise RuntimeError("synthetic visible fault")

    result = execute_cutover_saga(fixture.operation, fixture.adapter(), fault_hook=fault)
    assert result.outcome == "rolled_back"
    assert result.rollback_verified is True
    generation = fixture.generations.current(owner_id=fixture.owner, doc_id=fixture.doc_id)
    route = fixture.registry.current_route(fixture.owner, fixture.doc_id)
    assert generation.sequence == 3
    assert generation.state == "restored"
    assert generation.profile_fingerprint == fixture.source_profile.fingerprint
    assert route.revision == 3
    assert route.action == "rollback"
    assert route.collection_id == fixture.source_spec.collection_id
    assert route.generation_sequence == generation.sequence
    assert fixture.sparse.current == fixture.source_sparse
    assert fixture.provider.collection(fixture.target_spec).rows["target-row"][2] == pytest.approx(
        [0.9, 0.1, 0.0]
    )


def test_route_cas_failure_after_generation_publication_is_semantically_compensated(tmp_path):
    fixture = Fixture(tmp_path)
    original = fixture.registry.transition_route

    def fail_switch_once(**kwargs):
        if kwargs.get("action") == "switch":
            raise RuntimeError("synthetic CAS failure")
        return original(**kwargs)

    fixture.registry.transition_route = fail_switch_once  # type: ignore[method-assign]
    result = execute_cutover_saga(fixture.operation, fixture.adapter())
    assert result.outcome == "aborted"
    generation = fixture.generations.current(owner_id=fixture.owner, doc_id=fixture.doc_id)
    route = fixture.registry.current_route(fixture.owner, fixture.doc_id)
    assert generation.sequence == 3
    assert generation.state == "restored"
    assert generation.profile_fingerprint == fixture.source_profile.fingerprint
    assert route.collection_id == fixture.source_spec.collection_id
    assert route.generation_sequence == generation.sequence
    assert route.action == "generation_advance"
    assert fixture.sparse.current == fixture.source_sparse
    assert fixture.provider.collection(fixture.target_spec).rows == {}


def test_hidden_target_fault_discards_only_target_document_and_preserves_source(tmp_path):
    fixture = Fixture(tmp_path)
    source_before = dict(fixture.provider.collection(fixture.source_spec).rows)

    def fault(phase):
        if phase == "hidden_target_written":
            raise RuntimeError("synthetic hidden fault")

    result = execute_cutover_saga(fixture.operation, fixture.adapter(), fault_hook=fault)
    assert result.outcome == "aborted"
    assert fixture.provider.collection(fixture.source_spec).rows == source_before
    assert fixture.provider.collection(fixture.target_spec).rows == {}
    current = fixture.generations.current(owner_id=fixture.owner, doc_id=fixture.doc_id)
    route = fixture.registry.current_route(fixture.owner, fixture.doc_id)
    assert current.sequence == 1
    assert route.revision == 1


def test_unregistered_target_profile_aborts_before_hidden_publication(tmp_path):
    fixture = Fixture(tmp_path)
    fixture.registry.retire_collection(
        fixture.target_spec.collection_id,
        confirm_collection_id=fixture.target_spec.collection_id,
        now=4.0,
    )
    result = execute_cutover_saga(fixture.operation, fixture.adapter())
    assert result.outcome == "aborted"
    assert result.publication_id is None
    assert "hidden_target_written" not in result.phases
    assert fixture.registry.current_route(fixture.owner, fixture.doc_id).collection_id == (
        fixture.source_spec.collection_id
    )
