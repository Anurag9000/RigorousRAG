from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from tools.generation_store import GenerationStore
from tools.index_coordinator import DocumentGenerationManifest, IndexCoordinator
from tools.migration_cutover_control import CutoverOperation, CutoverPreparation
from tools.migration_cutover_local import LocalCutoverBackendAdapter
from tools.migration_cutover_preflight import _sparse_identity, _vector_identity
from tools.migration_cutover_saga import execute_cutover_saga
from tools.migration_shadow_store import MigrationShadowStore, ShadowBuild
from tools.sparse_index import (
    SparseDocumentSnapshot,
    SparseField,
    SparseFieldSnapshot,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class FakeCollection:
    def __init__(self, rows):
        self.rows = {row[0]: (row[1], dict(row[2]), list(row[3])) for row in rows}
        self.dimension = len(next(iter(self.rows.values()))[2])

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
            embeddings = [[0.0] * self.dimension for _ in ids]
        if not len(ids) == len(documents) == len(metadatas) == len(embeddings):
            raise ValueError("inconsistent vector arrays")
        for row_id, text, metadata, embedding in zip(
            ids,
            documents,
            metadatas,
            embeddings,
            strict=True,
        ):
            vector = [float(value) for value in embedding]
            if len(vector) != self.dimension:
                raise ValueError("vector dimension changed")
            self.rows[row_id] = (text, dict(metadata), vector)


class FakeRag:
    def __init__(self, collection):
        self.collection = collection

    def add_document(self, *args, **kwargs):
        raise AssertionError("local cutover must publish validated precomputed vectors")

    def delete_document(self, *, owner_id, doc_id):
        self.collection.delete(where={})

    def list_documents(self, *, owner_id, limit):
        if not self.collection.rows:
            return []
        return [{"doc_id": doc_id_from_rows(self.collection.rows), "owner_id": owner_id}]


def doc_id_from_rows(rows):
    return next(iter(rows.values()))[1]["doc_id"]


class FakeSparse:
    def __init__(self, snapshot: SparseDocumentSnapshot):
        self.current = snapshot

    def snapshot_document(self, *, owner_id, doc_id):
        if self.current is None:
            return None
        assert self.current.owner_id == owner_id and self.current.doc_id == doc_id
        return self.current

    def restore_document(self, *, owner_id, doc_id, snapshot):
        if snapshot is not None:
            assert snapshot.owner_id == owner_id and snapshot.doc_id == doc_id
        self.current = snapshot

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
        if self.current is None:
            current_generation = 0
        else:
            current_generation = self.current.generation
        if expected_generation is not None and expected_generation != current_generation:
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

    def delete_document(self, *, owner_id, doc_id):
        self.current = None

    def list_document_ids(self, *, owner_id, limit):
        return () if self.current is None else (self.current.doc_id,)


class Fixture:
    owner = "alice"
    doc_id = "doc-1"
    source_profile = digest("source-profile")
    target_profile = digest("target-profile")
    content = digest("content")

    def __init__(self, tmp_path, *, target_dimension=2):
        source_field = SparseFieldSnapshot(
            field_id="source-field",
            field_type="body",
            text="source evidence",
            position=0,
            token_count=2,
            page_number=1,
            section="Source",
            metadata={"kind": "source"},
        )
        sparse_snapshot = SparseDocumentSnapshot(
            owner_id=self.owner,
            doc_id=self.doc_id,
            generation=1,
            profile_fingerprint=self.source_profile,
            metadata={"owner_id": self.owner, "doc_id": self.doc_id},
            fields=(source_field,),
        )
        source_metadata = {
            "owner_id": self.owner,
            "doc_id": self.doc_id,
            "content_sha256": self.content,
            "embedding_profile_fingerprint": self.source_profile,
        }
        self.collection = FakeCollection(
            [("source-row", "source evidence", source_metadata, [0.2, 0.8])]
        )
        self.sparse = FakeSparse(sparse_snapshot)
        self.index = IndexCoordinator(rag=FakeRag(self.collection), sparse=self.sparse)
        self.generations = GenerationStore(tmp_path / "generations.sqlite3")
        self.generations.record_active(
            DocumentGenerationManifest(
                owner_id=self.owner,
                doc_id=self.doc_id,
                content_sha256=self.content,
                profile_fingerprint=self.source_profile,
                vector_rows=1,
                sparse_generation=1,
            ),
            expected_sequence=0,
            committed_at=1.0,
        )
        self.shadow = MigrationShadowStore(tmp_path / "shadow")
        self.task = SimpleNamespace(
            task_id="task-1",
            owner_id=self.owner,
            doc_id=self.doc_id,
            source_sequence=1,
            source_profile_fingerprint=self.source_profile,
            target_profile_name="target",
            target_profile_fingerprint=self.target_profile,
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
        target_embedding = [0.9, 0.1]
        if target_dimension == 3:
            target_embedding = [0.9, 0.1, 0.0]
        self.manifest = self.shadow.write(
            task=self.task,
            build=ShadowBuild(
                content_sha256=self.content,
                parser_fingerprint=digest("parser"),
                vector_rows=(
                    {
                        "row_id": "target-row",
                        "text": "target evidence",
                        "embedding": target_embedding,
                        "metadata": {
                            "owner_id": self.owner,
                            "doc_id": self.doc_id,
                            "source_sequence": 1,
                            "target_profile_name": "target",
                            "target_profile_fingerprint": self.target_profile,
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
        stores = self.index.snapshot(owner_id=self.owner, doc_id=self.doc_id)
        vector_digest, vector_rows = _vector_identity(stores.vector, self.owner, self.doc_id)
        sparse_digest, sparse_fields = _sparse_identity(
            stores.sparse,
            self.owner,
            self.doc_id,
            self.source_profile,
            1,
        )
        target_artifact_digest = json_digest(
            {
                "validation_digest": self.manifest.validation_digest,
                "target_profile_fingerprint": self.target_profile,
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
            source_profile_fingerprint=self.source_profile,
            target_profile_fingerprint=self.target_profile,
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
        return LocalCutoverBackendAdapter(
            index=self.index,
            generations=self.generations,
            shadow=self.shadow,
        )


def test_local_adapter_publishes_validated_precomputed_target(tmp_path):
    fixture = Fixture(tmp_path)
    result = execute_cutover_saga(fixture.operation, fixture.adapter())
    assert result.outcome == "published"
    assert result.failure_type is None
    assert result.phases[-1] == "visible_target_validated"
    current = fixture.generations.current(owner_id=fixture.owner, doc_id=fixture.doc_id)
    assert current.sequence == 2
    assert current.profile_fingerprint == fixture.target_profile
    assert current.content_sha256 == fixture.content
    assert list(fixture.collection.rows) == ["target-row"]
    target = fixture.collection.rows["target-row"]
    assert target[0] == "target evidence"
    assert target[2] == pytest.approx([0.9, 0.1])
    assert target[1]["embedding_profile_fingerprint"] == fixture.target_profile
    assert fixture.sparse.current.profile_fingerprint == fixture.target_profile
    assert fixture.sparse.current.fields[0].field_id == "target-field"
    assert fixture.shadow.validate(fixture.task.task_id) == fixture.manifest


def test_post_visibility_fault_restores_and_verifies_exact_source(tmp_path):
    fixture = Fixture(tmp_path)

    def fault(phase):
        if phase == "visibility_committed":
            raise RuntimeError("synthetic post-visibility fault")

    result = execute_cutover_saga(
        fixture.operation,
        fixture.adapter(),
        fault_hook=fault,
    )
    assert result.outcome == "rolled_back"
    assert result.rollback_verified is True
    assert result.failure_type == "RuntimeError"
    assert result.phases[-2:] == ("rollback_restored", "rollback_validated")
    current = fixture.generations.current(owner_id=fixture.owner, doc_id=fixture.doc_id)
    assert current.sequence == 3
    assert current.state == "restored"
    assert current.profile_fingerprint == fixture.source_profile
    assert current.sparse_generation == 1
    assert list(fixture.collection.rows) == ["source-row"]
    assert fixture.collection.rows["source-row"][2] == pytest.approx([0.2, 0.8])
    assert fixture.sparse.current.profile_fingerprint == fixture.source_profile
    assert fixture.sparse.current.fields[0].field_id == "source-field"


def test_dimension_change_fails_before_visibility_and_preserves_shadow(tmp_path):
    fixture = Fixture(tmp_path, target_dimension=3)
    result = execute_cutover_saga(fixture.operation, fixture.adapter())
    assert result.outcome == "aborted"
    assert "visibility_committed" not in result.phases
    assert result.publication_id is None
    current = fixture.generations.current(owner_id=fixture.owner, doc_id=fixture.doc_id)
    assert current.sequence == 1
    assert current.profile_fingerprint == fixture.source_profile
    assert list(fixture.collection.rows) == ["source-row"]
    assert fixture.shadow.validate(fixture.task.task_id) == fixture.manifest


def test_source_change_after_preparation_aborts_without_target_mutation(tmp_path):
    fixture = Fixture(tmp_path)
    fixture.collection.rows["source-row"][1]["content_sha256"] = digest("changed")
    result = execute_cutover_saga(fixture.operation, fixture.adapter())
    assert result.outcome == "aborted"
    assert result.phases == ("lock_acquired",)
    assert list(fixture.collection.rows) == ["source-row"]
    current = fixture.generations.current(owner_id=fixture.owner, doc_id=fixture.doc_id)
    assert current.sequence == 1


def test_adapter_instance_is_bound_to_one_operation(tmp_path):
    fixture = Fixture(tmp_path)
    adapter = fixture.adapter()
    with adapter.exclusive_lock(fixture.operation):
        adapter.current_identity(fixture.operation)
    other_preparation = CutoverPreparation(
        **{
            **fixture.preparation.__dict__,
            "task_id": "task-2",
            "prepared_at": 4.0,
        }
    )
    other = CutoverOperation(
        operation_id=other_preparation.operation_id,
        preparation=other_preparation,
        state="ready",
        attempt=1,
        created_at=4.0,
        updated_at=4.0,
    )
    with pytest.raises(RuntimeError, match="another operation"):
        adapter.exclusive_lock(other)
