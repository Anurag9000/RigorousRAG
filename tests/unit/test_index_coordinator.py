from __future__ import annotations

from copy import deepcopy

import pytest

from tools.index_coordinator import IndexCoordinationError, IndexCoordinator
from tools.sparse_index import SparseField, SparseIndex
from tools.vector_generation import (
    VectorGenerationSnapshot,
    capture_vector_generation,
    restore_vector_generation,
)

OWNER = "alice"
DOC = "doc-1"
PROFILE = "b" * 64
CONTENT = "c" * 64


class FakeCollection:
    def __init__(self):
        self.rows = {}
        self.fail_upsert = False

    @staticmethod
    def _scope(where):
        owner = None
        doc = None
        for item in (where or {}).get("$and", []):
            if "owner_id" in item:
                owner = item["owner_id"]["$eq"]
            if "doc_id" in item:
                doc = item["doc_id"]["$eq"]
        return owner, doc

    def get(self, *, where=None, include=None, limit=None, offset=0):
        owner, doc = self._scope(where)
        selected = [
            (identifier, row)
            for identifier, row in sorted(self.rows.items())
            if (owner is None or row["metadata"].get("owner_id") == owner)
            and (doc is None or row["metadata"].get("doc_id") == doc)
        ]
        if limit is not None:
            selected = selected[offset : offset + limit]
        return {
            "ids": [identifier for identifier, _ in selected],
            "documents": [row["document"] for _, row in selected],
            "metadatas": [deepcopy(row["metadata"]) for _, row in selected],
        }

    def upsert(self, *, ids, documents, metadatas):
        if self.fail_upsert:
            raise RuntimeError("forced restore failure")
        for identifier, document, metadata in zip(ids, documents, metadatas):
            self.rows[identifier] = {
                "document": document,
                "metadata": deepcopy(metadata),
            }

    def delete(self, *, ids=None, where=None):
        if ids is not None:
            for identifier in ids:
                self.rows.pop(identifier, None)
            return
        owner, doc = self._scope(where)
        for identifier, row in list(self.rows.items()):
            if row["metadata"].get("owner_id") == owner and row["metadata"].get("doc_id") == doc:
                del self.rows[identifier]


class FakeRag:
    def __init__(self):
        self.collection = FakeCollection()
        self.fail_add_after_write = False
        self.fail_delete_after_delete = False

    def add_document(self, *, doc_id, text, metadata, sections=None, replace=True, **_kwargs):
        assert replace is True
        self.delete_document(owner_id=metadata["owner_id"], doc_id=doc_id)
        rows = [
            (
                f"{doc_id}:c0",
                text,
                {**metadata, "doc_id": doc_id, "parent_id": f"{doc_id}:p0"},
            )
        ]
        if sections:
            rows.append(
                (
                    f"{doc_id}:c1",
                    str(getattr(sections[0], "content", "section")),
                    {**metadata, "doc_id": doc_id, "parent_id": f"{doc_id}:p1"},
                )
            )
        self.collection.upsert(
            ids=[row[0] for row in rows],
            documents=[row[1] for row in rows],
            metadatas=[row[2] for row in rows],
        )
        if self.fail_add_after_write:
            raise RuntimeError("forced vector failure")
        return len(rows)

    def delete_document(self, *, owner_id, doc_id):
        self.collection.delete(
            where={
                "$and": [
                    {"owner_id": {"$eq": owner_id}},
                    {"doc_id": {"$eq": doc_id}},
                ]
            }
        )
        if self.fail_delete_after_delete:
            raise RuntimeError("forced delete failure")

    def list_documents(self, *, owner_id, limit=5000):
        seen = {}
        for row in self.collection.rows.values():
            metadata = row["metadata"]
            if metadata.get("owner_id") == owner_id:
                seen[metadata["doc_id"]] = {"doc_id": metadata["doc_id"]}
        return list(seen.values())[:limit]


def sparse_field(text):
    return SparseField("body", "body", text, 0, page_number=1)


def vector_state(rag):
    return deepcopy(rag.collection.rows)


def seed_vector(rag, *, owner=OWNER, doc=DOC, text="old vector"):
    rag.collection.upsert(
        ids=[f"{doc}:old"],
        documents=[text],
        metadatas=[
            {
                "owner_id": owner,
                "doc_id": doc,
                "content_sha256": "a" * 64,
                "embedding_profile_fingerprint": "d" * 64,
            }
        ],
    )


def test_vector_snapshot_capture_and_exact_restore():
    rag = FakeRag()
    seed_vector(rag)
    snapshot = capture_vector_generation(rag, owner_id=OWNER, doc_id=DOC)
    assert snapshot.row_count == 1
    assert snapshot.documents == ("old vector",)
    rag.delete_document(owner_id=OWNER, doc_id=DOC)
    restore_vector_generation(rag, owner_id=OWNER, doc_id=DOC, snapshot=snapshot)
    assert capture_vector_generation(rag, owner_id=OWNER, doc_id=DOC) == snapshot


def test_vector_snapshot_rejects_cross_owner_and_malformed_arrays():
    rag = FakeRag()
    seed_vector(rag, owner="bob")
    assert capture_vector_generation(rag, owner_id=OWNER, doc_id=DOC).row_count == 0
    with pytest.raises(ValueError, match="owner scope"):
        VectorGenerationSnapshot(
            OWNER,
            DOC,
            ("x",),
            ("text",),
            ({"owner_id": "bob", "doc_id": DOC},),
        )


def test_successful_cross_store_replacement_and_manifest(tmp_path):
    rag = FakeRag()
    sparse = SparseIndex(tmp_path / "sparse.sqlite3")
    coordinator = IndexCoordinator(rag=rag, sparse=sparse)
    manifest = coordinator.replace_document(
        owner_id=OWNER,
        doc_id=DOC,
        text="new vector target",
        sections=None,
        metadata={"owner_id": OWNER, "filename": "paper.txt"},
        sparse_fields=[sparse_field("new sparse target")],
        content_sha256=CONTENT,
        profile_fingerprint=PROFILE,
    )
    assert manifest.vector_rows == 1
    assert manifest.sparse_generation == 1
    vector = capture_vector_generation(rag, owner_id=OWNER, doc_id=DOC)
    assert vector.metadatas[0]["content_sha256"] == CONTENT
    hit = sparse.search("target", owner_id=OWNER)[0]
    assert hit.doc_id == DOC
    assert hit.profile_fingerprint == PROFILE
    assert hit.metadata["vector_rows"] == 1


def test_sparse_failure_restores_prior_vector_and_sparse_generations(tmp_path, monkeypatch):
    rag = FakeRag()
    sparse = SparseIndex(tmp_path / "sparse.sqlite3")
    seed_vector(rag)
    sparse.replace_document(
        owner_id=OWNER,
        doc_id=DOC,
        fields=[sparse_field("old sparse")],
        profile_fingerprint="d" * 64,
    )
    coordinator = IndexCoordinator(rag=rag, sparse=sparse)
    old_vector = vector_state(rag)
    old_sparse = sparse.snapshot_document(owner_id=OWNER, doc_id=DOC)
    monkeypatch.setattr(
        sparse,
        "replace_document",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("forced sparse failure")),
    )
    with pytest.raises(IndexCoordinationError, match="replacement failed") as error:
        coordinator.replace_document(
            owner_id=OWNER,
            doc_id=DOC,
            text="new vector",
            sections=None,
            metadata={"owner_id": OWNER},
            sparse_fields=[sparse_field("new sparse")],
            content_sha256=CONTENT,
            profile_fingerprint=PROFILE,
        )
    assert error.value.rollback_errors == ()
    assert vector_state(rag) == old_vector
    assert sparse.snapshot_document(owner_id=OWNER, doc_id=DOC) == old_sparse


def test_partial_vector_failure_is_restored_before_error_publication(tmp_path):
    rag = FakeRag()
    sparse = SparseIndex(tmp_path / "sparse.sqlite3")
    seed_vector(rag)
    prior = vector_state(rag)
    coordinator = IndexCoordinator(rag=rag, sparse=sparse)
    rag.fail_add_after_write = True
    with pytest.raises(IndexCoordinationError):
        coordinator.replace_document(
            owner_id=OWNER,
            doc_id=DOC,
            text="partial new vector",
            sections=None,
            metadata={"owner_id": OWNER},
            sparse_fields=[sparse_field("new sparse")],
            content_sha256=CONTENT,
            profile_fingerprint=PROFILE,
        )
    assert vector_state(rag) == prior
    assert sparse.snapshot_document(owner_id=OWNER, doc_id=DOC) is None


def test_rollback_failure_is_explicit_and_never_hidden(tmp_path, monkeypatch):
    rag = FakeRag()
    sparse = SparseIndex(tmp_path / "sparse.sqlite3")
    seed_vector(rag)
    coordinator = IndexCoordinator(rag=rag, sparse=sparse)
    monkeypatch.setattr(
        sparse,
        "replace_document",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("sparse")),
    )
    rag.collection.fail_upsert = True
    with pytest.raises(IndexCoordinationError) as error:
        coordinator.replace_document(
            owner_id=OWNER,
            doc_id=DOC,
            text="new vector",
            sections=None,
            metadata={"owner_id": OWNER},
            sparse_fields=[sparse_field("new sparse")],
            content_sha256=CONTENT,
            profile_fingerprint=PROFILE,
        )
    assert error.value.rollback_errors == ("vector:RuntimeError",)
    assert "Rollback errors" in str(error.value)


def test_cross_store_delete_and_delete_failure_restore(tmp_path, monkeypatch):
    rag = FakeRag()
    sparse = SparseIndex(tmp_path / "sparse.sqlite3")
    seed_vector(rag)
    sparse.replace_document(
        owner_id=OWNER,
        doc_id=DOC,
        fields=[sparse_field("old sparse")],
    )
    coordinator = IndexCoordinator(rag=rag, sparse=sparse)
    prior_vector = vector_state(rag)
    prior_sparse = sparse.snapshot_document(owner_id=OWNER, doc_id=DOC)
    monkeypatch.setattr(
        sparse,
        "delete_document",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("delete")),
    )
    with pytest.raises(IndexCoordinationError, match="deletion failed"):
        coordinator.delete_document(owner_id=OWNER, doc_id=DOC)
    assert vector_state(rag) == prior_vector
    assert sparse.snapshot_document(owner_id=OWNER, doc_id=DOC) == prior_sparse


def test_delete_absent_and_reconciliation_are_owner_scoped(tmp_path):
    rag = FakeRag()
    sparse = SparseIndex(tmp_path / "sparse.sqlite3")
    coordinator = IndexCoordinator(rag=rag, sparse=sparse)
    assert coordinator.delete_document(owner_id=OWNER, doc_id=DOC) is False
    seed_vector(rag, doc="vector-only")
    sparse.replace_document(
        owner_id=OWNER,
        doc_id="sparse-only",
        fields=[sparse_field("sparse")],
    )
    seed_vector(rag, doc="aligned")
    sparse.replace_document(
        owner_id=OWNER,
        doc_id="aligned",
        fields=[sparse_field("aligned")],
    )
    seed_vector(rag, owner="bob", doc="bob-only")
    assert coordinator.scan_owner(owner_id=OWNER) == {
        "vector_only": ("vector-only",),
        "sparse_only": ("sparse-only",),
        "aligned": ("aligned",),
    }


def test_validation_fails_before_store_mutation(tmp_path):
    rag = FakeRag()
    sparse = SparseIndex(tmp_path / "sparse.sqlite3")
    coordinator = IndexCoordinator(rag=rag, sparse=sparse)
    with pytest.raises(ValueError, match="metadata.owner_id"):
        coordinator.replace_document(
            owner_id=OWNER,
            doc_id=DOC,
            text="text",
            sections=None,
            metadata={"owner_id": "bob"},
            sparse_fields=[sparse_field("sparse")],
            content_sha256=CONTENT,
            profile_fingerprint=PROFILE,
        )
    assert rag.collection.rows == {}
    assert sparse.list_document_ids(owner_id=OWNER) == ()
