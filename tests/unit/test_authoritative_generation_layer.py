from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass

import pytest

from tools.authoritative_document_index import commit_finalized_document
from tools.generation_store import GenerationStore
from tools.index_coordinator import IndexCoordinator
from tools.index_reconciliation import plan_repairs
from tools.sparse_fields import build_sparse_fields
from tools.sparse_index import SparseField, SparseIndex
from tools.three_store_coordinator import (
    AuthoritativeCoordinationError,
    AuthoritativeIndexCoordinator,
)
from tools.vector_generation import capture_vector_generation

CONTENT = "c" * 64
PROFILE = "b" * 64


class FakeCollection:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}

    @staticmethod
    def _scope(where):
        owner = document = None
        for item in (where or {}).get("$and", []):
            if "owner_id" in item:
                owner = item["owner_id"]["$eq"]
            if "doc_id" in item:
                document = item["doc_id"]["$eq"]
        return owner, document

    def get(self, *, where=None, include=None, limit=None, offset=0):
        owner, document = self._scope(where)
        selected = [
            (identifier, row)
            for identifier, row in sorted(self.rows.items())
            if (owner is None or row["metadata"].get("owner_id") == owner)
            and (document is None or row["metadata"].get("doc_id") == document)
        ]
        if limit is not None:
            selected = selected[offset : offset + limit]
        return {
            "ids": [identifier for identifier, _ in selected],
            "documents": [row["document"] for _, row in selected],
            "metadatas": [deepcopy(row["metadata"]) for _, row in selected],
        }

    def upsert(self, *, ids, documents, metadatas):
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
        owner, document = self._scope(where)
        for identifier, row in list(self.rows.items()):
            if (
                row["metadata"].get("owner_id") == owner
                and row["metadata"].get("doc_id") == document
            ):
                del self.rows[identifier]


class FakeRag:
    def __init__(self) -> None:
        self.collection = FakeCollection()

    def add_document(
        self,
        *,
        doc_id,
        text,
        metadata,
        sections=None,
        replace=True,
        **_kwargs,
    ):
        assert replace is True
        owner = metadata["owner_id"]
        self.delete_document(owner_id=owner, doc_id=doc_id)
        rows = [(f"{doc_id}:c0", text)]
        if sections:
            rows.append(
                (
                    f"{doc_id}:c1",
                    str(getattr(sections[0], "content", "section")),
                )
            )
        self.collection.upsert(
            ids=[row[0] for row in rows],
            documents=[row[1] for row in rows],
            metadatas=[
                {
                    **metadata,
                    "doc_id": doc_id,
                    "parent_id": f"{doc_id}:p{index}",
                }
                for index, _ in enumerate(rows)
            ],
        )
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

    def list_documents(self, *, owner_id, limit=5000):
        values = self.collection.get(
            where={"$and": [{"owner_id": {"$eq": owner_id}}]},
            include=["metadatas"],
            limit=limit,
        )
        return [
            {"doc_id": value["doc_id"]}
            for value in values["metadatas"]
            if isinstance(value, dict) and isinstance(value.get("doc_id"), str)
        ]


def field(text="evidence"):
    return SparseField("body", "body", text, 0, page_number=1)


def make_coordinator(tmp_path):
    rag = FakeRag()
    sparse = SparseIndex(tmp_path / "sparse.sqlite3")
    index = IndexCoordinator(rag=rag, sparse=sparse)
    generations = GenerationStore(tmp_path / "generations.sqlite3")
    return (
        AuthoritativeIndexCoordinator(index=index, generations=generations),
        index,
        generations,
    )


def replace(coordinator, *, content=CONTENT, profile=PROFILE):
    return coordinator.replace_document(
        owner_id="alice",
        doc_id="doc-1",
        text="evidence",
        sections=None,
        metadata={"owner_id": "alice"},
        sparse_fields=[field()],
        content_sha256=content,
        profile_fingerprint=profile,
    )


def test_three_store_replace_delete_and_reconcile(tmp_path):
    coordinator, _index, generations = make_coordinator(tmp_path)
    record = replace(coordinator)
    assert record.state == "active" and record.sequence == 1
    assert coordinator.reconcile_owner(owner_id="alice").healthy == ("doc-1",)
    assert coordinator.delete_document(owner_id="alice", doc_id="doc-1") is True
    current = generations.current(owner_id="alice", doc_id="doc-1")
    assert current is not None and current.state == "deleted"
    assert coordinator.reconcile_owner(owner_id="alice").clean


def test_manifest_failure_restores_vector_sparse_and_current_pointer(
    tmp_path,
    monkeypatch,
):
    coordinator, index, generations = make_coordinator(tmp_path)
    prior_record = replace(coordinator)
    prior_vector = capture_vector_generation(
        index.rag,
        owner_id="alice",
        doc_id="doc-1",
    )
    prior_sparse = index.sparse.snapshot_document(
        owner_id="alice",
        doc_id="doc-1",
    )
    original = generations.record_active

    def commit_then_fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("after commit")

    monkeypatch.setattr(generations, "record_active", commit_then_fail)
    with pytest.raises(AuthoritativeCoordinationError, match="manifest"):
        replace(coordinator, content="d" * 64, profile="e" * 64)
    assert capture_vector_generation(
        index.rag,
        owner_id="alice",
        doc_id="doc-1",
    ) == prior_vector
    assert index.sparse.snapshot_document(
        owner_id="alice",
        doc_id="doc-1",
    ) == prior_sparse
    restored = generations.current(owner_id="alice", doc_id="doc-1")
    assert restored is not None
    assert restored.content_sha256 == prior_record.content_sha256
    assert restored.state == "restored"


def test_reconciliation_plans_deleted_residue_as_automatic(tmp_path):
    coordinator, index, _generations = make_coordinator(tmp_path)
    replace(coordinator)
    coordinator.delete_document(owner_id="alice", doc_id="doc-1")
    index.rag.add_document(
        doc_id="doc-1",
        text="residue",
        metadata={
            "owner_id": "alice",
            "content_sha256": CONTENT,
            "embedding_profile_fingerprint": PROFILE,
        },
        replace=True,
    )
    report = coordinator.reconcile_owner(owner_id="alice")
    assert report.deleted_but_present == ("doc-1",)
    automatic = [action for action in plan_repairs(report) if action.automatic]
    assert [(action.action, action.doc_id) for action in automatic] == [
        ("delete_store_residue", "doc-1")
    ]


@dataclass
class Section:
    title: str
    content: str
    page_number: int | None = None
    metadata: dict | None = None


@dataclass
class Document:
    id: str
    title: str
    text: str
    sections: list[Section]


def test_sparse_fields_and_finalized_document_commit(tmp_path):
    document = Document(
        "doc-1",
        "Trial",
        "Abstract evidence and body evidence.",
        [Section("Abstract", "Abstract evidence", 1, {})],
    )
    fields = build_sparse_fields(document)
    assert {value.field_type for value in fields} >= {
        "title",
        "abstract",
        "heading",
    }
    coordinator, index, _generations = make_coordinator(tmp_path)
    content_hash = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
    result = commit_finalized_document(
        document,
        owner_id="alice",
        rag=index.rag,
        metadata={
            "owner_id": "alice",
            "content_sha256": content_hash,
        },
        coordinator=coordinator,
        profile_name="minilm-l6-v2",
    )
    assert result.vector_rows > 0
    assert result.sparse_field_count == len(fields)
    assert capture_vector_generation(
        index.rag,
        owner_id="alice",
        doc_id="doc-1",
    ).row_count == result.vector_rows
