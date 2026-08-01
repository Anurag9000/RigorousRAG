from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools import lifecycle_import_hook, lifecycle_source_context


def doc_id(owner: str, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rigorousrag:{owner}:{digest}",
        )
    )


def owner_file(tmp_path: Path, owner: str = "alice", payload: bytes = b"paper"):
    root = tmp_path / "uploads"
    directory = root / owner
    directory.mkdir(parents=True)
    source = directory / "paper.pdf"
    source.write_bytes(payload)
    return root, source


def test_retained_source_is_consumed_once_for_exact_owner_and_bytes(tmp_path):
    root, source = owner_file(tmp_path)
    identifier = doc_id("alice", b"paper")
    lifecycle_source_context.remember_retained_source(
        owner_id="alice",
        source_path=source,
    )
    assert lifecycle_source_context.consume_retained_source(
        owner_id="alice",
        doc_id=identifier,
        upload_root=root,
    ) == str(source)
    assert lifecycle_source_context.consume_retained_source(
        owner_id="alice",
        doc_id=identifier,
        upload_root=root,
    ) is None


def test_owner_or_document_mismatch_fails_closed_and_clears_intent(tmp_path):
    root, source = owner_file(tmp_path)
    lifecycle_source_context.remember_retained_source(
        owner_id="alice",
        source_path=source,
    )
    with pytest.raises(RuntimeError, match="another owner"):
        lifecycle_source_context.consume_retained_source(
            owner_id="bob",
            doc_id=doc_id("bob", b"paper"),
            upload_root=root,
        )
    assert lifecycle_source_context.consume_retained_source(
        owner_id="alice",
        doc_id=doc_id("alice", b"paper"),
        upload_root=root,
    ) is None

    lifecycle_source_context.remember_retained_source(
        owner_id="alice",
        source_path=source,
    )
    with pytest.raises(RuntimeError, match="does not match"):
        lifecycle_source_context.consume_retained_source(
            owner_id="alice",
            doc_id=doc_id("alice", b"different"),
            upload_root=root,
        )


def test_document_store_boundary_records_successful_copy_only(tmp_path):
    module = ModuleType("fake_document_store")
    copied = tmp_path / "copied.pdf"
    calls = []

    class DocumentStore:
        upload_root = tmp_path

        def copy_source(self, *, owner_id, source_path, max_bytes=10):
            calls.append((owner_id, source_path, max_bytes))
            return copied

        def register(self, **kwargs):
            return "registered"

        def get(self, **kwargs):
            return None

    module.DocumentStore = DocumentStore
    lifecycle_source_context.install_document_store_source_boundary(module)
    store = DocumentStore()
    assert store.copy_source(
        owner_id="alice",
        source_path="original.pdf",
        max_bytes=9,
    ) == copied
    assert calls == [("alice", "original.pdf", 9)]
    pending = lifecycle_source_context._PENDING.get()
    assert pending.owner_id == "alice"
    assert pending.source_path == str(copied)
    lifecycle_source_context.clear_retained_source()


def test_redundant_batch_registration_short_circuits_before_sqlite_write(tmp_path):
    root, source = owner_file(tmp_path)
    module = ModuleType("fake_document_store")
    writes = []

    class DocumentStore:
        upload_root = root

        def copy_source(self, **kwargs):
            return source

        def get(self, *, owner_id, doc_id, **kwargs):
            return {
                "owner_id": owner_id,
                "doc_id": doc_id,
                "source_path": str(source),
                "mime_type": "application/pdf",
            }

        def register(self, **kwargs):
            writes.append(kwargs)
            return "unexpected"

    module.DocumentStore = DocumentStore
    lifecycle_source_context.install_document_store_source_boundary(module)
    result = DocumentStore().register(
        owner_id="alice",
        doc_id="doc-1",
        filename="paper.pdf",
        mime_type="application/pdf",
        source_path=source,
    )
    assert result is None
    assert writes == []


def test_document_service_boundary_temporarily_binds_verified_copy(
    tmp_path, monkeypatch
):
    root, source = owner_file(tmp_path)
    identifier = doc_id("alice", b"paper")
    registry = SimpleNamespace(upload_root=root)
    import tools.document_store as document_store_module

    monkeypatch.setattr(
        document_store_module,
        "get_document_store",
        lambda: registry,
    )
    module = ModuleType("fake_document_service")
    observed = []

    def original(document, *, owner_id, **kwargs):
        observed.append((document.file_path, owner_id, kwargs))
        return "indexed"

    module.index_document = original
    lifecycle_source_context.install_document_service_source_boundary(module)
    document = SimpleNamespace(id=identifier, file_path="original.pdf")
    lifecycle_source_context.remember_retained_source(
        owner_id="alice",
        source_path=source,
    )
    assert module.index_document(
        document,
        owner_id="alice",
        job_id=None,
    ) == "indexed"
    assert observed == [(str(source), "alice", {"job_id": None})]
    assert document.file_path == "original.pdf"


def test_document_service_without_intent_is_backward_compatible(monkeypatch):
    lifecycle_source_context.clear_retained_source()
    module = ModuleType("fake_document_service")
    calls = []
    module.index_document = lambda document, *, owner_id, **kwargs: calls.append(
        (document.file_path, owner_id)
    ) or "indexed"
    lifecycle_source_context.install_document_service_source_boundary(module)
    monkeypatch.setattr(
        "tools.document_store.get_document_store",
        lambda: SimpleNamespace(upload_root="unused"),
    )
    document = SimpleNamespace(id="doc", file_path="original.pdf")
    assert module.index_document(document, owner_id="alice") == "indexed"
    assert calls == [("original.pdf", "alice")]


def test_import_hook_registers_store_and_service_boundaries():
    assert lifecycle_import_hook._TARGETS["tools.document_store"] == (
        "tools.lifecycle_source_context",
        "install_document_store_source_boundary",
    )
    assert lifecycle_import_hook._TARGETS["tools.document_service"] == (
        "tools.lifecycle_source_context",
        "install_document_service_source_boundary",
    )
