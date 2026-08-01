from __future__ import annotations

import importlib.abc
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools import lifecycle_boundary, lifecycle_import_hook, lifecycle_runtime
from tools.lifecycle_outbox import LifecycleOutbox, operation_id_for
from tools.lifecycle_reconciliation import LifecycleCleanupJournal

HASH = "a" * 64


class Generations:
    def __init__(self, value=None):
        self.value = value

    def current(self, *, owner_id, doc_id):
        return self.value


class Registry:
    def __init__(self, root: Path):
        self.upload_root = root
        self.rows = {}
        self.fail_register = False
        self.fail_delete = False

    def register(self, *, owner_id, doc_id, filename, mime_type, source_path):
        if self.fail_register:
            raise RuntimeError("private registry detail")
        prior = self.rows.get((owner_id, doc_id))
        self.rows[(owner_id, doc_id)] = {
            "owner_id": owner_id,
            "doc_id": doc_id,
            "filename": filename,
            "mime_type": mime_type,
            "source_path": source_path,
            "source_retained": bool(source_path),
            "updated_at": 2.0,
        }
        return (prior or {}).get("source_path")

    def get(self, *, owner_id, doc_id, **kwargs):
        return self.rows.get((owner_id, doc_id))

    def delete(self, *, owner_id, doc_id):
        if self.fail_delete:
            raise RuntimeError("private registry detail")
        return self.rows.pop((owner_id, doc_id), None)

    def remove_source(self, source_path):
        path = Path(source_path)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True


class Result:
    def __init__(self, generation, sparse_field_count):
        self.generation = generation
        self.sparse_field_count = sparse_field_count
        self.vector_rows = generation.vector_rows


def generation(sequence, state="active", content=HASH):
    return SimpleNamespace(
        sequence=sequence,
        state=state,
        content_sha256=content,
        vector_rows=2 if state != "deleted" else 0,
        sparse_generation=sequence if state != "deleted" else 0,
    )


def fake_authoritative(generations, calls, outbox):
    module = ModuleType("fake_authoritative")
    locks = {}

    def lock(owner_id, doc_id):
        return locks.setdefault((owner_id, doc_id), threading.RLock())

    def commit(document, **kwargs):
        pending = outbox.list_pending(owner_id=kwargs["owner_id"])
        assert pending and pending[0].state == "planned"
        calls["commit"] += 1
        generations.value = generation(calls["commit"])
        return Result(generations.value, 2)

    def delete(**kwargs):
        calls["delete"] += 1
        generations.value = generation(calls["delete"] + 10, state="deleted")
        return True

    module.commit_finalized_document = commit
    module.delete_authoritative_document = delete
    module._identifier = lambda value, label: str(value)
    module._text = lambda value, label: str(value)
    module.build_sparse_fields = lambda document, doc_id: ("title", "body")
    module.AuthoritativeIndexResult = Result
    module._document_lock = lock
    module.get_authoritative_index_coordinator = lambda rag: SimpleNamespace(
        generations=generations
    )
    return module


def document():
    return SimpleNamespace(
        id="doc-1",
        text="finalized text",
        filename="paper.pdf",
        mime_type="application/pdf",
        file_path="/ignored/snapshot.pdf",
    )


def metadata():
    return {
        "owner_id": "alice",
        "content_sha256": HASH,
        "filename": "paper.pdf",
        "mime_type": "application/pdf",
    }


def setup_boundary(tmp_path, monkeypatch):
    root = tmp_path / "uploads"
    owner = root / "alice"
    owner.mkdir(parents=True)
    source = owner / "paper.pdf"
    source.write_bytes(b"pdf")
    registry = Registry(root)
    outbox = LifecycleOutbox(tmp_path / "outbox.sqlite3")
    cleanup = LifecycleCleanupJournal(tmp_path / "cleanup.sqlite3")
    generations = Generations()
    calls = {"commit": 0, "delete": 0}
    module = fake_authoritative(generations, calls, outbox)
    monkeypatch.setattr(lifecycle_boundary, "get_lifecycle_outbox", lambda: outbox)
    monkeypatch.setattr(lifecycle_boundary, "get_cleanup_journal", lambda: cleanup)
    monkeypatch.setattr(lifecycle_boundary, "_document_store", lambda: registry)
    monkeypatch.setattr(
        lifecycle_boundary,
        "_source_context",
        lambda **kwargs: (registry, str(source), True, "job-1"),
    )
    lifecycle_boundary.install_authoritative_lifecycle_boundary(module)
    return module, outbox, generations, registry, source, calls


def replace_operation_id():
    return operation_id_for(
        kind="replace",
        owner_id="alice",
        doc_id="doc-1",
        content_sha256=HASH,
        idempotency_key="job:job-1",
    )


def test_replace_intent_precedes_index_and_completes_registry(tmp_path, monkeypatch):
    module, outbox, _generations, registry, source, calls = setup_boundary(
        tmp_path, monkeypatch
    )
    result = module.commit_finalized_document(
        document(),
        owner_id="alice",
        rag=object(),
        metadata=metadata(),
    )
    assert calls["commit"] == 1
    assert result.generation.sequence == 1
    operation = outbox.get(replace_operation_id())
    assert operation.state == "completed"
    assert operation.generation_sequence == 1
    assert registry.get(owner_id="alice", doc_id="doc-1")["source_path"] == str(
        source
    )


def test_registry_failure_replays_without_reindex(tmp_path, monkeypatch):
    module, outbox, generations, registry, _source, calls = setup_boundary(
        tmp_path, monkeypatch
    )
    registry.fail_register = True
    with pytest.raises(RuntimeError, match="private registry detail"):
        module.commit_finalized_document(
            document(),
            owner_id="alice",
            rag=object(),
            metadata=metadata(),
        )
    assert outbox.get(replace_operation_id()).state == "index_committed"
    assert calls["commit"] == 1

    registry.fail_register = False
    result = module.commit_finalized_document(
        document(),
        owner_id="alice",
        rag=object(),
        metadata=metadata(),
    )
    assert calls["commit"] == 1
    assert result.generation is generations.value
    assert outbox.get(replace_operation_id()).state == "completed"


def test_matching_generation_recovers_crash_before_index_mark(tmp_path, monkeypatch):
    module, outbox, generations, registry, source, calls = setup_boundary(
        tmp_path, monkeypatch
    )
    outbox.plan_replace(
        operation_id=replace_operation_id(),
        owner_id="alice",
        doc_id="doc-1",
        content_sha256=HASH,
        filename="paper.pdf",
        mime_type="application/pdf",
        source_path=str(source),
        retain_source=True,
    )
    generations.value = generation(7)
    result = module.commit_finalized_document(
        document(),
        owner_id="alice",
        rag=object(),
        metadata=metadata(),
    )
    assert calls["commit"] == 0
    assert result.generation.sequence == 7
    assert outbox.get(replace_operation_id()).state == "completed"
    assert registry.get(owner_id="alice", doc_id="doc-1") is not None


def test_nonplanned_operation_refuses_superseded_generation(tmp_path, monkeypatch):
    module, outbox, generations, _registry, source, calls = setup_boundary(
        tmp_path, monkeypatch
    )
    outbox.plan_replace(
        operation_id=replace_operation_id(),
        owner_id="alice",
        doc_id="doc-1",
        content_sha256=HASH,
        filename="paper.pdf",
        mime_type="application/pdf",
        source_path=str(source),
        retain_source=True,
    )
    outbox.mark_index_committed(replace_operation_id(), generation_sequence=2)
    generations.value = generation(3, content="b" * 64)
    with pytest.raises(RuntimeError, match="superseded"):
        module.commit_finalized_document(
            document(),
            owner_id="alice",
            rag=object(),
            metadata=metadata(),
        )
    assert calls["commit"] == 0


def test_delete_replays_registry_failure_without_second_index_delete(
    tmp_path, monkeypatch
):
    module, outbox, generations, registry, source, calls = setup_boundary(
        tmp_path, monkeypatch
    )
    generations.value = generation(5)
    registry.rows[("alice", "doc-1")] = {
        "source_path": str(source),
        "updated_at": 2.0,
    }
    registry.fail_delete = True
    with pytest.raises(RuntimeError, match="private registry detail"):
        module.delete_authoritative_document(
            owner_id="alice",
            doc_id="doc-1",
            rag=object(),
        )
    assert calls["delete"] == 1
    assert outbox.list_pending(owner_id="alice")[0].state == "index_committed"

    registry.fail_delete = False
    assert module.delete_authoritative_document(
        owner_id="alice",
        doc_id="doc-1",
        rag=object(),
    ) is True
    assert calls["delete"] == 1
    assert source.exists() is False


def test_rag_boundary_reconciles_before_constructing_layer(monkeypatch):
    module = ModuleType("fake_rag")
    events = []
    module.get_rag_layer = lambda *args, **kwargs: events.append("rag") or object()
    monkeypatch.setattr(
        lifecycle_boundary,
        "reconcile_lifecycle_before_retrieval",
        lambda: events.append("reconcile"),
    )
    lifecycle_boundary.install_rag_lifecycle_boundary(module)
    module.get_rag_layer()
    assert events == ["reconcile", "rag"]


def test_runtime_cleanup_is_idempotent_and_root_scoped(tmp_path):
    root = tmp_path / "uploads"
    root.mkdir()
    registry = Registry(root)
    absent = root / "absent.pdf"
    assert lifecycle_runtime.remove_source_idempotently(registry, str(absent)) is True
    present = root / "present.pdf"
    present.write_bytes(b"x")
    assert lifecycle_runtime.remove_source_idempotently(registry, str(present)) is True
    assert not present.exists()
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"x")
    assert lifecycle_runtime.remove_source_idempotently(registry, str(outside)) is False
    assert outside.exists()


def test_startup_reconciliation_marks_complete_only_after_success(monkeypatch, tmp_path):
    lifecycle_runtime.clear_lifecycle_runtime_caches()
    path = tmp_path / "outbox.sqlite3"
    calls = []

    def successful(**kwargs):
        calls.append("success")
        return ()

    monkeypatch.setattr(lifecycle_runtime, "reconcile_lifecycle_pending", successful)
    lifecycle_runtime.reconcile_lifecycle_before_retrieval(path)
    lifecycle_runtime.reconcile_lifecycle_before_retrieval(path)
    assert calls == ["success"]

    lifecycle_runtime.clear_lifecycle_runtime_caches()
    monkeypatch.setattr(
        lifecycle_runtime,
        "reconcile_lifecycle_pending",
        lambda **kwargs: (SimpleNamespace(outcome="error"),),
    )
    with pytest.raises(RuntimeError, match="failed before retrieval"):
        lifecycle_runtime.reconcile_lifecycle_before_retrieval(path)
    with pytest.raises(RuntimeError, match="failed before retrieval"):
        lifecycle_runtime.reconcile_lifecycle_before_retrieval(path)


def test_import_loader_patches_module_published_in_sys_modules(monkeypatch):
    events = []

    class Loader(importlib.abc.Loader):
        def exec_module(self, module):
            events.append("original")
            module.original = True

    monkeypatch.setattr(
        lifecycle_boundary,
        "install_rag_lifecycle_boundary",
        lambda target: events.append(("installed", target)),
    )
    loader = lifecycle_import_hook._LifecycleLoader(
        Loader(),
        "install_rag_lifecycle_boundary",
    )
    module = ModuleType("fake_lifecycle_target")
    published = ModuleType("fake_lifecycle_target")
    sys.modules[module.__name__] = published
    try:
        loader.exec_module(module)
    finally:
        sys.modules.pop(module.__name__, None)
    assert module.original is True
    assert events[0] == "original"
    assert events[1] == ("installed", published)
