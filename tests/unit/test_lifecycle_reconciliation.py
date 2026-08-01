from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.lifecycle_outbox import LifecycleOutbox
from tools.lifecycle_reconciliation import (
    LifecycleCleanupJournal,
    clear_cleanup_runtime_cache,
    get_cleanup_journal,
    reconcile_claimed_operations,
    reconcile_lifecycle_operation,
)

HASH = "a" * 64


class Generations:
    def __init__(self, value):
        self.value = value

    def current(self, *, owner_id, doc_id):
        return self.value


class Registry:
    def __init__(self, root: Path):
        self.upload_root = root
        self.rows = {}
        self.fail_register = False
        self.fail_delete = False

    def get(self, *, owner_id, doc_id, **kwargs):
        return self.rows.get((owner_id, doc_id))

    def register(self, *, owner_id, doc_id, filename, mime_type, source_path):
        if self.fail_register:
            raise RuntimeError("private registry failure")
        prior = self.rows.get((owner_id, doc_id))
        self.rows[(owner_id, doc_id)] = {
            "source_path": source_path,
            "filename": filename,
            "mime_type": mime_type,
        }
        return (prior or {}).get("source_path")

    def delete(self, *, owner_id, doc_id):
        if self.fail_delete:
            raise RuntimeError("private registry failure")
        return self.rows.pop((owner_id, doc_id), None)


def generation(sequence=1, state="active", content=HASH):
    return SimpleNamespace(
        sequence=sequence,
        state=state,
        content_sha256=content,
    )


def plan_replace(outbox, new_path):
    return outbox.plan_replace(
        operation_id="replace-1",
        owner_id="alice",
        doc_id="doc-1",
        content_sha256=HASH,
        filename="paper.pdf",
        mime_type="application/pdf",
        source_path=str(new_path),
        retain_source=True,
        max_attempts=2,
        now=1.0,
    )


def test_cleanup_intent_is_recorded_before_registry_mutation(tmp_path):
    outbox = LifecycleOutbox(tmp_path / "outbox.sqlite3")
    cleanup = LifecycleCleanupJournal(tmp_path / "cleanup.sqlite3")
    root = tmp_path / "uploads"
    root.mkdir()
    old_path = root / "old.pdf"
    old_path.write_bytes(b"old")
    new_path = root / "new.pdf"
    new_path.write_bytes(b"new")
    plan_replace(outbox, new_path)
    registry = Registry(root)
    registry.rows[("alice", "doc-1")] = {"source_path": str(old_path)}

    class InspectingRegistry(Registry):
        def register(self, **kwargs):
            intent = cleanup.get("replace-1")
            assert intent is not None
            assert intent.source_path == str(old_path)
            return super().register(**kwargs)

    inspecting = InspectingRegistry(root)
    inspecting.rows = registry.rows
    result = reconcile_lifecycle_operation(
        "replace-1",
        outbox=outbox,
        generations=Generations(generation()),
        registry=inspecting,
        cleanup=cleanup,
        remove_source=lambda _path: False,
    )
    assert result.outcome == "cleanup_required"
    assert outbox.get("replace-1").state == "registry_committed"
    assert cleanup.get("replace-1").source_path == str(old_path)


def test_cleanup_replay_is_idempotent_after_file_already_removed(tmp_path):
    outbox = LifecycleOutbox(tmp_path / "outbox.sqlite3")
    cleanup = LifecycleCleanupJournal(tmp_path / "cleanup.sqlite3")
    old_path = tmp_path / "old.pdf"
    old_path.write_bytes(b"old")
    new_path = tmp_path / "new.pdf"
    new_path.write_bytes(b"new")
    plan_replace(outbox, new_path)
    registry = Registry(tmp_path)
    registry.rows[("alice", "doc-1")] = {"source_path": str(old_path)}
    first = reconcile_lifecycle_operation(
        "replace-1",
        outbox=outbox,
        generations=Generations(generation()),
        registry=registry,
        cleanup=cleanup,
        remove_source=lambda path: Path(path).unlink() is None,
    )
    assert first.outcome == "completed"
    assert not old_path.exists()
    assert cleanup.get("replace-1") is None
    assert outbox.get("replace-1").state == "completed"


def test_crash_after_remove_before_clear_can_finish_on_absent_file(tmp_path):
    outbox = LifecycleOutbox(tmp_path / "outbox.sqlite3")
    cleanup = LifecycleCleanupJournal(tmp_path / "cleanup.sqlite3")
    old_path = tmp_path / "old.pdf"
    old_path.write_bytes(b"old")
    new_path = tmp_path / "new.pdf"
    new_path.write_bytes(b"new")
    plan_replace(outbox, new_path)
    outbox.mark_index_committed("replace-1", generation_sequence=1)
    cleanup.record("replace-1", str(old_path))
    outbox.mark_registry_committed("replace-1")
    old_path.unlink()
    result = reconcile_lifecycle_operation(
        "replace-1",
        outbox=outbox,
        generations=Generations(generation()),
        registry=Registry(tmp_path),
        cleanup=cleanup,
        remove_source=lambda path: not Path(path).exists(),
    )
    assert result.outcome == "completed"
    assert cleanup.get("replace-1") is None


def test_registry_failure_preserves_cleanup_intent_and_generic_outbox_error(tmp_path):
    outbox = LifecycleOutbox(tmp_path / "outbox.sqlite3")
    cleanup = LifecycleCleanupJournal(tmp_path / "cleanup.sqlite3")
    old_path = tmp_path / "old.pdf"
    old_path.write_bytes(b"old")
    new_path = tmp_path / "new.pdf"
    new_path.write_bytes(b"new")
    plan_replace(outbox, new_path)
    registry = Registry(tmp_path)
    registry.rows[("alice", "doc-1")] = {"source_path": str(old_path)}
    registry.fail_register = True
    claimed = outbox.claim(worker_id="worker", now=2.0)
    results = reconcile_claimed_operations(
        claimed,
        outbox=outbox,
        generations=Generations(generation()),
        registry=registry,
        worker_id="worker",
        cleanup=cleanup,
    )
    assert results[0].outcome == "error"
    assert cleanup.get("replace-1").source_path == str(old_path)
    operation = outbox.get("replace-1")
    assert operation.last_error_type == "RuntimeError"
    assert "private registry failure" not in repr(operation)


def test_delete_cleanup_intent_precedes_registry_delete(tmp_path):
    outbox = LifecycleOutbox(tmp_path / "outbox.sqlite3")
    cleanup = LifecycleCleanupJournal(tmp_path / "cleanup.sqlite3")
    old_path = tmp_path / "old.pdf"
    old_path.write_bytes(b"old")
    outbox.plan_delete(
        operation_id="delete-1",
        owner_id="alice",
        doc_id="doc-1",
        now=1.0,
    )

    class InspectingRegistry(Registry):
        def delete(self, **kwargs):
            assert cleanup.get("delete-1").source_path == str(old_path)
            return super().delete(**kwargs)

    registry = InspectingRegistry(tmp_path)
    registry.rows[("alice", "doc-1")] = {"source_path": str(old_path)}
    result = reconcile_lifecycle_operation(
        "delete-1",
        outbox=outbox,
        generations=Generations(generation(2, state="deleted")),
        registry=registry,
        cleanup=cleanup,
        remove_source=lambda path: Path(path).unlink() is None,
    )
    assert result.outcome == "completed"
    assert not old_path.exists()
    assert registry.get(owner_id="alice", doc_id="doc-1") is None


def test_cleanup_journal_conflicts_identity_and_runtime_cache(tmp_path, monkeypatch):
    path = tmp_path / "cleanup.sqlite3"
    journal = LifecycleCleanupJournal(path)
    first = journal.record("op", "/private/one")
    assert journal.record("op", "/private/one").source_path == first.source_path
    with pytest.raises(ValueError, match="different cleanup path"):
        journal.record("op", "/private/two")

    path.rename(tmp_path / "old.sqlite3")
    path.write_bytes(b"replacement")
    assert journal.ping() is False
    with pytest.raises(RuntimeError, match="identity changed"):
        journal.get("op")

    clear_cleanup_runtime_cache()
    monkeypatch.setenv(
        "LIFECYCLE_CLEANUP_DB_PATH",
        str(tmp_path / "runtime.sqlite3"),
    )
    assert get_cleanup_journal() is get_cleanup_journal()
