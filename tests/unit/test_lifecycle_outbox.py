from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from tools.lifecycle_outbox import (
    LifecycleOutbox,
    operation_id_for,
    reconcile_claimed_operations,
    reconcile_lifecycle_operation,
)

HASH = "a" * 64


class Generations:
    def __init__(self, current=None):
        self.value = current

    def current(self, *, owner_id, doc_id):
        return self.value


class Registry:
    def __init__(self):
        self.rows = {}
        self.previous = None

    def register(self, *, owner_id, doc_id, filename, mime_type, source_path):
        prior = self.rows.get((owner_id, doc_id))
        self.rows[(owner_id, doc_id)] = {
            "owner_id": owner_id,
            "doc_id": doc_id,
            "filename": filename,
            "mime_type": mime_type,
            "source_path": source_path,
            "source_retained": bool(source_path),
        }
        return self.previous or ((prior or {}).get("source_path"))

    def get(self, *, owner_id, doc_id, **kwargs):
        return self.rows.get((owner_id, doc_id))

    def delete(self, *, owner_id, doc_id):
        return self.rows.pop((owner_id, doc_id), None)


def replacement(outbox, operation_id="replace-1", max_attempts=3):
    return outbox.plan_replace(
        operation_id=operation_id,
        owner_id="alice",
        doc_id="doc-1",
        content_sha256=HASH,
        filename="paper.pdf",
        mime_type="application/pdf",
        source_path="/private/alice/paper.pdf",
        retain_source=True,
        max_attempts=max_attempts,
        now=1.0,
    )


def test_operation_ids_are_deterministic_and_owner_scoped():
    first = operation_id_for(
        kind="replace",
        owner_id="alice",
        doc_id="doc-1",
        content_sha256=HASH,
        idempotency_key="job-1",
    )
    second = operation_id_for(
        kind="replace",
        owner_id="alice",
        doc_id="doc-1",
        content_sha256=HASH,
        idempotency_key="job-1",
    )
    changed = operation_id_for(
        kind="replace",
        owner_id="bob",
        doc_id="doc-1",
        content_sha256=HASH,
        idempotency_key="job-1",
    )
    assert first == second
    assert first != changed
    assert first.startswith("lifecycle-")


def test_plan_is_idempotent_and_conflicts_fail_closed(tmp_path):
    store = LifecycleOutbox(tmp_path / "outbox.sqlite3")
    first = replacement(store)
    second = replacement(store)
    assert first == second
    with pytest.raises(ValueError, match="different lifecycle"):
        store.plan_replace(
            operation_id="replace-1",
            owner_id="alice",
            doc_id="doc-1",
            content_sha256="b" * 64,
            filename="paper.pdf",
            mime_type="application/pdf",
            source_path="/private/alice/paper.pdf",
            retain_source=True,
        )
    summary = store.list_pending(owner_id="alice")[0]
    assert not hasattr(summary, "source_path")
    assert "/private" not in repr(summary)


def test_transitions_are_strict_and_idempotent(tmp_path):
    store = LifecycleOutbox(tmp_path / "outbox.sqlite3")
    replacement(store)
    indexed = store.mark_index_committed(
        "replace-1",
        generation_sequence=7,
        now=2.0,
    )
    assert indexed.state == "index_committed"
    assert store.mark_index_committed(
        "replace-1",
        generation_sequence=7,
    ) == indexed
    with pytest.raises(ValueError, match="conflicts"):
        store.mark_index_committed("replace-1", generation_sequence=8)
    registered = store.mark_registry_committed("replace-1", now=3.0)
    assert registered.state == "registry_committed"
    assert store.complete("replace-1", now=4.0).state == "completed"
    with pytest.raises(ValueError, match="cannot transition"):
        store.mark_registry_committed("replace-1")


def test_claim_lease_failure_retry_and_owner_isolation(tmp_path):
    store = LifecycleOutbox(tmp_path / "outbox.sqlite3")
    replacement(store, "a", max_attempts=2)
    store.plan_delete(
        operation_id="b",
        owner_id="bob",
        doc_id="doc-2",
        max_attempts=2,
        now=1.0,
    )
    claimed = store.claim(
        worker_id="worker",
        limit=2,
        lease_seconds=10,
        now=2.0,
    )
    assert {item.operation_id for item in claimed} == {"a", "b"}
    assert all(item.lease_owner == "worker" for item in claimed)
    renewed = store.renew(
        "a",
        worker_id="worker",
        lease_seconds=10,
        now=3.0,
    )
    assert renewed.lease_expires_at == 13.0
    with pytest.raises(ValueError, match="another worker"):
        store.record_failure(
            "a",
            worker_id="other",
            error_type="RuntimeError",
            now=4.0,
        )
    assert store.record_failure(
        "a",
        worker_id="worker",
        error_type="RuntimeError",
        now=4.0,
    ).attempts == 1
    claimed_again = store.claim(
        worker_id="worker",
        limit=2,
        lease_seconds=10,
        now=20.0,
    )
    target = next(item for item in claimed_again if item.operation_id == "a")
    failed = store.record_failure(
        target.operation_id,
        worker_id="worker",
        error_type="ValueError",
        now=21.0,
    )
    assert failed.state == "failed"
    assert failed.last_error_type == "ValueError"
    assert store.retry_failed("a", now=22.0).state == "planned"
    assert {item.operation_id for item in store.list_pending(owner_id="alice")} == {
        "a"
    }


def test_replace_reconciliation_waits_for_exact_generation_then_commits_registry(
    tmp_path,
):
    store = LifecycleOutbox(tmp_path / "outbox.sqlite3")
    replacement(store)
    generations = Generations(
        SimpleNamespace(
            state="active",
            content_sha256="b" * 64,
            sequence=1,
        )
    )
    registry = Registry()
    waiting = reconcile_lifecycle_operation(
        "replace-1",
        outbox=store,
        generations=generations,
        registry=registry,
    )
    assert waiting.outcome == "waiting_for_matching_generation"
    assert registry.rows == {}

    generations.value = SimpleNamespace(
        state="active",
        content_sha256=HASH,
        sequence=9,
    )
    registry.previous = "/private/alice/old.pdf"
    removed = []
    completed = reconcile_lifecycle_operation(
        "replace-1",
        outbox=store,
        generations=generations,
        registry=registry,
        remove_source=lambda path: removed.append(path) or True,
    )
    assert completed.state == "completed"
    assert completed.source_cleanup_required is None
    assert removed == ["/private/alice/old.pdf"]
    row = registry.get(owner_id="alice", doc_id="doc-1")
    assert row["source_path"] == "/private/alice/paper.pdf"
    assert store.get("replace-1").generation_sequence == 9


def test_delete_reconciliation_requires_deleted_generation_and_removes_source(
    tmp_path,
):
    store = LifecycleOutbox(tmp_path / "outbox.sqlite3")
    store.plan_delete(
        operation_id="delete-1",
        owner_id="alice",
        doc_id="doc-1",
        now=1.0,
    )
    registry = Registry()
    registry.rows[("alice", "doc-1")] = {
        "source_path": "/private/alice/paper.pdf"
    }
    generations = Generations(SimpleNamespace(state="active", sequence=4))
    waiting = reconcile_lifecycle_operation(
        "delete-1",
        outbox=store,
        generations=generations,
        registry=registry,
    )
    assert waiting.outcome == "waiting_for_deleted_generation"
    assert registry.get(owner_id="alice", doc_id="doc-1") is not None

    generations.value = SimpleNamespace(state="deleted", sequence=5)
    removed = []
    completed = reconcile_lifecycle_operation(
        "delete-1",
        outbox=store,
        generations=generations,
        registry=registry,
        remove_source=lambda path: removed.append(path) or True,
    )
    assert completed.state == "completed"
    assert removed == ["/private/alice/paper.pdf"]
    assert registry.get(owner_id="alice", doc_id="doc-1") is None
    assert store.get("delete-1").generation_sequence == 5


def test_reconcile_claimed_records_only_generic_failures(tmp_path):
    store = LifecycleOutbox(tmp_path / "outbox.sqlite3")
    replacement(store, max_attempts=2)
    claimed = store.claim(worker_id="worker", now=2.0)

    class BrokenRegistry(Registry):
        def register(self, **kwargs):
            raise RuntimeError("private database path and provider detail")

    generations = Generations(
        SimpleNamespace(
            state="active",
            content_sha256=HASH,
            sequence=1,
        )
    )
    result = reconcile_claimed_operations(
        claimed,
        outbox=store,
        generations=generations,
        registry=BrokenRegistry(),
        worker_id="worker",
    )
    assert result[0].outcome == "error"
    record = store.get("replace-1")
    assert record.last_error_type == "RuntimeError"
    assert "private database path" not in repr(record)
    assert b"private database path" not in store.path.read_bytes()


def test_database_identity_replacement_and_corruption_fail_closed(tmp_path):
    path = tmp_path / "outbox.sqlite3"
    store = LifecycleOutbox(path)
    replacement(store)
    path.rename(tmp_path / "old.sqlite3")
    path.write_bytes(b"replacement")
    assert store.ping() is False
    with pytest.raises(RuntimeError, match="identity changed"):
        store.list_pending()

    corrupt_path = tmp_path / "corrupt.sqlite3"
    corrupt = LifecycleOutbox(corrupt_path)
    replacement(corrupt)
    with sqlite3.connect(corrupt_path) as connection:
        connection.execute(
            "UPDATE lifecycle_operations SET state='invented' "
            "WHERE operation_id='replace-1'"
        )
        connection.commit()
    with pytest.raises(ValueError, match="state is invalid"):
        corrupt.get("replace-1")


def test_symlink_path_and_boolean_limits_are_rejected(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links unavailable")
    with pytest.raises(ValueError, match="redirects"):
        LifecycleOutbox(link / "outbox.sqlite3")

    store = LifecycleOutbox(tmp_path / "outbox.sqlite3")
    with pytest.raises(ValueError, match="max_attempts"):
        store.plan_delete(
            operation_id="delete",
            owner_id="alice",
            doc_id="doc",
            max_attempts=True,
        )
