from __future__ import annotations

import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools import lifecycle_boundary
from tools.lifecycle_outbox import LifecycleOutbox
from tools.lifecycle_reconciliation import LifecycleCleanupJournal


class Generations:
    def __init__(self):
        self.value = SimpleNamespace(sequence=5, state="active")

    def current(self, *, owner_id, doc_id):
        return self.value


class Registry:
    def __init__(self, root: Path, source: Path):
        self.upload_root = root
        self.rows = {
            ("alice", "doc-1"): {
                "source_path": str(source),
                "updated_at": 2.0,
            }
        }
        self.fail_delete = True

    def get(self, *, owner_id, doc_id, **kwargs):
        return self.rows.get((owner_id, doc_id))

    def delete(self, *, owner_id, doc_id):
        if self.fail_delete:
            raise RuntimeError("registry unavailable")
        return self.rows.pop((owner_id, doc_id), None)

    def remove_source(self, source_path):
        path = Path(source_path)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True


def test_retry_reuses_pending_delete_after_generation_sequence_changes(
    tmp_path, monkeypatch
):
    root = tmp_path / "uploads"
    owner = root / "alice"
    owner.mkdir(parents=True)
    source = owner / "paper.pdf"
    source.write_bytes(b"pdf")
    registry = Registry(root, source)
    outbox = LifecycleOutbox(tmp_path / "outbox.sqlite3")
    cleanup = LifecycleCleanupJournal(tmp_path / "cleanup.sqlite3")
    generations = Generations()
    calls = []
    lock = threading.RLock()
    module = ModuleType("fake_authoritative_delete")

    def original_delete(**kwargs):
        calls.append("delete")
        generations.value = SimpleNamespace(sequence=6, state="deleted")
        return True

    module.commit_finalized_document = lambda *args, **kwargs: None
    module.delete_authoritative_document = original_delete
    module._identifier = lambda value, label: str(value)
    module._document_lock = lambda owner_id, doc_id: lock
    module.get_authoritative_index_coordinator = lambda rag: SimpleNamespace(
        generations=generations
    )
    monkeypatch.setattr(lifecycle_boundary, "get_lifecycle_outbox", lambda: outbox)
    monkeypatch.setattr(lifecycle_boundary, "get_cleanup_journal", lambda: cleanup)
    monkeypatch.setattr(lifecycle_boundary, "_document_store", lambda: registry)
    lifecycle_boundary.install_authoritative_lifecycle_boundary(module)

    with pytest.raises(RuntimeError, match="registry unavailable"):
        module.delete_authoritative_document(
            owner_id="alice",
            doc_id="doc-1",
            rag=object(),
        )
    pending = outbox.list_pending(owner_id="alice")
    assert len(pending) == 1
    original_operation_id = pending[0].operation_id
    assert pending[0].state == "index_committed"
    assert calls == ["delete"]

    registry.fail_delete = False
    assert module.delete_authoritative_document(
        owner_id="alice",
        doc_id="doc-1",
        rag=object(),
    ) is True
    assert calls == ["delete"]
    assert outbox.list_pending(owner_id="alice") == ()
    assert outbox.get(original_operation_id).state == "completed"
    assert not source.exists()
