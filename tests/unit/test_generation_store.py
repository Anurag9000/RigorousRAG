from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass

import pytest

from tools.generation_store import GenerationRecord, GenerationStore


@dataclass(frozen=True)
class Manifest:
    owner_id: str = "alice"
    doc_id: str = "doc-1"
    content_sha256: str = "a" * 64
    profile_fingerprint: str = "b" * 64
    vector_rows: int = 3
    sparse_generation: int = 1


def test_active_delete_restore_and_history_are_append_only(tmp_path):
    store = GenerationStore(tmp_path / "generations.sqlite3")
    active = store.record_active(
        Manifest(),
        metadata={"job_id": "j1"},
        committed_at=10.0,
    )
    assert active.sequence == 1 and active.state == "active"
    assert store.current(owner_id="alice", doc_id="doc-1") == active

    deleted = store.record_deleted(
        owner_id="alice",
        doc_id="doc-1",
        prior=active,
        expected_sequence=1,
        metadata={"reason": "user_delete"},
        committed_at=11.0,
    )
    assert deleted.sequence == 2 and deleted.state == "deleted"
    restored = store.restore_current(
        active,
        owner_id="alice",
        doc_id="doc-1",
        expected_sequence=2,
        committed_at=12.0,
    )
    assert restored is not None and restored.sequence == 3
    assert restored.state == "restored"
    assert restored.vector_rows == 3 and restored.sparse_generation == 1
    assert [
        item.sequence
        for item in store.history(owner_id="alice", doc_id="doc-1")
    ] == [3, 2, 1]


def test_optimistic_sequence_and_scope_validation(tmp_path):
    store = GenerationStore(tmp_path / "generations.sqlite3")
    store.record_active(Manifest())
    with pytest.raises(RuntimeError, match="concurrently"):
        store.record_active(Manifest(), expected_sequence=0)
    with pytest.raises(ValueError, match="scope"):
        store.restore_current(
            GenerationRecord(
                "bob",
                "doc-1",
                1,
                "active",
                "a" * 64,
                "b" * 64,
                1,
                1,
                1.0,
                {},
            ),
            owner_id="alice",
            doc_id="doc-1",
        )


def test_restore_absent_removes_only_current_pointer(tmp_path):
    store = GenerationStore(tmp_path / "generations.sqlite3")
    active = store.record_active(Manifest())
    store.restore_current(
        None,
        owner_id="alice",
        doc_id="doc-1",
        expected_sequence=active.sequence,
    )
    assert store.current(owner_id="alice", doc_id="doc-1") is None
    assert store.history(owner_id="alice", doc_id="doc-1")[0] == active


def test_corrupt_metadata_and_schema_fail_closed(tmp_path):
    path = tmp_path / "generations.sqlite3"
    store = GenerationStore(path)
    store.record_active(Manifest())
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE generation_history SET metadata_json = ? WHERE owner_id = ?",
            ('{"x": NaN}', "alice"),
        )
    with pytest.raises(RuntimeError, match="corrupt"):
        store.current(owner_id="alice", doc_id="doc-1")


def test_identity_replacement_and_symlink_path_are_rejected(tmp_path):
    path = tmp_path / "generations.sqlite3"
    store = GenerationStore(path)
    replacement = tmp_path / "replacement.sqlite3"
    replacement.write_bytes(path.read_bytes())
    os.replace(replacement, path)
    with pytest.raises(RuntimeError, match="identity changed"):
        store.current(owner_id="alice", doc_id="doc-1")

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="redirect"):
        GenerationStore(link / "generation.sqlite3")


def test_invalid_manifest_values_fail_before_write(tmp_path):
    store = GenerationStore(tmp_path / "generations.sqlite3")
    with pytest.raises(ValueError, match="active"):
        store.record_active(Manifest(vector_rows=0))
    assert store.list_current(owner_id="alice") == ()
