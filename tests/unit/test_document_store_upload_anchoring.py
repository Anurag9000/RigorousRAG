import os
from types import SimpleNamespace

import pytest

from tools.document_store import DocumentStore


def _store(monkeypatch, tmp_path):
    monkeypatch.setenv("ORPHAN_CLEANUP_ON_STARTUP", "false")
    root = tmp_path / "uploads"
    return DocumentStore(tmp_path / "documents.sqlite3", root), root


def test_copy_source_uses_random_owner_scoped_private_file(monkeypatch, tmp_path):
    store, root = _store(monkeypatch, tmp_path)
    source = tmp_path / "paper.txt"
    source.write_bytes(b"evidence")

    retained = store.copy_source(owner_id="alice", source_path=source)

    assert retained.parent == root.resolve() / "alice"
    assert retained.name != source.name
    assert retained.suffix == ".txt"
    assert retained.read_bytes() == b"evidence"
    if os.name != "nt":
        assert retained.stat().st_mode & 0o077 == 0
    assert store.remove_source(retained) is True
    assert not retained.exists()


def test_copy_source_refuses_symlinked_owner_directory(monkeypatch, tmp_path):
    store, root = _store(monkeypatch, tmp_path)
    source = tmp_path / "paper.txt"
    source.write_bytes(b"evidence")
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (root / "alice").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable in this environment.")

    with pytest.raises((OSError, ValueError)):
        store.copy_source(owner_id="alice", source_path=source)

    assert list(outside.iterdir()) == []


def test_orphan_cleanup_cannot_delete_through_swapped_owner_directory(
    monkeypatch,
    tmp_path,
):
    store, root = _store(monkeypatch, tmp_path)
    source = tmp_path / "paper.txt"
    source.write_bytes(b"inside")
    retained = store.copy_source(owner_id="alice", source_path=source)
    os.utime(retained, (1, 1))

    owner_dir = root / "alice"
    original_owner = root / "alice-original"
    owner_dir.rename(original_owner)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / retained.name
    outside_file.write_bytes(b"outside")
    try:
        owner_dir.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        original_owner.rename(owner_dir)
        pytest.skip("Symlinks are unavailable in this environment.")

    deleted = store.cleanup_orphans(
        now=10_000,
        job_store=SimpleNamespace(active_source_paths=lambda: set()),
    )

    assert deleted == 0
    assert outside_file.read_bytes() == b"outside"
    assert (original_owner / retained.name).read_bytes() == b"inside"
