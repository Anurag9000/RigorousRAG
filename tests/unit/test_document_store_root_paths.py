import sqlite3

import pytest

import tools.document_store as document_store_module
from tools.document_store import DocumentStore, get_document_store


def _symlink_or_skip(link, target, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable in this environment.")


def test_direct_store_rejects_symlinked_database_path(tmp_path):
    target = tmp_path / "target.sqlite3"
    with sqlite3.connect(target) as connection:
        connection.execute("SELECT 1")
    link = tmp_path / "documents.sqlite3"
    _symlink_or_skip(link, target)

    with pytest.raises(ValueError, match="DOCUMENT_DB_PATH"):
        DocumentStore(link, tmp_path / "uploads")


def test_direct_store_rejects_symlinked_database_parent(tmp_path):
    target = tmp_path / "real-data"
    target.mkdir()
    link = tmp_path / "data"
    _symlink_or_skip(link, target, directory=True)

    with pytest.raises(ValueError, match="DOCUMENT_DB_PATH"):
        DocumentStore(link / "documents.sqlite3", tmp_path / "uploads")


def test_factory_rejects_symlinked_upload_root_and_parent(tmp_path):
    target = tmp_path / "real-uploads"
    target.mkdir()
    link = tmp_path / "uploads"
    _symlink_or_skip(link, target, directory=True)

    with pytest.raises(ValueError, match="UPLOAD_DIR"):
        get_document_store(tmp_path / "documents.sqlite3", link)
    with pytest.raises(ValueError, match="UPLOAD_DIR"):
        get_document_store(tmp_path / "documents.sqlite3", link / "nested")


def test_registry_ping_fails_closed_after_database_path_swap(tmp_path):
    database = tmp_path / "documents.sqlite3"
    store = DocumentStore(database, tmp_path / "uploads")
    replacement = tmp_path / "replacement.sqlite3"
    with sqlite3.connect(replacement) as connection:
        connection.execute("SELECT 1")
    database.unlink()
    _symlink_or_skip(database, replacement)

    assert store.ping() is False


def test_cached_factory_revalidates_paths_before_reuse(tmp_path):
    document_store_module._DOCUMENT_STORES.clear()
    parent = tmp_path / "state"
    database = parent / "documents.sqlite3"
    uploads = tmp_path / "uploads"
    store = get_document_store(database, uploads)
    assert store.ping() is True

    moved = tmp_path / "state-moved"
    parent.rename(moved)
    _symlink_or_skip(parent, moved, directory=True)

    with pytest.raises(ValueError, match="DOCUMENT_DB_PATH"):
        get_document_store(database, uploads)


def test_visual_flag_cleanup_clock_and_identifiers_are_strict(tmp_path):
    store = DocumentStore(tmp_path / "documents.sqlite3", tmp_path / "uploads")
    with pytest.raises(ValueError, match="verify_visual"):
        store.get(owner_id="alice", doc_id="doc-1", verify_visual="yes")
    with pytest.raises(ValueError, match="doc_id"):
        store.get(owner_id="alice", doc_id="")
    with pytest.raises(ValueError, match="finite"):
        store.cleanup_orphans(now=float("nan"), job_store=object())
