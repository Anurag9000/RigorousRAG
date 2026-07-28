import sqlite3

import pytest

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


def test_factory_rejects_symlinked_upload_root(tmp_path):
    target = tmp_path / "real-uploads"
    target.mkdir()
    link = tmp_path / "uploads"
    _symlink_or_skip(link, target, directory=True)

    with pytest.raises(ValueError, match="UPLOAD_DIR"):
        get_document_store(tmp_path / "documents.sqlite3", link)


def test_registry_ping_fails_closed_after_database_path_swap(tmp_path):
    database = tmp_path / "documents.sqlite3"
    store = DocumentStore(database, tmp_path / "uploads")
    replacement = tmp_path / "replacement.sqlite3"
    with sqlite3.connect(replacement) as connection:
        connection.execute("SELECT 1")
    database.unlink()
    _symlink_or_skip(database, replacement)

    assert store.ping() is False
