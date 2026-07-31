import os
import sqlite3
from pathlib import Path

import tools.healthcheck as healthcheck


def test_healthcheck_paths_reject_all_ascii_controls(tmp_path):
    for path in ("bad\npath", "bad\rpath", "bad\tpath", "bad\x7fpath"):
        assert healthcheck.check_sqlite(path) is False
        assert healthcheck.check_writable_directory(path) is False


def test_sqlite_probe_handles_uri_metacharacters_in_filename(tmp_path):
    database = tmp_path / "state?mode=ro#fragment.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE sample(value INTEGER)")

    assert healthcheck.check_sqlite(database) is True


def test_reparse_flagged_database_and_directory_fail_closed(tmp_path, monkeypatch):
    database = tmp_path / "state.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE sample(value INTEGER)")
    directory = tmp_path / "storage"
    directory.mkdir()
    original_lstat = healthcheck.os.lstat

    class ReparseInfo:
        def __init__(self, info):
            self.st_mode = info.st_mode
            self.st_dev = info.st_dev
            self.st_ino = info.st_ino
            self.st_file_attributes = healthcheck._FILE_ATTRIBUTE_REPARSE_POINT

    targets = {database, directory}

    def fake_lstat(path):
        info = original_lstat(path)
        return ReparseInfo(info) if Path(path) in targets else info

    monkeypatch.setattr(healthcheck.os, "lstat", fake_lstat)

    assert healthcheck.check_sqlite(database) is False
    assert healthcheck.check_writable_directory(directory) is False


def test_posix_writable_probe_handles_short_writes_without_leaving_files(
    tmp_path,
    monkeypatch,
):
    if os.name == "nt":
        return
    directory = tmp_path / "storage"
    directory.mkdir()
    original_write = healthcheck.os.write

    def short_write(descriptor, payload):
        return original_write(descriptor, payload[:1])

    monkeypatch.setattr(healthcheck.os, "write", short_write)

    assert healthcheck.check_writable_directory(directory) is True
    assert list(directory.iterdir()) == []


def test_boolean_timeout_uses_bounded_default():
    assert healthcheck._finite_timeout(True) == 3.0
    assert healthcheck._finite_timeout(False) == 3.0
