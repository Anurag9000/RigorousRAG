import io
import os
import stat
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

import tools.upload_storage as upload_storage
from tools.upload_storage import (
    UploadStorageError,
    read_owner_file,
    remove_owner_file,
    store_owner_stream,
)


def test_store_owner_stream_writes_random_private_regular_file(tmp_path):
    root = tmp_path / "uploads"
    root.mkdir()

    stored = store_owner_stream(
        io.BytesIO(b"evidence"),
        upload_root=root,
        owner_id="alice",
        suffix=".txt",
        max_bytes=100,
    )

    assert stored.parent == root.absolute() / "alice"
    assert stored.name != "paper.txt"
    assert stored.suffix == ".txt"
    assert stored.read_bytes() == b"evidence"
    assert stat.S_ISREG(stored.stat().st_mode)
    if os.name != "nt":
        assert stat.S_IMODE(stored.stat().st_mode) & 0o077 == 0


def test_oversized_stream_removes_partial_owner_file(tmp_path):
    root = tmp_path / "uploads"
    root.mkdir()

    with pytest.raises(UploadStorageError, match="exceeds"):
        store_owner_stream(
            io.BytesIO(b"0123456789"),
            upload_root=root,
            owner_id="alice",
            suffix=".txt",
            max_bytes=5,
        )

    owner_dir = root / "alice"
    assert owner_dir.is_dir()
    assert list(owner_dir.iterdir()) == []


def test_invalid_limits_and_nonbyte_streams_fail_without_partial_file(tmp_path):
    root = tmp_path / "uploads"
    root.mkdir()
    for value in (
        0,
        True,
        1.5,
        Decimal("1.5"),
        Fraction(3, 2),
        "bad",
        1_000_000_001,
    ):
        with pytest.raises(UploadStorageError, match="max_bytes"):
            store_owner_stream(
                io.BytesIO(b"evidence"),
                upload_root=root,
                owner_id="alice",
                suffix=".txt",
                max_bytes=value,
            )

    with pytest.raises(UploadStorageError, match="produce bytes"):
        store_owner_stream(
            io.StringIO("not bytes"),
            upload_root=root,
            owner_id="alice",
            suffix=".txt",
            max_bytes=100,
        )
    assert list((root / "alice").iterdir()) == []


def test_exact_index_protocol_limit_is_accepted(tmp_path):
    class ExactInteger:
        def __index__(self):
            return 100

    root = tmp_path / "uploads"
    root.mkdir()

    stored = store_owner_stream(
        io.BytesIO(b"evidence"),
        upload_root=root,
        owner_id="alice",
        suffix=".txt",
        max_bytes=ExactInteger(),
    )

    assert stored.read_bytes() == b"evidence"


def test_store_refuses_symlinked_root_or_parent(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "uploads"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable in this environment.")

    for root in (link, link / "nested"):
        with pytest.raises(UploadStorageError, match="symbolic-link.*components"):
            store_owner_stream(
                io.BytesIO(b"secret"),
                upload_root=root,
                owner_id="alice",
                suffix=".txt",
                max_bytes=100,
            )
    assert list(outside.iterdir()) == []


def test_store_refuses_reparse_flagged_root(tmp_path, monkeypatch):
    root = tmp_path / "uploads"
    root.mkdir()
    original_lstat = upload_storage.os.lstat

    class ReparseInfo:
        def __init__(self, info):
            self.st_mode = info.st_mode
            self.st_dev = info.st_dev
            self.st_ino = info.st_ino
            self.st_file_attributes = upload_storage._FILE_ATTRIBUTE_REPARSE_POINT

    def fake_lstat(path):
        info = original_lstat(path)
        return ReparseInfo(info) if Path(path) == root else info

    monkeypatch.setattr(upload_storage.os, "lstat", fake_lstat)

    with pytest.raises(UploadStorageError, match="reparse-point"):
        store_owner_stream(
            io.BytesIO(b"secret"),
            upload_root=root,
            owner_id="alice",
            suffix=".txt",
            max_bytes=100,
        )


def test_store_refuses_symlinked_owner_directory(tmp_path):
    root = tmp_path / "uploads"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    try:
        (root / "alice").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable in this environment.")

    with pytest.raises((OSError, UploadStorageError)):
        store_owner_stream(
            io.BytesIO(b"secret"),
            upload_root=root,
            owner_id="alice",
            suffix=".txt",
            max_bytes=100,
        )

    assert list(outside.iterdir()) == []


def test_read_owner_file_rejects_invalid_limits_and_nonregular_entry(tmp_path):
    root = tmp_path / "uploads"
    root.mkdir()
    stored = store_owner_stream(
        io.BytesIO(b"evidence"),
        upload_root=root,
        owner_id="alice",
        suffix=".txt",
        max_bytes=100,
    )
    with pytest.raises(UploadStorageError, match="max_bytes"):
        read_owner_file(root, stored, max_bytes=float("nan"))

    if hasattr(os, "mkfifo"):
        fifo = root / "alice" / "pipe.txt"
        os.mkfifo(fifo)
        assert read_owner_file(root, fifo, max_bytes=100) is None


def test_remove_owner_file_is_descriptor_relative_and_owner_scoped(tmp_path):
    root = tmp_path / "uploads"
    root.mkdir()
    stored = store_owner_stream(
        io.BytesIO(b"evidence"),
        upload_root=root,
        owner_id="alice",
        suffix=".txt",
        max_bytes=100,
    )
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")

    assert remove_owner_file(root, outside) is False
    assert outside.exists()
    assert remove_owner_file(root, stored) is True
    assert not stored.exists()
    assert remove_owner_file(root, stored) is False


def test_remove_refuses_owner_directory_swapped_to_symlink(tmp_path):
    root = tmp_path / "uploads"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    stored = store_owner_stream(
        io.BytesIO(b"inside"),
        upload_root=root,
        owner_id="alice",
        suffix=".txt",
        max_bytes=100,
    )
    owner_dir = root / "alice"
    original_owner = root / "alice-original"
    owner_dir.rename(original_owner)
    outside_file = outside / stored.name
    outside_file.write_bytes(b"outside")
    try:
        owner_dir.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        original_owner.rename(owner_dir)
        pytest.skip("Symlinks are unavailable in this environment.")

    assert remove_owner_file(root, stored) is False
    assert outside_file.read_bytes() == b"outside"
    assert (original_owner / stored.name).read_bytes() == b"inside"
