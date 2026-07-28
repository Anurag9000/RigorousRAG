import io
import os
import stat

import pytest

from tools.upload_storage import (
    UploadStorageError,
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

    assert stored.parent == root.resolve() / "alice"
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
