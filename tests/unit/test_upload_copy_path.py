import os

import pytest

from tools.upload_storage import UploadStorageError, copy_path_to_owner


def test_copy_path_to_owner_refuses_symlinked_source(tmp_path):
    root = tmp_path / "uploads"
    root.mkdir()
    target = tmp_path / "target.txt"
    target.write_bytes(b"secret")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable in this environment.")

    with pytest.raises(UploadStorageError, match="symbolic links"):
        copy_path_to_owner(
            link,
            upload_root=root,
            owner_id="alice",
            max_bytes=100,
        )

    assert not (root / "alice").exists()


def test_copy_path_to_owner_enforces_opened_source_size(tmp_path):
    root = tmp_path / "uploads"
    root.mkdir()
    source = tmp_path / "source.txt"
    source.write_bytes(b"0123456789")

    with pytest.raises(UploadStorageError, match="exceeds"):
        copy_path_to_owner(
            source,
            upload_root=root,
            owner_id="alice",
            max_bytes=5,
        )

    assert not (root / "alice").exists()


def test_copy_path_to_owner_preserves_opened_inode_bytes(tmp_path):
    if os.name == "nt":
        pytest.skip("Atomic rename semantics differ on Windows.")
    root = tmp_path / "uploads"
    root.mkdir()
    source = tmp_path / "source.txt"
    source.write_bytes(b"original")

    retained = copy_path_to_owner(
        source,
        upload_root=root,
        owner_id="alice",
        max_bytes=100,
    )

    assert retained.read_bytes() == b"original"
