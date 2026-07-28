import io
import os

import pytest

from tools.ingestion_snapshot import materialize_ingestion_snapshot
from tools.upload_storage import UploadStorageError, store_owner_stream


def test_snapshot_contains_exact_anchored_bytes_and_is_deleted(tmp_path):
    root = tmp_path / "uploads"
    root.mkdir()
    stored = store_owner_stream(
        io.BytesIO(b"original evidence"),
        upload_root=root,
        owner_id="alice",
        suffix=".txt",
        max_bytes=100,
    )

    with materialize_ingestion_snapshot(
        upload_root=root,
        source_path=stored,
        max_bytes=100,
    ) as (snapshot, payload):
        assert payload == b"original evidence"
        assert snapshot.read_bytes() == payload
        stored.write_bytes(b"mutated after snapshot")
        assert snapshot.read_bytes() == b"original evidence"
        if os.name != "nt":
            assert snapshot.stat().st_mode & 0o077 == 0
        snapshot_path = snapshot

    assert not snapshot_path.exists()
    assert stored.read_bytes() == b"mutated after snapshot"


def test_snapshot_refuses_owner_directory_symlink_swap(tmp_path):
    root = tmp_path / "uploads"
    root.mkdir()
    stored = store_owner_stream(
        io.BytesIO(b"inside"),
        upload_root=root,
        owner_id="alice",
        suffix=".txt",
        max_bytes=100,
    )
    owner = root / "alice"
    original_owner = root / "alice-original"
    owner.rename(original_owner)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / stored.name).write_bytes(b"outside")
    try:
        owner.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        original_owner.rename(owner)
        pytest.skip("Symlinks are unavailable in this environment.")

    with pytest.raises(UploadStorageError, match="missing|symlinked"):
        with materialize_ingestion_snapshot(
            upload_root=root,
            source_path=stored,
            max_bytes=100,
        ):
            pass

    assert (outside / stored.name).read_bytes() == b"outside"


def test_snapshot_enforces_byte_ceiling_before_parser_file_creation(tmp_path):
    root = tmp_path / "uploads"
    root.mkdir()
    stored = store_owner_stream(
        io.BytesIO(b"0123456789"),
        upload_root=root,
        owner_id="alice",
        suffix=".txt",
        max_bytes=100,
    )

    for invalid in (5, 0, True, 1.5, "bad"):
        with pytest.raises(UploadStorageError):
            with materialize_ingestion_snapshot(
                upload_root=root,
                source_path=stored,
                max_bytes=invalid,
            ):
                pass


def test_snapshot_rejects_non_path_source_before_tempfile_creation(tmp_path):
    root = tmp_path / "uploads"
    root.mkdir()

    with pytest.raises(UploadStorageError):
        with materialize_ingestion_snapshot(
            upload_root=root,
            source_path=object(),
            max_bytes=100,
        ):
            pass
