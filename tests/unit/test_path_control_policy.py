import io

import pytest

import ingest_docs
import tools.internal_search as internal_search
from storage import StorageManager
from tools.ingestion import ingest_file
from tools.upload_storage import (
    UploadStorageError,
    copy_path_to_owner,
    read_owner_file,
    remove_owner_file,
    store_owner_stream,
    validated_owner_file_path,
)


CONTROL_VALUES = ("\t", "\n", "\r", "\x7f")


@pytest.mark.parametrize("control", CONTROL_VALUES)
def test_upload_root_controls_are_rejected_without_creation(tmp_path, control):
    root = tmp_path / f"uploads{control}unsafe"

    with pytest.raises(UploadStorageError, match="UPLOAD_DIR"):
        store_owner_stream(
            io.BytesIO(b"evidence"),
            upload_root=root,
            owner_id="alice",
            suffix=".txt",
            max_bytes=100,
        )

    assert not root.exists()


@pytest.mark.parametrize("control", CONTROL_VALUES)
def test_owner_source_controls_fail_closed(tmp_path, control):
    root = tmp_path / "uploads"
    root.mkdir()
    unsafe = root / "alice" / f"paper{control}unsafe.txt"

    assert validated_owner_file_path(root, unsafe) is None
    assert read_owner_file(root, unsafe, max_bytes=100) is None
    assert remove_owner_file(root, unsafe) is False


@pytest.mark.parametrize("control", CONTROL_VALUES)
def test_copy_source_control_path_is_rejected_before_read(tmp_path, control):
    root = tmp_path / "uploads"
    root.mkdir()
    unsafe = tmp_path / f"paper{control}unsafe.txt"

    with pytest.raises(UploadStorageError, match="source_path"):
        copy_path_to_owner(
            unsafe,
            upload_root=root,
            owner_id="alice",
            max_bytes=100,
        )

    assert list(root.iterdir()) == []


@pytest.mark.parametrize("control", CONTROL_VALUES)
def test_standalone_ingestion_control_path_returns_safe_failure(tmp_path, control):
    unsafe = tmp_path / f"paper{control}unsafe.txt"

    result = ingest_file(unsafe, owner_id="alice")

    assert result.success is False
    assert result.document is None
    assert "Input validation failed" in result.error
    assert not unsafe.exists()


@pytest.mark.parametrize("control", CONTROL_VALUES)
def test_classic_storage_control_root_is_rejected_before_creation(tmp_path, control):
    root = tmp_path / f"classic{control}unsafe"

    with pytest.raises(ValueError, match="CLASSIC_STORAGE_DIR"):
        StorageManager(root)

    assert not root.exists()


@pytest.mark.parametrize("control", CONTROL_VALUES)
def test_internal_search_control_storage_path_is_rejected(tmp_path, control):
    root = tmp_path / f"classic{control}unsafe"

    with pytest.raises(ValueError, match="CLASSIC_STORAGE_DIR"):
        internal_search._storage_signature(root)

    assert not root.exists()


@pytest.mark.parametrize("control", CONTROL_VALUES)
def test_batch_paths_and_vector_ids_reject_controls(tmp_path, control):
    unsafe = tmp_path / f"paper{control}unsafe.txt"
    with pytest.raises(ValueError, match="path"):
        ingest_docs._lexical_absolute(unsafe)

    rag = type(
        "Rag",
        (),
        {
            "collection": type(
                "Collection",
                (),
                {
                    "get": lambda self, **_kwargs: {
                        "ids": [f"vector{control}unsafe"],
                        "documents": ["evidence"],
                        "metadatas": [
                            {"owner_id": "alice", "doc_id": "doc-1"}
                        ],
                    }
                },
            )()
        },
    )()

    with pytest.raises(RuntimeError, match="invalid ID"):
        ingest_docs._capture_generation(rag, "alice", "doc-1")
