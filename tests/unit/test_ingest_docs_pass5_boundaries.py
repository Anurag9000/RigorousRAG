from pathlib import Path
from unittest.mock import MagicMock

import pytest

import ingest_docs


def test_directory_traversal_has_a_bounded_entry_inspection_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest_docs, "_MAX_INPUT_FILES", 1)
    for index in range(21):
        (tmp_path / f"unsupported-{index}.bin").write_bytes(b"x")

    with pytest.raises(ValueError, match="bounded inspection limit"):
        list(ingest_docs._directory_files(tmp_path, recursive=False))


def test_collect_files_rejects_non_string_path_members(tmp_path):
    with pytest.raises(ValueError, match="paths must be a list of strings"):
        ingest_docs._collect_files(
            [str(tmp_path), object()],
            recursive=False,
            output_path=None,
        )


def test_prior_generation_is_bounded_at_backend_request():
    rag = MagicMock()
    rag.collection.get.return_value = {
        "ids": [],
        "documents": [],
        "metadatas": [],
    }

    generation = ingest_docs._capture_generation(rag, "alice", "doc-1")

    assert generation.ids == ()
    rag.collection.get.assert_called_once_with(
        where={
            "$and": [
                {"owner_id": {"$eq": "alice"}},
                {"doc_id": {"$eq": "doc-1"}},
            ]
        },
        include=["documents", "metadatas"],
        limit=ingest_docs._MAX_VECTOR_ROWS + 1,
    )


def test_prior_generation_rejects_oversized_rows(monkeypatch):
    monkeypatch.setattr(ingest_docs, "_MAX_VECTOR_TEXT_CHARS", 5)
    monkeypatch.setattr(ingest_docs, "_MAX_VECTOR_METADATA_ITEMS", 2)
    rag = MagicMock()
    rag.collection.get.return_value = {
        "ids": ["old-1"],
        "documents": ["evidence"],
        "metadatas": [{"owner_id": "alice", "doc_id": "doc-1"}],
    }

    with pytest.raises(RuntimeError, match="invalid rows"):
        ingest_docs._capture_generation(rag, "alice", "doc-1")

    rag.collection.get.return_value = {
        "ids": ["old-1"],
        "documents": ["ok"],
        "metadatas": [
            {"owner_id": "alice", "doc_id": "doc-1", "extra": "too many"}
        ],
    }
    with pytest.raises(RuntimeError, match="invalid rows"):
        ingest_docs._capture_generation(rag, "alice", "doc-1")
