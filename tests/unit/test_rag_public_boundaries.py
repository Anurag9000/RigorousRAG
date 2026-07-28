import math
import threading

import pytest

import tools.rag as rag_module
from tools.rag import RAGLayer, get_rag_layer


class RecordingCollection:
    def __init__(self):
        self.upserts = []
        self.query_result = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    def get(self, **_kwargs):
        return {"ids": [], "documents": [], "metadatas": []}

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def delete(self, **_kwargs):
        return None

    def query(self, **_kwargs):
        return self.query_result


def _layer(collection=None):
    layer = RAGLayer.__new__(RAGLayer)
    layer.collection = collection or RecordingCollection()
    layer._write_lock = threading.RLock()
    return layer


def test_add_document_bounds_infinite_section_iterables():
    layer = _layer()

    def infinite_sections():
        while True:
            yield {"title": "Empty", "content": ""}

    with pytest.raises(ValueError, match="at most 10000 semantic sections"):
        layer.add_document(
            "doc-1",
            None,
            {"owner_id": "alice"},
            sections=infinite_sections(),
        )


def test_add_document_sanitizes_nonfinite_and_oversized_metadata_values():
    collection = RecordingCollection()
    layer = _layer(collection)

    count = layer.add_document(
        "doc-1",
        "bounded evidence",
        {
            "owner_id": "alice",
            "finite": 1.5,
            "nan": float("nan"),
            "infinity": float("inf"),
            "long": "x" * 5000,
            "nested": {"not": "stored"},
        },
    )

    assert count == 1
    metadata = collection.upserts[0]["metadatas"][0]
    assert metadata["owner_id"] == "alice"
    assert metadata["finite"] == 1.5
    assert "nan" not in metadata
    assert "infinity" not in metadata
    assert len(metadata["long"]) == 4000
    assert "nested" not in metadata


def test_query_discards_cross_owner_and_cross_document_rows():
    collection = RecordingCollection()
    collection.query_result = {
        "ids": [["good", "wrong-owner", "wrong-doc"]],
        "documents": [["good evidence", "secret", "other document"]],
        "metadatas": [[
            {"owner_id": "alice", "doc_id": "doc-1"},
            {"owner_id": "bob", "doc_id": "doc-1"},
            {"owner_id": "alice", "doc_id": "doc-2"},
        ]],
        "distances": [[0.1, 0.0, 0.0]],
    }
    layer = _layer(collection)

    results = layer.query(
        "question",
        owner_id="alice",
        doc_id="doc-1",
        n_results=10,
    )

    assert [chunk.id for chunk in results] == ["good"]
    assert results[0].text == "good evidence"


def test_query_normalizes_nonfinite_and_negative_distances():
    collection = RecordingCollection()
    collection.query_result = {
        "ids": [["nan", "negative", "finite"]],
        "documents": [["a", "b", "c"]],
        "metadatas": [[
            {"owner_id": "alice", "doc_id": "doc-1"},
            {"owner_id": "alice", "doc_id": "doc-1"},
            {"owner_id": "alice", "doc_id": "doc-1"},
        ]],
        "distances": [[float("nan"), -2.0, 0.25]],
    }
    layer = _layer(collection)

    results = layer.query("question", owner_id="alice", n_results=10)

    by_id = {chunk.id: chunk for chunk in results}
    assert by_id["finite"].distance == 0.25
    assert by_id["nan"].distance == 1.0
    assert by_id["negative"].distance == 1.0
    assert all(math.isfinite(chunk.score) for chunk in results)


def test_direct_query_and_document_identifiers_are_bounded():
    layer = _layer()

    with pytest.raises(ValueError, match="RAG queries"):
        layer.query("q" * 20_001, owner_id="alice")
    with pytest.raises(ValueError, match="doc_id"):
        layer.add_document("d" * 201, "text", {"owner_id": "alice"})
    with pytest.raises(ValueError, match="doc_id"):
        layer.delete_document(owner_id="alice", doc_id="d" * 201)


def test_symlinked_chroma_root_is_refused_before_initialization(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "vectors"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable in this environment.")

    with pytest.raises(ValueError, match="CHROMA_PATH"):
        RAGLayer(persist_directory=str(link))


def test_singleton_factory_refuses_symlinked_chroma_root(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "vectors"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable in this environment.")
    rag_module._RAG_INSTANCES.clear()

    with pytest.raises(ValueError, match="CHROMA_PATH"):
        get_rag_layer(str(link))
