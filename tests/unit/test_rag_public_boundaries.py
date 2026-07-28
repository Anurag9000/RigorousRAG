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
        self.get_result = {"ids": [], "documents": [], "metadatas": []}

    def get(self, **_kwargs):
        return self.get_result

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


def test_add_document_rejects_string_sections_and_invalid_chunk_parameters():
    layer = _layer()
    with pytest.raises(ValueError, match="sections must be"):
        layer.add_document(
            "doc-1",
            None,
            {"owner_id": "alice"},
            sections="not-a-section-list",
        )
    with pytest.raises(ValueError, match="Document text"):
        layer.add_document("doc-1", object(), {"owner_id": "alice"})
    with pytest.raises(ValueError, match="chunk_size"):
        layer.add_document(
            "doc-1", "text", {"owner_id": "alice"}, chunk_size="bad"
        )
    with pytest.raises(ValueError, match="smaller"):
        layer.add_document(
            "doc-1", "text", {"owner_id": "alice"}, chunk_size=100, overlap=100
        )
    with pytest.raises(ValueError, match="boolean"):
        layer.add_document(
            "doc-1", "text", {"owner_id": "alice"}, replace="yes"
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


def test_query_normalizes_nonfinite_distances_and_masks_result_metadata():
    collection = RecordingCollection()
    collection.query_result = {
        "ids": [["nan", "negative", "finite"]],
        "documents": [["a", "b", "c"]],
        "metadatas": [[
            {"owner_id": "alice", "doc_id": "doc-1"},
            {"owner_id": "alice", "doc_id": "doc-1"},
            {
                "owner_id": "alice",
                "doc_id": "doc-1",
                "source": "https://user:password@example.test?api_key=secret",
            },
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
    assert "password" not in by_id["finite"].metadata["source"]
    assert "api_key=secret" not in by_id["finite"].metadata["source"]


def test_query_rejects_invalid_direct_parameters_and_filters():
    layer = _layer()
    with pytest.raises(ValueError, match="n_results"):
        layer.query("question", owner_id="alice", n_results="bad")
    with pytest.raises(ValueError, match="boolean"):
        layer.query("question", owner_id="alice", use_multi_query="yes")
    with pytest.raises(ValueError, match="unsupported filter"):
        layer.query("question", owner_id="alice", where={"score": float("nan")})
    with pytest.raises(ValueError, match="must be strings"):
        layer.query(object(), owner_id="alice")


def test_malformed_backend_query_and_listing_fail_closed():
    collection = RecordingCollection()
    layer = _layer(collection)
    collection.query_result = "not-an-object"
    with pytest.raises(RuntimeError, match="retrieval is unavailable"):
        layer.query("question", owner_id="alice")

    collection.get_result = "not-an-object"
    with pytest.raises(RuntimeError, match="listing is unavailable"):
        layer.list_documents(owner_id="alice")


def test_document_listing_filters_owner_and_bounds_public_fields():
    collection = RecordingCollection()
    collection.get_result = {
        "ids": ["a", "b"],
        "metadatas": [
            {
                "owner_id": "alice",
                "doc_id": "doc-1",
                "filename": "alice@example.com-paper.pdf",
                "created_at": "2026-01-01",
            },
            {"owner_id": "bob", "doc_id": "secret"},
        ],
    }
    layer = _layer(collection)

    documents = layer.list_documents(owner_id="alice", limit=10)

    assert [item["doc_id"] for item in documents] == ["doc-1"]
    assert "alice@example.com" not in documents[0]["filename"]
    with pytest.raises(ValueError, match="limit"):
        layer.list_documents(owner_id="alice", limit=0)


def test_direct_query_and_document_identifiers_are_bounded():
    layer = _layer()

    with pytest.raises(ValueError, match="RAG queries"):
        layer.query("q" * 20_001, owner_id="alice")
    with pytest.raises(ValueError, match="doc_id"):
        layer.add_document("d" * 201, "text", {"owner_id": "alice"})
    with pytest.raises(ValueError, match="doc_id"):
        layer.delete_document(owner_id="alice", doc_id="d" * 201)


def test_symlinked_chroma_root_and_parent_are_refused_before_initialization(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "vectors"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable in this environment.")

    with pytest.raises(ValueError, match="CHROMA_PATH"):
        RAGLayer(persist_directory=str(link))
    with pytest.raises(ValueError, match="CHROMA_PATH"):
        RAGLayer(persist_directory=str(link / "nested"))


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
