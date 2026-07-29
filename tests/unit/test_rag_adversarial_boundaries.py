import threading
from unittest.mock import patch

import pytest

from tools.rag import RAGLayer


class Collection:
    def __init__(self):
        self.query_result = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        self.get_result = {"ids": [], "metadatas": []}
        self.upserts = []

    def query(self, **_kwargs):
        return self.query_result

    def get(self, **_kwargs):
        return self.get_result

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def delete(self, **_kwargs):
        return None


def _layer(collection=None):
    layer = RAGLayer.__new__(RAGLayer)
    layer.collection = collection or Collection()
    layer._write_lock = threading.RLock()
    return layer


def test_hostile_metadata_mapping_fails_before_vector_write():
    class BrokenMetadata(dict):
        def items(self):
            raise RuntimeError("private metadata detail")

    layer = _layer()
    with pytest.raises(ValueError, match="safely iterable"):
        layer.add_document("doc-1", "evidence", BrokenMetadata(owner_id="alice"))


def test_hostile_section_iterator_and_model_dump_fail_safely():
    class BrokenSections:
        def __iter__(self):
            raise RuntimeError("private section iterator")

    class BrokenSection:
        def model_dump(self):
            raise RuntimeError("private serialization detail")

    layer = _layer()
    with pytest.raises(ValueError, match="safely iterable"):
        layer.add_document(
            "doc-1",
            None,
            {"owner_id": "alice"},
            sections=BrokenSections(),
        )
    with pytest.raises(ValueError, match="could not be serialized"):
        layer.add_document(
            "doc-1",
            None,
            {"owner_id": "alice"},
            sections=[BrokenSection()],
        )


def test_metadata_owner_must_be_a_string():
    layer = _layer()
    with pytest.raises(ValueError, match="metadata.owner_id"):
        layer.add_document("doc-1", "evidence", {"owner_id": 123})


def test_query_rejects_nul_and_malformed_nested_rows():
    collection = Collection()
    layer = _layer(collection)
    with pytest.raises(ValueError, match="null character"):
        layer.query("question\x00hidden", owner_id="alice")

    collection.query_result = {
        "ids": [[], []],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    with pytest.raises(RuntimeError, match="retrieval is unavailable"):
        layer.query("question", owner_id="alice")


def test_boolean_distance_is_not_treated_as_zero():
    collection = Collection()
    collection.query_result = {
        "ids": [["chunk"]],
        "documents": [["evidence"]],
        "metadatas": [[{"owner_id": "alice", "doc_id": "doc-1"}]],
        "distances": [[False]],
    }
    layer = _layer(collection)

    result = layer.query("question", owner_id="alice")

    assert result[0].distance == 1.0


def test_listing_requires_parallel_identifier_and_metadata_arrays():
    collection = Collection()
    collection.get_result = {
        "ids": ["one", "two"],
        "metadatas": [{"owner_id": "alice", "doc_id": "doc-1"}],
    }
    layer = _layer(collection)

    with pytest.raises(RuntimeError, match="listing is unavailable"):
        layer.list_documents(owner_id="alice")


def test_provider_query_generators_fall_back_without_truthiness_or_iteration_leak():
    class BrokenGenerated:
        def __bool__(self):
            raise RuntimeError("truthiness must not be used")

        def __iter__(self):
            raise RuntimeError("private provider iterator")

    layer = _layer()
    base_class = RAGLayer.__mro__[1]
    with patch.object(
        base_class,
        "generate_expanded_queries",
        return_value=BrokenGenerated(),
    ):
        assert layer.generate_expanded_queries(
            "question",
            agent_client=object(),
        ) == ["question"]


def test_non_string_hyde_response_falls_back_to_original_query():
    layer = _layer()
    base_class = RAGLayer.__mro__[1]
    with patch.object(
        base_class,
        "generate_hyde_query",
        return_value=object(),
    ):
        assert layer.generate_hyde_query(
            "question",
            agent_client=object(),
        ) == "question"
