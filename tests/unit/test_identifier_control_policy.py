import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

import tools.document_service as document_service
import tools.integrity as integrity
import tools.rag_tool as rag_tool
from tools.document_store import DocumentStore
from tools.ingestion_models import DocumentSection, IngestedDocument
from tools.job_store import JobStore
from tools.rag import RAGLayer


CONTROL_VALUES = ("bad\tvalue", "bad\nvalue", "bad\rvalue", "bad\x7fvalue")


def _document(tmp_path):
    source = tmp_path / "paper.txt"
    source.write_text("evidence", encoding="utf-8")
    return IngestedDocument(
        id="doc-1",
        filename="paper.txt",
        file_path=str(source),
        mime_type="text/plain",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        title="Paper",
        text="evidence",
        sections=[DocumentSection(title="Full Text", content="evidence")],
        metadata={},
    )


def _rag_layer(collection=None):
    layer = RAGLayer.__new__(RAGLayer)
    layer.collection = collection or MagicMock()
    return layer


@pytest.mark.parametrize("value", CONTROL_VALUES)
def test_registry_rejects_control_bearing_document_ids_and_paths(tmp_path, value):
    store = DocumentStore(tmp_path / "documents.sqlite3", tmp_path / "uploads")

    with pytest.raises(ValueError, match="doc_id"):
        store.get(owner_id="alice", doc_id=value)
    with pytest.raises(ValueError, match="doc_id"):
        store.register(
            owner_id="alice",
            doc_id=value,
            filename="paper.txt",
            mime_type="text/plain",
        )
    with pytest.raises(ValueError, match="filename"):
        store.register(
            owner_id="alice",
            doc_id="doc-1",
            filename=value,
            mime_type="text/plain",
        )

    with pytest.raises(ValueError, match="DOCUMENT_DB_PATH"):
        DocumentStore(tmp_path / value, tmp_path / "other-uploads")


@pytest.mark.parametrize("value", CONTROL_VALUES)
def test_job_store_rejects_control_bearing_job_document_and_source_ids(tmp_path, value):
    store = JobStore(tmp_path / "jobs.sqlite3")

    with pytest.raises(ValueError, match="job_id"):
        store.update(value, "alice", status="queued", filename="paper.txt")
    with pytest.raises(ValueError, match="doc_id"):
        store.update(
            "job-1",
            "alice",
            status="success",
            filename="paper.txt",
            doc_id=value,
        )
    with pytest.raises(ValueError, match="source_path"):
        store.update(
            "job-2",
            "alice",
            status="queued",
            filename="paper.txt",
            source_path=str(tmp_path / value),
        )


def test_recoverable_jobs_skip_corrupted_identifiers(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    now = 1.0
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            INSERT INTO jobs(
                job_id, owner_id, status, filename, message, doc_id,
                source_path, attempts, next_attempt_at, created_at, updated_at
            ) VALUES (?, ?, 'queued', 'paper.txt', NULL, NULL, NULL, 0, 0, ?, ?)
            """,
            ("bad\njob", "alice", now, now),
        )
        connection.execute(
            """
            INSERT INTO jobs(
                job_id, owner_id, status, filename, message, doc_id,
                source_path, attempts, next_attempt_at, created_at, updated_at
            ) VALUES (?, ?, 'finalizing', 'paper.txt', NULL, ?, NULL, 1, 0, ?, ?)
            """,
            ("valid-job", "alice", "bad\tdoc", now + 1, now + 1),
        )

    assert store.recoverable() == []


@pytest.mark.parametrize("value", CONTROL_VALUES)
def test_document_service_rejects_control_bearing_job_model_and_source_path(
    tmp_path,
    value,
):
    document = _document(tmp_path)
    rag = MagicMock()

    with pytest.raises(ValueError, match="job_id"):
        document_service.index_document(
            document,
            owner_id="alice",
            rag=rag,
            job_id=value,
        )
    with pytest.raises(ValueError, match="summary model"):
        document_service.summarize_document(
            document,
            client=MagicMock(),
            model=value,
        )
    with pytest.raises(ValueError, match="source path"):
        document_service._bounded_source_sha256(
            str(tmp_path / value),
            max_bytes=100,
        )

    rag.add_document.assert_not_called()


@pytest.mark.parametrize("value", CONTROL_VALUES)
def test_rag_rejects_control_bearing_ids_metadata_keys_models_and_paths(
    tmp_path,
    value,
):
    layer = _rag_layer()

    with pytest.raises(ValueError, match="doc_id"):
        layer.add_document(value, "evidence", {"owner_id": "alice"})
    with pytest.raises(ValueError, match="metadata keys"):
        layer.add_document(
            "doc-1",
            "evidence",
            {"owner_id": "alice", value: "unsafe"},
        )
    with pytest.raises(ValueError, match="doc_id"):
        layer.delete_document(owner_id="alice", doc_id=value)
    with pytest.raises(ValueError, match="persist_directory"):
        RAGLayer(persist_directory=str(tmp_path / value))
    with pytest.raises(ValueError, match="model"):
        RAGLayer(
            persist_directory=str(tmp_path / "vectors"),
            embedding_model=value,
        )


@pytest.mark.parametrize("value", CONTROL_VALUES)
def test_scientific_identifiers_reject_controls_but_prose_remains_multiline(value):
    with pytest.raises(ValueError, match="figure_id"):
        integrity.check_visual_entailment(
            "claim",
            value,
            "doc-1",
            owner_id="alice",
        )
    with pytest.raises(ValueError, match="doc_id"):
        integrity.extract_limitations(value, owner_id="alice")
    with pytest.raises(ValueError, match="model"):
        integrity.run_scientific_debate("claim", "context", model=value)
    with pytest.raises(ValueError, match="doc_ids item"):
        integrity.compare_papers(["doc-1", value], "query", owner_id="alice")

    assert "No evidence context" in integrity.run_scientific_debate(
        "claim\ncontinued",
        "",
    )


@pytest.mark.parametrize("value", CONTROL_VALUES)
def test_uploaded_document_adapter_rejects_control_bearing_ids_before_vector_init(
    monkeypatch,
    value,
):
    initializer = MagicMock(side_effect=AssertionError("must not initialize vectors"))
    monkeypatch.setattr(rag_tool, "get_rag_layer", initializer)

    with pytest.raises(ValueError, match="doc_id"):
        rag_tool.search_uploaded_docs("question", owner_id="alice", doc_id=value)
    with pytest.raises(ValueError, match="expansion_model"):
        rag_tool.search_uploaded_docs(
            "question",
            owner_id="alice",
            expansion_model=value,
        )

    initializer.assert_not_called()


def test_malformed_backend_chunk_and_document_identifiers_are_dropped():
    collection = MagicMock()
    collection.query.return_value = {
        "ids": [["bad\tchunk", "good-chunk"]],
        "documents": [["unsafe", "evidence"]],
        "metadatas": [[
            {"owner_id": "alice", "doc_id": "doc-1"},
            {"owner_id": "alice", "doc_id": "doc-1"},
        ]],
        "distances": [[0.0, 0.1]],
    }
    layer = _rag_layer(collection)

    chunks = layer.query("question", owner_id="alice", n_results=2)

    assert [chunk.id for chunk in chunks] == ["good-chunk"]

    collection.get.return_value = {
        "ids": ["one", "two"],
        "metadatas": [
            {"owner_id": "alice", "doc_id": "bad\x7fdoc"},
            {"owner_id": "alice", "doc_id": "good-doc"},
        ],
    }
    documents = layer.list_documents(owner_id="alice", limit=10)
    assert [document["doc_id"] for document in documents] == ["good-doc"]
