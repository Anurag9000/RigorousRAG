import importlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools.ingestion_models import DocumentSection, IngestedDocument, IngestionResult


@pytest.fixture
def server_module(monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEY_OWNERS_JSON", json.dumps({"alice-key": "alice"}))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("JOB_DB_PATH", str(tmp_path / "jobs.sqlite3"))
    monkeypatch.setenv("DOCUMENT_DB_PATH", str(tmp_path / "documents.sqlite3"))
    monkeypatch.setenv("ORPHAN_CLEANUP_ON_STARTUP", "false")
    monkeypatch.setenv("REQUESTS_PER_MINUTE", "1000")
    monkeypatch.setenv("ALLOWED_MODELS", "test-model")
    monkeypatch.setenv("DEFAULT_MODEL", "test-model")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    for name in ("server", "server_app"):
        sys.modules.pop(name, None)
    module = importlib.import_module("server")
    yield module
    module._cancel_scheduled_ingestions()
    module._INGEST_EXECUTOR.shutdown(wait=False, cancel_futures=True)
    module._QUERY_EXECUTOR.shutdown(wait=False, cancel_futures=True)
    for name in ("server", "server_app"):
        sys.modules.pop(name, None)


def _owner_source(server_module, content=b"queued bytes") -> Path:
    source = server_module.UPLOAD_DIR / "alice" / "stored.txt"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(content)
    return source


def _document(snapshot: Path, *, doc_id: str = "doc-1") -> IngestedDocument:
    return IngestedDocument(
        id=doc_id,
        filename="paper.txt",
        file_path=str(snapshot),
        mime_type="text/plain",
        text="original queued bytes",
        sections=[
            DocumentSection(title="Full Text", content="original queued bytes")
        ],
    )


def test_worker_passes_private_snapshot_path_to_parser(server_module, monkeypatch, tmp_path):
    source = _owner_source(server_module)
    snapshot = tmp_path / "snapshot.txt"
    snapshot.write_bytes(b"immutable queued bytes")

    @contextmanager
    def fake_snapshot(**kwargs):
        assert kwargs["source_path"] == source
        yield snapshot, snapshot.read_bytes()

    parser = MagicMock(
        return_value=IngestionResult(success=False, error="parse failed")
    )
    failed = MagicMock(return_value=True)
    monkeypatch.setattr(server_module, "materialize_ingestion_snapshot", fake_snapshot)
    monkeypatch.setattr(server_module, "ingest_file", parser)
    monkeypatch.setattr(server_module, "_persist_failed_job", failed)
    monkeypatch.setattr(server_module._JOB_STORE, "claim", lambda *_args, **_kwargs: True)

    server_module.process_ingestion(
        str(source),
        "paper.txt",
        "job-1",
        "alice",
    )

    parser.assert_called_once_with(str(snapshot), owner_id="alice")
    assert parser.call_args.args[0] != str(source)
    assert failed.call_args.args[4] == "parse failed"


def test_mutated_queued_source_is_rejected_before_vector_publication(
    server_module,
    monkeypatch,
    tmp_path,
):
    source = _owner_source(server_module, b"original queued bytes")
    snapshot = tmp_path / "snapshot.txt"
    snapshot.write_bytes(source.read_bytes())
    document = _document(snapshot)

    @contextmanager
    def fake_snapshot(**_kwargs):
        yield snapshot, snapshot.read_bytes()

    def parse_then_mutate(*_args, **_kwargs):
        source.write_bytes(b"mutated queued bytes!")
        return IngestionResult(success=True, document=document)

    index_document = MagicMock()
    monkeypatch.setattr(server_module, "materialize_ingestion_snapshot", fake_snapshot)
    monkeypatch.setattr(server_module, "ingest_file", parse_then_mutate)
    monkeypatch.setattr(server_module, "index_document", index_document)
    monkeypatch.setattr(server_module._JOB_STORE, "claim", lambda *_args, **_kwargs: True)

    server_module.process_ingestion(
        str(source),
        "paper.txt",
        "job-mutated",
        "alice",
    )

    index_document.assert_not_called()
    status = server_module._JOB_STORE.get("job-mutated", "alice")
    assert status and status["status"] == "failed"
    assert "UploadStorageError" in status["message"]
    assert not source.exists()


def test_preexisting_registry_does_not_mask_current_vector_failure(
    server_module,
    monkeypatch,
    tmp_path,
):
    source = _owner_source(server_module, b"original queued bytes")
    snapshot = tmp_path / "snapshot.txt"
    snapshot.write_bytes(source.read_bytes())
    document = _document(snapshot, doc_id="existing-doc")

    @contextmanager
    def fake_snapshot(**_kwargs):
        yield snapshot, snapshot.read_bytes()

    submissions = []
    monkeypatch.setattr(server_module, "materialize_ingestion_snapshot", fake_snapshot)
    monkeypatch.setattr(
        server_module,
        "ingest_file",
        lambda *_args, **_kwargs: IngestionResult(success=True, document=document),
    )
    monkeypatch.setattr(
        server_module,
        "index_document",
        MagicMock(side_effect=RuntimeError("vector write failed")),
    )
    monkeypatch.setattr(
        server_module._DOCUMENT_STORE,
        "get",
        lambda **_kwargs: {
            "doc_id": "existing-doc",
            "source_path": str(source),
            "source_retained": 1,
        },
    )
    monkeypatch.setattr(server_module._JOB_STORE, "claim", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        server_module,
        "_submit_ingestion",
        lambda *args: submissions.append(args),
    )

    server_module.process_ingestion(
        str(source),
        "paper.txt",
        "job-vector-failed",
        "alice",
    )

    status = server_module._JOB_STORE.get("job-vector-failed", "alice")
    assert status and status["status"] == "queued"
    assert status["doc_id"] is None
    assert "RuntimeError" in status["message"]
    assert submissions == [
        (str(source), "paper.txt", "job-vector-failed", "alice")
    ]
    assert source.exists()


def test_unexpected_snapshot_failure_returns_job_to_durable_queue(
    server_module,
    monkeypatch,
):
    source = _owner_source(server_module)
    updates = []
    submissions = []
    monkeypatch.setattr(server_module._JOB_STORE, "claim", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        server_module,
        "materialize_ingestion_snapshot",
        MagicMock(side_effect=OSError("temporary volume unavailable")),
    )
    monkeypatch.setattr(
        server_module._JOB_STORE,
        "get_internal",
        lambda *_args, **_kwargs: {"attempts": 1},
    )
    monkeypatch.setattr(
        server_module._JOB_STORE,
        "update",
        lambda *args, **kwargs: updates.append((args, kwargs)),
    )
    monkeypatch.setattr(
        server_module,
        "_submit_ingestion",
        lambda *args: submissions.append(args),
    )

    server_module.process_ingestion(
        str(source),
        "paper.txt",
        "job-1",
        "alice",
    )

    assert updates
    assert updates[-1][1]["status"] == "queued"
    assert "snapshot failure" in updates[-1][1]["message"].lower()
    assert submissions == [(str(source), "paper.txt", "job-1", "alice")]


def test_snapshot_failure_after_attempt_limit_is_terminal(server_module, monkeypatch):
    source = _owner_source(server_module)
    failed = MagicMock(return_value=True)
    monkeypatch.setattr(server_module._JOB_STORE, "claim", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        server_module,
        "materialize_ingestion_snapshot",
        MagicMock(side_effect=OSError("temporary volume unavailable")),
    )
    monkeypatch.setattr(
        server_module._JOB_STORE,
        "get_internal",
        lambda *_args, **_kwargs: {"attempts": server_module.INGEST_MAX_ATTEMPTS},
    )
    monkeypatch.setattr(server_module, "_persist_failed_job", failed)

    server_module.process_ingestion(
        str(source),
        "paper.txt",
        "job-1",
        "alice",
    )

    failed.assert_called_once()
    assert "snapshot failed" in failed.call_args.args[4].lower()
