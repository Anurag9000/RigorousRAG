import importlib
import io
import json
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tools.ingestion_models import DocumentSection, IngestedDocument, IngestionResult


@pytest.fixture
def server_module(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "API_KEY_OWNERS_JSON",
        json.dumps({"alice-key": "alice"}),
    )
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("JOB_DB_PATH", str(tmp_path / "jobs.sqlite3"))
    monkeypatch.setenv("DOCUMENT_DB_PATH", str(tmp_path / "documents.sqlite3"))
    monkeypatch.setenv("ORPHAN_CLEANUP_ON_STARTUP", "false")
    monkeypatch.setenv("REQUESTS_PER_MINUTE", "1000")
    monkeypatch.setenv("ALLOWED_MODELS", "test-model")
    monkeypatch.setenv("DEFAULT_MODEL", "test-model")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    sys.modules.pop("server", None)
    module = importlib.import_module("server")
    yield module
    module._cancel_scheduled_ingestions()
    module._INGEST_EXECUTOR.shutdown(wait=False, cancel_futures=True)
    sys.modules.pop("server", None)


def test_request_id_is_echoed_only_when_header_safe(server_module):
    with TestClient(server_module.app) as client:
        safe = client.get("/health", headers={"X-Request-ID": "request_123-safe"})
        unsafe = client.get("/health", headers={"X-Request-ID": "bad value"})

    assert safe.headers["X-Request-ID"] == "request_123-safe"
    assert unsafe.headers["X-Request-ID"] != "bad value"
    assert len(unsafe.headers["X-Request-ID"]) == 32


def test_model_and_path_identifiers_are_bounded(server_module):
    with TestClient(server_module.app) as client:
        oversized_model = client.post(
            "/query",
            headers={"X-API-Key": "alice-key"},
            json={"query": "q", "model": "m" * 201},
        )
        oversized_job = client.get(
            f"/status/{'j' * 201}",
            headers={"X-API-Key": "alice-key"},
        )
        oversized_document = client.delete(
            f"/docs/{'d' * 201}",
            headers={"X-API-Key": "alice-key"},
        )

    assert oversized_model.status_code == 422
    assert oversized_job.status_code == 400
    assert oversized_document.status_code == 400


def test_upload_is_fsynced_before_job_submission(server_module, monkeypatch):
    fsync_calls = []
    submitted = []
    monkeypatch.setattr(server_module.os, "fsync", lambda descriptor: fsync_calls.append(descriptor))
    monkeypatch.setattr(
        server_module,
        "_submit_ingestion",
        lambda *args: submitted.append(args),
    )

    with TestClient(server_module.app) as client:
        response = client.post(
            "/ingest",
            headers={"X-API-Key": "alice-key"},
            files={"file": ("paper.txt", io.BytesIO(b"evidence"), "text/plain")},
        )

    assert response.status_code == 200
    assert fsync_calls
    assert len(submitted) == 1


def test_internal_indexing_error_is_redacted_from_public_job_status(
    server_module,
    monkeypatch,
):
    source = server_module.UPLOAD_DIR / "alice" / "paper.txt"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("evidence", encoding="utf-8")
    server_module._JOB_STORE.update(
        "job-1",
        "alice",
        status="queued",
        filename="paper.txt",
        source_path=source,
    )
    document = IngestedDocument(
        id="doc-1",
        filename="paper.txt",
        file_path=str(source),
        mime_type="text/plain",
        text="evidence",
        sections=[DocumentSection(title="Full Text", content="evidence")],
    )
    monkeypatch.setattr(server_module, "INGEST_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(
        server_module,
        "ingest_file",
        lambda *_args, **_kwargs: IngestionResult(success=True, document=document),
    )
    monkeypatch.setattr(
        server_module,
        "_new_agent",
        lambda *_args, **_kwargs: SimpleNamespace(client=None),
    )
    monkeypatch.setattr(
        server_module,
        "index_document",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("provider secret at /private/alice/state.sqlite3")
        ),
    )

    server_module.process_ingestion(str(source), "paper.txt", "job-1", "alice")
    status = server_module._JOB_STORE.get("job-1", "alice")

    assert status and status["status"] == "failed"
    assert status["message"] == "Ingestion failed (RuntimeError)."
    assert "secret" not in status["message"]
    assert "/private" not in status["message"]
    assert not source.exists()
