import importlib
import io
import json
import os
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from tools.ingestion_models import DocumentSection, IngestedDocument, IngestionResult
from tools.models import AgentAnswer


@pytest.fixture
def server_module(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "API_KEY_OWNERS_JSON",
        json.dumps({"alice-key": "alice", "bob-key": "bob"}),
    )
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("JOB_DB_PATH", str(tmp_path / "jobs.sqlite3"))
    monkeypatch.setenv("DOCUMENT_DB_PATH", str(tmp_path / "documents.sqlite3"))
    monkeypatch.setenv("RETAIN_SOURCE_FILES", "false")
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


def _patch_recovery_submission(monkeypatch, server_module, submitted):
    monkeypatch.setitem(
        server_module._recover_interrupted_jobs.__globals__,
        "_submit_ingestion",
        lambda *args: submitted.append(args),
    )


def _patch_retry_submission(monkeypatch, server_module, submitted):
    monkeypatch.setitem(
        server_module._retry_or_fail_job.__globals__,
        "_submit_ingestion",
        lambda *args: submitted.append(args),
    )


def test_health_config_and_security_headers_are_public(server_module):
    with TestClient(server_module.app) as client:
        response = client.get("/health")
        assert response.json()["status"] == "ok"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert "default-src 'self'" in response.headers["Content-Security-Policy"]
        config = client.get("/config").json()
    assert config["auth_required"] is True
    assert config["allowed_models"] == ["test-model"]
    assert config["retain_source_files"] is False
    assert config["ingest_max_attempts"] >= 1


def test_api_key_selects_server_owned_tenant(server_module, monkeypatch):
    captured = []

    class FakeAgent:
        def __init__(self, owner):
            self.owner = owner

        def run(self, query):
            captured.append((self.owner, query))
            return AgentAnswer(answer=f"owner={self.owner}")

    monkeypatch.setattr(
        server_module,
        "_new_agent",
        lambda owner_id, model=None: FakeAgent(owner_id),
    )
    with TestClient(server_module.app) as client:
        assert client.post("/query", json={"query": "q"}).status_code == 401
        alice = client.post(
            "/query",
            headers={"X-API-Key": "alice-key"},
            json={"query": "q"},
        )
        bob = client.post(
            "/query",
            headers={"X-API-Key": "bob-key"},
            json={"query": "q"},
        )
    assert alice.json()["answer"] == "owner=alice"
    assert bob.json()["answer"] == "owner=bob"
    assert captured == [("alice", "q"), ("bob", "q")]


def test_client_supplied_owner_header_is_ignored(server_module, monkeypatch):
    monkeypatch.setattr(
        server_module,
        "_new_agent",
        lambda owner_id, model=None: MagicMock(
            run=lambda _query: AgentAnswer(answer=owner_id)
        ),
    )
    with TestClient(server_module.app) as client:
        response = client.post(
            "/query",
            headers={"X-API-Key": "alice-key", "X-Owner-ID": "bob"},
            json={"query": "q"},
        )
    assert response.json()["answer"] == "alice"


def test_upload_uses_generated_name_and_durable_owner_scoped_queue(
    server_module,
    monkeypatch,
):
    submitted = []
    monkeypatch.setattr(
        server_module,
        "_submit_ingestion",
        lambda *args: submitted.append(args),
    )
    with TestClient(server_module.app) as client:
        response = client.post(
            "/ingest",
            headers={"X-API-Key": "alice-key"},
            files={"file": ("../../paper.txt", io.BytesIO(b"evidence"), "text/plain")},
        )
        assert response.status_code == 200
        payload = response.json()
        stored = list((server_module.UPLOAD_DIR / "alice").iterdir())
        assert len(stored) == 1
        assert stored[0].name != "paper.txt"
        assert stored[0].suffix == ".txt"
        status = client.get(
            f"/status/{payload['job_id']}",
            headers={"X-API-Key": "alice-key"},
        )
        hidden = client.get(
            f"/status/{payload['job_id']}",
            headers={"X-API-Key": "bob-key"},
        )
    assert status.status_code == 200
    assert status.json()["status"] == "queued"
    assert hidden.status_code == 404
    assert len(submitted) == 1
    reloaded = server_module.JobStore(
        path=server_module._JOB_STORE.path,
        ttl_seconds=3600,
    )
    internal = reloaded.get_internal(payload["job_id"], "alice")
    assert internal and internal["status"] == "queued"
    assert internal["source_path"] == str(stored[0].resolve())


def test_startup_recovery_submits_valid_job_and_cleans_exhausted_source(
    server_module,
    monkeypatch,
):
    owner_dir = server_module.UPLOAD_DIR / "alice"
    owner_dir.mkdir(parents=True, exist_ok=True)
    valid = owner_dir / "valid.txt"
    exhausted = owner_dir / "exhausted.txt"
    valid.write_text("valid", encoding="utf-8")
    exhausted.write_text("exhausted", encoding="utf-8")
    server_module._JOB_STORE.update(
        "valid-job",
        "alice",
        status="queued",
        filename="valid.txt",
        source_path=str(valid),
    )
    server_module._JOB_STORE.update(
        "exhausted-job",
        "alice",
        status="queued",
        filename="exhausted.txt",
        source_path=str(exhausted),
    )
    assert server_module._JOB_STORE.claim("exhausted-job", "alice", 1) is True
    monkeypatch.setattr(server_module, "INGEST_MAX_ATTEMPTS", 1)
    submitted = []
    _patch_recovery_submission(monkeypatch, server_module, submitted)

    server_module._recover_interrupted_jobs()

    assert submitted == [(str(valid.resolve()), "valid.txt", "valid-job", "alice")]
    assert server_module._JOB_STORE.get("valid-job", "alice")["status"] == "queued"
    exhausted_status = server_module._JOB_STORE.get("exhausted-job", "alice")
    assert exhausted_status and exhausted_status["status"] == "failed"
    assert not exhausted.exists()


def test_finalizing_job_with_registry_is_replayed(
    server_module,
    monkeypatch,
):
    source = server_module.UPLOAD_DIR / "alice" / "paper.pdf"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"%PDF-test")
    server_module._DOCUMENT_STORE.register(
        owner_id="alice",
        doc_id="doc-1",
        filename="paper.pdf",
        mime_type="application/pdf",
        source_path=source,
    )
    server_module._JOB_STORE.update(
        "job-1",
        "alice",
        status="finalizing",
        filename="paper.pdf",
        source_path=str(source),
        doc_id="doc-1",
    )
    submitted = []
    _patch_recovery_submission(monkeypatch, server_module, submitted)

    server_module._recover_interrupted_jobs()

    status = server_module._JOB_STORE.get("job-1", "alice")
    assert status and status["status"] == "queued"
    assert status["doc_id"] is None
    assert submitted == [(str(source.resolve()), "paper.pdf", "job-1", "alice")]
    assert source.exists()


def test_losing_duplicate_worker_does_not_delete_winner_source(
    server_module,
    monkeypatch,
):
    source = server_module.UPLOAD_DIR / "alice" / "paper.txt"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("evidence", encoding="utf-8")
    monkeypatch.setattr(server_module._JOB_STORE, "claim", lambda *_args, **_kwargs: False)

    server_module.process_ingestion(str(source), "paper.txt", "job-1", "alice")

    assert source.exists()


def test_transient_index_failure_requeues_and_preserves_source(
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
        source_path=str(source),
    )
    document = IngestedDocument(
        id="doc-1",
        filename="paper.txt",
        file_path=str(source),
        mime_type="text/plain",
        text="evidence",
        sections=[DocumentSection(title="Full Text", content="evidence")],
    )
    monkeypatch.setattr(
        server_module,
        "ingest_file",
        lambda *_args, **_kwargs: IngestionResult(success=True, document=document),
    )
    monkeypatch.setattr(
        server_module,
        "index_document",
        MagicMock(side_effect=RuntimeError("down")),
    )
    monkeypatch.setattr(
        server_module,
        "_new_agent",
        lambda *_args, **_kwargs: MagicMock(client=None),
    )
    submitted = []
    _patch_retry_submission(monkeypatch, server_module, submitted)

    server_module.process_ingestion(str(source), "paper.txt", "job-1", "alice")

    status = server_module._JOB_STORE.get("job-1", "alice")
    assert status and status["status"] == "queued"
    assert status["doc_id"] is None
    assert source.exists()
    assert submitted == [(str(source.resolve()), "paper.txt", "job-1", "alice")]


def test_safe_unlink_refuses_symlink(server_module, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = server_module.UPLOAD_DIR / "alice" / "link.txt"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(outside, link)
    except OSError:
        pytest.skip("Symlinks are unavailable on this platform.")

    assert server_module._safe_unlink_upload(link) is False
    assert outside.exists()
    assert link.is_symlink()


def test_document_delete_cleans_vectors_registry_and_retained_source(
    server_module,
    monkeypatch,
):
    source = server_module.UPLOAD_DIR / "alice" / "paper.pdf"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"%PDF-test")
    server_module._DOCUMENT_STORE.register(
        owner_id="alice",
        doc_id="doc-1",
        filename="paper.pdf",
        mime_type="application/pdf",
        source_path=source,
    )

    class FakeCollection:
        def __init__(self, exists=True):
            self.exists = exists

        def get(self, **_kwargs):
            return {
                "metadatas": (
                    [{"doc_id": "doc-1", "owner_id": "alice"}]
                    if self.exists
                    else []
                )
            }

    class FakeRag:
        def __init__(self, exists=True):
            self.collection = FakeCollection(exists)
            self.deleted = []

        def delete_document(self, *, owner_id, doc_id):
            self.deleted.append((owner_id, doc_id))

    fake_rag = FakeRag()
    monkeypatch.setattr(server_module, "get_rag_layer", lambda: fake_rag)
    with TestClient(server_module.app) as client:
        response = client.delete(
            "/docs/doc-1",
            headers={"X-API-Key": "alice-key"},
        )
    assert response.status_code == 200
    assert fake_rag.deleted == [("alice", "doc-1")]
    assert not source.exists()
    assert server_module._DOCUMENT_STORE.get(owner_id="alice", doc_id="doc-1") is None


def test_document_delete_retries_registry_only_cleanup(server_module, monkeypatch):
    source = server_module.UPLOAD_DIR / "alice" / "paper.pdf"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"%PDF-test")
    server_module._DOCUMENT_STORE.register(
        owner_id="alice",
        doc_id="doc-1",
        filename="paper.pdf",
        mime_type="application/pdf",
        source_path=source,
    )

    class EmptyCollection:
        def get(self, **_kwargs):
            return {"metadatas": []}

    fake_rag = MagicMock(collection=EmptyCollection())
    monkeypatch.setattr(server_module, "get_rag_layer", lambda: fake_rag)
    with TestClient(server_module.app) as client:
        response = client.delete(
            "/docs/doc-1",
            headers={"X-API-Key": "alice-key"},
        )
    assert response.status_code == 200
    fake_rag.delete_document.assert_not_called()
    assert not source.exists()
    assert server_module._DOCUMENT_STORE.get(owner_id="alice", doc_id="doc-1") is None


def test_model_override_is_allowlisted(server_module):
    with TestClient(server_module.app) as client:
        response = client.post(
            "/query",
            headers={"X-API-Key": "alice-key"},
            json={"query": "q", "model": "unapproved-model"},
        )
    assert response.status_code == 400
