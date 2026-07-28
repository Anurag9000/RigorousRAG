import importlib
import json
import sys
import threading

import pytest
from fastapi.testclient import TestClient


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
    monkeypatch.setenv("QUERY_WORKERS", "1")
    monkeypatch.setenv("QUERY_MAX_PENDING", "2")
    monkeypatch.setenv("QUERY_TIMEOUT_SECONDS", "2")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    sys.modules.pop("server", None)
    module = importlib.import_module("server")
    yield module
    module._cancel_scheduled_ingestions()
    module._INGEST_EXECUTOR.shutdown(wait=False, cancel_futures=True)
    module._QUERY_EXECUTOR.shutdown(wait=False, cancel_futures=True)
    sys.modules.pop("server", None)


def test_document_list_initialization_scan_and_registry_join_use_bounded_worker(
    server_module,
    monkeypatch,
):
    calls = []

    class FakeRag:
        def list_documents(self, *, owner_id, limit):
            calls.append(("list", threading.current_thread().name, owner_id, limit))
            return [{"doc_id": "doc-1", "filename": "paper.pdf"}]

    def fake_get_rag_layer():
        calls.append(("init", threading.current_thread().name))
        return FakeRag()

    def fake_registry_get(*, owner_id, doc_id):
        calls.append(("registry", threading.current_thread().name, owner_id, doc_id))
        return {
            "source_retained": True,
            "visual_source_available": True,
            "visual_source_verified": False,
        }

    monkeypatch.setattr(server_module, "get_rag_layer", fake_get_rag_layer)
    monkeypatch.setattr(server_module._DOCUMENT_STORE, "get", fake_registry_get)

    with TestClient(server_module.app) as client:
        response = client.get(
            "/docs/list",
            headers={"X-API-Key": "alice-key"},
        )

    assert response.status_code == 200
    assert response.json() == [{
        "doc_id": "doc-1",
        "filename": "paper.pdf",
        "source_retained": True,
        "visual_source_available": True,
        "visual_source_verified": False,
    }]
    assert calls
    assert all(call[1].startswith("rigorousrag-query") for call in calls)


def test_document_delete_vector_registry_and_source_work_use_bounded_worker(
    server_module,
    monkeypatch,
):
    calls = []

    class FakeCollection:
        def get(self, **kwargs):
            calls.append(("vector_get", threading.current_thread().name, kwargs))
            return {"metadatas": [{"owner_id": "alice", "doc_id": "doc-1"}]}

    class FakeRag:
        collection = FakeCollection()

        def delete_document(self, *, owner_id, doc_id):
            calls.append(("vector_delete", threading.current_thread().name, owner_id, doc_id))

    def fake_get_rag_layer():
        calls.append(("init", threading.current_thread().name))
        return FakeRag()

    def fake_registry_get(*, owner_id, doc_id):
        calls.append(("registry_get", threading.current_thread().name, owner_id, doc_id))
        return {"source_path": ""}

    def fake_registry_delete(*, owner_id, doc_id):
        calls.append(("registry_delete", threading.current_thread().name, owner_id, doc_id))
        return {"source_path": ""}

    def fake_unlink(path):
        calls.append(("unlink", threading.current_thread().name, path))
        return False

    monkeypatch.setattr(server_module, "get_rag_layer", fake_get_rag_layer)
    monkeypatch.setattr(server_module._DOCUMENT_STORE, "get", fake_registry_get)
    monkeypatch.setattr(server_module._DOCUMENT_STORE, "delete", fake_registry_delete)
    monkeypatch.setattr(server_module, "_safe_unlink_upload", fake_unlink)

    with TestClient(server_module.app) as client:
        response = client.delete(
            "/docs/doc-1",
            headers={"X-API-Key": "alice-key"},
        )

    assert response.status_code == 200
    assert response.json() == {"status": "deleted", "doc_id": "doc-1"}
    assert calls
    assert all(call[1].startswith("rigorousrag-query") for call in calls)


def test_document_routes_fail_closed_when_shared_executor_is_saturated(
    server_module,
    monkeypatch,
):
    class UnavailableExecutor:
        def submit(self, *_args, **_kwargs):
            return None

        def shutdown(self, **_kwargs):
            return None

    monkeypatch.setattr(server_module, "_QUERY_EXECUTOR", UnavailableExecutor())

    with TestClient(server_module.app) as client:
        listing = client.get(
            "/docs/list",
            headers={"X-API-Key": "alice-key"},
        )
        deletion = client.delete(
            "/docs/doc-1",
            headers={"X-API-Key": "alice-key"},
        )

    assert listing.status_code == 503
    assert deletion.status_code == 503
    assert listing.headers["Retry-After"] == "1"
    assert deletion.headers["Retry-After"] == "1"
