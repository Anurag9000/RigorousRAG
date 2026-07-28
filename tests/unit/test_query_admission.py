import importlib
import json
import sys
import threading

import pytest
from fastapi.testclient import TestClient

from tools.models import AgentAnswer


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
    monkeypatch.setenv("QUERY_MAX_PENDING", "1")
    monkeypatch.setenv("QUERY_TIMEOUT_SECONDS", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    sys.modules.pop("server", None)
    module = importlib.import_module("server")
    yield module
    module._cancel_scheduled_ingestions()
    module._INGEST_EXECUTOR.shutdown(wait=False, cancel_futures=True)
    module._QUERY_EXECUTOR.shutdown(wait=False, cancel_futures=True)
    sys.modules.pop("server", None)


def test_query_success_uses_bounded_executor(server_module, monkeypatch):
    class Agent:
        def run(self, query):
            return AgentAnswer(answer=f"answered:{query}")

    monkeypatch.setattr(
        server_module,
        "_new_agent",
        lambda *_args, **_kwargs: Agent(),
    )

    with TestClient(server_module.app) as client:
        response = client.post(
            "/query",
            headers={"X-API-Key": "alice-key"},
            json={"query": "evidence"},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "answered:evidence"


def test_timed_out_query_retains_capacity_until_worker_finishes(
    server_module,
    monkeypatch,
):
    started = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(server_module, "QUERY_TIMEOUT_SECONDS", 0.03)

    class BlockingAgent:
        def run(self, _query):
            started.set()
            release.wait(2.0)
            return AgentAnswer(answer="late")

    monkeypatch.setattr(
        server_module,
        "_new_agent",
        lambda *_args, **_kwargs: BlockingAgent(),
    )

    with TestClient(server_module.app) as client:
        timed_out = client.post(
            "/query",
            headers={"X-API-Key": "alice-key"},
            json={"query": "slow"},
        )
        assert started.is_set()
        saturated = client.post(
            "/query",
            headers={"X-API-Key": "alice-key"},
            json={"query": "second"},
        )
        release.set()

    assert timed_out.status_code == 504
    assert timed_out.json()["detail"] == "The research operation exceeded the server time limit."
    assert saturated.status_code == 503
    assert saturated.headers["Retry-After"] == "1"
    assert saturated.json()["detail"] == "The research executor is at capacity."


def test_direct_protocol_route_uses_same_bounded_executor(server_module, monkeypatch):
    called = []

    def fake_protocol(text, doc_id, **_kwargs):
        called.append((text, doc_id))
        return json.dumps({"steps": [], "warnings": []})

    monkeypatch.setattr(server_module, "extract_protocol", fake_protocol)

    with TestClient(server_module.app) as client:
        response = client.post(
            "/tool/protocol",
            headers={"X-API-Key": "alice-key"},
            json={"text": "Add buffer.", "doc_id": "doc-1"},
        )

    assert response.status_code == 200
    assert response.json() == {"steps": [], "warnings": []}
    assert called == [("Add buffer.", "doc-1")]


def test_direct_visual_missing_document_is_owner_safe_404(server_module, monkeypatch):
    def missing_document(*_args, **_kwargs):
        raise ValueError("The requested document was not found for this owner.")

    monkeypatch.setattr(server_module, "check_visual_entailment", missing_document)

    with TestClient(server_module.app, raise_server_exceptions=False) as client:
        response = client.post(
            "/tool/visual-entailment",
            headers={"X-API-Key": "alice-key"},
            json={
                "claim_text": "Accuracy increased.",
                "figure_id": "Figure 1",
                "doc_id": "missing-doc",
            },
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found."}
    assert "owner" not in response.text


def test_research_executor_failure_is_generic_for_query_and_direct_tool(
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
        query = client.post(
            "/query",
            headers={"X-API-Key": "alice-key"},
            json={"query": "question"},
        )
        protocol = client.post(
            "/tool/protocol",
            headers={"X-API-Key": "alice-key"},
            json={"text": "Add buffer."},
        )

    assert query.status_code == 503
    assert protocol.status_code == 503
    assert query.json()["detail"] == "The research executor is at capacity."
    assert protocol.json()["detail"] == "The research executor is at capacity."
