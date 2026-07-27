import importlib
import io
import json
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from tools.models import AgentAnswer


@pytest.fixture
def server_module(monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEY_OWNERS_JSON", json.dumps({"alice-key": "alice", "bob-key": "bob"}))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("ALLOWED_MODELS", "test-model")
    monkeypatch.setenv("DEFAULT_MODEL", "test-model")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    sys.modules.pop("server", None)
    module = importlib.import_module("server")
    yield module
    sys.modules.pop("server", None)


def test_health_and_config_are_public(server_module):
    client = TestClient(server_module.app)
    assert client.get("/health").json()["status"] == "ok"
    config = client.get("/config").json()
    assert config["auth_required"] is True
    assert config["allowed_models"] == ["test-model"]


def test_api_key_selects_server_owned_tenant(server_module, monkeypatch):
    captured = []

    class FakeAgent:
        def __init__(self, owner):
            self.owner = owner

        def run(self, query):
            captured.append((self.owner, query))
            return AgentAnswer(answer=f"owner={self.owner}")

    monkeypatch.setattr(server_module, "_new_agent", lambda owner_id, model=None: FakeAgent(owner_id))
    client = TestClient(server_module.app)
    missing = client.post("/query", json={"query": "q"})
    assert missing.status_code == 401
    alice = client.post("/query", headers={"X-API-Key": "alice-key"}, json={"query": "q"})
    bob = client.post("/query", headers={"X-API-Key": "bob-key"}, json={"query": "q"})
    assert alice.json()["answer"] == "owner=alice"
    assert bob.json()["answer"] == "owner=bob"
    assert captured == [("alice", "q"), ("bob", "q")]


def test_client_supplied_owner_header_is_ignored(server_module, monkeypatch):
    monkeypatch.setattr(
        server_module,
        "_new_agent",
        lambda owner_id, model=None: MagicMock(run=lambda _query: AgentAnswer(answer=owner_id)),
    )
    client = TestClient(server_module.app)
    response = client.post(
        "/query",
        headers={"X-API-Key": "alice-key", "X-Owner-ID": "bob"},
        json={"query": "q"},
    )
    assert response.json()["answer"] == "alice"


def test_upload_uses_generated_storage_name_and_owner_scoped_job(server_module, monkeypatch):
    monkeypatch.setattr(server_module, "process_ingestion", lambda *_args, **_kwargs: None)
    client = TestClient(server_module.app)
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
    status = client.get(f"/status/{payload['job_id']}", headers={"X-API-Key": "alice-key"})
    assert status.status_code == 200
    hidden = client.get(f"/status/{payload['job_id']}", headers={"X-API-Key": "bob-key"})
    assert hidden.status_code == 404


def test_model_override_is_allowlisted(server_module):
    client = TestClient(server_module.app)
    response = client.post(
        "/query",
        headers={"X-API-Key": "alice-key"},
        json={"query": "q", "model": "unapproved-model"},
    )
    assert response.status_code == 400
