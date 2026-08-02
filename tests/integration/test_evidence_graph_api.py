from __future__ import annotations

import importlib
import json
import sys

import pytest
from fastapi.testclient import TestClient

from tools.models import AgentAnswer, Citation


@pytest.fixture
def graph_server(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "API_KEY_OWNERS_JSON",
        json.dumps({"alice-key": "alice"}),
    )
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("JOB_DB_PATH", str(tmp_path / "jobs.sqlite3"))
    monkeypatch.setenv(
        "DOCUMENT_DB_PATH",
        str(tmp_path / "documents.sqlite3"),
    )
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


def _graph_answer() -> AgentAnswer:
    return AgentAnswer(
        answer="The reviewed graph supports this statement [1].",
        citations=[
            Citation(
                label="[1]",
                title="Primary result",
                url="local://doc-1",
                source_type="uploaded_document",
                snippet="Privacy-finalized evidence text.",
                quote="Privacy-finalized evidence text.",
                source_id="e" * 64,
                doc_id="doc-1",
                chunk_id="a" * 64,
                page_number=2,
                metadata={
                    "retrieval_strategy": "graph",
                    "graph_set_key": "review",
                    "graph_set_id": "b" * 64,
                    "graph_set_digest": "c" * 64,
                    "graph_authority_digest": "d" * 64,
                    "graph_selection_digest": "f" * 64,
                    "graph_generation": 3,
                    "graph_node_type": "claim",
                    "graph_origin": "cross_document",
                    "graph_lineage_step_digests": ["1" * 64],
                    "graph_matched_term_count": 2,
                    "graph_matched_terms_digest": "2" * 64,
                },
            )
        ],
        metadata={"model": "test-model"},
    )


def test_query_serializes_canonical_graph_citations_without_private_fields(
    graph_server,
    monkeypatch,
):
    captured = []

    class FakeAgent:
        def __init__(self, owner_id):
            self.owner_id = owner_id

        def run(self, query):
            captured.append((self.owner_id, query))
            return _graph_answer()

    monkeypatch.setattr(
        graph_server,
        "_new_agent",
        lambda owner_id, model=None: FakeAgent(owner_id),
    )
    with TestClient(graph_server.app) as client:
        response = client.post(
            "/query",
            headers={
                "X-API-Key": "alice-key",
                "X-Owner-ID": "mallory",
            },
            json={"query": "Use the reviewed evidence graph."},
        )

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    payload = response.json()
    assert captured == [("alice", "Use the reviewed evidence graph.")]
    assert payload["answer"].endswith("[1].")
    assert len(payload["citations"]) == 1
    citation = payload["citations"][0]
    assert citation["source_type"] == "uploaded_document"
    assert citation["url"] == "local://doc-1"
    assert citation["doc_id"] == "doc-1"
    assert citation["chunk_id"] == "a" * 64
    assert citation["metadata"]["retrieval_strategy"] == "graph"
    assert citation["metadata"]["graph_generation"] == 3
    assert citation["metadata"]["graph_origin"] == "cross_document"
    assert citation["metadata"]["graph_lineage_step_digests"] == [
        "1" * 64
    ]
    assert "owner_id" not in citation["metadata"]
    assert "graph_matched_terms" not in citation["metadata"]
    assert "query" not in citation["metadata"]
    assert "source_path" not in citation["metadata"]
