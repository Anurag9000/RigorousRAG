import importlib
import json
import sys

import pytest


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


def _source(module):
    path = module.UPLOAD_DIR / "alice" / "paper.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("evidence", encoding="utf-8")
    return path


def test_queued_job_is_not_promoted_by_preexisting_registry(
    server_module,
    monkeypatch,
):
    source = _source(server_module)
    server_module._JOB_STORE.update(
        "job-queued",
        "alice",
        status="queued",
        filename="paper.txt",
        source_path=source,
        doc_id="existing-doc",
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
    submissions = []
    monkeypatch.setattr(
        server_module,
        "_submit_ingestion",
        lambda *args: submissions.append(args),
    )

    server_module._recover_interrupted_jobs()

    status = server_module._JOB_STORE.get("job-queued", "alice")
    assert status and status["status"] == "queued"
    assert submissions == [(str(source), "paper.txt", "job-queued", "alice")]


def test_finalizing_job_is_replayed_instead_of_inferred_success(
    server_module,
    monkeypatch,
):
    source = _source(server_module)
    server_module._JOB_STORE.update(
        "job-finalizing",
        "alice",
        status="queued",
        filename="paper.txt",
        source_path=source,
    )
    assert server_module._JOB_STORE.claim(
        "job-finalizing",
        "alice",
        max_attempts=3,
    ) is True
    server_module._JOB_STORE.update(
        "job-finalizing",
        "alice",
        status="finalizing",
        filename="paper.txt",
        source_path=source,
        doc_id="committed-doc",
    )
    monkeypatch.setattr(
        server_module._DOCUMENT_STORE,
        "get",
        lambda **_kwargs: {
            "doc_id": "committed-doc",
            "source_path": str(source),
            "source_retained": 1,
        },
    )
    submissions = []
    monkeypatch.setattr(
        server_module,
        "_submit_ingestion",
        lambda *args: submissions.append(args),
    )

    server_module._recover_interrupted_jobs()

    status = server_module._JOB_STORE.get("job-finalizing", "alice")
    assert status and status["status"] == "queued"
    assert submissions == [
        (str(source), "paper.txt", "job-finalizing", "alice")
    ]
    assert source.exists()
