import importlib
import io
import json
import math
import sys
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def server_module(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "API_KEY_OWNERS_JSON",
        json.dumps({"alice-key": "alice"}),
    )
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("JOB_DB_PATH", str(tmp_path / "jobs.sqlite3"))
    monkeypatch.setenv("DOCUMENT_DB_PATH", str(tmp_path / "documents.sqlite3"))
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "vectors"))
    monkeypatch.setenv("CLASSIC_STORAGE_DIR", str(tmp_path / "classic"))
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
    module._QUERY_EXECUTOR.shutdown(wait=False, cancel_futures=True)
    sys.modules.pop("server", None)


def test_unsupported_upload_type_is_generic_bad_request(server_module):
    with TestClient(server_module.app) as client:
        response = client.post(
            "/ingest",
            headers={"X-API-Key": "alice-key"},
            files={
                "file": (
                    "malware.exe",
                    io.BytesIO(b"not executed"),
                    "application/octet-stream",
                )
            },
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid request."}
    assert "exe" not in response.text.lower()
    owner_directory = server_module.UPLOAD_DIR / "alice"
    assert not owner_directory.exists() or not list(owner_directory.iterdir())


def test_disabled_model_error_does_not_echo_requested_name(server_module):
    requested = "private-model-name"
    with TestClient(server_module.app) as client:
        response = client.post(
            "/query",
            headers={"X-API-Key": "alice-key"},
            json={"query": "question", "model": requested},
        )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "The requested model is not enabled by the server."
    }
    assert requested not in response.text


def test_identifier_helpers_reject_hostile_truthiness_without_stringifying(server_module):
    class Hostile:
        def __bool__(self):
            raise RuntimeError("must not call bool")

        def __str__(self):
            raise RuntimeError("private identifier")

    with pytest.raises(server_module.HTTPException) as captured:
        server_module._bounded_identifier(Hostile(), "doc_id")
    assert captured.value.status_code == 400
    assert "private identifier" not in captured.value.detail
    assert len(server_module._safe_request_id(Hostile())) == 32


def test_future_callback_releases_admission_exactly_once(server_module, monkeypatch):
    future = Future()
    releases = []
    monkeypatch.setattr(
        server_module,
        "_INGEST_ADMISSION",
        SimpleNamespace(release=lambda: releases.append(True)),
    )
    server_module._INGEST_FUTURES.clear()
    server_module._INGEST_FUTURES.add(future)

    server_module._forget_future(future)
    server_module._forget_future(future)

    assert releases == [True]
    assert future not in server_module._INGEST_FUTURES


def test_safe_due_at_rejects_nonfinite_and_caps_remote_future(server_module):
    now = 100.0
    assert server_module._safe_due_at(float("nan"), now) == 0.0
    assert server_module._safe_due_at(float("inf"), now) == 0.0
    assert server_module._safe_due_at(-1, now) == 0.0
    bounded = server_module._safe_due_at(10**100, now)
    assert math.isfinite(bounded)
    assert bounded <= now + max(
        server_module._JOB_STORE.retry_max_seconds,
        604_800.0,
    )


def test_corrupt_due_timestamp_is_processed_without_scheduler_crash(
    server_module,
    monkeypatch,
):
    submitted = []
    cancelled = []
    future = Future()
    monkeypatch.setattr(
        server_module._JOB_STORE,
        "get_internal",
        lambda *_args: {"status": "queued", "next_attempt_at": float("nan")},
    )
    monkeypatch.setattr(
        server_module._INGEST_SCHEDULER,
        "cancel",
        lambda key: cancelled.append(key) or False,
    )
    monkeypatch.setattr(
        server_module._INGEST_EXECUTOR,
        "submit",
        lambda *args: submitted.append(args) or future,
    )

    server_module._submit_ingestion(
        "/tmp/source.txt",
        "source.txt",
        "job-1",
        "alice",
    )

    assert cancelled == ["job-1"]
    assert len(submitted) == 1
    future.set_result(None)


def test_callback_registration_failure_releases_slot_and_reschedules(
    server_module,
    monkeypatch,
):
    releases = []
    scheduled = []

    class BrokenFuture:
        def add_done_callback(self, _callback):
            raise RuntimeError("callback registration failed")

        def cancel(self):
            return True

    future = BrokenFuture()
    monkeypatch.setattr(
        server_module._JOB_STORE,
        "get_internal",
        lambda *_args: {"status": "queued", "next_attempt_at": 0},
    )
    monkeypatch.setattr(
        server_module,
        "_INGEST_ADMISSION",
        SimpleNamespace(
            acquire=lambda **_kwargs: True,
            release=lambda: releases.append(True),
        ),
    )
    monkeypatch.setattr(
        server_module._INGEST_EXECUTOR,
        "submit",
        lambda *_args: future,
    )
    monkeypatch.setitem(
        server_module._submit_ingestion.__globals__,
        "_schedule_ingestion_attempt",
        lambda *args: scheduled.append(args),
    )
    server_module._INGEST_FUTURES.clear()

    server_module._submit_ingestion(
        "/tmp/source.txt",
        "source.txt",
        "job-1",
        "alice",
    )

    assert releases == [True]
    assert scheduled
    assert future not in server_module._INGEST_FUTURES


def test_delete_document_rejects_malformed_vector_backend(server_module, monkeypatch):
    rag = SimpleNamespace(
        collection=SimpleNamespace(get=lambda **_kwargs: "not-an-object"),
        delete_document=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("delete should not run")
        ),
    )
    monkeypatch.setattr(server_module, "get_rag_layer", lambda: rag)

    with pytest.raises(RuntimeError, match="invalid response"):
        server_module._delete_document_for_owner("alice", "doc-1")


def test_delete_document_ignores_cross_owner_vector_metadata(server_module, monkeypatch):
    deleted = []
    rag = SimpleNamespace(
        collection=SimpleNamespace(
            get=lambda **_kwargs: {
                "metadatas": [{"owner_id": "bob", "doc_id": "doc-1"}]
            }
        ),
        delete_document=lambda **kwargs: deleted.append(kwargs),
    )
    monkeypatch.setattr(server_module, "get_rag_layer", lambda: rag)
    monkeypatch.setattr(
        server_module._DOCUMENT_STORE,
        "get",
        lambda **_kwargs: None,
    )

    assert server_module._delete_document_for_owner("alice", "doc-1") is False
    assert deleted == []


def test_recovery_replays_even_when_an_older_registry_row_exists(
    server_module,
    monkeypatch,
):
    source = server_module.UPLOAD_DIR / "alice" / "source.pdf"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"%PDF-1.4\n")
    updates = []
    submissions = []
    registry_reads = []
    monkeypatch.setattr(
        server_module._JOB_STORE,
        "recoverable",
        lambda: [
            {
                "job_id": "job-1",
                "owner_id": "alice",
                "status": "finalizing",
                "filename": "paper.pdf",
                "doc_id": "doc-1",
                "source_path": str(source),
                "attempts": 1,
            }
        ],
    )
    monkeypatch.setattr(
        server_module._JOB_STORE,
        "update",
        lambda *args, **kwargs: updates.append((args, kwargs)),
    )
    monkeypatch.setattr(
        server_module._DOCUMENT_STORE,
        "get",
        lambda **kwargs: registry_reads.append(kwargs) or {
            "source_path": str(source),
            "source_retained": True,
        },
    )
    monkeypatch.setitem(
        server_module._recover_interrupted_jobs.__globals__,
        "_submit_ingestion",
        lambda *args: submissions.append(args),
    )

    server_module._recover_interrupted_jobs()

    assert updates[0][1]["status"] == "queued"
    assert updates[0][1]["source_path"] == str(source)
    assert submissions == [(str(source), "paper.pdf", "job-1", "alice")]
    assert registry_reads == []
    assert source.exists()


def test_same_retained_source_is_not_treated_as_stale_replacement(server_module):
    current = server_module.UPLOAD_DIR / "alice" / "source.pdf"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.write_bytes(b"%PDF-1.4\n")
    other = server_module.UPLOAD_DIR / "alice" / "other.pdf"
    other.write_bytes(b"%PDF-1.4\n")

    assert server_module._same_retained_source(str(current), current) is True
    assert server_module._same_retained_source(str(other), current) is False
    assert server_module._same_retained_source("", current) is False
