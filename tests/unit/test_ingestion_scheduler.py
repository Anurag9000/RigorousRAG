import importlib
import json
import sys
import time

import pytest


class DummyFuture:
    def add_done_callback(self, callback):
        self.callback = callback


class FakeScheduler:
    def __init__(self):
        self.scheduled = {}
        self.cancelled = []
        self.shutdown_called = False

    def schedule(self, key, due_at, callback, *args):
        self.scheduled[key] = (due_at, callback, args)
        return True

    def cancel(self, key):
        self.cancelled.append(key)
        return self.scheduled.pop(key, None) is not None

    def shutdown(self, *, wait=True, timeout=2.0):
        self.shutdown_called = True
        self.scheduled.clear()

    def pending_count(self):
        return len(self.scheduled)


class RejectAdmission:
    def acquire(self, *, blocking):
        assert blocking is False
        return False


@pytest.fixture
def server_module(monkeypatch, tmp_path):
    monkeypatch.setenv("API_KEY_OWNERS_JSON", json.dumps({"alice-key": "alice"}))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("JOB_DB_PATH", str(tmp_path / "jobs.sqlite3"))
    monkeypatch.setenv("DOCUMENT_DB_PATH", str(tmp_path / "documents.sqlite3"))
    monkeypatch.setenv("ORPHAN_CLEANUP_ON_STARTUP", "false")
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


def _queued_source(module, *, next_attempt_at):
    source = module.UPLOAD_DIR / "alice" / "paper.txt"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("evidence", encoding="utf-8")
    module._JOB_STORE.update(
        "job-1",
        "alice",
        status="queued",
        filename="paper.txt",
        source_path=source,
        next_attempt_at=next_attempt_at,
    )
    return source


def test_future_retry_uses_one_deduplicated_scheduler_entry_not_executor(
    server_module,
    monkeypatch,
):
    source = _queued_source(server_module, next_attempt_at=time.time() + 30)
    submitted = []
    scheduler = FakeScheduler()
    monkeypatch.setattr(server_module, "_INGEST_SCHEDULER", scheduler)
    monkeypatch.setattr(
        server_module._INGEST_EXECUTOR,
        "submit",
        lambda *args: submitted.append(args) or DummyFuture(),
    )

    server_module._submit_ingestion(str(source), "paper.txt", "job-1", "alice")
    server_module._submit_ingestion(str(source), "paper.txt", "job-1", "alice")

    assert submitted == []
    assert list(scheduler.scheduled) == ["job-1"]
    assert scheduler.pending_count() == 1
    assert scheduler.scheduled["job-1"][0] > time.time()


def test_scheduler_release_submits_due_job_once(server_module, monkeypatch):
    source = _queued_source(server_module, next_attempt_at=time.time() + 30)
    submitted = []
    scheduler = FakeScheduler()
    monkeypatch.setattr(server_module, "_INGEST_SCHEDULER", scheduler)
    monkeypatch.setattr(
        server_module._INGEST_EXECUTOR,
        "submit",
        lambda *args: submitted.append(args) or DummyFuture(),
    )
    server_module._submit_ingestion(str(source), "paper.txt", "job-1", "alice")
    _due_at, callback, args = scheduler.scheduled["job-1"]
    server_module._JOB_STORE.update(
        "job-1",
        "alice",
        status="queued",
        filename="paper.txt",
        source_path=source,
        next_attempt_at=0,
    )

    callback(*args)

    assert len(submitted) == 1
    assert submitted[0][0] is server_module.process_ingestion
    assert "job-1" not in scheduler.scheduled


def test_executor_saturation_retries_without_entering_unbounded_queue(
    server_module,
    monkeypatch,
):
    source = _queued_source(server_module, next_attempt_at=0)
    submitted = []
    scheduler = FakeScheduler()
    monkeypatch.setattr(server_module, "_INGEST_SCHEDULER", scheduler)
    monkeypatch.setattr(server_module, "_INGEST_ADMISSION", RejectAdmission())
    monkeypatch.setattr(
        server_module._INGEST_EXECUTOR,
        "submit",
        lambda *args: submitted.append(args) or DummyFuture(),
    )

    server_module._submit_ingestion(str(source), "paper.txt", "job-1", "alice")

    assert submitted == []
    assert list(scheduler.scheduled) == ["job-1"]
    assert scheduler.scheduled["job-1"][0] > time.time()
    assert server_module._JOB_STORE.get("job-1", "alice")["status"] == "queued"


def test_shutdown_stops_central_scheduler(server_module, monkeypatch):
    scheduler = FakeScheduler()
    monkeypatch.setattr(server_module, "_INGEST_SCHEDULER", scheduler)

    server_module._cancel_scheduled_ingestions()

    assert scheduler.shutdown_called is True
    assert server_module._INGEST_SHUTDOWN.is_set()
