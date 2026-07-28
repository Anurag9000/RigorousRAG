import importlib
import json
import sys
import time

import pytest


class DummyFuture:
    def add_done_callback(self, callback):
        self.callback = callback


class FakeTimer:
    created = []

    def __init__(self, delay, function, args=()):
        self.delay = delay
        self.function = function
        self.args = args
        self.daemon = False
        self.started = False
        self.cancelled = False
        FakeTimer.created.append(self)

    def start(self):
        self.started = True

    def is_alive(self):
        return self.started and not self.cancelled

    def cancel(self):
        self.cancelled = True


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


def test_future_retry_uses_one_deduplicated_timer_not_executor(
    server_module,
    monkeypatch,
):
    FakeTimer.created = []
    source = _queued_source(server_module, next_attempt_at=time.time() + 30)
    submitted = []
    monkeypatch.setattr(server_module.threading, "Timer", FakeTimer)
    monkeypatch.setattr(
        server_module._INGEST_EXECUTOR,
        "submit",
        lambda *args: submitted.append(args) or DummyFuture(),
    )

    server_module._submit_ingestion(str(source), "paper.txt", "job-1", "alice")
    server_module._submit_ingestion(str(source), "paper.txt", "job-1", "alice")

    assert submitted == []
    assert len(FakeTimer.created) == 1
    assert FakeTimer.created[0].started is True
    assert FakeTimer.created[0].delay > 0


def test_timer_release_submits_due_job_once(server_module, monkeypatch):
    FakeTimer.created = []
    source = _queued_source(server_module, next_attempt_at=time.time() + 30)
    submitted = []
    monkeypatch.setattr(server_module.threading, "Timer", FakeTimer)
    monkeypatch.setattr(
        server_module._INGEST_EXECUTOR,
        "submit",
        lambda *args: submitted.append(args) or DummyFuture(),
    )
    server_module._submit_ingestion(str(source), "paper.txt", "job-1", "alice")
    timer = FakeTimer.created[0]
    server_module._JOB_STORE.update(
        "job-1",
        "alice",
        status="queued",
        filename="paper.txt",
        source_path=source,
        next_attempt_at=0,
    )

    timer.function(*timer.args)

    assert len(submitted) == 1
    assert submitted[0][0] is server_module.process_ingestion
    assert "job-1" not in server_module._INGEST_TIMERS


def test_shutdown_cancels_pending_timers(server_module, monkeypatch):
    FakeTimer.created = []
    source = _queued_source(server_module, next_attempt_at=time.time() + 30)
    monkeypatch.setattr(server_module.threading, "Timer", FakeTimer)
    server_module._submit_ingestion(str(source), "paper.txt", "job-1", "alice")

    server_module._cancel_scheduled_ingestions()

    assert FakeTimer.created[0].cancelled is True
    assert server_module._INGEST_TIMERS == {}
