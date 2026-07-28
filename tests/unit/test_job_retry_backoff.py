import sqlite3

import tools.job_store as job_store_module
from tools.job_store import JobStore


def test_retry_transition_persists_exponential_due_time(monkeypatch, tmp_path):
    monkeypatch.setenv("INGEST_RETRY_BASE_SECONDS", "4")
    monkeypatch.setenv("INGEST_RETRY_MAX_SECONDS", "10")
    clock = {"now": 100.0}
    monkeypatch.setattr(job_store_module.time, "time", lambda: clock["now"])

    source = tmp_path / "source.txt"
    source.write_text("evidence", encoding="utf-8")
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.update(
        "job-1",
        "alice",
        status="queued",
        filename="source.txt",
        source_path=source,
    )
    assert store.claim("job-1", "alice", max_attempts=5, now=100.0) is True

    clock["now"] = 101.0
    store.update(
        "job-1",
        "alice",
        status="queued",
        filename="source.txt",
        source_path=source,
    )
    first_retry = store.get_internal("job-1", "alice")
    assert first_retry and first_retry["next_attempt_at"] == 105.0
    assert store.claim("job-1", "alice", max_attempts=5, now=104.99) is False
    assert store.claim("job-1", "alice", max_attempts=5, now=105.0) is True

    clock["now"] = 106.0
    store.update(
        "job-1",
        "alice",
        status="queued",
        filename="source.txt",
        source_path=source,
    )
    second_retry = store.get_internal("job-1", "alice")
    assert second_retry and second_retry["next_attempt_at"] == 114.0
    assert store.claim("job-1", "alice", max_attempts=5, now=114.0) is True

    clock["now"] = 115.0
    store.update(
        "job-1",
        "alice",
        status="queued",
        filename="source.txt",
        source_path=source,
    )
    capped_retry = store.get_internal("job-1", "alice")
    assert capped_retry and capped_retry["next_attempt_at"] == 125.0


def test_existing_queued_deadline_survives_recovery_update(monkeypatch, tmp_path):
    monkeypatch.setenv("INGEST_RETRY_BASE_SECONDS", "2")
    clock = {"now": 200.0}
    monkeypatch.setattr(job_store_module.time, "time", lambda: clock["now"])
    source = tmp_path / "source.txt"
    source.write_text("evidence", encoding="utf-8")
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.update(
        "job-1",
        "alice",
        status="queued",
        filename="source.txt",
        source_path=source,
        next_attempt_at=250.0,
    )

    clock["now"] = 210.0
    store.update(
        "job-1",
        "alice",
        status="queued",
        filename="source.txt",
        source_path=source,
        message="Recovered after restart.",
    )
    internal = store.get_internal("job-1", "alice")
    assert internal and internal["next_attempt_at"] == 250.0
    assert store.claim("job-1", "alice", max_attempts=3, now=249.0) is False


def test_job_store_migrates_schema_without_retry_column(tmp_path):
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                status TEXT NOT NULL,
                filename TEXT NOT NULL,
                message TEXT,
                doc_id TEXT,
                source_path TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
    store = JobStore(database)
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
    assert "next_attempt_at" in columns
    assert store.ping() is True
