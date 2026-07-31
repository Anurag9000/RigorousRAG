import time

from tools.job_store import JobStore
from tools.operator_repair import list_corrupt_jobs


def _insert_raw_job(store: JobStore, **overrides):
    now = time.time()
    values = {
        "job_id": "corrupt-read",
        "owner_id": "operator-owner",
        "status": "queued",
        "filename": "paper.pdf",
        "message": None,
        "doc_id": None,
        "source_path": None,
        "attempts": 0,
        "next_attempt_at": 0.0,
        "created_at": now,
        "updated_at": now,
    }
    values.update(overrides)
    with store._lock, store._connect() as connection:  # noqa: SLF001
        connection.execute(
            """
            INSERT INTO jobs(
                job_id, owner_id, status, filename, message, doc_id,
                source_path, attempts, next_attempt_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(values[key] for key in (
                "job_id",
                "owner_id",
                "status",
                "filename",
                "message",
                "doc_id",
                "source_path",
                "attempts",
                "next_attempt_at",
                "created_at",
                "updated_at",
            )),
        )


def test_public_and_internal_point_reads_fail_closed_on_control_bearing_text(tmp_path):
    store = JobStore(path=tmp_path / "jobs.sqlite3")
    _insert_raw_job(store, filename="paper\nprivate.pdf", message="bad\tmessage")

    assert store.get("corrupt-read", "operator-owner") is None
    assert store.get_internal("corrupt-read", "operator-owner") is None
    records = list_corrupt_jobs(store)
    assert len(records) == 1
    assert "invalid_filename" in records[0].reasons
    assert "invalid_message" in records[0].reasons


def test_point_reads_fail_closed_on_invalid_numeric_state(tmp_path):
    store = JobStore(path=tmp_path / "jobs.sqlite3")
    _insert_raw_job(store, attempts=-1, next_attempt_at=float("nan"))

    assert store.get("corrupt-read", "operator-owner") is None
    assert store.get_internal("corrupt-read", "operator-owner") is None
    assert store.recoverable() == []


def test_valid_terminal_rows_remain_readable(tmp_path):
    store = JobStore(path=tmp_path / "jobs.sqlite3")
    store.update(
        "valid-success",
        "operator-owner",
        status="success",
        filename="paper.pdf",
        message="Indexed successfully.",
        doc_id="doc-1",
    )

    public = store.get("valid-success", "operator-owner")
    internal = store.get_internal("valid-success", "operator-owner")

    assert public == {
        "job_id": "valid-success",
        "status": "success",
        "filename": "paper.pdf",
        "message": "Indexed successfully.",
        "doc_id": "doc-1",
    }
    assert internal is not None
    assert internal["status"] == "success"
    assert internal["doc_id"] == "doc-1"
