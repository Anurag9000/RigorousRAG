import time

from tools.job_store import JobStore
from tools.operator_repair import list_corrupt_jobs


def _insert_relative_source_row(store: JobStore) -> None:
    now = time.time()
    with store._lock, store._connect() as connection:  # noqa: SLF001
        connection.execute(
            """
            INSERT INTO jobs(
                job_id, owner_id, status, filename, message, doc_id,
                source_path, attempts, next_attempt_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "relative-source",
                "operator-owner",
                "queued",
                "paper.pdf",
                None,
                None,
                "uploads/operator-owner/paper.pdf",
                0,
                0.0,
                now,
                now,
            ),
        )


def test_relative_durable_source_path_fails_closed_everywhere(tmp_path):
    store = JobStore(path=tmp_path / "jobs.sqlite3")
    _insert_relative_source_row(store)

    assert store.get("relative-source", "operator-owner") is None
    assert store.get_internal("relative-source", "operator-owner") is None
    assert store.recoverable() == []
    assert store.active_source_paths() == set()

    corrupt = list_corrupt_jobs(store)
    assert len(corrupt) == 1
    assert "invalid_source_path" in corrupt[0].reasons
