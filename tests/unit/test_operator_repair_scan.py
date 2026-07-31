import time

from tools.job_store import JobStore
from tools.operator_repair import list_corrupt_jobs


def _insert_corrupt_row(store: JobStore, *, job_id: str) -> int:
    now = time.time()
    with store._lock, store._connect() as connection:  # noqa: SLF001
        cursor = connection.execute(
            """
            INSERT INTO jobs(
                job_id, owner_id, status, filename, message, doc_id,
                source_path, attempts, next_attempt_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                "operator-owner",
                "processing",
                "paper.pdf",
                None,
                None,
                None,
                -1,
                0.0,
                now,
                now,
            ),
        )
        return int(cursor.lastrowid)


def test_corrupt_listing_scans_past_valid_rows_to_fill_result_limit(tmp_path):
    store = JobStore(path=tmp_path / "jobs.sqlite3")
    for index in range(20):
        store.update(
            f"valid-{index}",
            "operator-owner",
            status="queued",
            filename=f"paper-{index}.pdf",
        )
    corrupt_rowid = _insert_corrupt_row(store, job_id="corrupt-after-valid-prefix")

    records = list_corrupt_jobs(store, limit=1)

    assert len(records) == 1
    assert records[0].rowid == corrupt_rowid
    assert "invalid_attempts" in records[0].reasons


def test_corrupt_listing_result_limit_applies_to_corrupt_rows_not_scanned_rows(tmp_path):
    store = JobStore(path=tmp_path / "jobs.sqlite3")
    first = _insert_corrupt_row(store, job_id="corrupt-one")
    second = _insert_corrupt_row(store, job_id="corrupt-two")

    records = list_corrupt_jobs(store, limit=1)

    assert [record.rowid for record in records] == [first]
    assert second > first
