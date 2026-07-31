import time

import pytest

from tools.job_store import JobStore
from tools.operator_repair import list_corrupt_jobs, scan_corrupt_jobs


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

    scan = scan_corrupt_jobs(store, limit=1)

    assert [record.rowid for record in scan.records] == [first]
    assert scan.scanned_rows == 1
    assert scan.next_after_rowid == first
    assert scan.complete is False
    assert second > first


def test_scan_ceiling_exposes_cursor_and_next_page_finds_later_corruption(tmp_path):
    store = JobStore(path=tmp_path / "jobs.sqlite3")
    valid_rowids = []
    for index in range(3):
        store.update(
            f"valid-{index}",
            "operator-owner",
            status="queued",
            filename=f"paper-{index}.pdf",
        )
        with store._lock, store._connect() as connection:  # noqa: SLF001
            row = connection.execute(
                "SELECT rowid FROM jobs WHERE job_id=?",
                (f"valid-{index}",),
            ).fetchone()
        valid_rowids.append(int(row["rowid"]))
    corrupt_rowid = _insert_corrupt_row(store, job_id="corrupt-after-scan-page")

    first = scan_corrupt_jobs(store, limit=10, scan_limit=2)

    assert first.records == ()
    assert first.scanned_rows == 2
    assert first.next_after_rowid == valid_rowids[1]
    assert first.complete is False

    second = scan_corrupt_jobs(
        store,
        limit=10,
        after_rowid=first.next_after_rowid,
        scan_limit=2,
    )

    assert [record.rowid for record in second.records] == [corrupt_rowid]
    assert second.scanned_rows == 2
    assert second.next_after_rowid is None
    assert second.complete is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"limit": 0}, "limit"),
        ({"after_rowid": -1}, "after_rowid"),
        ({"scan_limit": 0}, "scan_limit"),
        ({"scan_limit": 100_001}, "scan_limit"),
    ],
)
def test_corrupt_scan_rejects_invalid_bounds(tmp_path, kwargs, message):
    store = JobStore(path=tmp_path / "jobs.sqlite3")

    with pytest.raises(ValueError, match=message):
        scan_corrupt_jobs(store, **kwargs)
