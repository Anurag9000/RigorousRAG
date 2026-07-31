import json
import math
import time

import pytest

from tools.job_store import JobStore
from tools.operator_repair import (
    _row_fingerprint,
    list_corrupt_jobs,
    main,
    retire_corrupt_job,
)


def _insert_corrupt_row(store: JobStore, source_path: str) -> int:
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
                "bad\njob",
                "operator-owner",
                "processing",
                "paper.pdf",
                None,
                None,
                source_path,
                -1,
                0.0,
                now,
                now,
            ),
        )
        return int(cursor.lastrowid)


def test_corrupt_job_listing_is_sanitized_and_hides_source_path(tmp_path):
    store = JobStore(path=tmp_path / "jobs.sqlite3")
    private_source = str(tmp_path / "private" / "secret-paper.pdf")
    rowid = _insert_corrupt_row(store, private_source)

    records = list_corrupt_jobs(store)

    assert len(records) == 1
    record = records[0]
    assert record.rowid == rowid
    assert "invalid_job_id" in record.reasons
    assert "invalid_attempts" in record.reasons
    payload = json.dumps(record.public_dict())
    assert private_source not in payload
    assert "secret-paper.pdf" not in payload
    assert record.source_recorded is True


def test_retire_corrupt_job_requires_exact_confirmation_and_preserves_source(tmp_path):
    store = JobStore(path=tmp_path / "jobs.sqlite3")
    source = tmp_path / "private.pdf"
    source.write_bytes(b"retained")
    rowid = _insert_corrupt_row(store, str(source))
    record = list_corrupt_jobs(store)[0]

    with pytest.raises(ValueError, match="confirmation"):
        retire_corrupt_job(
            store,
            rowid=rowid,
            fingerprint=record.fingerprint,
            confirmation="wrong",
            reason="retire malformed recovery row",
        )

    result = retire_corrupt_job(
        store,
        rowid=rowid,
        fingerprint=record.fingerprint,
        confirmation=f"RETIRE-{rowid}-{record.fingerprint[:12]}",
        reason="retire malformed recovery row",
    )

    assert result["retired"] is True
    assert result["source_preserved"] is True
    assert source.read_bytes() == b"retained"
    with store._lock, store._connect() as connection:  # noqa: SLF001
        assert connection.execute(
            "SELECT 1 FROM jobs WHERE rowid=?", (rowid,)
        ).fetchone() is None
        audit = connection.execute(
            "SELECT action, job_rowid, row_fingerprint, source_preserved "
            "FROM operator_repairs"
        ).fetchone()
    assert dict(audit) == {
        "action": "retire_corrupt_job",
        "job_rowid": rowid,
        "row_fingerprint": record.fingerprint,
        "source_preserved": 1,
    }


def test_retirement_fails_when_row_changed_after_listing(tmp_path):
    store = JobStore(path=tmp_path / "jobs.sqlite3")
    rowid = _insert_corrupt_row(store, str(tmp_path / "private.pdf"))
    record = list_corrupt_jobs(store)[0]
    with store._lock, store._connect() as connection:  # noqa: SLF001
        connection.execute(
            "UPDATE jobs SET filename='changed.pdf' WHERE rowid=?", (rowid,)
        )

    with pytest.raises(RuntimeError, match="changed after inspection"):
        retire_corrupt_job(
            store,
            rowid=rowid,
            fingerprint=record.fingerprint,
            confirmation=f"RETIRE-{rowid}-{record.fingerprint[:12]}",
            reason="retire malformed recovery row",
        )


def test_valid_job_cannot_be_retired_by_corrupt_row_tool(tmp_path):
    store = JobStore(path=tmp_path / "jobs.sqlite3")
    store.update(
        "valid-job",
        "operator-owner",
        status="queued",
        filename="paper.pdf",
        source_path=str(tmp_path / "paper.pdf"),
    )
    with store._lock, store._connect() as connection:  # noqa: SLF001
        row = connection.execute(
            """
            SELECT rowid, job_id, owner_id, status, filename, message, doc_id,
                   source_path, attempts, next_attempt_at, created_at, updated_at
            FROM jobs WHERE job_id='valid-job'
            """
        ).fetchone()
    record = dict(row)
    rowid = int(record["rowid"])
    fingerprint = _row_fingerprint(record)

    assert list_corrupt_jobs(store) == []
    with pytest.raises(ValueError, match="valid"):
        retire_corrupt_job(
            store,
            rowid=rowid,
            fingerprint=fingerprint,
            confirmation=f"RETIRE-{rowid}-{fingerprint[:12]}",
            reason="must not retire valid rows",
        )


def test_cli_lists_corrupt_rows_with_scan_metadata_and_without_private_path(
    tmp_path,
    capsys,
):
    database = tmp_path / "jobs.sqlite3"
    store = JobStore(path=database)
    private_source = str(tmp_path / "private" / "secret.pdf")
    _insert_corrupt_row(store, private_source)

    assert main(["--job-db", str(database), "list"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert len(payload["records"]) == 1
    assert payload["scanned_rows"] == 1
    assert payload["next_after_rowid"] is None
    assert payload["complete"] is True
    assert private_source not in captured.out
    assert "secret.pdf" not in captured.out

def test_retirement_audit_reason_is_single_line_masked_and_clock_is_finite(
    tmp_path,
):
    store = JobStore(path=tmp_path / "jobs.sqlite3")
    rowid = _insert_corrupt_row(store, str(tmp_path / "private.pdf"))
    record = list_corrupt_jobs(store)[0]

    retire_corrupt_job(
        store,
        rowid=rowid,
        fingerprint=record.fingerprint,
        confirmation=f"RETIRE-{rowid}-{record.fingerprint[:12]}",
        reason="retire\napi_key=top-secret\x1b[31m row",
    )

    with store._lock, store._connect() as connection:  # noqa: SLF001
        audit = connection.execute(
            "SELECT repaired_at, reason FROM operator_repairs"
        ).fetchone()
    assert audit is not None
    assert math.isfinite(float(audit["repaired_at"]))
    assert float(audit["repaired_at"]) >= 0
    assert "\n" not in audit["reason"]
    assert "\r" not in audit["reason"]
    assert "\x1b" not in audit["reason"]
    assert "top-secret" not in audit["reason"]


def test_nonfinite_repair_clock_preserves_corrupt_row(tmp_path, monkeypatch):
    store = JobStore(path=tmp_path / "jobs.sqlite3")
    rowid = _insert_corrupt_row(store, str(tmp_path / "private.pdf"))
    record = list_corrupt_jobs(store)[0]
    monkeypatch.setattr("tools.operator_repair.time.time", lambda: float("nan"))

    with pytest.raises(RuntimeError, match="audit clock"):
        retire_corrupt_job(
            store,
            rowid=rowid,
            fingerprint=record.fingerprint,
            confirmation=f"RETIRE-{rowid}-{record.fingerprint[:12]}",
            reason="retire malformed recovery row",
        )

    with store._lock, store._connect() as connection:  # noqa: SLF001
        assert connection.execute(
            "SELECT 1 FROM jobs WHERE rowid=?", (rowid,)
        ).fetchone() is not None
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='operator_repairs'"
        ).fetchone()
    assert table is None
