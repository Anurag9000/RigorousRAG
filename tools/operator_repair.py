"""Explicit operator tooling for inspecting and retiring corrupt durable job rows.

The commands in this module are deliberately conservative:

* private source paths and raw database values are never printed;
* rows are identified by SQLite ``rowid`` plus a bounded content fingerprint;
* retirement requires an exact fingerprint and confirmation token;
* source files, vectors, and document-registry rows are never deleted implicitly;
* every retirement is recorded in an append-only audit table in the job database.

This is recovery tooling for malformed rows that normal startup replay skips. It is not a
remote API and should only be run by an operator with direct access to the service state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.job_store import JobStore
from tools.privacy import mask_metadata_text
from tools.security import normalize_owner_id

_ALLOWED_STATUSES = frozenset({"queued", "processing", "finalizing", "success", "failed"})
_MAX_ROWS = 10_000
_MAX_CELL_CHARS = 1_000_000
_FINGERPRINT_SAMPLE_CHARS = 4096
_AUDIT_REASON_CHARS = 500


@dataclass(frozen=True)
class CorruptJobRecord:
    rowid: int
    fingerprint: str
    reasons: tuple[str, ...]
    status: str
    filename: str
    source_recorded: bool
    updated_at: float | None

    def public_dict(self) -> dict[str, Any]:
        return {
            "rowid": self.rowid,
            "fingerprint": self.fingerprint,
            "reasons": list(self.reasons),
            "status": self.status,
            "filename": self.filename,
            "source_recorded": self.source_recorded,
            "updated_at": self.updated_at,
        }


def _contains_ascii_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _valid_identifier(value: Any, *, maximum: int) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and len(value.strip()) <= maximum
        and not _contains_ascii_control(value.strip())
    )


def _finite_nonnegative(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(numeric) and numeric >= 0


def _bounded_integer(value: Any, *, minimum: int, maximum: int) -> bool:
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    return minimum <= value <= maximum


def _safe_public_text(value: Any, *, limit: int, default: str) -> str:
    if isinstance(value, str):
        rendered = value
    elif value is None:
        rendered = default
    else:
        rendered = type(value).__name__
    return mask_metadata_text(rendered).strip()[:limit] or default


def _cell_fingerprint_bytes(value: Any) -> bytes:
    """Return a bounded deterministic representation without exposing the value."""

    if value is None:
        return b"null"
    if isinstance(value, bytes):
        payload = value
        type_name = b"blob"
    elif isinstance(value, str):
        payload = value.encode("utf-8", errors="surrogatepass")
        type_name = b"text"
    elif isinstance(value, bool):
        payload = b"1" if value else b"0"
        type_name = b"bool"
    elif isinstance(value, int):
        payload = str(value).encode("ascii")
        type_name = b"int"
    elif isinstance(value, float):
        payload = value.hex().encode("ascii")
        type_name = b"float"
    else:
        payload = type(value).__name__.encode("utf-8", errors="replace")
        type_name = b"other"

    length = len(payload)
    if length <= _MAX_CELL_CHARS:
        sampled = payload
    else:
        sampled = payload[:_FINGERPRINT_SAMPLE_CHARS] + payload[-_FINGERPRINT_SAMPLE_CHARS:]
    return b"|".join((type_name, str(length).encode("ascii"), sampled))


def _row_fingerprint(row: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in (
        "rowid",
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
    ):
        digest.update(key.encode("ascii"))
        digest.update(b"\0")
        digest.update(_cell_fingerprint_bytes(row.get(key)))
        digest.update(b"\0")
    return digest.hexdigest()


def _row_reasons(row: Mapping[str, Any]) -> tuple[str, ...]:
    reasons: list[str] = []
    job_id = row.get("job_id")
    owner_id = row.get("owner_id")
    status = row.get("status")
    filename = row.get("filename")
    message = row.get("message")
    doc_id = row.get("doc_id")
    source_path = row.get("source_path")

    if not _valid_identifier(job_id, maximum=200):
        reasons.append("invalid_job_id")
    try:
        normalize_owner_id(owner_id)
    except (TypeError, ValueError):
        reasons.append("invalid_owner_id")
    if not isinstance(status, str) or status not in _ALLOWED_STATUSES:
        reasons.append("invalid_status")
    if (
        not isinstance(filename, str)
        or not filename
        or len(filename) > 500
        or _contains_ascii_control(filename)
    ):
        reasons.append("invalid_filename")
    if message is not None and (
        not isinstance(message, str)
        or len(message) > 2000
        or _contains_ascii_control(message)
    ):
        reasons.append("invalid_message")
    if doc_id not in (None, "") and not _valid_identifier(doc_id, maximum=200):
        reasons.append("invalid_doc_id")
    if status in {"finalizing", "success"} and doc_id in (None, ""):
        reasons.append("missing_doc_id")
    if status not in {"finalizing", "success"} and doc_id not in (None, ""):
        reasons.append("unexpected_doc_id")
    if source_path not in (None, "") and (
        not isinstance(source_path, str)
        or len(source_path) > 4000
        or _contains_ascii_control(source_path)
        or not Path(source_path).is_absolute()
    ):
        reasons.append("invalid_source_path")
    if not _bounded_integer(row.get("attempts"), minimum=0, maximum=1_000_000):
        reasons.append("invalid_attempts")
    if not _finite_nonnegative(row.get("next_attempt_at")):
        reasons.append("invalid_next_attempt_at")
    if not _finite_nonnegative(row.get("created_at")):
        reasons.append("invalid_created_at")
    if not _finite_nonnegative(row.get("updated_at")):
        reasons.append("invalid_updated_at")

    return tuple(dict.fromkeys(reasons))


def _select_rows(connection: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT rowid, job_id, owner_id, status, filename, message, doc_id,
               source_path, attempts, next_attempt_at, created_at, updated_at
        FROM jobs
        ORDER BY rowid ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def list_corrupt_jobs(store: JobStore, *, limit: int = 1000) -> list[CorruptJobRecord]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_ROWS:
        raise ValueError(f"limit must be an integer between 1 and {_MAX_ROWS}.")

    records: list[CorruptJobRecord] = []
    with store._lock, store._connect() as connection:  # noqa: SLF001 - operator boundary
        rows = _select_rows(connection, limit)
    for sqlite_row in rows:
        row = dict(sqlite_row)
        reasons = _row_reasons(row)
        if not reasons:
            continue
        updated_at = None
        if _finite_nonnegative(row.get("updated_at")):
            updated_at = float(row["updated_at"])
        records.append(
            CorruptJobRecord(
                rowid=int(row["rowid"]),
                fingerprint=_row_fingerprint(row),
                reasons=reasons,
                status=_safe_public_text(row.get("status"), limit=50, default="invalid"),
                filename=_safe_public_text(row.get("filename"), limit=200, default="upload"),
                source_recorded=row.get("source_path") not in (None, ""),
                updated_at=updated_at,
            )
        )
    return records


def _ensure_audit_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS operator_repairs (
            repair_id INTEGER PRIMARY KEY AUTOINCREMENT,
            repaired_at REAL NOT NULL,
            action TEXT NOT NULL,
            job_rowid INTEGER NOT NULL,
            row_fingerprint TEXT NOT NULL,
            reason TEXT NOT NULL,
            source_preserved INTEGER NOT NULL
        )
        """
    )


def retire_corrupt_job(
    store: JobStore,
    *,
    rowid: int,
    fingerprint: str,
    confirmation: str,
    reason: str,
) -> dict[str, Any]:
    if isinstance(rowid, bool) or not isinstance(rowid, int) or rowid <= 0:
        raise ValueError("rowid must be a positive integer.")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("fingerprint must be a lowercase SHA-256 value.")
    expected_confirmation = f"RETIRE-{rowid}-{fingerprint[:12]}"
    if confirmation != expected_confirmation:
        raise ValueError(f"confirmation must exactly equal {expected_confirmation}.")
    if not isinstance(reason, str):
        raise ValueError("reason must be a string.")
    public_reason = mask_metadata_text(reason).strip()[:_AUDIT_REASON_CHARS]
    if not public_reason:
        raise ValueError("reason must not be empty.")

    with store._lock, store._connect() as connection:  # noqa: SLF001 - operator boundary
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT rowid, job_id, owner_id, status, filename, message, doc_id,
                   source_path, attempts, next_attempt_at, created_at, updated_at
            FROM jobs WHERE rowid=?
            """,
            (rowid,),
        ).fetchone()
        if row is None:
            raise LookupError("The selected job row no longer exists.")
        record = dict(row)
        current_fingerprint = _row_fingerprint(record)
        if current_fingerprint != fingerprint:
            raise RuntimeError("The selected job row changed after inspection.")
        reasons = _row_reasons(record)
        if not reasons:
            raise ValueError("The selected job row is valid and cannot be retired here.")

        _ensure_audit_table(connection)
        source_preserved = record.get("source_path") not in (None, "")
        connection.execute(
            """
            INSERT INTO operator_repairs(
                repaired_at, action, job_rowid, row_fingerprint, reason, source_preserved
            ) VALUES (?, 'retire_corrupt_job', ?, ?, ?, ?)
            """,
            (time.time(), rowid, fingerprint, public_reason, int(source_preserved)),
        )
        deleted = connection.execute("DELETE FROM jobs WHERE rowid=?", (rowid,))
        if deleted.rowcount != 1:
            raise RuntimeError("The selected job row could not be retired atomically.")

    return {
        "retired": True,
        "rowid": rowid,
        "fingerprint": fingerprint,
        "corruption_reasons": list(reasons),
        "source_preserved": source_preserved,
    }


def _bounded_argv(argv: Iterable[str] | None) -> list[str] | None:
    if argv is None:
        return None
    values: list[str] = []
    for index, value in enumerate(argv):
        if index >= 100:
            raise ValueError("Too many command-line arguments.")
        if not isinstance(value, str) or len(value) > 4096 or _contains_ascii_control(value):
            raise ValueError("Command-line argument is invalid or too long.")
        values.append(value)
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect or retire corrupt RigorousRAG durable job rows safely."
    )
    parser.add_argument("--job-db", default=None, help="Override JOB_DB_PATH.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List sanitized corrupt-row records.")
    list_parser.add_argument("--limit", type=int, default=1000)

    retire_parser = subparsers.add_parser(
        "retire", help="Retire one unchanged corrupt row without deleting its source."
    )
    retire_parser.add_argument("--rowid", type=int, required=True)
    retire_parser.add_argument("--fingerprint", required=True)
    retire_parser.add_argument("--confirm", required=True)
    retire_parser.add_argument("--reason", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = build_parser().parse_args(_bounded_argv(argv))
        store = JobStore(path=arguments.job_db)
        if arguments.command == "list":
            records = list_corrupt_jobs(store, limit=arguments.limit)
            print(json.dumps([record.public_dict() for record in records], indent=2))
            return 0
        result = retire_corrupt_job(
            store,
            rowid=arguments.rowid,
            fingerprint=arguments.fingerprint,
            confirmation=arguments.confirm,
            reason=arguments.reason,
        )
        print(json.dumps(result, indent=2))
        return 0
    except (LookupError, OSError, RuntimeError, sqlite3.Error, TypeError, ValueError) as exc:
        print(f"operator repair failed: {mask_metadata_text(str(exc))[:500]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
