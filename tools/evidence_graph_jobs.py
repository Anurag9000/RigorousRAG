"""Durable exact-generation jobs for derived evidence-graph reconciliation.

The journal is deliberately operator driven. A job describes one immutable
owner/document authoritative generation and may only build the derived graph
for that exact generation. It does not mutate vector, sparse, retained-source,
or authoritative generation state.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tools.security import normalize_owner_id

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_MAX_LIMIT = 10_000
_SCHEMA_VERSION = 1
_SOURCE_STATES = frozenset({"active", "restored", "deleted"})
_JOB_STATES = frozenset({"planned", "running", "completed", "failed", "cancelled"})


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("evidence graph job database path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("evidence graph job database path is invalid.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for part in (absolute, *absolute.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError("evidence graph job database path could not be validated.") from exc
        if _redirecting(info):
            raise ValueError("evidence graph job database path may not contain redirects.")
    return absolute


def _identifier(value: Any, label: str, maximum: int = 200) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip()
    if (
        not cleaned
        or len(cleaned) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in cleaned)
    ):
        raise ValueError(f"{label} is invalid.")
    return cleaned


def _digest(value: Any, label: str) -> str:
    cleaned = _identifier(value, label, 64).lower()
    if len(cleaned) != 64 or any(character not in "0123456789abcdef" for character in cleaned):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return cleaned


def _optional_digest(value: Any, label: str) -> str | None:
    return None if value is None else _digest(value, label)


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return result


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_graph_job_id(
    *,
    owner_id: str,
    doc_id: str,
    source_sequence: int,
    source_state: str,
    content_sha256: str,
    profile_fingerprint: str,
    sparse_generation: int,
) -> str:
    owner = normalize_owner_id(owner_id)
    document = _identifier(doc_id, "doc_id")
    sequence = _integer(source_sequence, "source_sequence", 1, 2**63 - 1)
    state = _identifier(source_state, "source_state", 20)
    if state not in _SOURCE_STATES:
        raise ValueError("source_state is unsupported.")
    content = _digest(content_sha256, "content_sha256")
    profile = _digest(profile_fingerprint, "profile_fingerprint")
    sparse = _integer(sparse_generation, "sparse_generation", 0, 2**63 - 1)
    if state in {"active", "restored"} and sparse <= 0:
        raise ValueError("active/restored graph jobs require a sparse generation.")
    if state == "deleted" and sparse != 0:
        raise ValueError("deleted graph jobs require sparse_generation=0.")
    return _canonical_digest(
        {
            "scope": "rigorousrag-derived-evidence-graph-job-v1",
            "owner_id": owner,
            "doc_id": document,
            "source_sequence": sequence,
            "source_state": state,
            "content_sha256": content,
            "profile_fingerprint": profile,
            "sparse_generation": sparse,
        }
    )


@dataclass(frozen=True)
class EvidenceGraphJob:
    job_id: str
    owner_id: str
    doc_id: str
    source_sequence: int
    source_state: str
    content_sha256: str
    profile_fingerprint: str
    sparse_generation: int
    state: str
    attempt_count: int
    max_attempts: int
    lease_owner: str | None
    lease_expires_at: float | None
    graph_digest: str | None
    failure_type: str | None
    created_at: float
    updated_at: float
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        document = _identifier(self.doc_id, "doc_id")
        sequence = _integer(self.source_sequence, "source_sequence", 1, 2**63 - 1)
        source_state = _identifier(self.source_state, "source_state", 20)
        if source_state not in _SOURCE_STATES:
            raise ValueError("source_state is unsupported.")
        content = _digest(self.content_sha256, "content_sha256")
        profile = _digest(self.profile_fingerprint, "profile_fingerprint")
        sparse = _integer(self.sparse_generation, "sparse_generation", 0, 2**63 - 1)
        expected = deterministic_graph_job_id(
            owner_id=owner,
            doc_id=document,
            source_sequence=sequence,
            source_state=source_state,
            content_sha256=content,
            profile_fingerprint=profile,
            sparse_generation=sparse,
        )
        if _digest(self.job_id, "job_id") != expected:
            raise ValueError("job_id does not match immutable job identity.")
        state = _identifier(self.state, "state", 20)
        if state not in _JOB_STATES:
            raise ValueError("job state is unsupported.")
        attempts = _integer(self.attempt_count, "attempt_count", 0, 1_000_000)
        maximum = _integer(self.max_attempts, "max_attempts", 1, 1_000_000)
        if attempts > maximum:
            raise ValueError("attempt_count may not exceed max_attempts.")
        lease_owner = None if self.lease_owner is None else _identifier(
            self.lease_owner, "lease_owner", 200
        )
        lease_expires = None if self.lease_expires_at is None else _timestamp(
            self.lease_expires_at, "lease_expires_at"
        )
        graph_digest = _optional_digest(self.graph_digest, "graph_digest")
        failure = None if self.failure_type is None else _identifier(
            self.failure_type, "failure_type", 200
        )
        created = _timestamp(self.created_at, "created_at")
        updated = _timestamp(self.updated_at, "updated_at")
        if updated < created:
            raise ValueError("updated_at may not precede created_at.")
        if state == "running":
            if lease_owner is None or lease_expires is None:
                raise ValueError("running jobs require an active lease.")
        elif lease_owner is not None or lease_expires is not None:
            raise ValueError("only running jobs may retain a lease.")
        if state == "completed":
            if graph_digest is None or failure is not None:
                raise ValueError("completed jobs require graph_digest and no failure.")
        elif graph_digest is not None:
            raise ValueError("only completed jobs may contain graph_digest.")
        if state == "failed" and failure is None:
            raise ValueError("failed jobs require a generic failure_type.")
        if state != "failed" and failure is not None:
            raise ValueError("only failed jobs may contain failure_type.")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("job schema is unsupported.")
        object.__setattr__(self, "job_id", expected)
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "doc_id", document)
        object.__setattr__(self, "source_sequence", sequence)
        object.__setattr__(self, "source_state", source_state)
        object.__setattr__(self, "content_sha256", content)
        object.__setattr__(self, "profile_fingerprint", profile)
        object.__setattr__(self, "sparse_generation", sparse)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "attempt_count", attempts)
        object.__setattr__(self, "max_attempts", maximum)
        object.__setattr__(self, "lease_owner", lease_owner)
        object.__setattr__(self, "lease_expires_at", lease_expires)
        object.__setattr__(self, "graph_digest", graph_digest)
        object.__setattr__(self, "failure_type", failure)
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)

    @classmethod
    def from_generation(
        cls,
        generation: Any,
        *,
        max_attempts: int = 3,
        now: float | None = None,
    ) -> "EvidenceGraphJob":
        timestamp = _timestamp(time.time() if now is None else now, "now")
        owner = getattr(generation, "owner_id", None)
        document = getattr(generation, "doc_id", None)
        sequence = getattr(generation, "sequence", None)
        state = getattr(generation, "state", None)
        content = getattr(generation, "content_sha256", None)
        profile = getattr(generation, "profile_fingerprint", None)
        sparse = getattr(generation, "sparse_generation", None)
        job_id = deterministic_graph_job_id(
            owner_id=owner,
            doc_id=document,
            source_sequence=sequence,
            source_state=state,
            content_sha256=content,
            profile_fingerprint=profile,
            sparse_generation=sparse,
        )
        return cls(
            job_id=job_id,
            owner_id=owner,
            doc_id=document,
            source_sequence=sequence,
            source_state=state,
            content_sha256=content,
            profile_fingerprint=profile,
            sparse_generation=sparse,
            state="planned",
            attempt_count=0,
            max_attempts=max_attempts,
            lease_owner=None,
            lease_expires_at=None,
            graph_digest=None,
            failure_type=None,
            created_at=timestamp,
            updated_at=timestamp,
        )

    @property
    def immutable_digest(self) -> str:
        value = asdict(self)
        for key in (
            "state",
            "attempt_count",
            "lease_owner",
            "lease_expires_at",
            "graph_digest",
            "failure_type",
            "created_at",
            "updated_at",
        ):
            value.pop(key, None)
        return _canonical_digest(value)


class EvidenceGraphJobJournal:
    """SQLite job journal with exclusive leases and idempotent exact jobs."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("evidence graph job database parent must be a directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("evidence graph job database is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
        ):
            raise RuntimeError("evidence graph job database parent identity changed.")
        if self._file_identity() != self._database_identity:
            raise RuntimeError("evidence graph job database identity changed.")

    def _connect(self) -> sqlite3.Connection:
        self._verify()
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with sqlite3.connect(self.path, timeout=30.0, isolation_level=None) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS evidence_graph_jobs (
                    job_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    source_sequence INTEGER NOT NULL,
                    source_state TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    profile_fingerprint TEXT NOT NULL,
                    sparse_generation INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    graph_digest TEXT,
                    failure_type TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    schema_version INTEGER NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS evidence_graph_job_identity
                    ON evidence_graph_jobs(
                        owner_id, doc_id, source_sequence, source_state,
                        content_sha256, profile_fingerprint, sparse_generation
                    );
                CREATE INDEX IF NOT EXISTS evidence_graph_job_queue
                    ON evidence_graph_jobs(owner_id, state, created_at, job_id);
                """
            )

    @staticmethod
    def _job(row: sqlite3.Row) -> EvidenceGraphJob:
        return EvidenceGraphJob(
            job_id=row["job_id"],
            owner_id=row["owner_id"],
            doc_id=row["doc_id"],
            source_sequence=int(row["source_sequence"]),
            source_state=row["source_state"],
            content_sha256=row["content_sha256"],
            profile_fingerprint=row["profile_fingerprint"],
            sparse_generation=int(row["sparse_generation"]),
            state=row["state"],
            attempt_count=int(row["attempt_count"]),
            max_attempts=int(row["max_attempts"]),
            lease_owner=row["lease_owner"],
            lease_expires_at=row["lease_expires_at"],
            graph_digest=row["graph_digest"],
            failure_type=row["failure_type"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            schema_version=int(row["schema_version"]),
        )

    def seed(self, job: EvidenceGraphJob) -> EvidenceGraphJob:
        if not isinstance(job, EvidenceGraphJob) or job.state != "planned":
            raise ValueError("seed requires a planned EvidenceGraphJob.")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT * FROM evidence_graph_jobs WHERE job_id=?",
                    (job.job_id,),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO evidence_graph_jobs VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                        """,
                        (
                            job.job_id,
                            job.owner_id,
                            job.doc_id,
                            job.source_sequence,
                            job.source_state,
                            job.content_sha256,
                            job.profile_fingerprint,
                            job.sparse_generation,
                            job.state,
                            job.attempt_count,
                            job.max_attempts,
                            None,
                            None,
                            None,
                            None,
                            job.created_at,
                            job.updated_at,
                            job.schema_version,
                        ),
                    )
                else:
                    stored = self._job(existing)
                    if stored.immutable_digest != job.immutable_digest:
                        raise RuntimeError("evidence graph job identity collision detected.")
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        result = self.get(job.job_id)
        if result is None:
            raise RuntimeError("seeded evidence graph job disappeared.")
        return result

    def get(self, job_id: str) -> EvidenceGraphJob | None:
        selected = _digest(job_id, "job_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM evidence_graph_jobs WHERE job_id=?", (selected,)
            ).fetchone()
        return None if row is None else self._job(row)

    def list(
        self,
        *,
        owner_id: str | None = None,
        state: str | None = None,
        limit: int = 100,
    ) -> tuple[EvidenceGraphJob, ...]:
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        clauses: list[str] = []
        values: list[Any] = []
        if owner_id is not None:
            clauses.append("owner_id=?")
            values.append(normalize_owner_id(owner_id))
        if state is not None:
            selected_state = _identifier(state, "state", 20)
            if selected_state not in _JOB_STATES:
                raise ValueError("job state is unsupported.")
            clauses.append("state=?")
            values.append(selected_state)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        values.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM evidence_graph_jobs{where} "
                "ORDER BY created_at, job_id LIMIT ?",
                values,
            ).fetchall()
        return tuple(self._job(row) for row in rows)

    def claim(
        self,
        *,
        owner_id: str,
        worker_id: str,
        lease_seconds: int = 60,
        now: float | None = None,
    ) -> EvidenceGraphJob | None:
        owner = normalize_owner_id(owner_id)
        worker = _identifier(worker_id, "worker_id", 200)
        duration = _integer(lease_seconds, "lease_seconds", 1, 86_400)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        expires = timestamp + duration
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT * FROM evidence_graph_jobs
                    WHERE owner_id=?
                      AND attempt_count < max_attempts
                      AND (
                        state='planned'
                        OR state='failed'
                        OR (state='running' AND lease_expires_at <= ?)
                      )
                    ORDER BY created_at, job_id
                    LIMIT 1
                    """,
                    (owner, timestamp),
                ).fetchone()
                if row is None:
                    connection.execute("COMMIT")
                    return None
                job = self._job(row)
                connection.execute(
                    """
                    UPDATE evidence_graph_jobs
                    SET state='running', attempt_count=?, lease_owner=?,
                        lease_expires_at=?, graph_digest=NULL, failure_type=NULL,
                        updated_at=?
                    WHERE job_id=?
                    """,
                    (
                        job.attempt_count + 1,
                        worker,
                        expires,
                        timestamp,
                        job.job_id,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        claimed = self.get(job.job_id)
        if claimed is None:
            raise RuntimeError("claimed evidence graph job disappeared.")
        return claimed

    def renew(
        self,
        job_id: str,
        *,
        worker_id: str,
        lease_seconds: int = 60,
        now: float | None = None,
    ) -> EvidenceGraphJob:
        selected = _digest(job_id, "job_id")
        worker = _identifier(worker_id, "worker_id", 200)
        duration = _integer(lease_seconds, "lease_seconds", 1, 86_400)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE evidence_graph_jobs
                SET lease_expires_at=?, updated_at=?
                WHERE job_id=? AND state='running' AND lease_owner=?
                  AND lease_expires_at > ?
                """,
                (timestamp + duration, timestamp, selected, worker, timestamp),
            )
        if cursor.rowcount != 1:
            raise RuntimeError("evidence graph job lease is unavailable.")
        result = self.get(selected)
        if result is None:
            raise RuntimeError("renewed evidence graph job disappeared.")
        return result

    def _finish(
        self,
        job_id: str,
        *,
        worker_id: str,
        state: str,
        graph_digest: str | None,
        failure_type: str | None,
        now: float | None,
    ) -> EvidenceGraphJob:
        selected = _digest(job_id, "job_id")
        worker = _identifier(worker_id, "worker_id", 200)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        digest_value = _optional_digest(graph_digest, "graph_digest")
        failure = None if failure_type is None else _identifier(
            failure_type, "failure_type", 200
        )
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE evidence_graph_jobs
                SET state=?, lease_owner=NULL, lease_expires_at=NULL,
                    graph_digest=?, failure_type=?, updated_at=?
                WHERE job_id=? AND state='running' AND lease_owner=?
                  AND lease_expires_at > ?
                """,
                (
                    state,
                    digest_value,
                    failure,
                    timestamp,
                    selected,
                    worker,
                    timestamp,
                ),
            )
        if cursor.rowcount != 1:
            raise RuntimeError("evidence graph job lease is unavailable.")
        result = self.get(selected)
        if result is None:
            raise RuntimeError("finished evidence graph job disappeared.")
        return result

    def complete(
        self,
        job_id: str,
        *,
        worker_id: str,
        graph_digest: str,
        now: float | None = None,
    ) -> EvidenceGraphJob:
        return self._finish(
            job_id,
            worker_id=worker_id,
            state="completed",
            graph_digest=graph_digest,
            failure_type=None,
            now=now,
        )

    def fail(
        self,
        job_id: str,
        *,
        worker_id: str,
        failure_type: str,
        now: float | None = None,
    ) -> EvidenceGraphJob:
        return self._finish(
            job_id,
            worker_id=worker_id,
            state="failed",
            graph_digest=None,
            failure_type=failure_type,
            now=now,
        )

    def cancel(
        self,
        job_id: str,
        *,
        owner_id: str,
        now: float | None = None,
    ) -> EvidenceGraphJob:
        selected = _digest(job_id, "job_id")
        owner = normalize_owner_id(owner_id)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE evidence_graph_jobs
                SET state='cancelled', failure_type=NULL, updated_at=?
                WHERE job_id=? AND owner_id=? AND state IN ('planned','failed')
                """,
                (timestamp, selected, owner),
            )
        if cursor.rowcount != 1:
            raise RuntimeError("evidence graph job cannot be cancelled.")
        result = self.get(selected)
        if result is None:
            raise RuntimeError("cancelled evidence graph job disappeared.")
        return result

    def retry_failed(
        self,
        job_id: str,
        *,
        owner_id: str,
        now: float | None = None,
    ) -> EvidenceGraphJob:
        selected = _digest(job_id, "job_id")
        owner = normalize_owner_id(owner_id)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE evidence_graph_jobs
                SET state='planned', attempt_count=0, failure_type=NULL,
                    updated_at=?
                WHERE job_id=? AND owner_id=? AND state='failed'
                """,
                (timestamp, selected, owner),
            )
        if cursor.rowcount != 1:
            raise RuntimeError("failed evidence graph job cannot be retried.")
        result = self.get(selected)
        if result is None:
            raise RuntimeError("retried evidence graph job disappeared.")
        return result


__all__ = [
    "EvidenceGraphJob",
    "EvidenceGraphJobJournal",
    "deterministic_graph_job_id",
]
