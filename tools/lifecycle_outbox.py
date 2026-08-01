"""Durable fourth-store lifecycle journal and registry reconciliation.

Vector, sparse, and generation state are committed by the authoritative index
coordinator. This module records the retained-source registry side effect as an
idempotent durable operation. A replay worker advances an operation only after
it can verify the exact current generation expected by the operation.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import operator
import os
import sqlite3
import stat
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from tools.security import normalize_owner_id

_SCHEMA_VERSION = 1
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_KINDS = {"replace", "delete"}
_STATES = {"planned", "index_committed", "registry_committed", "completed", "failed"}
_MAX_PATH = 4096
_MAX_LIMIT = 10_000
_MAX_ATTEMPTS = 100


def _redirecting(value: os.stat_result) -> bool:
    return stat.S_ISLNK(value.st_mode) or bool(
        int(getattr(value, "st_file_attributes", 0)) & _REPARSE
    )


def _identifier(value: Any, label: str, maximum: int = 200) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    rendered = value.strip()
    if (
        not rendered
        or len(rendered) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError(f"{label} is invalid.")
    return rendered


def _optional_text(value: Any, label: str, maximum: int) -> str | None:
    if value is None:
        return None
    return _identifier(value, label, maximum)


def _digest(value: Any, label: str) -> str:
    rendered = _identifier(value, label, 64).lower()
    if len(rendered) != 64 or any(
        character not in "0123456789abcdef" for character in rendered
    ):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return rendered


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        parsed = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return parsed


def _timestamp(value: Any, label: str = "timestamp") -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return parsed


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("lifecycle outbox path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("lifecycle outbox path is invalid.")
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
            raise ValueError("lifecycle outbox path could not be validated.") from exc
        if _redirecting(info):
            raise ValueError("lifecycle outbox path may not contain redirects.")
    return absolute


def operation_id_for(
    *,
    kind: str,
    owner_id: str,
    doc_id: str,
    content_sha256: str | None = None,
    idempotency_key: str | None = None,
) -> str:
    if kind not in _KINDS:
        raise ValueError("kind must be replace or delete.")
    owner = normalize_owner_id(owner_id)
    document = _identifier(doc_id, "doc_id")
    content = _digest(content_sha256, "content_sha256") if content_sha256 else "-"
    key = _optional_text(idempotency_key, "idempotency_key", 500) or "-"
    digest = hashlib.sha256(
        f"{kind}\0{owner}\0{document}\0{content}\0{key}".encode("utf-8")
    ).hexdigest()
    return f"lifecycle-{digest}"


@dataclass(frozen=True)
class LifecycleOperation:
    operation_id: str
    owner_id: str
    doc_id: str
    kind: str
    state: str
    content_sha256: str | None
    generation_sequence: int | None
    filename: str | None
    mime_type: str | None
    source_path: str | None
    retain_source: bool
    attempts: int
    max_attempts: int
    lease_owner: str | None
    lease_expires_at: float | None
    last_error_type: str | None
    created_at: float
    updated_at: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation_id",
            _identifier(self.operation_id, "operation_id", 200),
        )
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "doc_id", _identifier(self.doc_id, "doc_id"))
        if self.kind not in _KINDS:
            raise ValueError("kind is invalid.")
        if self.state not in _STATES:
            raise ValueError("state is invalid.")
        if self.content_sha256 is not None:
            object.__setattr__(
                self,
                "content_sha256",
                _digest(self.content_sha256, "content_sha256"),
            )
        if self.generation_sequence is not None:
            object.__setattr__(
                self,
                "generation_sequence",
                _integer(
                    self.generation_sequence,
                    "generation_sequence",
                    0,
                    2**63 - 1,
                ),
            )
        object.__setattr__(
            self,
            "filename",
            _optional_text(self.filename, "filename", 500),
        )
        object.__setattr__(
            self,
            "mime_type",
            _optional_text(self.mime_type, "mime_type", 200),
        )
        object.__setattr__(
            self,
            "source_path",
            _optional_text(self.source_path, "source_path", _MAX_PATH),
        )
        if not isinstance(self.retain_source, bool):
            raise ValueError("retain_source must be a boolean.")
        object.__setattr__(
            self,
            "attempts",
            _integer(self.attempts, "attempts", 0, _MAX_ATTEMPTS),
        )
        object.__setattr__(
            self,
            "max_attempts",
            _integer(self.max_attempts, "max_attempts", 1, _MAX_ATTEMPTS),
        )
        object.__setattr__(
            self,
            "lease_owner",
            _optional_text(self.lease_owner, "lease_owner", 200),
        )
        if self.lease_expires_at is not None:
            object.__setattr__(
                self,
                "lease_expires_at",
                _timestamp(self.lease_expires_at, "lease_expires_at"),
            )
        object.__setattr__(
            self,
            "last_error_type",
            _optional_text(self.last_error_type, "last_error_type", 200),
        )
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _timestamp(self.updated_at, "updated_at"))
        if self.kind == "replace":
            if (
                self.content_sha256 is None
                or self.filename is None
                or self.mime_type is None
            ):
                raise ValueError(
                    "replace operations require content, filename, and MIME type."
                )
            if self.retain_source and self.source_path is None:
                raise ValueError("retained replace operations require source_path.")
        elif any((self.content_sha256, self.filename, self.mime_type, self.source_path)):
            raise ValueError("delete operations may not carry replace metadata.")


@dataclass(frozen=True)
class LifecyclePublicSummary:
    operation_id: str
    owner_id: str
    doc_id: str
    kind: str
    state: str
    generation_sequence: int | None
    retain_source: bool
    attempts: int
    max_attempts: int
    last_error_type: str | None
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class LifecycleReconcileResult:
    operation_id: str
    outcome: str
    state: str
    source_cleanup_required: str | None = None


class _GenerationStore(Protocol):
    def current(self, *, owner_id: str, doc_id: str) -> Any: ...


class _Registry(Protocol):
    def register(self, **kwargs: Any) -> str | None: ...

    def get(
        self,
        *,
        owner_id: str,
        doc_id: str,
        **kwargs: Any,
    ) -> Mapping[str, Any] | None: ...

    def delete(
        self,
        *,
        owner_id: str,
        doc_id: str,
    ) -> Mapping[str, Any] | None: ...


class LifecycleOutbox:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("lifecycle outbox parent must be a regular directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._identity()

    def _identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("lifecycle outbox is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
        ):
            raise RuntimeError("lifecycle outbox parent identity changed.")
        if self._identity() != self._database_identity:
            raise RuntimeError("lifecycle outbox identity changed.")

    def _connect(self) -> sqlite3.Connection:
        self._verify()
        connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
        ) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS lifecycle_schema(
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    schema_version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS lifecycle_operations(
                    operation_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    state TEXT NOT NULL,
                    content_sha256 TEXT,
                    generation_sequence INTEGER,
                    filename TEXT,
                    mime_type TEXT,
                    source_path TEXT,
                    retain_source INTEGER NOT NULL,
                    attempts INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    last_error_type TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_lifecycle_pending
                    ON lifecycle_operations(state, lease_expires_at, updated_at);
                CREATE INDEX IF NOT EXISTS idx_lifecycle_owner_doc
                    ON lifecycle_operations(owner_id, doc_id, updated_at);
                """
            )
            row = connection.execute(
                "SELECT schema_version FROM lifecycle_schema WHERE singleton=1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO lifecycle_schema(singleton, schema_version) "
                    "VALUES(1, ?)",
                    (_SCHEMA_VERSION,),
                )
            elif int(row[0]) != _SCHEMA_VERSION:
                raise RuntimeError("lifecycle outbox schema is unsupported.")

    @staticmethod
    def _record(row: sqlite3.Row) -> LifecycleOperation:
        return LifecycleOperation(
            operation_id=row["operation_id"],
            owner_id=row["owner_id"],
            doc_id=row["doc_id"],
            kind=row["kind"],
            state=row["state"],
            content_sha256=row["content_sha256"],
            generation_sequence=row["generation_sequence"],
            filename=row["filename"],
            mime_type=row["mime_type"],
            source_path=row["source_path"],
            retain_source=bool(row["retain_source"]),
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            lease_owner=row["lease_owner"],
            lease_expires_at=row["lease_expires_at"],
            last_error_type=row["last_error_type"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def public(operation: LifecycleOperation) -> LifecyclePublicSummary:
        if not isinstance(operation, LifecycleOperation):
            raise ValueError("operation must be a LifecycleOperation.")
        return LifecyclePublicSummary(
            operation.operation_id,
            operation.owner_id,
            operation.doc_id,
            operation.kind,
            operation.state,
            operation.generation_sequence,
            operation.retain_source,
            operation.attempts,
            operation.max_attempts,
            operation.last_error_type,
            operation.created_at,
            operation.updated_at,
        )

    def get(self, operation_id: str) -> LifecycleOperation | None:
        identifier = _identifier(operation_id, "operation_id", 200)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM lifecycle_operations WHERE operation_id=?",
                (identifier,),
            ).fetchone()
        return None if row is None else self._record(row)

    def _plan(self, operation: LifecycleOperation) -> LifecycleOperation:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM lifecycle_operations WHERE operation_id=?",
                (operation.operation_id,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO lifecycle_operations(
                        operation_id, owner_id, doc_id, kind, state,
                        content_sha256, generation_sequence, filename, mime_type,
                        source_path, retain_source, attempts, max_attempts,
                        lease_owner, lease_expires_at, last_error_type,
                        created_at, updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        operation.operation_id,
                        operation.owner_id,
                        operation.doc_id,
                        operation.kind,
                        operation.state,
                        operation.content_sha256,
                        operation.generation_sequence,
                        operation.filename,
                        operation.mime_type,
                        operation.source_path,
                        int(operation.retain_source),
                        operation.attempts,
                        operation.max_attempts,
                        operation.lease_owner,
                        operation.lease_expires_at,
                        operation.last_error_type,
                        operation.created_at,
                        operation.updated_at,
                    ),
                )
                connection.execute("COMMIT")
                return operation
            current = self._record(existing)
            immutable = (
                "owner_id",
                "doc_id",
                "kind",
                "content_sha256",
                "filename",
                "mime_type",
                "source_path",
                "retain_source",
                "max_attempts",
            )
            if any(
                getattr(current, name) != getattr(operation, name)
                for name in immutable
            ):
                connection.execute("ROLLBACK")
                raise ValueError(
                    "operation_id already identifies a different lifecycle operation."
                )
            connection.execute("COMMIT")
            return current

    def plan_replace(
        self,
        *,
        operation_id: str,
        owner_id: str,
        doc_id: str,
        content_sha256: str,
        filename: str,
        mime_type: str,
        source_path: str | None,
        retain_source: bool,
        max_attempts: int = 8,
        now: float | None = None,
    ) -> LifecycleOperation:
        timestamp = time.time() if now is None else _timestamp(now)
        operation = LifecycleOperation(
            operation_id,
            owner_id,
            doc_id,
            "replace",
            "planned",
            content_sha256,
            None,
            filename,
            mime_type,
            source_path,
            retain_source,
            0,
            max_attempts,
            None,
            None,
            None,
            timestamp,
            timestamp,
        )
        return self._plan(operation)

    def plan_delete(
        self,
        *,
        operation_id: str,
        owner_id: str,
        doc_id: str,
        max_attempts: int = 8,
        now: float | None = None,
    ) -> LifecycleOperation:
        timestamp = time.time() if now is None else _timestamp(now)
        operation = LifecycleOperation(
            operation_id,
            owner_id,
            doc_id,
            "delete",
            "planned",
            None,
            None,
            None,
            None,
            None,
            False,
            0,
            max_attempts,
            None,
            None,
            None,
            timestamp,
            timestamp,
        )
        return self._plan(operation)

    def _transition(
        self,
        operation_id: str,
        *,
        allowed: set[str],
        target: str,
        generation_sequence: int | None = None,
        now: float | None = None,
    ) -> LifecycleOperation:
        identifier = _identifier(operation_id, "operation_id", 200)
        timestamp = time.time() if now is None else _timestamp(now)
        sequence = (
            _integer(generation_sequence, "generation_sequence", 0, 2**63 - 1)
            if generation_sequence is not None
            else None
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM lifecycle_operations WHERE operation_id=?",
                (identifier,),
            ).fetchone()
            if row is None:
                connection.execute("ROLLBACK")
                raise ValueError("lifecycle operation was not found.")
            current = self._record(row)
            if current.state == target:
                if sequence is not None and current.generation_sequence != sequence:
                    connection.execute("ROLLBACK")
                    raise ValueError(
                        "generation_sequence conflicts with the committed transition."
                    )
                connection.execute("COMMIT")
                return current
            if current.state not in allowed:
                connection.execute("ROLLBACK")
                raise ValueError(
                    f"cannot transition lifecycle operation from "
                    f"{current.state} to {target}."
                )
            if target == "completed":
                connection.execute(
                    """
                    UPDATE lifecycle_operations
                       SET state=?,
                           generation_sequence=COALESCE(?, generation_sequence),
                           lease_owner=NULL,
                           lease_expires_at=NULL,
                           last_error_type=NULL,
                           updated_at=?
                     WHERE operation_id=?
                    """,
                    (target, sequence, timestamp, identifier),
                )
            else:
                connection.execute(
                    """
                    UPDATE lifecycle_operations
                       SET state=?,
                           generation_sequence=COALESCE(?, generation_sequence),
                           last_error_type=NULL,
                           updated_at=?
                     WHERE operation_id=?
                    """,
                    (target, sequence, timestamp, identifier),
                )
            row = connection.execute(
                "SELECT * FROM lifecycle_operations WHERE operation_id=?",
                (identifier,),
            ).fetchone()
            connection.execute("COMMIT")
        return self._record(row)

    def mark_index_committed(
        self,
        operation_id: str,
        *,
        generation_sequence: int,
        now: float | None = None,
    ) -> LifecycleOperation:
        return self._transition(
            operation_id,
            allowed={"planned"},
            target="index_committed",
            generation_sequence=generation_sequence,
            now=now,
        )

    def mark_registry_committed(
        self,
        operation_id: str,
        *,
        now: float | None = None,
    ) -> LifecycleOperation:
        return self._transition(
            operation_id,
            allowed={"index_committed"},
            target="registry_committed",
            now=now,
        )

    def complete(
        self,
        operation_id: str,
        *,
        now: float | None = None,
    ) -> LifecycleOperation:
        return self._transition(
            operation_id,
            allowed={"registry_committed"},
            target="completed",
            now=now,
        )

    def claim(
        self,
        *,
        worker_id: str,
        limit: int = 50,
        lease_seconds: float = 60.0,
        now: float | None = None,
    ) -> tuple[LifecycleOperation, ...]:
        worker = _identifier(worker_id, "worker_id", 200)
        maximum = _integer(limit, "limit", 1, _MAX_LIMIT)
        current = time.time() if now is None else _timestamp(now)
        duration = _timestamp(lease_seconds, "lease_seconds")
        if duration <= 0 or duration > 86_400:
            raise ValueError(
                "lease_seconds must be greater than zero and at most 86400."
            )
        expires = current + duration
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT operation_id FROM lifecycle_operations
                 WHERE state IN ('planned','index_committed','registry_committed')
                   AND attempts < max_attempts
                   AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                 ORDER BY updated_at, operation_id
                 LIMIT ?
                """,
                (current, maximum),
            ).fetchall()
            identifiers = [row["operation_id"] for row in rows]
            for identifier in identifiers:
                connection.execute(
                    """
                    UPDATE lifecycle_operations
                       SET lease_owner=?, lease_expires_at=?, updated_at=?
                     WHERE operation_id=?
                    """,
                    (worker, expires, current, identifier),
                )
            claimed = [
                connection.execute(
                    "SELECT * FROM lifecycle_operations WHERE operation_id=?",
                    (identifier,),
                ).fetchone()
                for identifier in identifiers
            ]
            connection.execute("COMMIT")
        return tuple(self._record(row) for row in claimed if row is not None)

    def renew(
        self,
        operation_id: str,
        *,
        worker_id: str,
        lease_seconds: float = 60.0,
        now: float | None = None,
    ) -> LifecycleOperation:
        identifier = _identifier(operation_id, "operation_id", 200)
        worker = _identifier(worker_id, "worker_id", 200)
        current = time.time() if now is None else _timestamp(now)
        duration = _timestamp(lease_seconds, "lease_seconds")
        if duration <= 0 or duration > 86_400:
            raise ValueError(
                "lease_seconds must be greater than zero and at most 86400."
            )
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE lifecycle_operations
                   SET lease_expires_at=?, updated_at=?
                 WHERE operation_id=? AND lease_owner=?
                   AND lease_expires_at > ?
                   AND state IN ('planned','index_committed','registry_committed')
                """,
                (current + duration, current, identifier, worker, current),
            )
            if cursor.rowcount != 1:
                raise ValueError(
                    "lifecycle lease is absent, expired, or owned by another worker."
                )
            row = connection.execute(
                "SELECT * FROM lifecycle_operations WHERE operation_id=?",
                (identifier,),
            ).fetchone()
        return self._record(row)

    def release(
        self,
        operation_id: str,
        *,
        worker_id: str,
        now: float | None = None,
    ) -> LifecycleOperation:
        identifier = _identifier(operation_id, "operation_id", 200)
        worker = _identifier(worker_id, "worker_id", 200)
        current = time.time() if now is None else _timestamp(now)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE lifecycle_operations
                   SET lease_owner=NULL, lease_expires_at=NULL, updated_at=?
                 WHERE operation_id=? AND lease_owner=?
                   AND state IN ('planned','index_committed','registry_committed')
                """,
                (current, identifier, worker),
            )
            if cursor.rowcount != 1:
                raise ValueError(
                    "lifecycle lease is absent or owned by another worker."
                )
            row = connection.execute(
                "SELECT * FROM lifecycle_operations WHERE operation_id=?",
                (identifier,),
            ).fetchone()
        return self._record(row)

    def record_failure(
        self,
        operation_id: str,
        *,
        worker_id: str,
        error_type: str,
        now: float | None = None,
    ) -> LifecycleOperation:
        identifier = _identifier(operation_id, "operation_id", 200)
        worker = _identifier(worker_id, "worker_id", 200)
        generic = _identifier(error_type, "error_type", 200)
        current = time.time() if now is None else _timestamp(now)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM lifecycle_operations WHERE operation_id=?",
                (identifier,),
            ).fetchone()
            if row is None:
                connection.execute("ROLLBACK")
                raise ValueError("lifecycle operation was not found.")
            operation = self._record(row)
            if operation.lease_owner != worker:
                connection.execute("ROLLBACK")
                raise ValueError("lifecycle lease is owned by another worker.")
            attempts = operation.attempts + 1
            state = (
                "failed"
                if attempts >= operation.max_attempts
                else operation.state
            )
            connection.execute(
                """
                UPDATE lifecycle_operations
                   SET attempts=?, state=?, lease_owner=NULL,
                       lease_expires_at=NULL, last_error_type=?, updated_at=?
                 WHERE operation_id=?
                """,
                (attempts, state, generic, current, identifier),
            )
            row = connection.execute(
                "SELECT * FROM lifecycle_operations WHERE operation_id=?",
                (identifier,),
            ).fetchone()
            connection.execute("COMMIT")
        return self._record(row)

    def retry_failed(
        self,
        operation_id: str,
        *,
        now: float | None = None,
    ) -> LifecycleOperation:
        identifier = _identifier(operation_id, "operation_id", 200)
        current = time.time() if now is None else _timestamp(now)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE lifecycle_operations
                   SET state='planned', attempts=0, lease_owner=NULL,
                       lease_expires_at=NULL, last_error_type=NULL,
                       generation_sequence=NULL, updated_at=?
                 WHERE operation_id=? AND state='failed'
                """,
                (current, identifier),
            )
            if cursor.rowcount != 1:
                raise ValueError(
                    "only failed lifecycle operations may be retried."
                )
            row = connection.execute(
                "SELECT * FROM lifecycle_operations WHERE operation_id=?",
                (identifier,),
            ).fetchone()
        return self._record(row)

    def list_pending(
        self,
        *,
        owner_id: str | None = None,
        limit: int = 100,
    ) -> tuple[LifecyclePublicSummary, ...]:
        maximum = _integer(limit, "limit", 1, _MAX_LIMIT)
        owner = normalize_owner_id(owner_id) if owner_id is not None else None
        query = (
            "SELECT * FROM lifecycle_operations "
            "WHERE state IN ('planned','index_committed','registry_committed') "
            + ("AND owner_id=? " if owner is not None else "")
            + "ORDER BY updated_at, operation_id LIMIT ?"
        )
        parameters: tuple[Any, ...] = (
            (owner, maximum) if owner is not None else (maximum,)
        )
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(self.public(self._record(row)) for row in rows)

    def ping(self) -> bool:
        try:
            with self._lock, self._connect() as connection:
                row = connection.execute("SELECT 1").fetchone()
            return bool(row and int(row[0]) == 1)
        except Exception:
            return False


def _generation_matches_replace(
    operation: LifecycleOperation,
    generation: Any,
) -> bool:
    return bool(
        generation is not None
        and getattr(generation, "state", None) in {"active", "restored"}
        and getattr(generation, "content_sha256", None) == operation.content_sha256
        and (
            operation.generation_sequence is None
            or getattr(generation, "sequence", None)
            == operation.generation_sequence
        )
    )


def _generation_matches_delete(
    operation: LifecycleOperation,
    generation: Any,
) -> bool:
    if generation is None:
        return operation.generation_sequence in {None, 0}
    return bool(
        getattr(generation, "state", None) == "deleted"
        and (
            operation.generation_sequence is None
            or getattr(generation, "sequence", None)
            == operation.generation_sequence
        )
    )


def reconcile_lifecycle_operation(
    operation_id: str,
    *,
    outbox: LifecycleOutbox,
    generations: _GenerationStore,
    registry: _Registry,
    remove_source: Callable[[str], bool] | None = None,
) -> LifecycleReconcileResult:
    operation = outbox.get(operation_id)
    if operation is None:
        raise ValueError("lifecycle operation was not found.")
    if operation.state == "completed":
        return LifecycleReconcileResult(
            operation.operation_id,
            "already_completed",
            operation.state,
        )
    if operation.state == "failed":
        return LifecycleReconcileResult(
            operation.operation_id,
            "failed",
            operation.state,
        )

    generation = generations.current(
        owner_id=operation.owner_id,
        doc_id=operation.doc_id,
    )
    if operation.kind == "replace":
        if not _generation_matches_replace(operation, generation):
            return LifecycleReconcileResult(
                operation.operation_id,
                "waiting_for_matching_generation",
                operation.state,
            )
        if operation.state == "planned":
            operation = outbox.mark_index_committed(
                operation.operation_id,
                generation_sequence=int(getattr(generation, "sequence")),
            )
        if operation.state == "index_committed":
            previous = registry.register(
                owner_id=operation.owner_id,
                doc_id=operation.doc_id,
                filename=operation.filename,
                mime_type=operation.mime_type,
                source_path=(
                    operation.source_path if operation.retain_source else None
                ),
            )
            operation = outbox.mark_registry_committed(operation.operation_id)
            cleanup = None
            if (
                isinstance(previous, str)
                and previous
                and previous != operation.source_path
            ):
                cleanup = previous
                if remove_source is not None:
                    try:
                        if remove_source(previous):
                            cleanup = None
                    except Exception:
                        pass
            operation = outbox.complete(operation.operation_id)
            return LifecycleReconcileResult(
                operation.operation_id,
                "completed",
                operation.state,
                cleanup,
            )
        if operation.state == "registry_committed":
            operation = outbox.complete(operation.operation_id)
            return LifecycleReconcileResult(
                operation.operation_id,
                "completed",
                operation.state,
            )

    if operation.kind == "delete":
        if not _generation_matches_delete(operation, generation):
            return LifecycleReconcileResult(
                operation.operation_id,
                "waiting_for_deleted_generation",
                operation.state,
            )
        if operation.state == "planned":
            sequence = (
                int(getattr(generation, "sequence", 0))
                if generation is not None
                else 0
            )
            operation = outbox.mark_index_committed(
                operation.operation_id,
                generation_sequence=sequence,
            )
        if operation.state == "index_committed":
            record = registry.delete(
                owner_id=operation.owner_id,
                doc_id=operation.doc_id,
            )
            operation = outbox.mark_registry_committed(operation.operation_id)
            source_path = str((record or {}).get("source_path") or "")
            cleanup = source_path or None
            if cleanup is not None and remove_source is not None:
                try:
                    if remove_source(cleanup):
                        cleanup = None
                except Exception:
                    pass
            operation = outbox.complete(operation.operation_id)
            return LifecycleReconcileResult(
                operation.operation_id,
                "completed",
                operation.state,
                cleanup,
            )
        if operation.state == "registry_committed":
            operation = outbox.complete(operation.operation_id)
            return LifecycleReconcileResult(
                operation.operation_id,
                "completed",
                operation.state,
            )

    raise RuntimeError("lifecycle operation entered an unsupported state.")


def reconcile_claimed_operations(
    operations: Iterable[LifecycleOperation],
    *,
    outbox: LifecycleOutbox,
    generations: _GenerationStore,
    registry: _Registry,
    worker_id: str,
    remove_source: Callable[[str], bool] | None = None,
) -> tuple[LifecycleReconcileResult, ...]:
    worker = _identifier(worker_id, "worker_id", 200)
    if isinstance(operations, (str, bytes, bytearray)):
        raise ValueError("operations must be an iterable.")
    results: list[LifecycleReconcileResult] = []
    try:
        rows = list(itertools.islice(iter(operations), _MAX_LIMIT + 1))
    except Exception as exc:
        raise ValueError("operations must be safely iterable.") from exc
    if len(rows) > _MAX_LIMIT or any(
        not isinstance(row, LifecycleOperation) for row in rows
    ):
        raise ValueError("operations are invalid or exceed the limit.")
    for operation in rows:
        try:
            result = reconcile_lifecycle_operation(
                operation.operation_id,
                outbox=outbox,
                generations=generations,
                registry=registry,
                remove_source=remove_source,
            )
            results.append(result)
            if result.outcome.startswith("waiting_for_"):
                outbox.release(operation.operation_id, worker_id=worker)
        except Exception as exc:
            outbox.record_failure(
                operation.operation_id,
                worker_id=worker,
                error_type=type(exc).__name__,
            )
            results.append(
                LifecycleReconcileResult(
                    operation.operation_id,
                    "error",
                    (outbox.get(operation.operation_id) or operation).state,
                )
            )
    return tuple(results)


__all__ = [
    "LifecycleOperation",
    "LifecycleOutbox",
    "LifecyclePublicSummary",
    "LifecycleReconcileResult",
    "operation_id_for",
    "reconcile_claimed_operations",
    "reconcile_lifecycle_operation",
]
