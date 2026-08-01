"""Crash-safe retained-source cleanup and fourth-store reconciliation.

The primary lifecycle outbox records replacement/deletion phases. This module
adds a private cleanup journal so a prior retained source cannot be forgotten
between registry mutation and file removal. Cleanup intent is persisted before
registry mutation and is cleared only after idempotent removal succeeds.
"""

from __future__ import annotations

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

from tools.lifecycle_outbox import (
    LifecycleOperation,
    LifecycleOutbox,
    LifecycleReconcileResult,
)

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_MAX_LIMIT = 10_000
_LOCK = threading.RLock()
_JOURNALS: dict[str, "LifecycleCleanupJournal"] = {}


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


def _path_text(value: Any, label: str = "source_path") -> str:
    return _identifier(value, label, _MAX_PATH)


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


def _timestamp(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("timestamp must be finite and non-negative.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("timestamp must be finite and non-negative.") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError("timestamp must be finite and non-negative.")
    return parsed


def _database_path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("cleanup journal path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("cleanup journal path is invalid.")
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
            raise ValueError("cleanup journal path could not be validated.") from exc
        if _redirecting(info):
            raise ValueError("cleanup journal path may not contain redirects.")
    return absolute


@dataclass(frozen=True)
class CleanupIntent:
    operation_id: str
    source_path: str
    created_at: float
    updated_at: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation_id",
            _identifier(self.operation_id, "operation_id", 200),
        )
        object.__setattr__(
            self,
            "source_path",
            _path_text(self.source_path),
        )
        object.__setattr__(self, "created_at", _timestamp(self.created_at))
        object.__setattr__(self, "updated_at", _timestamp(self.updated_at))


class LifecycleCleanupJournal:
    """Private path journal with identity-bound SQLite storage."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _database_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("cleanup journal parent must be a regular directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._identity()

    def _identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("cleanup journal is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
        ):
            raise RuntimeError("cleanup journal parent identity changed.")
        if self._identity() != self._database_identity:
            raise RuntimeError("cleanup journal identity changed.")

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
                CREATE TABLE IF NOT EXISTS lifecycle_cleanup_intents(
                    operation_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> CleanupIntent:
        return CleanupIntent(
            row["operation_id"],
            row["source_path"],
            row["created_at"],
            row["updated_at"],
        )

    def record(
        self,
        operation_id: str,
        source_path: str,
        *,
        now: float | None = None,
    ) -> CleanupIntent:
        identifier = _identifier(operation_id, "operation_id", 200)
        source = _path_text(source_path)
        current = time.time() if now is None else _timestamp(now)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM lifecycle_cleanup_intents WHERE operation_id=?",
                (identifier,),
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO lifecycle_cleanup_intents(
                        operation_id, source_path, created_at, updated_at
                    ) VALUES(?,?,?,?)
                    """,
                    (identifier, source, current, current),
                )
            else:
                existing = self._record(row)
                if existing.source_path != source:
                    connection.execute("ROLLBACK")
                    raise ValueError(
                        "operation_id already identifies a different cleanup path."
                    )
                connection.execute(
                    "UPDATE lifecycle_cleanup_intents SET updated_at=? "
                    "WHERE operation_id=?",
                    (current, identifier),
                )
            row = connection.execute(
                "SELECT * FROM lifecycle_cleanup_intents WHERE operation_id=?",
                (identifier,),
            ).fetchone()
            connection.execute("COMMIT")
        return self._record(row)

    def get(self, operation_id: str) -> CleanupIntent | None:
        identifier = _identifier(operation_id, "operation_id", 200)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM lifecycle_cleanup_intents WHERE operation_id=?",
                (identifier,),
            ).fetchone()
        return None if row is None else self._record(row)

    def clear(self, operation_id: str) -> bool:
        identifier = _identifier(operation_id, "operation_id", 200)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM lifecycle_cleanup_intents WHERE operation_id=?",
                (identifier,),
            )
        return cursor.rowcount == 1

    def ping(self) -> bool:
        try:
            with self._lock, self._connect() as connection:
                row = connection.execute("SELECT 1").fetchone()
            return bool(row and int(row[0]) == 1)
        except Exception:
            return False


def get_cleanup_journal(
    path: str | os.PathLike[str] | None = None,
) -> LifecycleCleanupJournal:
    selected = path or os.getenv(
        "LIFECYCLE_CLEANUP_DB_PATH",
        "data/lifecycle_cleanup.sqlite3",
    )
    absolute = str(_database_path(selected))
    with _LOCK:
        journal = _JOURNALS.get(absolute)
        if journal is None:
            journal = LifecycleCleanupJournal(absolute)
            _JOURNALS[absolute] = journal
        elif not journal.ping():
            raise RuntimeError("cleanup journal is unavailable.")
        return journal


def clear_cleanup_runtime_cache() -> None:
    with _LOCK:
        _JOURNALS.clear()


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


def _replace_generation_matches(operation: LifecycleOperation, generation: Any) -> bool:
    return bool(
        generation is not None
        and getattr(generation, "state", None) in {"active", "restored"}
        and getattr(generation, "content_sha256", None)
        == operation.content_sha256
        and (
            operation.generation_sequence is None
            or getattr(generation, "sequence", None)
            == operation.generation_sequence
        )
    )


def _delete_generation_matches(operation: LifecycleOperation, generation: Any) -> bool:
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


def _cleanup_or_wait(
    operation: LifecycleOperation,
    *,
    outbox: LifecycleOutbox,
    cleanup: LifecycleCleanupJournal,
    remove_source: Callable[[str], bool] | None,
) -> LifecycleReconcileResult:
    intent = cleanup.get(operation.operation_id)
    if intent is not None:
        removed = False
        if remove_source is not None:
            try:
                removed = bool(remove_source(intent.source_path))
            except Exception:
                removed = False
        if not removed:
            return LifecycleReconcileResult(
                operation.operation_id,
                "cleanup_required",
                operation.state,
                intent.source_path,
            )
        cleanup.clear(operation.operation_id)
    operation = outbox.complete(operation.operation_id)
    return LifecycleReconcileResult(
        operation.operation_id,
        "completed",
        operation.state,
    )


def reconcile_lifecycle_operation(
    operation_id: str,
    *,
    outbox: LifecycleOutbox,
    generations: _GenerationStore,
    registry: _Registry,
    cleanup: LifecycleCleanupJournal | None = None,
    remove_source: Callable[[str], bool] | None = None,
) -> LifecycleReconcileResult:
    operation = outbox.get(operation_id)
    if operation is None:
        raise ValueError("lifecycle operation was not found.")
    cleanup_store = cleanup or get_cleanup_journal()
    if operation.state == "completed":
        intent = cleanup_store.get(operation.operation_id)
        if intent is not None:
            raise RuntimeError("completed lifecycle operation retains cleanup intent.")
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
        if not _replace_generation_matches(operation, generation):
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
            prior = registry.get(
                owner_id=operation.owner_id,
                doc_id=operation.doc_id,
            )
            prior_path = str((prior or {}).get("source_path") or "")
            if prior_path and prior_path != operation.source_path:
                cleanup_store.record(operation.operation_id, prior_path)
            registry.register(
                owner_id=operation.owner_id,
                doc_id=operation.doc_id,
                filename=operation.filename,
                mime_type=operation.mime_type,
                source_path=(
                    operation.source_path if operation.retain_source else None
                ),
            )
            operation = outbox.mark_registry_committed(operation.operation_id)
        if operation.state == "registry_committed":
            return _cleanup_or_wait(
                operation,
                outbox=outbox,
                cleanup=cleanup_store,
                remove_source=remove_source,
            )

    if operation.kind == "delete":
        if not _delete_generation_matches(operation, generation):
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
            prior = registry.get(
                owner_id=operation.owner_id,
                doc_id=operation.doc_id,
            )
            prior_path = str((prior or {}).get("source_path") or "")
            if prior_path:
                cleanup_store.record(operation.operation_id, prior_path)
            registry.delete(
                owner_id=operation.owner_id,
                doc_id=operation.doc_id,
            )
            operation = outbox.mark_registry_committed(operation.operation_id)
        if operation.state == "registry_committed":
            return _cleanup_or_wait(
                operation,
                outbox=outbox,
                cleanup=cleanup_store,
                remove_source=remove_source,
            )

    raise RuntimeError("lifecycle operation entered an unsupported state.")


def reconcile_claimed_operations(
    operations: Iterable[LifecycleOperation],
    *,
    outbox: LifecycleOutbox,
    generations: _GenerationStore,
    registry: _Registry,
    worker_id: str,
    cleanup: LifecycleCleanupJournal | None = None,
    remove_source: Callable[[str], bool] | None = None,
) -> tuple[LifecycleReconcileResult, ...]:
    worker = _identifier(worker_id, "worker_id", 200)
    if isinstance(operations, (str, bytes, bytearray)):
        raise ValueError("operations must be an iterable.")
    try:
        rows = list(itertools.islice(iter(operations), _MAX_LIMIT + 1))
    except Exception as exc:
        raise ValueError("operations must be safely iterable.") from exc
    if len(rows) > _MAX_LIMIT or any(
        not isinstance(row, LifecycleOperation) for row in rows
    ):
        raise ValueError("operations are invalid or exceed the limit.")
    cleanup_store = cleanup or get_cleanup_journal()
    results: list[LifecycleReconcileResult] = []
    for operation in rows:
        try:
            result = reconcile_lifecycle_operation(
                operation.operation_id,
                outbox=outbox,
                generations=generations,
                registry=registry,
                cleanup=cleanup_store,
                remove_source=remove_source,
            )
            results.append(result)
            if (
                result.outcome.startswith("waiting_for_")
                or result.outcome == "cleanup_required"
            ):
                outbox.release(operation.operation_id, worker_id=worker)
        except Exception as exc:
            outbox.record_failure(
                operation.operation_id,
                worker_id=worker,
                error_type=type(exc).__name__,
            )
            current = outbox.get(operation.operation_id) or operation
            results.append(
                LifecycleReconcileResult(
                    operation.operation_id,
                    "error",
                    current.state,
                )
            )
    return tuple(results)


__all__ = [
    "CleanupIntent",
    "LifecycleCleanupJournal",
    "clear_cleanup_runtime_cache",
    "get_cleanup_journal",
    "reconcile_claimed_operations",
    "reconcile_lifecycle_operation",
]
