"""Durable work-queue contracts and bounded reference/provider implementations.

``InMemoryDurableQueue`` is deterministic and suitable for tests or single-process
execution. ``SQLiteDurableQueue`` provides same-host, cross-process durability using
short SQLite transactions, WAL journaling, idempotent enqueue, visibility leases,
bounded retries, acknowledgements, and dead-lettering.

Queue payloads are deliberately JSON-only. Domain authority must remain in the
application's durable control plane; queue messages should contain opaque identifiers
rather than private source/evidence content.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import stat
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_IDENTIFIER = 512
_DEFAULT_MAX_PAYLOAD_BYTES = 64 * 1024


class CoordinationError(RuntimeError):
    """Raised when a queue state transition is invalid or durable state is corrupt."""


@dataclass(frozen=True)
class QueueMessage:
    message_id: str
    payload: Mapping[str, object]
    idempotency_key: str
    attempts: int


@dataclass(frozen=True)
class ClaimedMessage(QueueMessage):
    receipt: str
    owner: str
    visibility_deadline: float


class DurableQueue(Protocol):
    def enqueue(self, payload: Mapping[str, object], *, idempotency_key: str) -> str: ...

    def claim(self, owner: str, *, visibility_timeout: float) -> ClaimedMessage | None: ...

    def ack(self, receipt: str) -> None: ...

    def nack(self, receipt: str, *, retry_delay: float = 0.0) -> None: ...


def _identifier(value: str, label: str, *, maximum: int = _MAX_IDENTIFIER) -> str:
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


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def _positive_seconds(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be positive.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be positive and finite.")
    return result


def _nonnegative_seconds(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must not be negative.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and not negative.")
    return result


def _safe_sqlite_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    if len(str(absolute)) > 4096:
        raise ValueError("durable queue database path is too long")
    for component in (absolute, *absolute.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        if stat.S_ISLNK(metadata.st_mode) or bool(attributes & _WINDOWS_REPARSE_POINT):
            raise RuntimeError("durable queue path may not traverse symlinks/reparse points")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    return absolute


def _encode_payload(payload: Mapping[str, object], *, maximum_bytes: int) -> str:
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")
    value = dict(payload)
    if any(not isinstance(key, str) for key in value):
        raise ValueError("payload keys must be strings")
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("payload must be JSON serializable") from exc
    if len(encoded.encode("utf-8")) > maximum_bytes:
        raise ValueError("payload exceeds durable queue size limit")
    return encoded


def _decode_payload(payload_json: str) -> dict[str, object]:
    try:
        value = json.loads(payload_json)
    except (TypeError, ValueError) as exc:
        raise CoordinationError("durable queue payload is corrupt") from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise CoordinationError("durable queue payload is corrupt")
    return value


@dataclass
class _QueuedState:
    message_id: str
    payload: dict[str, object]
    idempotency_key: str
    sequence: int
    available_at: float
    invisible_until: float = 0.0
    receipt: str | None = None
    owner: str | None = None
    attempts: int = 0
    acked: bool = False
    dead_lettered: bool = False


class InMemoryDurableQueue:
    """Reference at-least-once queue with idempotency, visibility, retries, and a DLQ."""

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_attempts = _positive_integer(max_attempts, "max_attempts")
        self._clock = clock
        self._sequence = 0
        self._receipt_sequence = 0
        self._messages: dict[str, _QueuedState] = {}
        self._idempotency: dict[str, str] = {}

    def enqueue(self, payload: Mapping[str, object], *, idempotency_key: str) -> str:
        key = _identifier(idempotency_key, "idempotency_key")
        existing = self._idempotency.get(key)
        if existing is not None:
            return existing
        self._sequence += 1
        message_id = f"msg-{self._sequence:016d}"
        self._messages[message_id] = _QueuedState(
            message_id=message_id,
            payload=dict(payload),
            idempotency_key=key,
            sequence=self._sequence,
            available_at=self._clock(),
        )
        self._idempotency[key] = message_id
        return message_id

    def _expire_or_dead_letter(self, state: _QueuedState, now: float) -> None:
        if state.receipt is None or state.invisible_until > now:
            return
        state.receipt = None
        state.owner = None
        state.invisible_until = 0.0
        if state.attempts >= self._max_attempts:
            state.dead_lettered = True

    def claim(self, owner: str, *, visibility_timeout: float) -> ClaimedMessage | None:
        holder = _identifier(owner, "owner")
        timeout = _positive_seconds(visibility_timeout, "visibility_timeout")
        now = self._clock()
        for state in sorted(self._messages.values(), key=lambda item: item.sequence):
            self._expire_or_dead_letter(state, now)
            if (
                state.acked
                or state.dead_lettered
                or state.available_at > now
                or state.receipt is not None
            ):
                continue
            state.attempts += 1
            self._receipt_sequence += 1
            state.receipt = f"receipt-{self._receipt_sequence:016d}"
            state.owner = holder
            state.invisible_until = now + timeout
            return ClaimedMessage(
                state.message_id,
                dict(state.payload),
                state.idempotency_key,
                state.attempts,
                state.receipt,
                holder,
                state.invisible_until,
            )
        return None

    def _by_receipt(self, receipt: str) -> _QueuedState:
        token = _identifier(receipt, "receipt")
        for state in self._messages.values():
            if state.receipt == token and not state.acked and not state.dead_lettered:
                if state.invisible_until <= self._clock():
                    self._expire_or_dead_letter(state, self._clock())
                    break
                return state
        raise CoordinationError("receipt is invalid or expired")

    def ack(self, receipt: str) -> None:
        state = self._by_receipt(receipt)
        state.acked = True
        state.receipt = None
        state.owner = None
        state.invisible_until = 0.0

    def nack(self, receipt: str, *, retry_delay: float = 0.0) -> None:
        delay = _nonnegative_seconds(retry_delay, "retry_delay")
        state = self._by_receipt(receipt)
        state.receipt = None
        state.owner = None
        state.invisible_until = 0.0
        if state.attempts >= self._max_attempts:
            state.dead_lettered = True
        else:
            state.available_at = self._clock() + delay

    def dead_letters(self) -> tuple[QueueMessage, ...]:
        now = self._clock()
        for state in self._messages.values():
            self._expire_or_dead_letter(state, now)
        return tuple(
            QueueMessage(
                item.message_id,
                dict(item.payload),
                item.idempotency_key,
                item.attempts,
            )
            for item in sorted(self._messages.values(), key=lambda value: value.sequence)
            if item.dead_lettered
        )


class SQLiteDurableQueue:
    """SQLite-backed same-host durable queue for independent producer/worker processes.

    Each operation opens a short-lived connection. ``BEGIN IMMEDIATE`` serializes the
    claim transition so two processes cannot lease the same record. Visibility receipts
    fence stale workers: acknowledgements and negative acknowledgements are accepted only
    while the exact receipt is still live.

    SQLite is intentionally documented as a same-host transport. Multi-host deployments
    should bind ``DurableQueue`` to a networked provider rather than placing this database
    on an unsupported shared/network filesystem.
    """

    _SCHEMA_VERSION = 1

    def __init__(
        self,
        path: str | Path,
        *,
        namespace: str = "default",
        max_attempts: int = 3,
        max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = _safe_sqlite_path(path)
        self.namespace = _identifier(namespace, "namespace", maximum=128)
        self._max_attempts = _positive_integer(max_attempts, "max_attempts")
        self._max_payload_bytes = _positive_integer(max_payload_bytes, "max_payload_bytes")
        self._clock = clock
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            # sqlite3.executescript() may commit a pending transaction, so schema
            # creation must precede the transaction that fences namespace bootstrap.
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS durable_queue_namespaces (
                    namespace TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS durable_queue_messages (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    message_id TEXT NOT NULL UNIQUE,
                    idempotency_key TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    available_at REAL NOT NULL,
                    invisible_until REAL NOT NULL DEFAULT 0,
                    receipt TEXT,
                    owner TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    acked INTEGER NOT NULL DEFAULT 0,
                    dead_lettered INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(namespace, idempotency_key),
                    FOREIGN KEY(namespace) REFERENCES durable_queue_namespaces(namespace)
                );
                CREATE INDEX IF NOT EXISTS durable_queue_ready_idx
                  ON durable_queue_messages(namespace, acked, dead_lettered, available_at, sequence);
                CREATE INDEX IF NOT EXISTS durable_queue_receipt_idx
                  ON durable_queue_messages(namespace, receipt);
                """
            )
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT schema_version,max_attempts FROM durable_queue_namespaces WHERE namespace=?",
                (self.namespace,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO durable_queue_namespaces(namespace,schema_version,max_attempts,created_at) VALUES(?,?,?,?)",
                    (self.namespace, self._SCHEMA_VERSION, self._max_attempts, self._clock()),
                )
            elif int(row["schema_version"]) != self._SCHEMA_VERSION:
                raise RuntimeError("durable queue schema version mismatch")
            elif int(row["max_attempts"]) != self._max_attempts:
                raise RuntimeError("durable queue namespace max_attempts mismatch")
            connection.commit()
        except Exception:
            try:
                connection.rollback()
            except Exception:
                pass
            raise
        finally:
            connection.close()

    def _recover_expired(self, connection: sqlite3.Connection, now: float) -> None:
        connection.execute(
            """UPDATE durable_queue_messages
               SET receipt=NULL,owner=NULL,invisible_until=0,dead_lettered=1,updated_at=?
               WHERE namespace=? AND acked=0 AND dead_lettered=0
                 AND receipt IS NOT NULL AND invisible_until<=? AND attempts>=?""",
            (now, self.namespace, now, self._max_attempts),
        )
        connection.execute(
            """UPDATE durable_queue_messages
               SET receipt=NULL,owner=NULL,invisible_until=0,updated_at=?
               WHERE namespace=? AND acked=0 AND dead_lettered=0
                 AND receipt IS NOT NULL AND invisible_until<=? AND attempts<?""",
            (now, self.namespace, now, self._max_attempts),
        )

    @staticmethod
    def _message_from_row(row: sqlite3.Row) -> QueueMessage:
        return QueueMessage(
            str(row["message_id"]),
            _decode_payload(str(row["payload_json"])),
            str(row["idempotency_key"]),
            int(row["attempts"]),
        )

    def enqueue(self, payload: Mapping[str, object], *, idempotency_key: str) -> str:
        key = _identifier(idempotency_key, "idempotency_key")
        payload_json = _encode_payload(payload, maximum_bytes=self._max_payload_bytes)
        now = self._clock()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """SELECT message_id,payload_json FROM durable_queue_messages
                   WHERE namespace=? AND idempotency_key=?""",
                (self.namespace, key),
            ).fetchone()
            if existing is not None:
                if str(existing["payload_json"]) != payload_json:
                    raise CoordinationError("idempotency key already exists with different payload")
                connection.commit()
                return str(existing["message_id"])
            message_id = f"msg-{uuid.uuid4().hex}"
            connection.execute(
                """INSERT INTO durable_queue_messages(
                       namespace,message_id,idempotency_key,payload_json,available_at,
                       invisible_until,attempts,acked,dead_lettered,created_at,updated_at
                   ) VALUES(?,?,?,?,?,0,0,0,0,?,?)""",
                (self.namespace, message_id, key, payload_json, now, now, now),
            )
            connection.commit()
            return message_id
        except Exception:
            try:
                connection.rollback()
            except Exception:
                pass
            raise
        finally:
            connection.close()

    def claim(self, owner: str, *, visibility_timeout: float) -> ClaimedMessage | None:
        holder = _identifier(owner, "owner")
        timeout = _positive_seconds(visibility_timeout, "visibility_timeout")
        now = self._clock()
        receipt = f"receipt-{uuid.uuid4().hex}"
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._recover_expired(connection, now)
            row = connection.execute(
                """SELECT * FROM durable_queue_messages
                   WHERE namespace=? AND acked=0 AND dead_lettered=0
                     AND receipt IS NULL AND available_at<=?
                   ORDER BY sequence ASC LIMIT 1""",
                (self.namespace, now),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            deadline = now + timeout
            cursor = connection.execute(
                """UPDATE durable_queue_messages
                   SET attempts=attempts+1,receipt=?,owner=?,invisible_until=?,updated_at=?
                   WHERE sequence=? AND namespace=? AND acked=0 AND dead_lettered=0 AND receipt IS NULL""",
                (receipt, holder, deadline, now, int(row["sequence"]), self.namespace),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            claimed = connection.execute(
                "SELECT * FROM durable_queue_messages WHERE sequence=?",
                (int(row["sequence"]),),
            ).fetchone()
            if claimed is None:
                raise CoordinationError("claimed durable queue message disappeared")
            connection.commit()
            return ClaimedMessage(
                str(claimed["message_id"]),
                _decode_payload(str(claimed["payload_json"])),
                str(claimed["idempotency_key"]),
                int(claimed["attempts"]),
                receipt,
                holder,
                deadline,
            )
        except Exception:
            try:
                connection.rollback()
            except Exception:
                pass
            raise
        finally:
            connection.close()

    def _active_receipt_row(
        self,
        connection: sqlite3.Connection,
        receipt: str,
        now: float,
    ) -> sqlite3.Row:
        token = _identifier(receipt, "receipt")
        row = connection.execute(
            """SELECT * FROM durable_queue_messages
               WHERE namespace=? AND receipt=? AND acked=0 AND dead_lettered=0""",
            (self.namespace, token),
        ).fetchone()
        if row is None:
            raise CoordinationError("receipt is invalid or expired")
        if float(row["invisible_until"]) <= now:
            sequence = int(row["sequence"])
            attempts = int(row["attempts"])
            if attempts >= self._max_attempts:
                connection.execute(
                    """UPDATE durable_queue_messages
                       SET receipt=NULL,owner=NULL,invisible_until=0,dead_lettered=1,updated_at=?
                       WHERE sequence=? AND namespace=? AND receipt=?""",
                    (now, sequence, self.namespace, token),
                )
            else:
                connection.execute(
                    """UPDATE durable_queue_messages
                       SET receipt=NULL,owner=NULL,invisible_until=0,updated_at=?
                       WHERE sequence=? AND namespace=? AND receipt=?""",
                    (now, sequence, self.namespace, token),
                )
            raise CoordinationError("receipt is invalid or expired")
        return row

    def ack(self, receipt: str) -> None:
        now = self._clock()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._active_receipt_row(connection, receipt, now)
            except CoordinationError:
                # _active_receipt_row may have reclaimed/dead-lettered an expired
                # lease. Commit that authoritative transition before rejecting the
                # stale receipt.
                connection.commit()
                raise
            token = _identifier(receipt, "receipt")
            cursor = connection.execute(
                """UPDATE durable_queue_messages
                   SET acked=1,receipt=NULL,owner=NULL,invisible_until=0,updated_at=?
                   WHERE sequence=? AND namespace=? AND receipt=?""",
                (now, int(row["sequence"]), self.namespace, token),
            )
            if cursor.rowcount != 1:
                raise CoordinationError("receipt lost ownership before acknowledgement")
            connection.commit()
        except Exception:
            try:
                connection.rollback()
            except Exception:
                pass
            raise
        finally:
            connection.close()

    def nack(self, receipt: str, *, retry_delay: float = 0.0) -> None:
        delay = _nonnegative_seconds(retry_delay, "retry_delay")
        now = self._clock()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._active_receipt_row(connection, receipt, now)
            except CoordinationError:
                connection.commit()
                raise
            token = _identifier(receipt, "receipt")
            if int(row["attempts"]) >= self._max_attempts:
                cursor = connection.execute(
                    """UPDATE durable_queue_messages
                       SET receipt=NULL,owner=NULL,invisible_until=0,dead_lettered=1,updated_at=?
                       WHERE sequence=? AND namespace=? AND receipt=?""",
                    (now, int(row["sequence"]), self.namespace, token),
                )
            else:
                cursor = connection.execute(
                    """UPDATE durable_queue_messages
                       SET receipt=NULL,owner=NULL,invisible_until=0,available_at=?,updated_at=?
                       WHERE sequence=? AND namespace=? AND receipt=?""",
                    (now + delay, now, int(row["sequence"]), self.namespace, token),
                )
            if cursor.rowcount != 1:
                raise CoordinationError("receipt lost ownership before negative acknowledgement")
            connection.commit()
        except Exception:
            try:
                connection.rollback()
            except Exception:
                pass
            raise
        finally:
            connection.close()

    def dead_letters(self) -> tuple[QueueMessage, ...]:
        now = self._clock()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._recover_expired(connection, now)
            rows = connection.execute(
                """SELECT * FROM durable_queue_messages
                   WHERE namespace=? AND dead_lettered=1 ORDER BY sequence ASC""",
                (self.namespace,),
            ).fetchall()
            values = tuple(self._message_from_row(row) for row in rows)
            connection.commit()
            return values
        except Exception:
            try:
                connection.rollback()
            except Exception:
                pass
            raise
        finally:
            connection.close()


__all__ = [
    "ClaimedMessage",
    "CoordinationError",
    "DurableQueue",
    "InMemoryDurableQueue",
    "QueueMessage",
    "SQLiteDurableQueue",
]
