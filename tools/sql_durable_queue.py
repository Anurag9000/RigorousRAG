"""SQLite-backed durable queue with idempotency, visibility, retry, and DLQ semantics."""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from tools.durable_queue import ClaimedMessage, CoordinationError, QueueMessage


class SQLiteDurableQueue:
    """Persistent at-least-once queue suitable for one host or a shared SQLite volume.

    Claims use ``BEGIN IMMEDIATE`` so concurrent worker processes serialize the claim mutation.
    Deployments requiring cross-host broker availability should use a broker adapter with the
    same ``DurableQueue`` contract rather than treating SQLite as a distributed consensus system.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_attempts: int = 3,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._max_attempts = max_attempts
        self._clock = clock
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS durable_queue (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT NOT NULL UNIQUE,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    available_at REAL NOT NULL,
                    invisible_until REAL NOT NULL DEFAULT 0,
                    receipt TEXT,
                    owner TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    acked INTEGER NOT NULL DEFAULT 0,
                    dead_lettered INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_durable_queue_claim
                ON durable_queue(acked, dead_lettered, available_at, sequence);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_durable_queue_receipt
                ON durable_queue(receipt) WHERE receipt IS NOT NULL;
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=10.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _identifier(value: str, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be non-empty")
        rendered = value.strip()
        if len(rendered) > 500 or any(ord(ch) < 32 or ord(ch) == 127 for ch in rendered):
            raise ValueError(f"{label} is invalid")
        return rendered

    @staticmethod
    def _payload(payload: Mapping[str, object]) -> str:
        try:
            return json.dumps(
                dict(payload),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("queue payload must be deterministic JSON") from exc

    @staticmethod
    def _decode(payload_json: str) -> dict[str, object]:
        value = json.loads(payload_json)
        if not isinstance(value, dict):
            raise CoordinationError("stored queue payload is invalid")
        return value

    def enqueue(self, payload: Mapping[str, object], *, idempotency_key: str) -> str:
        key = self._identifier(idempotency_key, "idempotency_key")
        encoded = self._payload(payload)
        message_id = f"msg-{secrets.token_hex(16)}"
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT message_id FROM durable_queue WHERE idempotency_key=?", (key,)
            ).fetchone()
            if existing is not None:
                connection.commit()
                return str(existing["message_id"])
            connection.execute(
                "INSERT INTO durable_queue(message_id,idempotency_key,payload_json,available_at) "
                "VALUES(?,?,?,?)",
                (message_id, key, encoded, float(self._clock())),
            )
            connection.commit()
            return message_id
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _expire(self, connection: sqlite3.Connection, now: float) -> None:
        connection.execute(
            "UPDATE durable_queue SET dead_lettered=1,receipt=NULL,owner=NULL,invisible_until=0 "
            "WHERE acked=0 AND dead_lettered=0 AND receipt IS NOT NULL "
            "AND invisible_until<=? AND attempts>=?",
            (now, self._max_attempts),
        )
        connection.execute(
            "UPDATE durable_queue SET receipt=NULL,owner=NULL,invisible_until=0 "
            "WHERE acked=0 AND dead_lettered=0 AND receipt IS NOT NULL "
            "AND invisible_until<=? AND attempts<?",
            (now, self._max_attempts),
        )

    def claim(self, owner: str, *, visibility_timeout: float) -> ClaimedMessage | None:
        holder = self._identifier(owner, "owner")
        timeout = float(visibility_timeout)
        if timeout <= 0.0:
            raise ValueError("visibility_timeout must be positive")
        now = float(self._clock())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._expire(connection, now)
            row = connection.execute(
                "SELECT sequence,message_id,idempotency_key,payload_json,attempts "
                "FROM durable_queue WHERE acked=0 AND dead_lettered=0 AND receipt IS NULL "
                "AND available_at<=? AND attempts<? ORDER BY sequence LIMIT 1",
                (now, self._max_attempts),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            attempts = int(row["attempts"]) + 1
            receipt = f"receipt-{secrets.token_hex(16)}"
            deadline = now + timeout
            connection.execute(
                "UPDATE durable_queue SET attempts=?,receipt=?,owner=?,invisible_until=? "
                "WHERE sequence=?",
                (attempts, receipt, holder, deadline, int(row["sequence"])),
            )
            connection.commit()
            return ClaimedMessage(
                message_id=str(row["message_id"]),
                payload=self._decode(str(row["payload_json"])),
                idempotency_key=str(row["idempotency_key"]),
                attempts=attempts,
                receipt=receipt,
                owner=holder,
                visibility_deadline=deadline,
            )
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def ack(self, receipt: str) -> None:
        token = self._identifier(receipt, "receipt")
        now = float(self._clock())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE durable_queue SET acked=1,receipt=NULL,owner=NULL,invisible_until=0 "
                "WHERE receipt=? AND acked=0 AND dead_lettered=0 AND invisible_until>?",
                (token, now),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise CoordinationError("receipt is invalid or expired")
            connection.commit()
        except CoordinationError:
            raise
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def nack(self, receipt: str, *, retry_delay: float = 0.0) -> None:
        token = self._identifier(receipt, "receipt")
        delay = float(retry_delay)
        if delay < 0.0:
            raise ValueError("retry_delay must not be negative")
        now = float(self._clock())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT sequence,attempts FROM durable_queue WHERE receipt=? AND acked=0 "
                "AND dead_lettered=0 AND invisible_until>?",
                (token, now),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise CoordinationError("receipt is invalid or expired")
            attempts = int(row["attempts"])
            if attempts >= self._max_attempts:
                connection.execute(
                    "UPDATE durable_queue SET dead_lettered=1,receipt=NULL,owner=NULL,"
                    "invisible_until=0 WHERE sequence=?",
                    (int(row["sequence"]),),
                )
            else:
                connection.execute(
                    "UPDATE durable_queue SET receipt=NULL,owner=NULL,invisible_until=0,"
                    "available_at=? WHERE sequence=?",
                    (now + delay, int(row["sequence"])),
                )
            connection.commit()
        except CoordinationError:
            raise
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def dead_letters(self) -> tuple[QueueMessage, ...]:
        now = float(self._clock())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._expire(connection, now)
            rows = connection.execute(
                "SELECT message_id,idempotency_key,payload_json,attempts FROM durable_queue "
                "WHERE dead_lettered=1 ORDER BY sequence"
            ).fetchall()
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return tuple(
            QueueMessage(
                message_id=str(row["message_id"]),
                payload=self._decode(str(row["payload_json"])),
                idempotency_key=str(row["idempotency_key"]),
                attempts=int(row["attempts"]),
            )
            for row in rows
        )


__all__ = ["SQLiteDurableQueue"]
