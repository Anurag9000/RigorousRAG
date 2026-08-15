"""Owner-scoped durable agent trajectories with idempotent redacted checkpoints."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


_SENSITIVE_KEYS = {
    "authorization",
    "api_key",
    "apikey",
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "cookie",
}


def default_trajectory_redactor(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            text_key = str(key)
            if text_key.lower() in _SENSITIVE_KEYS:
                result[text_key] = "[REDACTED]"
            else:
                result[text_key] = default_trajectory_redactor(item)
        return result
    if isinstance(value, (list, tuple)):
        return [default_trajectory_redactor(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)[:1000]


def _identifier(value: Any, label: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").replace("\r", " ").replace("\n", " ").split())
    if not cleaned:
        raise ValueError(f"{label} must be non-empty")
    if len(cleaned) > maximum:
        raise ValueError(f"{label} exceeds {maximum} characters")
    return cleaned


@dataclass(frozen=True)
class TrajectoryEvent:
    owner_id: str
    trajectory_id: str
    sequence: int
    event_type: str
    agent: str
    payload: Mapping[str, Any]
    created_at: float
    idempotency_key: str


class AgentTrajectoryStore:
    def __init__(
        self,
        path: str | Path,
        *,
        redactor: Callable[[Any], Any] = default_trajectory_redactor,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not callable(redactor):
            raise TypeError("redactor must be callable")
        self._redactor = redactor
        self._clock = clock
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_trajectory_events (
                    owner_id TEXT NOT NULL,
                    trajectory_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    PRIMARY KEY (owner_id, trajectory_id, sequence),
                    UNIQUE (owner_id, trajectory_id, idempotency_key)
                )
                """
            )

    def append(
        self,
        *,
        owner_id: str,
        trajectory_id: str,
        event_type: str,
        agent: str,
        payload: Mapping[str, Any],
        idempotency_key: str,
    ) -> TrajectoryEvent:
        owner = _identifier(owner_id, "owner_id")
        trajectory = _identifier(trajectory_id, "trajectory_id")
        kind = _identifier(event_type, "event_type", maximum=100)
        agent_name = _identifier(agent, "agent", maximum=100)
        idem = _identifier(idempotency_key, "idempotency_key")
        if not isinstance(payload, Mapping):
            raise ValueError("payload must be a mapping")
        redacted = self._redactor(dict(payload))
        if not isinstance(redacted, Mapping):
            raise ValueError("redactor must return a mapping")
        encoded = json.dumps(dict(redacted), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        created = float(self._clock())

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                """SELECT * FROM agent_trajectory_events
                   WHERE owner_id=? AND trajectory_id=? AND idempotency_key=?""",
                (owner, trajectory, idem),
            ).fetchone()
            if existing is not None:
                conn.rollback()
                return self._row(existing)
            row = conn.execute(
                """SELECT COALESCE(MAX(sequence), 0) AS sequence
                   FROM agent_trajectory_events WHERE owner_id=? AND trajectory_id=?""",
                (owner, trajectory),
            ).fetchone()
            sequence = int(row["sequence"]) + 1
            conn.execute(
                """INSERT INTO agent_trajectory_events
                   (owner_id, trajectory_id, sequence, event_type, agent, payload_json, created_at, idempotency_key)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (owner, trajectory, sequence, kind, agent_name, encoded, created, idem),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return TrajectoryEvent(owner, trajectory, sequence, kind, agent_name, dict(redacted), created, idem)

    def list_events(self, *, owner_id: str, trajectory_id: str) -> tuple[TrajectoryEvent, ...]:
        owner = _identifier(owner_id, "owner_id")
        trajectory = _identifier(trajectory_id, "trajectory_id")
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM agent_trajectory_events
                   WHERE owner_id=? AND trajectory_id=? ORDER BY sequence ASC""",
                (owner, trajectory),
            ).fetchall()
        return tuple(self._row(row) for row in rows)

    def latest_checkpoint(self, *, owner_id: str, trajectory_id: str) -> Optional[TrajectoryEvent]:
        owner = _identifier(owner_id, "owner_id")
        trajectory = _identifier(trajectory_id, "trajectory_id")
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM agent_trajectory_events
                   WHERE owner_id=? AND trajectory_id=? AND event_type='checkpoint'
                   ORDER BY sequence DESC LIMIT 1""",
                (owner, trajectory),
            ).fetchone()
        return self._row(row) if row is not None else None

    @staticmethod
    def _row(row: sqlite3.Row) -> TrajectoryEvent:
        return TrajectoryEvent(
            owner_id=str(row["owner_id"]),
            trajectory_id=str(row["trajectory_id"]),
            sequence=int(row["sequence"]),
            event_type=str(row["event_type"]),
            agent=str(row["agent"]),
            payload=json.loads(str(row["payload_json"])),
            created_at=float(row["created_at"]),
            idempotency_key=str(row["idempotency_key"]),
        )
