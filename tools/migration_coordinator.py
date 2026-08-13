"""Durable, fenced migration saga for heterogeneous stores.

Participants must make prepare/commit/rollback idempotent for a transaction ID. This
module provides recovery and compensation; it does not claim strict cross-store ACID.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from tools.lease_coordination import LeaseCoordinator, LeaseRecord
from tools.security import normalize_owner_id

_TERMINAL = {"committed", "rolled_back", "manual_intervention"}


def _id(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    text = value.strip()
    if not text or len(text) > 500 or any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError(f"{label} is invalid.")
    return text


def _json(value: Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("payload must be a mapping.")
    try:
        rendered = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("payload must be JSON serializable.") from exc
    if len(rendered.encode()) > 1_000_000:
        raise ValueError("payload exceeds the byte limit.")
    return rendered


class MigrationParticipant(Protocol):
    name: str
    def prepare(self, *, transaction_id: str, fencing_token: int, payload: Mapping[str, Any]) -> None: ...
    def commit(self, *, transaction_id: str, fencing_token: int, payload: Mapping[str, Any]) -> None: ...
    def rollback(self, *, transaction_id: str, fencing_token: int, payload: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True)
class MigrationRecord:
    owner_id: str
    transaction_id: str
    resource_id: str
    state: str
    participant_names: tuple[str, ...]
    payload: Mapping[str, Any]
    fencing_token: int
    last_error: str | None


class MigrationCoordinator:
    def __init__(self, path: str | Path, *, leases: LeaseCoordinator | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.leases = leases or LeaseCoordinator(self.path.with_name(self.path.stem + "_leases.sqlite3"))
        with self._connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS migration_tx (
              owner_id TEXT NOT NULL, tx_id TEXT NOT NULL, resource_id TEXT NOT NULL,
              state TEXT NOT NULL, names_json TEXT NOT NULL, payload_json TEXT NOT NULL,
              fencing_token INTEGER NOT NULL DEFAULT 0, updated_at REAL NOT NULL,
              last_error TEXT, PRIMARY KEY(owner_id, tx_id));
            CREATE TABLE IF NOT EXISTS migration_step (
              owner_id TEXT NOT NULL, tx_id TEXT NOT NULL, name TEXT NOT NULL,
              prepared INTEGER NOT NULL DEFAULT 0, committed INTEGER NOT NULL DEFAULT 0,
              rolled_back INTEGER NOT NULL DEFAULT 0, updated_at REAL NOT NULL,
              PRIMARY KEY(owner_id, tx_id, name),
              FOREIGN KEY(owner_id, tx_id) REFERENCES migration_tx(owner_id, tx_id) ON DELETE CASCADE);
            """)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(self.path), timeout=10.0, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=10000")
        db.execute("PRAGMA foreign_keys=ON")
        return db

    @staticmethod
    def _record(row: sqlite3.Row) -> MigrationRecord:
        return MigrationRecord(str(row["owner_id"]), str(row["tx_id"]), str(row["resource_id"]),
            str(row["state"]), tuple(json.loads(str(row["names_json"]))),
            dict(json.loads(str(row["payload_json"]))), int(row["fencing_token"]),
            None if row["last_error"] is None else str(row["last_error"]))

    @staticmethod
    def _parts(parts: Sequence[MigrationParticipant]) -> tuple[MigrationParticipant, ...]:
        if not isinstance(parts, Sequence) or not 1 <= len(parts) <= 128:
            raise ValueError("participants must contain 1-128 entries.")
        selected = tuple(parts)
        names = [_id(getattr(item, "name", None), "participant name") for item in selected]
        if len(names) != len(set(names)):
            raise ValueError("participant names must be unique.")
        for item in selected:
            if not all(callable(getattr(item, method, None)) for method in ("prepare", "commit", "rollback")):
                raise ValueError("participant is incomplete.")
        return selected

    def begin(self, *, owner_id: str, transaction_id: str, resource_id: str,
              participants: Sequence[MigrationParticipant], payload: Mapping[str, Any]) -> MigrationRecord:
        owner, tx, resource = normalize_owner_id(owner_id), _id(transaction_id, "transaction_id"), _id(resource_id, "resource_id")
        parts = self._parts(participants)
        names = tuple(item.name for item in parts)
        names_json, payload_json, now = json.dumps(names, separators=(",", ":")), _json(payload), time.time()
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM migration_tx WHERE owner_id=? AND tx_id=?", (owner, tx)).fetchone()
            if row is not None:
                record = self._record(row)
                if record.resource_id != resource or record.participant_names != names or dict(record.payload) != dict(payload):
                    raise ValueError("transaction_id is already bound to different inputs.")
                db.execute("COMMIT")
                return record
            db.execute("INSERT INTO migration_tx VALUES(?,?,?,'created',?,?,0,?,NULL)",
                       (owner, tx, resource, names_json, payload_json, now))
            for name in names:
                db.execute("INSERT INTO migration_step VALUES(?,?,?,0,0,0,?)", (owner, tx, name, now))
            row = db.execute("SELECT * FROM migration_tx WHERE owner_id=? AND tx_id=?", (owner, tx)).fetchone()
            db.execute("COMMIT")
        except Exception:
            try: db.execute("ROLLBACK")
            except sqlite3.Error: pass
            raise
        finally:
            db.close()
        if row is None:
            raise RuntimeError("migration journal creation failed.")
        return self._record(row)

    def get(self, *, owner_id: str, transaction_id: str) -> MigrationRecord | None:
        owner, tx = normalize_owner_id(owner_id), _id(transaction_id, "transaction_id")
        with self._connect() as db:
            row = db.execute("SELECT * FROM migration_tx WHERE owner_id=? AND tx_id=?", (owner, tx)).fetchone()
        return None if row is None else self._record(row)

    def _state(self, record: MigrationRecord, state: str, token: int, error: str | None = None) -> None:
        with self._connect() as db:
            db.execute("UPDATE migration_tx SET state=?,fencing_token=?,updated_at=?,last_error=? WHERE owner_id=? AND tx_id=?",
                       (state, token, time.time(), None if error is None else error[:2000], record.owner_id, record.transaction_id))

    def _steps(self, record: MigrationRecord) -> dict[str, sqlite3.Row]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM migration_step WHERE owner_id=? AND tx_id=?",
                              (record.owner_id, record.transaction_id)).fetchall()
        return {str(row["name"]): row for row in rows}

    def _done(self, record: MigrationRecord, name: str, column: str) -> None:
        if column not in {"prepared", "committed", "rolled_back"}:
            raise ValueError("invalid step column.")
        with self._connect() as db:
            db.execute(f"UPDATE migration_step SET {column}=1,updated_at=? WHERE owner_id=? AND tx_id=? AND name=?",
                       (time.time(), record.owner_id, record.transaction_id, name))

    def _rollback(self, record: MigrationRecord, parts: tuple[MigrationParticipant, ...], lease: LeaseRecord,
                  cause: BaseException) -> MigrationRecord:
        self._state(record, "rolling_back", lease.fencing_token, f"{type(cause).__name__}: {cause}")
        errors: list[str] = []
        for item in reversed(parts):
            step = self._steps(record)[item.name]
            if not bool(step["prepared"]) or bool(step["rolled_back"]):
                continue
            if not self.leases.validate_fence(lease):
                errors.append("coordinator lease expired during rollback")
                break
            try:
                item.rollback(transaction_id=record.transaction_id, fencing_token=lease.fencing_token, payload=record.payload)
                self._done(record, item.name, "rolled_back")
            except Exception as exc:
                errors.append(f"{item.name}: {type(exc).__name__}: {exc}")
        self._state(record, "manual_intervention" if errors else "rolled_back", lease.fencing_token,
                    "; ".join(errors) if errors else f"compensated after {type(cause).__name__}")
        result = self.get(owner_id=record.owner_id, transaction_id=record.transaction_id)
        if result is None: raise RuntimeError("migration journal disappeared.")
        return result

    def execute(self, *, owner_id: str, transaction_id: str, participants: Sequence[MigrationParticipant],
                coordinator_id: str, lease_ttl_seconds: float = 300.0) -> MigrationRecord:
        record = self.get(owner_id=owner_id, transaction_id=transaction_id)
        if record is None: raise KeyError("unknown migration transaction.")
        parts = self._parts(participants)
        if tuple(item.name for item in parts) != record.participant_names:
            raise ValueError("participants do not match the durable journal.")
        if record.state in _TERMINAL: return record
        lease = self.leases.acquire(owner_id=record.owner_id, resource_id=f"migration:{record.resource_id}",
                                    holder_id=_id(coordinator_id, "coordinator_id"), ttl_seconds=lease_ttl_seconds)
        if lease is None: raise RuntimeError("migration resource is leased by another coordinator.")
        try:
            try:
                self._state(record, "preparing", lease.fencing_token)
                for item in parts:
                    if bool(self._steps(record)[item.name]["prepared"]): continue
                    if not self.leases.validate_fence(lease): raise RuntimeError("coordinator lease expired during prepare.")
                    item.prepare(transaction_id=record.transaction_id, fencing_token=lease.fencing_token, payload=record.payload)
                    self._done(record, item.name, "prepared")
                self._state(record, "committing", lease.fencing_token)
                for item in parts:
                    if bool(self._steps(record)[item.name]["committed"]): continue
                    if not self.leases.validate_fence(lease): raise RuntimeError("coordinator lease expired during commit.")
                    item.commit(transaction_id=record.transaction_id, fencing_token=lease.fencing_token, payload=record.payload)
                    self._done(record, item.name, "committed")
            except Exception as exc:
                return self._rollback(record, parts, lease, exc)
            self._state(record, "committed", lease.fencing_token)
            result = self.get(owner_id=record.owner_id, transaction_id=record.transaction_id)
            if result is None: raise RuntimeError("migration journal disappeared.")
            return result
        finally:
            self.leases.release(lease)

    def recover(self, *, participants_by_name: Mapping[str, MigrationParticipant], coordinator_id: str,
                owner_id: str | None = None, limit: int = 100) -> tuple[MigrationRecord, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit is invalid.")
        owner = None if owner_id is None else normalize_owner_id(owner_id)
        with self._connect() as db:
            if owner is None:
                rows = db.execute("SELECT * FROM migration_tx WHERE state NOT IN ('committed','rolled_back','manual_intervention') ORDER BY updated_at LIMIT ?", (limit,)).fetchall()
            else:
                rows = db.execute("SELECT * FROM migration_tx WHERE owner_id=? AND state NOT IN ('committed','rolled_back','manual_intervention') ORDER BY updated_at LIMIT ?", (owner, limit)).fetchall()
        recovered: list[MigrationRecord] = []
        for row in rows:
            record = self._record(row)
            try: parts = tuple(participants_by_name[name] for name in record.participant_names)
            except KeyError: continue
            recovered.append(self.execute(owner_id=record.owner_id, transaction_id=record.transaction_id,
                                          participants=parts, coordinator_id=coordinator_id))
        return tuple(recovered)


__all__ = ["MigrationCoordinator", "MigrationParticipant", "MigrationRecord"]
