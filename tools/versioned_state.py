"""Versioned owner-scoped state with compare-and-set semantics.

The SQLite implementation is suitable for one shared filesystem/database.  The callback
adapter is an integration boundary for Redis/PostgreSQL/etcd-style backends; it does not
claim those services are configured or certified by this repository.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from tools.security import normalize_owner_id

_MAX_KEY = 500
_MAX_VALUE_BYTES = 1_000_000


def _key(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("key must be a string.")
    text = value.strip()
    if not text or len(text) > _MAX_KEY or any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError("key is invalid.")
    return text


def _encode(value: Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("value must be a mapping.")
    try:
        rendered = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("value must be JSON serializable.") from exc
    if len(rendered.encode("utf-8")) > _MAX_VALUE_BYTES:
        raise ValueError("value exceeds the byte limit.")
    return rendered


@dataclass(frozen=True)
class VersionedValue:
    owner_id: str
    key: str
    version: int
    value: Mapping[str, Any]
    updated_at: float


@runtime_checkable
class VersionedStateBackend(Protocol):
    def get(self, *, owner_id: str, key: str) -> VersionedValue | None: ...

    def compare_and_set(
        self,
        *,
        owner_id: str,
        key: str,
        expected_version: int | None,
        value: Mapping[str, Any],
    ) -> VersionedValue | None: ...

    def delete(self, *, owner_id: str, key: str, expected_version: int | None = None) -> bool: ...


class SQLiteVersionedState:
    """Durable compare-and-set state using SQLite BEGIN IMMEDIATE transactions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS versioned_state (
                    owner_id TEXT NOT NULL,
                    state_key TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    value_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(owner_id, state_key)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=10.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    @staticmethod
    def _record(row: sqlite3.Row) -> VersionedValue:
        return VersionedValue(
            owner_id=str(row["owner_id"]),
            key=str(row["state_key"]),
            version=int(row["version"]),
            value=dict(json.loads(str(row["value_json"]))),
            updated_at=float(row["updated_at"]),
        )

    def get(self, *, owner_id: str, key: str) -> VersionedValue | None:
        owner = normalize_owner_id(owner_id)
        selected_key = _key(key)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM versioned_state WHERE owner_id=? AND state_key=?",
                (owner, selected_key),
            ).fetchone()
        return None if row is None else self._record(row)

    def compare_and_set(
        self,
        *,
        owner_id: str,
        key: str,
        expected_version: int | None,
        value: Mapping[str, Any],
    ) -> VersionedValue | None:
        owner = normalize_owner_id(owner_id)
        selected_key = _key(key)
        rendered = _encode(value)
        if expected_version is not None and (
            isinstance(expected_version, bool) or not isinstance(expected_version, int) or expected_version < 1
        ):
            raise ValueError("expected_version must be None or a positive integer.")
        now = time.time()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT version FROM versioned_state WHERE owner_id=? AND state_key=?",
                (owner, selected_key),
            ).fetchone()
            if row is None:
                if expected_version is not None:
                    connection.execute("ROLLBACK")
                    return None
                version = 1
                connection.execute(
                    "INSERT INTO versioned_state(owner_id,state_key,version,value_json,updated_at) VALUES(?,?,?,?,?)",
                    (owner, selected_key, version, rendered, now),
                )
            else:
                current = int(row["version"])
                if expected_version != current:
                    connection.execute("ROLLBACK")
                    return None
                version = current + 1
                connection.execute(
                    "UPDATE versioned_state SET version=?,value_json=?,updated_at=? WHERE owner_id=? AND state_key=?",
                    (version, rendered, now, owner, selected_key),
                )
            result = connection.execute(
                "SELECT * FROM versioned_state WHERE owner_id=? AND state_key=?",
                (owner, selected_key),
            ).fetchone()
            connection.execute("COMMIT")
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.close()
        if result is None:
            raise RuntimeError("state write failed.")
        return self._record(result)

    def delete(self, *, owner_id: str, key: str, expected_version: int | None = None) -> bool:
        owner = normalize_owner_id(owner_id)
        selected_key = _key(key)
        if expected_version is not None and (
            isinstance(expected_version, bool) or not isinstance(expected_version, int) or expected_version < 1
        ):
            raise ValueError("expected_version must be None or a positive integer.")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if expected_version is None:
                changed = connection.execute(
                    "DELETE FROM versioned_state WHERE owner_id=? AND state_key=?",
                    (owner, selected_key),
                ).rowcount
            else:
                changed = connection.execute(
                    "DELETE FROM versioned_state WHERE owner_id=? AND state_key=? AND version=?",
                    (owner, selected_key, expected_version),
                ).rowcount
            connection.execute("COMMIT")
        return changed == 1

    def delete_owner(self, *, owner_id: str) -> int:
        owner = normalize_owner_id(owner_id)
        with self._connect() as connection:
            return connection.execute("DELETE FROM versioned_state WHERE owner_id=?", (owner,)).rowcount


GetCallback = Callable[[str, str], VersionedValue | None]
CasCallback = Callable[[str, str, int | None, Mapping[str, Any]], VersionedValue | None]
DeleteCallback = Callable[[str, str, int | None], bool]


class CallbackVersionedState:
    """Validated adapter for an externally managed atomic-CAS implementation."""

    def __init__(self, *, get: GetCallback, compare_and_set: CasCallback, delete: DeleteCallback) -> None:
        if not all(callable(item) for item in (get, compare_and_set, delete)):
            raise ValueError("all state callbacks must be callable.")
        self._get = get
        self._cas = compare_and_set
        self._delete = delete

    @staticmethod
    def _validate(record: VersionedValue | None, owner: str, selected_key: str) -> VersionedValue | None:
        if record is None:
            return None
        if not isinstance(record, VersionedValue):
            raise RuntimeError("external state backend returned an invalid value.")
        if record.owner_id != owner or record.key != selected_key or record.version < 1:
            raise RuntimeError("external state backend violated owner/key/version invariants.")
        _encode(record.value)
        return record

    def get(self, *, owner_id: str, key: str) -> VersionedValue | None:
        owner = normalize_owner_id(owner_id)
        selected_key = _key(key)
        return self._validate(self._get(owner, selected_key), owner, selected_key)

    def compare_and_set(
        self,
        *,
        owner_id: str,
        key: str,
        expected_version: int | None,
        value: Mapping[str, Any],
    ) -> VersionedValue | None:
        owner = normalize_owner_id(owner_id)
        selected_key = _key(key)
        _encode(value)
        return self._validate(self._cas(owner, selected_key, expected_version, dict(value)), owner, selected_key)

    def delete(self, *, owner_id: str, key: str, expected_version: int | None = None) -> bool:
        owner = normalize_owner_id(owner_id)
        selected_key = _key(key)
        result = self._delete(owner, selected_key, expected_version)
        if not isinstance(result, bool):
            raise RuntimeError("external delete callback must return bool.")
        return result


__all__ = [
    "CallbackVersionedState",
    "SQLiteVersionedState",
    "VersionedStateBackend",
    "VersionedValue",
]
