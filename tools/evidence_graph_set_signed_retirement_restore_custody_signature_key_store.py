"""SQLite registry for governed Ed25519 custody signer public keys."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import threading
import time
from dataclasses import asdict
from typing import Any

from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _identifier,
    _integer,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature_key_contracts import (
    STATES,
    CustodySignerKeyRecord,
    redirecting,
    validated_path,
)
from tools.security import normalize_owner_id

_TABLE = "evidence_graph_restore_custody_signer_keys"
_MAX_LIMIT = 10_000


def _strict_pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


class CustodySignerKeyRegistry:
    """Owner-scoped monotonic Ed25519 public-key governance registry."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = validated_path(path, label="signer_key_registry_path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("signer key registry parent must be a directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("signer key registry is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
        ):
            raise RuntimeError("signer key registry parent identity changed.")
        if self._file_identity() != self._database_identity:
            raise RuntimeError("signer key registry identity changed.")

    def _connect(self) -> sqlite3.Connection:
        self._verify()
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with sqlite3.connect(self.path, timeout=30.0, isolation_level=None) as connection:
            connection.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS {_TABLE} (
                    owner_id TEXT NOT NULL,
                    key_id TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    registered_at REAL NOT NULL,
                    retired_at REAL,
                    PRIMARY KEY(owner_id, key_id)
                );
                CREATE INDEX IF NOT EXISTS {_TABLE}_owner_state
                    ON {_TABLE}(owner_id, state, registered_at, key_id);
                """
            )

    @staticmethod
    def _encode(value: CustodySignerKeyRecord) -> str:
        return json.dumps(
            asdict(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @staticmethod
    def _decode(value: Any) -> CustodySignerKeyRecord:
        if not isinstance(value, str) or len(value) > 1_000_000:
            raise RuntimeError("stored signer key record is corrupt.")
        try:
            raw = json.loads(
                value,
                object_pairs_hook=_strict_pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
            )
        except (json.JSONDecodeError, ValueError, RecursionError) as exc:
            raise RuntimeError("stored signer key record is corrupt.") from exc
        if not isinstance(raw, dict) or set(raw) != set(
            CustodySignerKeyRecord.__dataclass_fields__
        ):
            raise RuntimeError("stored signer key record schema is corrupt.")
        try:
            return CustodySignerKeyRecord(**raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("stored signer key record is corrupt.") from exc

    def register(self, value: CustodySignerKeyRecord) -> CustodySignerKeyRecord:
        if not isinstance(value, CustodySignerKeyRecord) or value.state != "active":
            raise ValueError("active signer key record is required.")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    f"SELECT record_json FROM {_TABLE} WHERE owner_id=? AND key_id=?",
                    (value.owner_id, value.key_id),
                ).fetchone()
                if row is not None:
                    stored = self._decode(row["record_json"])
                    if stored.record_digest != value.record_digest:
                        raise RuntimeError("signer key identity collision.")
                    connection.execute("COMMIT")
                    return stored
                connection.execute(
                    f"INSERT INTO {_TABLE} VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        value.owner_id,
                        value.key_id,
                        self._encode(value),
                        value.state,
                        value.registered_at,
                        value.retired_at,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return value

    def get(self, *, owner_id: str, key_id: str) -> CustodySignerKeyRecord:
        owner = normalize_owner_id(owner_id)
        selected_key = _identifier(key_id, "key_id", 200)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT record_json FROM {_TABLE} WHERE owner_id=? AND key_id=?",
                (owner, selected_key),
            ).fetchone()
        if row is None:
            raise KeyError(selected_key)
        return self._decode(row["record_json"])

    def list(
        self,
        *,
        owner_id: str,
        state: str | None = None,
        limit: int = 100,
    ) -> tuple[CustodySignerKeyRecord, ...]:
        owner = normalize_owner_id(owner_id)
        selected_state = None if state is None else _identifier(state, "state", 30)
        if selected_state is not None and selected_state not in STATES:
            raise ValueError("signer key state is unsupported.")
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        query = f"SELECT record_json FROM {_TABLE} WHERE owner_id=?"
        params: list[Any] = [owner]
        if selected_state is not None:
            query += " AND state=?"
            params.append(selected_state)
        query += " ORDER BY registered_at DESC, key_id DESC LIMIT ?"
        params.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(self._decode(row["record_json"]) for row in rows)

    def retire(
        self,
        *,
        owner_id: str,
        key_id: str,
        actor: ReviewActorBinding,
        now: float | None = None,
    ) -> CustodySignerKeyRecord:
        owner = normalize_owner_id(owner_id)
        selected_key = _identifier(key_id, "key_id", 200)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    f"SELECT record_json FROM {_TABLE} WHERE owner_id=? AND key_id=?",
                    (owner, selected_key),
                ).fetchone()
                if row is None:
                    raise KeyError(selected_key)
                current = self._decode(row["record_json"])
                retired = current.retire(actor_binding=actor, now=timestamp)
                connection.execute(
                    f"UPDATE {_TABLE} SET record_json=?, state=?, retired_at=? "
                    "WHERE owner_id=? AND key_id=?",
                    (
                        self._encode(retired),
                        retired.state,
                        retired.retired_at,
                        owner,
                        selected_key,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return retired


__all__ = ["CustodySignerKeyRegistry"]
