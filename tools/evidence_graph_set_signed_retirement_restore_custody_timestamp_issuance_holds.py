"""Integrity-backed legal holds for custody timestamp issuance records."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _integer,
    _timestamp,
)
from tools.security import normalize_owner_id

_SCHEMA_VERSION = 1
_MAX_LIMIT = 10_000
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_TABLE = "evidence_graph_restore_custody_timestamp_issuance_holds"


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def deterministic_timestamp_issuance_hold_id(
    *,
    owner_id: str,
    issuance_id: str,
    hold_key: str,
) -> str:
    return _canonical_digest(
        {
            "scope": "rigorousrag-custody-timestamp-issuance-hold-v1",
            "owner_id": normalize_owner_id(owner_id),
            "issuance_id": _digest(issuance_id, "issuance_id"),
            "hold_key": _identifier(hold_key, "hold_key", 200),
        }
    )


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str]) -> Path:
    rendered = os.fspath(value)
    if not isinstance(rendered, str) or not rendered or len(rendered) > 4096:
        raise ValueError("timestamp issuance hold database path is invalid.")
    if any(ord(character) < 32 or ord(character) == 127 for character in rendered):
        raise ValueError("timestamp issuance hold database path is invalid.")
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
            raise ValueError("timestamp issuance hold path could not be validated.") from exc
        if _redirecting(info):
            raise ValueError("timestamp issuance hold path may not contain redirects.")
    return absolute


@dataclass(frozen=True)
class CustodyTimestampIssuanceHold:
    hold_id: str
    owner_id: str
    issuance_id: str
    hold_key: str
    reason_code: str
    status: str
    created_actor_id: str
    created_binding_method: str
    created_binding_digest: str
    created_at: float
    released_actor_id: str | None
    released_binding_method: str | None
    released_binding_digest: str | None
    released_at: float | None
    hold_digest: str
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        issuance = _digest(self.issuance_id, "issuance_id")
        key = _identifier(self.hold_key, "hold_key", 200)
        hold_id = _digest(self.hold_id, "hold_id")
        expected_id = deterministic_timestamp_issuance_hold_id(
            owner_id=owner,
            issuance_id=issuance,
            hold_key=key,
        )
        if hold_id != expected_id:
            raise ValueError("hold_id differs from timestamp issuance hold scope.")
        reason = _identifier(self.reason_code, "reason_code", 100)
        status = _identifier(self.status, "status", 20)
        if status not in {"active", "released"}:
            raise ValueError("timestamp issuance hold status is unsupported.")
        created_actor = _identifier(self.created_actor_id, "created_actor_id", 200)
        created_method = _identifier(
            self.created_binding_method,
            "created_binding_method",
            50,
        )
        created_digest = _digest(
            self.created_binding_digest,
            "created_binding_digest",
        )
        created_at = _timestamp(self.created_at, "created_at")
        if status == "active":
            if any(
                value is not None
                for value in (
                    self.released_actor_id,
                    self.released_binding_method,
                    self.released_binding_digest,
                    self.released_at,
                )
            ):
                raise ValueError("active timestamp issuance hold has release fields.")
            released_actor = released_method = released_digest = released_at = None
        else:
            if any(
                value is None
                for value in (
                    self.released_actor_id,
                    self.released_binding_method,
                    self.released_binding_digest,
                    self.released_at,
                )
            ):
                raise ValueError("released timestamp issuance hold is incomplete.")
            released_actor = _identifier(
                self.released_actor_id,
                "released_actor_id",
                200,
            )
            released_method = _identifier(
                self.released_binding_method,
                "released_binding_method",
                50,
            )
            released_digest = _digest(
                self.released_binding_digest,
                "released_binding_digest",
            )
            released_at = _timestamp(self.released_at, "released_at")
            if released_at < created_at:
                raise ValueError("timestamp issuance hold release predates creation.")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("timestamp issuance hold schema is unsupported.")
        stable = {
            "scope": "rigorousrag-custody-timestamp-issuance-hold-record-v1",
            "hold_id": hold_id,
            "owner_id": owner,
            "issuance_id": issuance,
            "hold_key": key,
            "reason_code": reason,
            "status": status,
            "created_actor_id": created_actor,
            "created_binding_method": created_method,
            "created_binding_digest": created_digest,
            "created_at": created_at,
            "released_actor_id": released_actor,
            "released_binding_method": released_method,
            "released_binding_digest": released_digest,
            "released_at": released_at,
            "schema_version": self.schema_version,
        }
        digest = _digest(self.hold_digest, "hold_digest")
        if digest != _canonical_digest(stable):
            raise ValueError("hold_digest differs from timestamp issuance hold.")
        for name, value in stable.items():
            if name != "scope":
                object.__setattr__(self, name, value)
        object.__setattr__(self, "hold_digest", digest)

    @classmethod
    def active(
        cls,
        *,
        owner_id: str,
        issuance_id: str,
        hold_key: str,
        reason_code: str,
        actor: ReviewActorBinding,
        now: float,
    ) -> "CustodyTimestampIssuanceHold":
        if not isinstance(actor, ReviewActorBinding):
            raise ValueError("actor must be ReviewActorBinding.")
        timestamp = _timestamp(now, "now")
        if actor.expires_at is not None and actor.expires_at < timestamp:
            raise PermissionError("review actor binding expired before hold placement.")
        owner = normalize_owner_id(owner_id)
        issuance = _digest(issuance_id, "issuance_id")
        key = _identifier(hold_key, "hold_key", 200)
        values = {
            "hold_id": deterministic_timestamp_issuance_hold_id(
                owner_id=owner,
                issuance_id=issuance,
                hold_key=key,
            ),
            "owner_id": owner,
            "issuance_id": issuance,
            "hold_key": key,
            "reason_code": _identifier(reason_code, "reason_code", 100),
            "status": "active",
            "created_actor_id": actor.actor_id,
            "created_binding_method": actor.binding_method,
            "created_binding_digest": actor.binding_digest,
            "created_at": timestamp,
            "released_actor_id": None,
            "released_binding_method": None,
            "released_binding_digest": None,
            "released_at": None,
            "schema_version": _SCHEMA_VERSION,
        }
        return cls(
            **values,
            hold_digest=_canonical_digest(
                {"scope": "rigorousrag-custody-timestamp-issuance-hold-record-v1", **values}
            ),
        )

    def release(
        self,
        *,
        actor: ReviewActorBinding,
        now: float,
    ) -> "CustodyTimestampIssuanceHold":
        if not isinstance(actor, ReviewActorBinding):
            raise ValueError("actor must be ReviewActorBinding.")
        timestamp = _timestamp(now, "now")
        if actor.expires_at is not None and actor.expires_at < timestamp:
            raise PermissionError("review actor binding expired before hold release.")
        if self.status == "released":
            return self
        values = {
            "hold_id": self.hold_id,
            "owner_id": self.owner_id,
            "issuance_id": self.issuance_id,
            "hold_key": self.hold_key,
            "reason_code": self.reason_code,
            "status": "released",
            "created_actor_id": self.created_actor_id,
            "created_binding_method": self.created_binding_method,
            "created_binding_digest": self.created_binding_digest,
            "created_at": self.created_at,
            "released_actor_id": actor.actor_id,
            "released_binding_method": actor.binding_method,
            "released_binding_digest": actor.binding_digest,
            "released_at": max(timestamp, self.created_at),
            "schema_version": self.schema_version,
        }
        return CustodyTimestampIssuanceHold(
            **values,
            hold_digest=_canonical_digest(
                {"scope": "rigorousrag-custody-timestamp-issuance-hold-record-v1", **values}
            ),
        )


class CustodyTimestampIssuanceHoldStore:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("timestamp issuance hold parent is invalid.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("timestamp issuance hold database is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
            or self._file_identity() != self._database_identity
        ):
            raise RuntimeError("timestamp issuance hold database identity changed.")

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
                    hold_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    issuance_id TEXT NOT NULL,
                    hold_key TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_actor_id TEXT NOT NULL,
                    created_binding_method TEXT NOT NULL,
                    created_binding_digest TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    released_actor_id TEXT,
                    released_binding_method TEXT,
                    released_binding_digest TEXT,
                    released_at REAL,
                    hold_digest TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    UNIQUE(owner_id, issuance_id, hold_key)
                );
                CREATE INDEX IF NOT EXISTS custody_timestamp_issuance_hold_scope
                    ON {_TABLE}(owner_id, issuance_id, status, created_at, hold_id);
                """
            )

    @staticmethod
    def _value(row: sqlite3.Row) -> CustodyTimestampIssuanceHold:
        try:
            return CustodyTimestampIssuanceHold(**dict(row))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("stored timestamp issuance hold is corrupt.") from exc

    def place(
        self,
        *,
        owner_id: str,
        issuance_id: str,
        hold_key: str,
        reason_code: str,
        actor: ReviewActorBinding,
        issuance_journal: Any,
        now: float | None = None,
    ) -> CustodyTimestampIssuanceHold:
        owner = normalize_owner_id(owner_id)
        issuance = _digest(issuance_id, "issuance_id")
        if not callable(getattr(issuance_journal, "get", None)):
            raise ValueError("issuance_journal lacks the required read boundary.")
        issuance_value = issuance_journal.get(issuance)
        if issuance_value.owner_id != owner:
            raise RuntimeError("timestamp issuance escaped legal-hold owner scope.")
        value = CustodyTimestampIssuanceHold.active(
            owner_id=owner,
            issuance_id=issuance,
            hold_key=hold_key,
            reason_code=reason_code,
            actor=actor,
            now=time.time() if now is None else now,
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    f"SELECT * FROM {_TABLE} WHERE hold_id=?",
                    (value.hold_id,),
                ).fetchone()
                if row is not None:
                    stored = self._value(row)
                    if (
                        stored.owner_id != value.owner_id
                        or stored.issuance_id != value.issuance_id
                        or stored.hold_key != value.hold_key
                        or stored.reason_code != value.reason_code
                        or stored.created_actor_id != value.created_actor_id
                        or stored.created_binding_method != value.created_binding_method
                        or stored.created_binding_digest != value.created_binding_digest
                    ):
                        raise RuntimeError("timestamp issuance hold identity collision detected.")
                    connection.execute("COMMIT")
                    return stored
                connection.execute(
                    f"INSERT INTO {_TABLE} VALUES ({','.join('?' for _ in range(16))})",
                    tuple(value.__dict__.values()),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(value.hold_id)

    def get(self, hold_id: str) -> CustodyTimestampIssuanceHold:
        selected = _digest(hold_id, "hold_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM {_TABLE} WHERE hold_id=?",
                (selected,),
            ).fetchone()
        if row is None:
            raise KeyError(selected)
        return self._value(row)

    def release(
        self,
        hold_id: str,
        *,
        owner_id: str,
        confirm_hold_id: str,
        actor: ReviewActorBinding,
        now: float | None = None,
    ) -> CustodyTimestampIssuanceHold:
        selected = _digest(hold_id, "hold_id")
        if selected != _digest(confirm_hold_id, "confirm_hold_id"):
            raise ValueError("timestamp issuance hold confirmation differs.")
        owner = normalize_owner_id(owner_id)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    f"SELECT * FROM {_TABLE} WHERE hold_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise KeyError(selected)
                current = self._value(row)
                if current.owner_id != owner:
                    raise RuntimeError("timestamp issuance hold escaped owner scope.")
                released = current.release(
                    actor=actor,
                    now=time.time() if now is None else now,
                )
                if released != current:
                    connection.execute(
                        f"""UPDATE {_TABLE} SET status=?, released_actor_id=?,
                        released_binding_method=?, released_binding_digest=?, released_at=?,
                        hold_digest=? WHERE hold_id=? AND status='active'""",
                        (
                            released.status,
                            released.released_actor_id,
                            released.released_binding_method,
                            released.released_binding_digest,
                            released.released_at,
                            released.hold_digest,
                            selected,
                        ),
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def list(
        self,
        *,
        owner_id: str,
        issuance_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> tuple[CustodyTimestampIssuanceHold, ...]:
        owner = normalize_owner_id(owner_id)
        issuance = None if issuance_id is None else _digest(issuance_id, "issuance_id")
        selected_status = None if status is None else _identifier(status, "status", 20)
        if selected_status is not None and selected_status not in {"active", "released"}:
            raise ValueError("timestamp issuance hold status is unsupported.")
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        query = f"SELECT * FROM {_TABLE} WHERE owner_id=?"
        parameters: list[Any] = [owner]
        if issuance is not None:
            query += " AND issuance_id=?"
            parameters.append(issuance)
        if selected_status is not None:
            query += " AND status=?"
            parameters.append(selected_status)
        query += " ORDER BY created_at DESC, hold_id DESC LIMIT ?"
        parameters.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return tuple(self._value(row) for row in rows)

    def active_issuance_ids(
        self,
        *,
        owner_id: str,
        limit: int = _MAX_LIMIT,
    ) -> frozenset[str]:
        values = self.list(owner_id=owner_id, status="active", limit=limit)
        if len(values) >= limit:
            raise RuntimeError("active timestamp issuance holds reached the bounded limit.")
        return frozenset(value.issuance_id for value in values)


__all__ = [
    "CustodyTimestampIssuanceHold",
    "CustodyTimestampIssuanceHoldStore",
    "deterministic_timestamp_issuance_hold_id",
]
