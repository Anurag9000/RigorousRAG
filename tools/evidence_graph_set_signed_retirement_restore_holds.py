"""Durable process-owned legal holds for signed retirement restore intents."""

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

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_MAX_LIMIT = 10_000
_SCHEMA_VERSION = 1


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


def deterministic_restore_hold_id(
    *,
    owner_id: str,
    restore_id: str,
    hold_key: str,
) -> str:
    return _canonical_digest(
        {
            "scope": "rigorousrag-signed-retirement-restore-hold-v1",
            "owner_id": normalize_owner_id(owner_id),
            "restore_id": _digest(restore_id, "restore_id"),
            "hold_key": _identifier(hold_key, "hold_key", 200),
        }
    )


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("restore hold database path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("restore hold database path is invalid.")
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
            raise ValueError(
                "restore hold database path could not be validated."
            ) from exc
        if _redirecting(info):
            raise ValueError(
                "restore hold database path may not contain redirects."
            )
    return absolute


@dataclass(frozen=True)
class SignedRetirementRestoreHold:
    hold_id: str
    owner_id: str
    restore_id: str
    hold_key: str
    reason_code: str
    status: str
    created_actor_id: str
    created_binding_method: str
    created_binding_digest: str
    created_at: float
    released_actor_id: str | None = None
    released_binding_method: str | None = None
    released_binding_digest: str | None = None
    released_at: float | None = None
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        restore = _digest(self.restore_id, "restore_id")
        key = _identifier(self.hold_key, "hold_key", 200)
        hold = _digest(self.hold_id, "hold_id")
        expected = deterministic_restore_hold_id(
            owner_id=owner,
            restore_id=restore,
            hold_key=key,
        )
        if hold != expected:
            raise ValueError("hold_id differs from immutable hold scope.")
        reason = _identifier(self.reason_code, "reason_code", 100)
        status = _identifier(self.status, "status", 20)
        if status not in {"active", "released"}:
            raise ValueError("restore hold status is unsupported.")
        created_actor = _identifier(
            self.created_actor_id,
            "created_actor_id",
            200,
        )
        created_method = _identifier(
            self.created_binding_method,
            "created_binding_method",
            50,
        )
        created_digest = _digest(
            self.created_binding_digest,
            "created_binding_digest",
        )
        created = _timestamp(self.created_at, "created_at")
        release_fields = (
            self.released_actor_id,
            self.released_binding_method,
            self.released_binding_digest,
            self.released_at,
        )
        if status == "active":
            if any(value is not None for value in release_fields):
                raise ValueError("active hold may not contain release fields.")
            released_actor = released_method = released_digest = released_at = None
        else:
            if any(value is None for value in release_fields):
                raise ValueError("released hold requires complete release fields.")
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
            if released_at < created:
                raise ValueError("hold release predates creation.")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("restore hold schema is unsupported.")
        object.__setattr__(self, "hold_id", hold)
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "restore_id", restore)
        object.__setattr__(self, "hold_key", key)
        object.__setattr__(self, "reason_code", reason)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "created_actor_id", created_actor)
        object.__setattr__(self, "created_binding_method", created_method)
        object.__setattr__(self, "created_binding_digest", created_digest)
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "released_actor_id", released_actor)
        object.__setattr__(self, "released_binding_method", released_method)
        object.__setattr__(self, "released_binding_digest", released_digest)
        object.__setattr__(self, "released_at", released_at)

    @property
    def hold_digest(self) -> str:
        return _canonical_digest(
            {
                "scope": "rigorousrag-signed-retirement-restore-hold-record-v1",
                "hold_id": self.hold_id,
                "owner_id": self.owner_id,
                "restore_id": self.restore_id,
                "hold_key": self.hold_key,
                "reason_code": self.reason_code,
                "status": self.status,
                "created_actor_id": self.created_actor_id,
                "created_binding_method": self.created_binding_method,
                "created_binding_digest": self.created_binding_digest,
                "created_at": self.created_at,
                "released_actor_id": self.released_actor_id,
                "released_binding_method": self.released_binding_method,
                "released_binding_digest": self.released_binding_digest,
                "released_at": self.released_at,
                "schema_version": self.schema_version,
            }
        )

    @classmethod
    def create(
        cls,
        *,
        owner_id: str,
        restore_id: str,
        hold_key: str,
        reason_code: str,
        actor: ReviewActorBinding,
        now: float | None = None,
    ) -> "SignedRetirementRestoreHold":
        if not isinstance(actor, ReviewActorBinding):
            raise ValueError("actor must be ReviewActorBinding.")
        owner = normalize_owner_id(owner_id)
        restore = _digest(restore_id, "restore_id")
        key = _identifier(hold_key, "hold_key", 200)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        if actor.expires_at is not None and actor.expires_at < timestamp:
            raise PermissionError("review actor binding expired before hold creation.")
        return cls(
            hold_id=deterministic_restore_hold_id(
                owner_id=owner,
                restore_id=restore,
                hold_key=key,
            ),
            owner_id=owner,
            restore_id=restore,
            hold_key=key,
            reason_code=reason_code,
            status="active",
            created_actor_id=actor.actor_id,
            created_binding_method=actor.binding_method,
            created_binding_digest=actor.binding_digest,
            created_at=timestamp,
        )


class SignedRetirementRestoreHoldStore:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("restore hold database parent must be a directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("restore hold database is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
            or self._file_identity() != self._database_identity
        ):
            raise RuntimeError("restore hold database identity changed.")

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
                CREATE TABLE IF NOT EXISTS evidence_graph_set_signed_restore_holds (
                    hold_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    restore_id TEXT NOT NULL,
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
                    schema_version INTEGER NOT NULL,
                    UNIQUE(owner_id, restore_id, hold_key)
                );
                CREATE INDEX IF NOT EXISTS signed_restore_hold_scope
                    ON evidence_graph_set_signed_restore_holds(
                        owner_id, restore_id, status, created_at, hold_id
                    );
                """
            )

    @staticmethod
    def _value(row: sqlite3.Row) -> SignedRetirementRestoreHold:
        try:
            return SignedRetirementRestoreHold(
                hold_id=row["hold_id"],
                owner_id=row["owner_id"],
                restore_id=row["restore_id"],
                hold_key=row["hold_key"],
                reason_code=row["reason_code"],
                status=row["status"],
                created_actor_id=row["created_actor_id"],
                created_binding_method=row["created_binding_method"],
                created_binding_digest=row["created_binding_digest"],
                created_at=row["created_at"],
                released_actor_id=row["released_actor_id"],
                released_binding_method=row["released_binding_method"],
                released_binding_digest=row["released_binding_digest"],
                released_at=row["released_at"],
                schema_version=int(row["schema_version"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("stored restore hold is corrupt.") from exc

    def place(
        self,
        *,
        owner_id: str,
        restore_id: str,
        hold_key: str,
        reason_code: str,
        actor: ReviewActorBinding,
        restore_journal: Any,
        now: float | None = None,
    ) -> SignedRetirementRestoreHold:
        owner = normalize_owner_id(owner_id)
        restore = _digest(restore_id, "restore_id")
        if not callable(getattr(restore_journal, "get", None)):
            raise ValueError("restore_journal lacks the required read boundary.")
        restore_value = restore_journal.get(restore)
        if restore_value.owner_id != owner:
            raise RuntimeError("restore escaped legal-hold owner scope.")
        value = SignedRetirementRestoreHold.create(
            owner_id=owner,
            restore_id=restore,
            hold_key=hold_key,
            reason_code=reason_code,
            actor=actor,
            now=now,
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_set_signed_restore_holds "
                    "WHERE hold_id=?",
                    (value.hold_id,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO evidence_graph_set_signed_restore_holds "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
                        (
                            value.hold_id,
                            value.owner_id,
                            value.restore_id,
                            value.hold_key,
                            value.reason_code,
                            value.status,
                            value.created_actor_id,
                            value.created_binding_method,
                            value.created_binding_digest,
                            value.created_at,
                            None,
                            None,
                            None,
                            None,
                        ),
                    )
                    connection.execute("COMMIT")
                    return value
                stored = self._value(row)
                if (
                    stored.owner_id != value.owner_id
                    or stored.restore_id != value.restore_id
                    or stored.hold_key != value.hold_key
                    or stored.reason_code != value.reason_code
                    or stored.created_actor_id != value.created_actor_id
                    or stored.created_binding_digest != value.created_binding_digest
                ):
                    raise RuntimeError("restore hold identity collision detected.")
                connection.execute("COMMIT")
                return stored
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def get(self, hold_id: str) -> SignedRetirementRestoreHold:
        selected = _digest(hold_id, "hold_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM evidence_graph_set_signed_restore_holds "
                "WHERE hold_id=?",
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
    ) -> SignedRetirementRestoreHold:
        selected = _digest(hold_id, "hold_id")
        if selected != _digest(confirm_hold_id, "confirm_hold_id"):
            raise ValueError("hold confirmation differs.")
        owner = normalize_owner_id(owner_id)
        if not isinstance(actor, ReviewActorBinding):
            raise ValueError("actor must be ReviewActorBinding.")
        timestamp = _timestamp(time.time() if now is None else now, "now")
        if actor.expires_at is not None and actor.expires_at < timestamp:
            raise PermissionError("review actor binding expired before hold release.")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_set_signed_restore_holds "
                    "WHERE hold_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise KeyError(selected)
                current = self._value(row)
                if current.owner_id != owner:
                    raise RuntimeError("restore hold escaped owner scope.")
                if current.status == "released":
                    connection.execute("COMMIT")
                    return current
                connection.execute(
                    "UPDATE evidence_graph_set_signed_restore_holds "
                    "SET status='released', released_actor_id=?, "
                    "released_binding_method=?, released_binding_digest=?, "
                    "released_at=? WHERE hold_id=? AND status='active'",
                    (
                        actor.actor_id,
                        actor.binding_method,
                        actor.binding_digest,
                        timestamp,
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
        restore_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> tuple[SignedRetirementRestoreHold, ...]:
        owner = normalize_owner_id(owner_id)
        restore = None if restore_id is None else _digest(restore_id, "restore_id")
        selected_status = (
            None if status is None else _identifier(status, "status", 20)
        )
        if selected_status is not None and selected_status not in {
            "active",
            "released",
        }:
            raise ValueError("restore hold status is unsupported.")
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        query = (
            "SELECT * FROM evidence_graph_set_signed_restore_holds "
            "WHERE owner_id=?"
        )
        params: list[Any] = [owner]
        if restore is not None:
            query += " AND restore_id=?"
            params.append(restore)
        if selected_status is not None:
            query += " AND status=?"
            params.append(selected_status)
        query += " ORDER BY created_at DESC, hold_id DESC LIMIT ?"
        params.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(self._value(row) for row in rows)

    def active_restore_ids(
        self,
        *,
        owner_id: str,
        limit: int = _MAX_LIMIT,
    ) -> frozenset[str]:
        values = self.list(
            owner_id=owner_id,
            status="active",
            limit=limit,
        )
        if len(values) >= limit:
            raise RuntimeError("active restore hold list reached the bounded limit.")
        return frozenset(value.restore_id for value in values)


__all__ = [
    "SignedRetirementRestoreHold",
    "SignedRetirementRestoreHoldStore",
    "deterministic_restore_hold_id",
]
