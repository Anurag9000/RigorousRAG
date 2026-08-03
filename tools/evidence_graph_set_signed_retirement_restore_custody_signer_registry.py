"""Durable public-key registry for external restore custody signers."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _integer,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature import (
    _load_public,
    _public_fingerprint,
)
from tools.security import normalize_owner_id

_SCHEMA_VERSION = 1
_STATES = frozenset({"active", "retired"})
_MAX_LIMIT = 10_000
_MAX_PATH = 4096
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_TABLE = "evidence_graph_restore_custody_signers"


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


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("signer registry path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("signer registry path is invalid.")
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
            raise ValueError("signer registry path could not be validated.") from exc
        if _redirecting(info):
            raise ValueError("signer registry path may not contain redirects.")
    return absolute


@dataclass(frozen=True)
class CustodySignerKeyRecord:
    owner_id: str
    key_id: str
    issuer: str
    algorithm: str
    public_key_sha256: str
    state: str
    registered_actor_id: str
    registered_binding_method: str
    registered_binding_digest: str
    registered_at: float
    retired_actor_id: str | None
    retired_binding_method: str | None
    retired_binding_digest: str | None
    retired_at: float | None
    record_digest: str
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        key_id = _identifier(self.key_id, "key_id", 200)
        issuer = _identifier(self.issuer, "issuer", 200)
        algorithm = _identifier(self.algorithm, "algorithm", 30)
        if algorithm != "ed25519":
            raise ValueError("signer algorithm is unsupported.")
        fingerprint = _digest(self.public_key_sha256, "public_key_sha256")
        state = _identifier(self.state, "state", 30)
        if state not in _STATES:
            raise ValueError("signer state is unsupported.")
        registered_actor = _identifier(
            self.registered_actor_id,
            "registered_actor_id",
            200,
        )
        registered_method = _identifier(
            self.registered_binding_method,
            "registered_binding_method",
            50,
        )
        registered_binding = _digest(
            self.registered_binding_digest,
            "registered_binding_digest",
        )
        registered_at = _timestamp(self.registered_at, "registered_at")
        retired_values = (
            self.retired_actor_id,
            self.retired_binding_method,
            self.retired_binding_digest,
            self.retired_at,
        )
        if state == "active":
            if any(value is not None for value in retired_values):
                raise ValueError("active signer may not contain retirement fields.")
            retired_actor = retired_method = retired_binding = retired_at = None
        else:
            if any(value is None for value in retired_values):
                raise ValueError("retired signer requires complete retirement fields.")
            retired_actor = _identifier(
                self.retired_actor_id,
                "retired_actor_id",
                200,
            )
            retired_method = _identifier(
                self.retired_binding_method,
                "retired_binding_method",
                50,
            )
            retired_binding = _digest(
                self.retired_binding_digest,
                "retired_binding_digest",
            )
            retired_at = _timestamp(self.retired_at, "retired_at")
            if retired_at < registered_at:
                raise ValueError("signer retirement predates registration.")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("signer registry schema is unsupported.")
        stable = {
            "scope": "rigorousrag-restore-custody-signer-key-v1",
            "owner_id": owner,
            "key_id": key_id,
            "issuer": issuer,
            "algorithm": algorithm,
            "public_key_sha256": fingerprint,
            "state": state,
            "registered_actor_id": registered_actor,
            "registered_binding_method": registered_method,
            "registered_binding_digest": registered_binding,
            "registered_at": registered_at,
            "retired_actor_id": retired_actor,
            "retired_binding_method": retired_method,
            "retired_binding_digest": retired_binding,
            "retired_at": retired_at,
            "schema_version": self.schema_version,
        }
        digest = _digest(self.record_digest, "record_digest")
        if digest != _canonical_digest(stable):
            raise ValueError("record_digest differs from signer record.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "key_id", key_id)
        object.__setattr__(self, "issuer", issuer)
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "public_key_sha256", fingerprint)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "registered_actor_id", registered_actor)
        object.__setattr__(self, "registered_binding_method", registered_method)
        object.__setattr__(self, "registered_binding_digest", registered_binding)
        object.__setattr__(self, "registered_at", registered_at)
        object.__setattr__(self, "retired_actor_id", retired_actor)
        object.__setattr__(self, "retired_binding_method", retired_method)
        object.__setattr__(self, "retired_binding_digest", retired_binding)
        object.__setattr__(self, "retired_at", retired_at)
        object.__setattr__(self, "record_digest", digest)

    @classmethod
    def active(
        cls,
        *,
        owner_id: str,
        key_id: str,
        issuer: str,
        public_key_sha256: str,
        actor: ReviewActorBinding,
        now: float,
    ) -> "CustodySignerKeyRecord":
        if not isinstance(actor, ReviewActorBinding):
            raise ValueError("actor must be ReviewActorBinding.")
        timestamp = _timestamp(now, "now")
        values = {
            "owner_id": normalize_owner_id(owner_id),
            "key_id": _identifier(key_id, "key_id", 200),
            "issuer": _identifier(issuer, "issuer", 200),
            "algorithm": "ed25519",
            "public_key_sha256": _digest(
                public_key_sha256,
                "public_key_sha256",
            ),
            "state": "active",
            "registered_actor_id": actor.actor_id,
            "registered_binding_method": actor.binding_method,
            "registered_binding_digest": actor.binding_digest,
            "registered_at": timestamp,
            "retired_actor_id": None,
            "retired_binding_method": None,
            "retired_binding_digest": None,
            "retired_at": None,
            "schema_version": _SCHEMA_VERSION,
        }
        return cls(
            **values,
            record_digest=_canonical_digest(
                {
                    "scope": "rigorousrag-restore-custody-signer-key-v1",
                    **values,
                }
            ),
        )

    def retire(
        self,
        *,
        actor: ReviewActorBinding,
        now: float,
    ) -> "CustodySignerKeyRecord":
        if self.state == "retired":
            if (
                self.retired_actor_id != actor.actor_id
                or self.retired_binding_method != actor.binding_method
                or self.retired_binding_digest != actor.binding_digest
            ):
                raise RuntimeError("signer is already retired by another actor binding.")
            return self
        timestamp = max(_timestamp(now, "now"), self.registered_at)
        values = {
            **asdict(self),
            "state": "retired",
            "retired_actor_id": actor.actor_id,
            "retired_binding_method": actor.binding_method,
            "retired_binding_digest": actor.binding_digest,
            "retired_at": timestamp,
        }
        values.pop("record_digest", None)
        return CustodySignerKeyRecord(
            **values,
            record_digest=_canonical_digest(
                {
                    "scope": "rigorousrag-restore-custody-signer-key-v1",
                    **values,
                }
            ),
        )


class CustodySignerKeyRegistry:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("signer registry parent must be a directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("signer registry is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
            or self._file_identity() != self._database_identity
        ):
            raise RuntimeError("signer registry identity changed.")

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
                f"""
                CREATE TABLE IF NOT EXISTS {_TABLE} (
                    owner_id TEXT NOT NULL,
                    key_id TEXT NOT NULL,
                    issuer TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    public_key_sha256 TEXT NOT NULL,
                    state TEXT NOT NULL,
                    registered_actor_id TEXT NOT NULL,
                    registered_binding_method TEXT NOT NULL,
                    registered_binding_digest TEXT NOT NULL,
                    registered_at REAL NOT NULL,
                    retired_actor_id TEXT,
                    retired_binding_method TEXT,
                    retired_binding_digest TEXT,
                    retired_at REAL,
                    record_digest TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    PRIMARY KEY(owner_id, key_id)
                );
                CREATE INDEX IF NOT EXISTS restore_custody_signer_state
                    ON {_TABLE}(owner_id, state, registered_at, key_id);
                CREATE UNIQUE INDEX IF NOT EXISTS restore_custody_signer_fingerprint
                    ON {_TABLE}(owner_id, public_key_sha256);
                """
            )

    @staticmethod
    def _value(row: sqlite3.Row) -> CustodySignerKeyRecord:
        try:
            return CustodySignerKeyRecord(
                owner_id=row["owner_id"],
                key_id=row["key_id"],
                issuer=row["issuer"],
                algorithm=row["algorithm"],
                public_key_sha256=row["public_key_sha256"],
                state=row["state"],
                registered_actor_id=row["registered_actor_id"],
                registered_binding_method=row["registered_binding_method"],
                registered_binding_digest=row["registered_binding_digest"],
                registered_at=row["registered_at"],
                retired_actor_id=row["retired_actor_id"],
                retired_binding_method=row["retired_binding_method"],
                retired_binding_digest=row["retired_binding_digest"],
                retired_at=row["retired_at"],
                record_digest=row["record_digest"],
                schema_version=int(row["schema_version"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("stored signer record is corrupt.") from exc

    @staticmethod
    def _registration_scope(value: CustodySignerKeyRecord) -> tuple[Any, ...]:
        return (
            value.owner_id,
            value.key_id,
            value.issuer,
            value.algorithm,
            value.public_key_sha256,
            value.registered_actor_id,
            value.registered_binding_method,
            value.registered_binding_digest,
        )

    def register(
        self,
        *,
        owner_id: str,
        key_id: str,
        issuer: str,
        public_key_path: str | os.PathLike[str],
        actor: ReviewActorBinding,
        now: float | None = None,
    ) -> CustodySignerKeyRecord:
        fingerprint = _public_fingerprint(_load_public(public_key_path))
        value = CustodySignerKeyRecord.active(
            owner_id=owner_id,
            key_id=key_id,
            issuer=issuer,
            public_key_sha256=fingerprint,
            actor=actor,
            now=time.time() if now is None else now,
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    f"SELECT * FROM {_TABLE} WHERE owner_id=? AND key_id=?",
                    (value.owner_id, value.key_id),
                ).fetchone()
                if row is None:
                    connection.execute(
                        f"INSERT INTO {_TABLE} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
                        (
                            value.owner_id,
                            value.key_id,
                            value.issuer,
                            value.algorithm,
                            value.public_key_sha256,
                            value.state,
                            value.registered_actor_id,
                            value.registered_binding_method,
                            value.registered_binding_digest,
                            value.registered_at,
                            value.retired_actor_id,
                            value.retired_binding_method,
                            value.retired_binding_digest,
                            value.retired_at,
                            value.record_digest,
                        ),
                    )
                    connection.execute("COMMIT")
                    return value
                stored = self._value(row)
                if self._registration_scope(stored) != self._registration_scope(value):
                    raise RuntimeError("signer key identity collision detected.")
                connection.execute("COMMIT")
                return stored
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def get(self, *, owner_id: str, key_id: str) -> CustodySignerKeyRecord:
        owner = normalize_owner_id(owner_id)
        selected_key = _identifier(key_id, "key_id", 200)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM {_TABLE} WHERE owner_id=? AND key_id=?",
                (owner, selected_key),
            ).fetchone()
        if row is None:
            raise KeyError((owner, selected_key))
        return self._value(row)

    def list(
        self,
        *,
        owner_id: str,
        state: str | None = None,
        limit: int = 100,
    ) -> tuple[CustodySignerKeyRecord, ...]:
        owner = normalize_owner_id(owner_id)
        selected_state = None if state is None else _identifier(state, "state", 30)
        if selected_state is not None and selected_state not in _STATES:
            raise ValueError("signer state is unsupported.")
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        query = f"SELECT * FROM {_TABLE} WHERE owner_id=?"
        parameters: list[Any] = [owner]
        if selected_state is not None:
            query += " AND state=?"
            parameters.append(selected_state)
        query += " ORDER BY registered_at DESC, key_id DESC LIMIT ?"
        parameters.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return tuple(self._value(row) for row in rows)

    def retire(
        self,
        *,
        owner_id: str,
        key_id: str,
        confirm_key_id: str,
        actor: ReviewActorBinding,
        now: float | None = None,
    ) -> CustodySignerKeyRecord:
        owner = normalize_owner_id(owner_id)
        selected_key = _identifier(key_id, "key_id", 200)
        confirmation = _identifier(confirm_key_id, "confirm_key_id", 200)
        if confirmation != selected_key:
            raise ValueError("signer retirement confirmation differs.")
        timestamp = time.time() if now is None else now
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    f"SELECT * FROM {_TABLE} WHERE owner_id=? AND key_id=?",
                    (owner, selected_key),
                ).fetchone()
                if row is None:
                    raise KeyError((owner, selected_key))
                current = self._value(row)
                retired = current.retire(actor=actor, now=timestamp)
                if retired == current:
                    connection.execute("COMMIT")
                    return current
                connection.execute(
                    f"UPDATE {_TABLE} SET state=?, retired_actor_id=?, "
                    "retired_binding_method=?, retired_binding_digest=?, "
                    "retired_at=?, record_digest=? WHERE owner_id=? AND key_id=?",
                    (
                        retired.state,
                        retired.retired_actor_id,
                        retired.retired_binding_method,
                        retired.retired_binding_digest,
                        retired.retired_at,
                        retired.record_digest,
                        owner,
                        selected_key,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(owner_id=owner, key_id=selected_key)


__all__ = [
    "CustodySignerKeyRecord",
    "CustodySignerKeyRegistry",
]
