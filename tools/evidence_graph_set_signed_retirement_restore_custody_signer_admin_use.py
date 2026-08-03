"""One-operation reservations for signed custody signer administration."""

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
_STATES = frozenset({"reserved", "committed"})
_ACTIONS = frozenset({"register", "retire"})
_SIGNED_METHODS = frozenset(
    {
        "signed_assertion",
        "hmac_assertion",
        "hmac_signed_assertion",
    }
)
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_TABLE = "evidence_graph_restore_custody_signer_admin_uses"


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
        raise ValueError("signer admin-use database path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("signer admin-use database path is invalid.")
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
            raise ValueError("signer admin-use path could not be validated.") from exc
        if _redirecting(info):
            raise ValueError("signer admin-use path may not contain redirects.")
    return absolute


def _assertion_fields(binding: ReviewActorBinding) -> tuple[str, str, float]:
    if not isinstance(binding, ReviewActorBinding):
        raise ValueError("binding must be ReviewActorBinding.")
    method = _identifier(binding.binding_method, "binding_method", 50)
    if method not in _SIGNED_METHODS:
        raise PermissionError("signer admin-use requires a supported signed assertion.")
    assertion_digest = getattr(binding, "assertion_digest", None)
    if assertion_digest is None:
        assertion_digest = binding.binding_digest
    issuer = getattr(binding, "assertion_issuer", None)
    if issuer is None:
        issuer = getattr(binding, "issuer", None)
    expires_at = getattr(binding, "assertion_expires_at", None)
    if expires_at is None:
        expires_at = getattr(binding, "expires_at", None)
    if issuer is None or expires_at is None:
        raise PermissionError(
            "signed signer administration requires issuer and expiry provenance."
        )
    return (
        _digest(assertion_digest, "assertion_digest"),
        _identifier(issuer, "assertion_issuer", 200),
        _timestamp(expires_at, "assertion_expires_at"),
    )


def deterministic_signer_admin_use_id(
    *,
    binding_digest: str,
    owner_id: str,
    action: str,
    key_id: str,
    action_digest: str,
) -> str:
    selected_action = _identifier(action, "action", 30)
    if selected_action not in _ACTIONS:
        raise ValueError("signer administration action is unsupported.")
    return _canonical_digest(
        {
            "scope": "rigorousrag-custody-signer-admin-use-v1",
            "binding_digest": _digest(binding_digest, "binding_digest"),
            "owner_id": normalize_owner_id(owner_id),
            "action": selected_action,
            "key_id": _identifier(key_id, "key_id", 200),
            "action_digest": _digest(action_digest, "action_digest"),
        }
    )


@dataclass(frozen=True)
class CustodySignerAdminUse:
    use_id: str
    binding_digest: str
    assertion_digest: str
    assertion_issuer: str
    assertion_expires_at: float
    actor_id: str
    binding_method: str
    owner_id: str
    action: str
    key_id: str
    action_digest: str
    state: str
    reserved_at: float
    committed_at: float | None
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        binding = _digest(self.binding_digest, "binding_digest")
        assertion = _digest(self.assertion_digest, "assertion_digest")
        issuer = _identifier(self.assertion_issuer, "assertion_issuer", 200)
        expiry = _timestamp(self.assertion_expires_at, "assertion_expires_at")
        actor = _identifier(self.actor_id, "actor_id", 200)
        method = _identifier(self.binding_method, "binding_method", 50)
        if method not in _SIGNED_METHODS:
            raise ValueError("signer admin-use binding method is unsupported.")
        owner = normalize_owner_id(self.owner_id)
        action = _identifier(self.action, "action", 30)
        if action not in _ACTIONS:
            raise ValueError("signer administration action is unsupported.")
        key_id = _identifier(self.key_id, "key_id", 200)
        action_digest = _digest(self.action_digest, "action_digest")
        use_id = _digest(self.use_id, "use_id")
        if use_id != deterministic_signer_admin_use_id(
            binding_digest=binding,
            owner_id=owner,
            action=action,
            key_id=key_id,
            action_digest=action_digest,
        ):
            raise ValueError("use_id differs from signer administration scope.")
        state = _identifier(self.state, "state", 30)
        if state not in _STATES:
            raise ValueError("signer admin-use state is unsupported.")
        reserved = _timestamp(self.reserved_at, "reserved_at")
        committed = (
            None
            if self.committed_at is None
            else _timestamp(self.committed_at, "committed_at")
        )
        if expiry <= reserved:
            raise ValueError("signed assertion expired before reservation.")
        if state == "reserved" and committed is not None:
            raise ValueError("reserved signer admin-use may not be committed.")
        if state == "committed":
            if committed is None or committed < reserved:
                raise ValueError("committed signer admin-use timestamp is invalid.")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("signer admin-use schema is unsupported.")
        object.__setattr__(self, "use_id", use_id)
        object.__setattr__(self, "binding_digest", binding)
        object.__setattr__(self, "assertion_digest", assertion)
        object.__setattr__(self, "assertion_issuer", issuer)
        object.__setattr__(self, "assertion_expires_at", expiry)
        object.__setattr__(self, "actor_id", actor)
        object.__setattr__(self, "binding_method", method)
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "key_id", key_id)
        object.__setattr__(self, "action_digest", action_digest)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "reserved_at", reserved)
        object.__setattr__(self, "committed_at", committed)

    @classmethod
    def reserve(
        cls,
        *,
        binding: ReviewActorBinding,
        owner_id: str,
        action: str,
        key_id: str,
        action_digest: str,
        now: float,
    ) -> "CustodySignerAdminUse":
        assertion, issuer, expiry = _assertion_fields(binding)
        timestamp = _timestamp(now, "now")
        return cls(
            use_id=deterministic_signer_admin_use_id(
                binding_digest=binding.binding_digest,
                owner_id=owner_id,
                action=action,
                key_id=key_id,
                action_digest=action_digest,
            ),
            binding_digest=binding.binding_digest,
            assertion_digest=assertion,
            assertion_issuer=issuer,
            assertion_expires_at=expiry,
            actor_id=binding.actor_id,
            binding_method=binding.binding_method,
            owner_id=owner_id,
            action=action,
            key_id=key_id,
            action_digest=action_digest,
            state="reserved",
            reserved_at=timestamp,
            committed_at=None,
        )


class CustodySignerAdminUseStore:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("signer admin-use database parent must be a directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("signer admin-use database is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
            or self._file_identity() != self._database_identity
        ):
            raise RuntimeError("signer admin-use database identity changed.")

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
                    use_id TEXT PRIMARY KEY,
                    binding_digest TEXT NOT NULL UNIQUE,
                    assertion_digest TEXT NOT NULL,
                    assertion_issuer TEXT NOT NULL,
                    assertion_expires_at REAL NOT NULL,
                    actor_id TEXT NOT NULL,
                    binding_method TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    key_id TEXT NOT NULL,
                    action_digest TEXT NOT NULL,
                    state TEXT NOT NULL,
                    reserved_at REAL NOT NULL,
                    committed_at REAL,
                    schema_version INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS custody_signer_admin_use_scope
                    ON {_TABLE}(owner_id, action, key_id, reserved_at, use_id);
                """
            )

    @staticmethod
    def _value(row: sqlite3.Row) -> CustodySignerAdminUse:
        try:
            return CustodySignerAdminUse(
                use_id=row["use_id"],
                binding_digest=row["binding_digest"],
                assertion_digest=row["assertion_digest"],
                assertion_issuer=row["assertion_issuer"],
                assertion_expires_at=row["assertion_expires_at"],
                actor_id=row["actor_id"],
                binding_method=row["binding_method"],
                owner_id=row["owner_id"],
                action=row["action"],
                key_id=row["key_id"],
                action_digest=row["action_digest"],
                state=row["state"],
                reserved_at=row["reserved_at"],
                committed_at=row["committed_at"],
                schema_version=int(row["schema_version"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("stored signer admin-use record is corrupt.") from exc

    def reserve(self, value: CustodySignerAdminUse) -> CustodySignerAdminUse:
        if not isinstance(value, CustodySignerAdminUse):
            raise ValueError("value must be CustodySignerAdminUse.")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    f"SELECT * FROM {_TABLE} WHERE binding_digest=?",
                    (value.binding_digest,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        f"INSERT INTO {_TABLE} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
                        (
                            value.use_id,
                            value.binding_digest,
                            value.assertion_digest,
                            value.assertion_issuer,
                            value.assertion_expires_at,
                            value.actor_id,
                            value.binding_method,
                            value.owner_id,
                            value.action,
                            value.key_id,
                            value.action_digest,
                            value.state,
                            value.reserved_at,
                            value.committed_at,
                        ),
                    )
                    connection.execute("COMMIT")
                    return value
                stored = self._value(row)
                if stored.use_id != value.use_id:
                    raise RuntimeError(
                        "signed signer assertion is already reserved for another action."
                    )
                connection.execute("COMMIT")
                return stored
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def get(self, use_id: str) -> CustodySignerAdminUse:
        selected = _digest(use_id, "use_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM {_TABLE} WHERE use_id=?",
                (selected,),
            ).fetchone()
        if row is None:
            raise KeyError(selected)
        return self._value(row)

    def commit(
        self,
        use_id: str,
        *,
        confirm_use_id: str,
        now: float | None = None,
    ) -> CustodySignerAdminUse:
        selected = _digest(use_id, "use_id")
        confirmation = _digest(confirm_use_id, "confirm_use_id")
        if confirmation != selected:
            raise ValueError("signer admin-use confirmation differs.")
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    f"SELECT * FROM {_TABLE} WHERE use_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise KeyError(selected)
                current = self._value(row)
                if current.state == "committed":
                    connection.execute("COMMIT")
                    return current
                committed_at = max(timestamp, current.reserved_at)
                connection.execute(
                    f"UPDATE {_TABLE} SET state='committed', committed_at=? "
                    "WHERE use_id=?",
                    (committed_at, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)


__all__ = [
    "CustodySignerAdminUse",
    "CustodySignerAdminUseStore",
    "deterministic_signer_admin_use_id",
]
