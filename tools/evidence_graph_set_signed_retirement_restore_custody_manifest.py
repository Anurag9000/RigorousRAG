"""Durable custody manifests binding restore intents to pre/post receipts."""

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
from tools.evidence_graph_set_signed_retirement_restore_custody_boundary import (
    verify_post_restore_comparison_receipt,
    verify_pre_restore_backup_receipt,
)
from tools.security import normalize_owner_id

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_SCHEMA_VERSION = 1
_MAX_PATH = 4096
_MAX_LIMIT = 10_000
_METHODS = frozenset(
    {"process_environment", "descriptor_file", "hmac_assertion"}
)


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


def deterministic_restore_custody_id(
    *,
    owner_id: str,
    restore_id: str,
    pre_receipt_digest: str,
    backup_sha256: str,
) -> str:
    return _canonical_digest(
        {
            "scope": "rigorousrag-signed-retirement-restore-custody-v1",
            "owner_id": normalize_owner_id(owner_id),
            "restore_id": _digest(restore_id, "restore_id"),
            "pre_receipt_digest": _digest(
                pre_receipt_digest,
                "pre_receipt_digest",
            ),
            "backup_sha256": _digest(backup_sha256, "backup_sha256"),
        }
    )


def _actor_fields(
    actor: ReviewActorBinding,
    *,
    now: float,
) -> tuple[str, str, str]:
    if not isinstance(actor, ReviewActorBinding):
        raise ValueError("actor must be ReviewActorBinding.")
    if actor.expires_at is not None and actor.expires_at < now:
        raise PermissionError("custody actor binding expired.")
    method = _identifier(actor.binding_method, "binding_method", 50)
    if method not in _METHODS:
        raise ValueError("custody actor binding method is unsupported.")
    return (
        _identifier(actor.actor_id, "actor_id", 200),
        method,
        _digest(actor.binding_digest, "binding_digest"),
    )


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("custody database path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("custody database path is invalid.")
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
            raise ValueError("custody database path could not be validated.") from exc
        if _redirecting(info):
            raise ValueError("custody database path may not contain redirects.")
    return absolute


@dataclass(frozen=True)
class SignedRetirementRestoreCustodyManifest:
    custody_id: str
    owner_id: str
    restore_id: str
    snapshot_digest: str
    target_path_digest: str
    pre_receipt_digest: str
    backup_sha256: str
    backup_size_bytes: int
    state: str
    pre_bound_actor_id: str
    pre_bound_method: str
    pre_bound_binding_digest: str
    pre_bound_at: float
    post_receipt_digest: str | None
    target_verification_digest: str | None
    post_bound_actor_id: str | None
    post_bound_method: str | None
    post_bound_binding_digest: str | None
    post_bound_at: float | None
    manifest_digest: str
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        restore = _digest(self.restore_id, "restore_id")
        snapshot = _digest(self.snapshot_digest, "snapshot_digest")
        target = _digest(self.target_path_digest, "target_path_digest")
        pre_receipt = _digest(
            self.pre_receipt_digest,
            "pre_receipt_digest",
        )
        backup = _digest(self.backup_sha256, "backup_sha256")
        size = _integer(
            self.backup_size_bytes,
            "backup_size_bytes",
            1,
            1024 * 1024 * 1024 * 1024,
        )
        custody = _digest(self.custody_id, "custody_id")
        if custody != deterministic_restore_custody_id(
            owner_id=owner,
            restore_id=restore,
            pre_receipt_digest=pre_receipt,
            backup_sha256=backup,
        ):
            raise ValueError("custody_id differs from immutable custody scope.")
        state = _identifier(self.state, "state", 20)
        if state not in {"pre_bound", "post_bound"}:
            raise ValueError("custody state is unsupported.")
        pre_actor = _identifier(
            self.pre_bound_actor_id,
            "pre_bound_actor_id",
            200,
        )
        pre_method = _identifier(
            self.pre_bound_method,
            "pre_bound_method",
            50,
        )
        if pre_method not in _METHODS:
            raise ValueError("pre-bound actor method is unsupported.")
        pre_binding = _digest(
            self.pre_bound_binding_digest,
            "pre_bound_binding_digest",
        )
        pre_at = _timestamp(self.pre_bound_at, "pre_bound_at")
        post_fields = (
            self.post_receipt_digest,
            self.target_verification_digest,
            self.post_bound_actor_id,
            self.post_bound_method,
            self.post_bound_binding_digest,
            self.post_bound_at,
        )
        if state == "pre_bound":
            if any(value is not None for value in post_fields):
                raise ValueError("pre-bound custody may not contain post fields.")
            post_receipt = post_verification = post_actor = post_method = None
            post_binding = post_at = None
        else:
            if any(value is None for value in post_fields):
                raise ValueError("post-bound custody requires complete post fields.")
            post_receipt = _digest(
                self.post_receipt_digest,
                "post_receipt_digest",
            )
            post_verification = _digest(
                self.target_verification_digest,
                "target_verification_digest",
            )
            post_actor = _identifier(
                self.post_bound_actor_id,
                "post_bound_actor_id",
                200,
            )
            post_method = _identifier(
                self.post_bound_method,
                "post_bound_method",
                50,
            )
            if post_method not in _METHODS:
                raise ValueError("post-bound actor method is unsupported.")
            post_binding = _digest(
                self.post_bound_binding_digest,
                "post_bound_binding_digest",
            )
            post_at = _timestamp(self.post_bound_at, "post_bound_at")
            if post_at < pre_at:
                raise ValueError("post custody binding predates pre binding.")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("custody schema is unsupported.")
        stable = {
            "scope": "rigorousrag-signed-retirement-restore-custody-record-v1",
            "custody_id": custody,
            "owner_id": owner,
            "restore_id": restore,
            "snapshot_digest": snapshot,
            "target_path_digest": target,
            "pre_receipt_digest": pre_receipt,
            "backup_sha256": backup,
            "backup_size_bytes": size,
            "state": state,
            "pre_bound_actor_id": pre_actor,
            "pre_bound_method": pre_method,
            "pre_bound_binding_digest": pre_binding,
            "pre_bound_at": pre_at,
            "post_receipt_digest": post_receipt,
            "target_verification_digest": post_verification,
            "post_bound_actor_id": post_actor,
            "post_bound_method": post_method,
            "post_bound_binding_digest": post_binding,
            "post_bound_at": post_at,
            "schema_version": self.schema_version,
        }
        manifest = _digest(self.manifest_digest, "manifest_digest")
        if manifest != _canonical_digest(stable):
            raise ValueError("manifest_digest differs from custody content.")
        object.__setattr__(self, "custody_id", custody)
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "restore_id", restore)
        object.__setattr__(self, "snapshot_digest", snapshot)
        object.__setattr__(self, "target_path_digest", target)
        object.__setattr__(self, "pre_receipt_digest", pre_receipt)
        object.__setattr__(self, "backup_sha256", backup)
        object.__setattr__(self, "backup_size_bytes", size)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "pre_bound_actor_id", pre_actor)
        object.__setattr__(self, "pre_bound_method", pre_method)
        object.__setattr__(self, "pre_bound_binding_digest", pre_binding)
        object.__setattr__(self, "pre_bound_at", pre_at)
        object.__setattr__(self, "post_receipt_digest", post_receipt)
        object.__setattr__(
            self,
            "target_verification_digest",
            post_verification,
        )
        object.__setattr__(self, "post_bound_actor_id", post_actor)
        object.__setattr__(self, "post_bound_method", post_method)
        object.__setattr__(self, "post_bound_binding_digest", post_binding)
        object.__setattr__(self, "post_bound_at", post_at)
        object.__setattr__(self, "manifest_digest", manifest)

    @classmethod
    def create_pre_bound(
        cls,
        *,
        owner_id: str,
        restore_id: str,
        snapshot_digest: str,
        target_path_digest: str,
        pre_receipt_digest: str,
        backup_sha256: str,
        backup_size_bytes: int,
        actor: ReviewActorBinding,
        now: float,
    ) -> "SignedRetirementRestoreCustodyManifest":
        timestamp = _timestamp(now, "now")
        actor_id, method, binding = _actor_fields(actor, now=timestamp)
        values = {
            "custody_id": deterministic_restore_custody_id(
                owner_id=owner_id,
                restore_id=restore_id,
                pre_receipt_digest=pre_receipt_digest,
                backup_sha256=backup_sha256,
            ),
            "owner_id": owner_id,
            "restore_id": restore_id,
            "snapshot_digest": snapshot_digest,
            "target_path_digest": target_path_digest,
            "pre_receipt_digest": pre_receipt_digest,
            "backup_sha256": backup_sha256,
            "backup_size_bytes": backup_size_bytes,
            "state": "pre_bound",
            "pre_bound_actor_id": actor_id,
            "pre_bound_method": method,
            "pre_bound_binding_digest": binding,
            "pre_bound_at": timestamp,
            "post_receipt_digest": None,
            "target_verification_digest": None,
            "post_bound_actor_id": None,
            "post_bound_method": None,
            "post_bound_binding_digest": None,
            "post_bound_at": None,
            "schema_version": _SCHEMA_VERSION,
        }
        stable = {
            "scope": "rigorousrag-signed-retirement-restore-custody-record-v1",
            **values,
        }
        return cls(
            **values,
            manifest_digest=_canonical_digest(stable),
        )


class SignedRetirementRestoreCustodyStore:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("custody database parent must be a directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("custody database is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
            or self._file_identity() != self._database_identity
        ):
            raise RuntimeError("custody database identity changed.")

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
                CREATE TABLE IF NOT EXISTS evidence_graph_set_signed_restore_custody (
                    custody_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    restore_id TEXT NOT NULL UNIQUE,
                    snapshot_digest TEXT NOT NULL,
                    target_path_digest TEXT NOT NULL,
                    pre_receipt_digest TEXT NOT NULL,
                    backup_sha256 TEXT NOT NULL,
                    backup_size_bytes INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    pre_bound_actor_id TEXT NOT NULL,
                    pre_bound_method TEXT NOT NULL,
                    pre_bound_binding_digest TEXT NOT NULL,
                    pre_bound_at REAL NOT NULL,
                    post_receipt_digest TEXT,
                    target_verification_digest TEXT,
                    post_bound_actor_id TEXT,
                    post_bound_method TEXT,
                    post_bound_binding_digest TEXT,
                    post_bound_at REAL,
                    manifest_digest TEXT NOT NULL,
                    schema_version INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS signed_restore_custody_scope
                    ON evidence_graph_set_signed_restore_custody(
                        owner_id, state, pre_bound_at, custody_id
                    );
                """
            )

    @staticmethod
    def _value(row: sqlite3.Row) -> SignedRetirementRestoreCustodyManifest:
        try:
            return SignedRetirementRestoreCustodyManifest(
                custody_id=row["custody_id"],
                owner_id=row["owner_id"],
                restore_id=row["restore_id"],
                snapshot_digest=row["snapshot_digest"],
                target_path_digest=row["target_path_digest"],
                pre_receipt_digest=row["pre_receipt_digest"],
                backup_sha256=row["backup_sha256"],
                backup_size_bytes=int(row["backup_size_bytes"]),
                state=row["state"],
                pre_bound_actor_id=row["pre_bound_actor_id"],
                pre_bound_method=row["pre_bound_method"],
                pre_bound_binding_digest=row["pre_bound_binding_digest"],
                pre_bound_at=row["pre_bound_at"],
                post_receipt_digest=row["post_receipt_digest"],
                target_verification_digest=row["target_verification_digest"],
                post_bound_actor_id=row["post_bound_actor_id"],
                post_bound_method=row["post_bound_method"],
                post_bound_binding_digest=row["post_bound_binding_digest"],
                post_bound_at=row["post_bound_at"],
                manifest_digest=row["manifest_digest"],
                schema_version=int(row["schema_version"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("stored custody manifest is corrupt.") from exc

    def bind_pre(
        self,
        *,
        restore_id: str,
        pre_receipt_path: str | os.PathLike[str],
        backup_path: str | os.PathLike[str],
        restore_journal: Any,
        actor: ReviewActorBinding,
        now: float | None = None,
    ) -> SignedRetirementRestoreCustodyManifest:
        selected = _digest(restore_id, "restore_id")
        if not callable(getattr(restore_journal, "get", None)):
            raise ValueError("restore_journal lacks the required read boundary.")
        restore = restore_journal.get(selected)
        if restore.state not in {"planned", "failed"} or restore.phase != "planned":
            raise RuntimeError("custody must be bound before target work begins.")
        receipt = verify_pre_restore_backup_receipt(
            receipt_path=pre_receipt_path,
            backup_path=backup_path,
        )
        if (
            restore.owner_id != receipt.owner_id
            or restore.snapshot_digest != receipt.snapshot_digest
            or restore.target_path_digest != receipt.target_path_digest
        ):
            raise RuntimeError("pre custody evidence differs from restore scope.")
        timestamp = _timestamp(time.time() if now is None else now, "now")
        value = SignedRetirementRestoreCustodyManifest.create_pre_bound(
            owner_id=restore.owner_id,
            restore_id=selected,
            snapshot_digest=restore.snapshot_digest,
            target_path_digest=restore.target_path_digest,
            pre_receipt_digest=receipt.receipt_digest,
            backup_sha256=receipt.backup_sha256,
            backup_size_bytes=receipt.backup_size_bytes,
            actor=actor,
            now=timestamp,
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_set_signed_restore_custody "
                    "WHERE restore_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO evidence_graph_set_signed_restore_custody "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
                        (
                            value.custody_id,
                            value.owner_id,
                            value.restore_id,
                            value.snapshot_digest,
                            value.target_path_digest,
                            value.pre_receipt_digest,
                            value.backup_sha256,
                            value.backup_size_bytes,
                            value.state,
                            value.pre_bound_actor_id,
                            value.pre_bound_method,
                            value.pre_bound_binding_digest,
                            value.pre_bound_at,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            value.manifest_digest,
                        ),
                    )
                    connection.execute("COMMIT")
                    return value
                stored = self._value(row)
                if stored != value:
                    raise RuntimeError("restore custody manifest collision detected.")
                connection.execute("COMMIT")
                return stored
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def get(self, custody_id: str) -> SignedRetirementRestoreCustodyManifest:
        selected = _digest(custody_id, "custody_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM evidence_graph_set_signed_restore_custody "
                "WHERE custody_id=?",
                (selected,),
            ).fetchone()
        if row is None:
            raise KeyError(selected)
        return self._value(row)

    def get_for_restore(
        self,
        restore_id: str,
    ) -> SignedRetirementRestoreCustodyManifest:
        selected = _digest(restore_id, "restore_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM evidence_graph_set_signed_restore_custody "
                "WHERE restore_id=?",
                (selected,),
            ).fetchone()
        if row is None:
            raise KeyError(selected)
        return self._value(row)

    def require_pre_bound(
        self,
        *,
        restore_id: str,
        pre_receipt_path: str | os.PathLike[str],
        backup_path: str | os.PathLike[str],
        restore_journal: Any,
    ) -> SignedRetirementRestoreCustodyManifest:
        selected = _digest(restore_id, "restore_id")
        manifest = self.get_for_restore(selected)
        receipt = verify_pre_restore_backup_receipt(
            receipt_path=pre_receipt_path,
            backup_path=backup_path,
        )
        restore = restore_journal.get(selected)
        if (
            manifest.owner_id != restore.owner_id
            or manifest.snapshot_digest != restore.snapshot_digest
            or manifest.target_path_digest != restore.target_path_digest
            or manifest.pre_receipt_digest != receipt.receipt_digest
            or manifest.backup_sha256 != receipt.backup_sha256
            or manifest.backup_size_bytes != receipt.backup_size_bytes
        ):
            raise RuntimeError("live custody evidence differs from restore manifest.")
        return manifest

    def bind_post(
        self,
        *,
        restore_id: str,
        post_receipt_path: str | os.PathLike[str],
        restore_journal: Any,
        actor: ReviewActorBinding,
        now: float | None = None,
    ) -> SignedRetirementRestoreCustodyManifest:
        selected = _digest(restore_id, "restore_id")
        receipt = verify_post_restore_comparison_receipt(post_receipt_path)
        restore = restore_journal.get(selected)
        current = self.get_for_restore(selected)
        if (
            restore.state != "completed"
            or restore.phase != "verified"
            or receipt.restore_id != selected
            or receipt.owner_id != current.owner_id
            or receipt.snapshot_digest != current.snapshot_digest
            or receipt.target_path_digest != current.target_path_digest
            or receipt.pre_restore_receipt_digest != current.pre_receipt_digest
            or receipt.backup_sha256 != current.backup_sha256
            or receipt.target_verification_digest
            != restore.target_verification_digest
        ):
            raise RuntimeError("post custody evidence differs from completed restore.")
        timestamp = _timestamp(time.time() if now is None else now, "now")
        actor_id, method, binding = _actor_fields(actor, now=timestamp)
        values = {
            "custody_id": current.custody_id,
            "owner_id": current.owner_id,
            "restore_id": current.restore_id,
            "snapshot_digest": current.snapshot_digest,
            "target_path_digest": current.target_path_digest,
            "pre_receipt_digest": current.pre_receipt_digest,
            "backup_sha256": current.backup_sha256,
            "backup_size_bytes": current.backup_size_bytes,
            "state": "post_bound",
            "pre_bound_actor_id": current.pre_bound_actor_id,
            "pre_bound_method": current.pre_bound_method,
            "pre_bound_binding_digest": current.pre_bound_binding_digest,
            "pre_bound_at": current.pre_bound_at,
            "post_receipt_digest": receipt.receipt_digest,
            "target_verification_digest": receipt.target_verification_digest,
            "post_bound_actor_id": actor_id,
            "post_bound_method": method,
            "post_bound_binding_digest": binding,
            "post_bound_at": timestamp,
            "schema_version": _SCHEMA_VERSION,
        }
        stable = {
            "scope": "rigorousrag-signed-retirement-restore-custody-record-v1",
            **values,
        }
        updated = SignedRetirementRestoreCustodyManifest(
            **values,
            manifest_digest=_canonical_digest(stable),
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_set_signed_restore_custody "
                    "WHERE restore_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise KeyError(selected)
                stored = self._value(row)
                if stored.state == "post_bound":
                    if stored != updated:
                        raise RuntimeError(
                            "post custody manifest collision detected."
                        )
                    connection.execute("COMMIT")
                    return stored
                connection.execute(
                    "UPDATE evidence_graph_set_signed_restore_custody SET "
                    "state='post_bound', post_receipt_digest=?, "
                    "target_verification_digest=?, post_bound_actor_id=?, "
                    "post_bound_method=?, post_bound_binding_digest=?, "
                    "post_bound_at=?, manifest_digest=? WHERE restore_id=?",
                    (
                        updated.post_receipt_digest,
                        updated.target_verification_digest,
                        updated.post_bound_actor_id,
                        updated.post_bound_method,
                        updated.post_bound_binding_digest,
                        updated.post_bound_at,
                        updated.manifest_digest,
                        selected,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get_for_restore(selected)

    def list(
        self,
        *,
        owner_id: str,
        state: str | None = None,
        limit: int = 100,
    ) -> tuple[SignedRetirementRestoreCustodyManifest, ...]:
        owner = normalize_owner_id(owner_id)
        selected_state = None if state is None else _identifier(state, "state", 20)
        if selected_state is not None and selected_state not in {
            "pre_bound",
            "post_bound",
        }:
            raise ValueError("custody state is unsupported.")
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        query = (
            "SELECT * FROM evidence_graph_set_signed_restore_custody "
            "WHERE owner_id=?"
        )
        params: list[Any] = [owner]
        if selected_state is not None:
            query += " AND state=?"
            params.append(selected_state)
        query += " ORDER BY pre_bound_at DESC, custody_id DESC LIMIT ?"
        params.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(self._value(row) for row in rows)


__all__ = [
    "SignedRetirementRestoreCustodyManifest",
    "SignedRetirementRestoreCustodyStore",
    "deterministic_restore_custody_id",
]
