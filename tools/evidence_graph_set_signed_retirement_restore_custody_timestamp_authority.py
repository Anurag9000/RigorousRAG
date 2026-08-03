"""Durable governance registry for custody timestamp-authority public keys."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import threading
import time
from dataclasses import asdict, dataclass
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
    _load_private,
    _load_public,
    _public_fingerprint,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp import (
    CustodyTimestampAttestation,
    issue_custody_timestamp_attestation,
    verify_custody_timestamp_attestation,
)
from tools.security import normalize_owner_id

_SCHEMA_VERSION = 1
_STATES = frozenset({"active", "retired"})
_MAX_LIMIT = 10_000
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_TABLE = "evidence_graph_restore_custody_timestamp_authorities"


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
    rendered = os.fspath(value)
    if not isinstance(rendered, str) or not rendered or len(rendered) > 4096:
        raise ValueError("timestamp authority registry path is invalid.")
    if any(ord(character) < 32 or ord(character) == 127 for character in rendered):
        raise ValueError("timestamp authority registry path is invalid.")
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
            raise ValueError("timestamp authority registry path could not be validated.") from exc
        if _redirecting(info):
            raise ValueError("timestamp authority registry path may not contain redirects.")
    return absolute


@dataclass(frozen=True)
class CustodyTimestampAuthorityRecord:
    owner_id: str
    authority_id: str
    key_id: str
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
        authority = _identifier(self.authority_id, "authority_id", 200)
        key_id = _identifier(self.key_id, "key_id", 200)
        algorithm = _identifier(self.algorithm, "algorithm", 30)
        if algorithm != "ed25519":
            raise ValueError("timestamp authority algorithm is unsupported.")
        fingerprint = _digest(self.public_key_sha256, "public_key_sha256")
        state = _identifier(self.state, "state", 30)
        if state not in _STATES:
            raise ValueError("timestamp authority state is unsupported.")
        registered_actor = _identifier(self.registered_actor_id, "registered_actor_id", 200)
        registered_method = _identifier(
            self.registered_binding_method, "registered_binding_method", 50
        )
        registered_binding = _digest(
            self.registered_binding_digest, "registered_binding_digest"
        )
        registered_at = _timestamp(self.registered_at, "registered_at")
        if state == "active":
            if any(
                value is not None
                for value in (
                    self.retired_actor_id,
                    self.retired_binding_method,
                    self.retired_binding_digest,
                    self.retired_at,
                )
            ):
                raise ValueError("active timestamp authority may not contain retirement fields.")
            retired_actor = retired_method = retired_binding = retired_at = None
        else:
            if any(
                value is None
                for value in (
                    self.retired_actor_id,
                    self.retired_binding_method,
                    self.retired_binding_digest,
                    self.retired_at,
                )
            ):
                raise ValueError("retired timestamp authority requires complete retirement fields.")
            retired_actor = _identifier(self.retired_actor_id, "retired_actor_id", 200)
            retired_method = _identifier(
                self.retired_binding_method, "retired_binding_method", 50
            )
            retired_binding = _digest(
                self.retired_binding_digest, "retired_binding_digest"
            )
            retired_at = _timestamp(self.retired_at, "retired_at")
            if retired_at < registered_at:
                raise ValueError("timestamp authority retirement predates registration.")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("timestamp authority schema is unsupported.")
        stable = {
            "scope": "rigorousrag-custody-timestamp-authority-key-v1",
            "owner_id": owner,
            "authority_id": authority,
            "key_id": key_id,
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
            raise ValueError("record_digest differs from timestamp authority record.")
        for name, value in stable.items():
            if name != "scope":
                object.__setattr__(self, name, value)
        object.__setattr__(self, "record_digest", digest)

    @classmethod
    def active(
        cls,
        *,
        owner_id: str,
        authority_id: str,
        key_id: str,
        public_key_sha256: str,
        actor: ReviewActorBinding,
        now: float,
    ) -> "CustodyTimestampAuthorityRecord":
        if not isinstance(actor, ReviewActorBinding):
            raise ValueError("actor must be ReviewActorBinding.")
        values = {
            "owner_id": normalize_owner_id(owner_id),
            "authority_id": _identifier(authority_id, "authority_id", 200),
            "key_id": _identifier(key_id, "key_id", 200),
            "algorithm": "ed25519",
            "public_key_sha256": _digest(public_key_sha256, "public_key_sha256"),
            "state": "active",
            "registered_actor_id": actor.actor_id,
            "registered_binding_method": actor.binding_method,
            "registered_binding_digest": actor.binding_digest,
            "registered_at": _timestamp(now, "now"),
            "retired_actor_id": None,
            "retired_binding_method": None,
            "retired_binding_digest": None,
            "retired_at": None,
            "schema_version": _SCHEMA_VERSION,
        }
        return cls(
            **values,
            record_digest=_canonical_digest(
                {"scope": "rigorousrag-custody-timestamp-authority-key-v1", **values}
            ),
        )

    def retire(
        self,
        *,
        actor: ReviewActorBinding,
        now: float,
    ) -> "CustodyTimestampAuthorityRecord":
        if not isinstance(actor, ReviewActorBinding):
            raise ValueError("actor must be ReviewActorBinding.")
        if self.state == "retired":
            if (
                self.retired_actor_id != actor.actor_id
                or self.retired_binding_method != actor.binding_method
                or self.retired_binding_digest != actor.binding_digest
            ):
                raise RuntimeError("timestamp authority already retired by another actor binding.")
            return self
        values = asdict(self)
        values.pop("record_digest")
        values.update(
            state="retired",
            retired_actor_id=actor.actor_id,
            retired_binding_method=actor.binding_method,
            retired_binding_digest=actor.binding_digest,
            retired_at=max(_timestamp(now, "now"), self.registered_at),
        )
        return CustodyTimestampAuthorityRecord(
            **values,
            record_digest=_canonical_digest(
                {"scope": "rigorousrag-custody-timestamp-authority-key-v1", **values}
            ),
        )


class CustodyTimestampAuthorityRegistry:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("timestamp authority registry parent must be a directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("timestamp authority registry is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
            or self._file_identity() != self._database_identity
        ):
            raise RuntimeError("timestamp authority registry identity changed.")

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
                    authority_id TEXT NOT NULL,
                    key_id TEXT NOT NULL,
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
                    PRIMARY KEY(owner_id, authority_id, key_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS custody_timestamp_authority_fingerprint
                    ON {_TABLE}(owner_id, public_key_sha256);
                CREATE INDEX IF NOT EXISTS custody_timestamp_authority_state
                    ON {_TABLE}(owner_id, state, registered_at, authority_id, key_id);
                """
            )

    @staticmethod
    def _value(row: sqlite3.Row) -> CustodyTimestampAuthorityRecord:
        try:
            return CustodyTimestampAuthorityRecord(**dict(row))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("stored timestamp authority record is corrupt.") from exc

    def register(
        self,
        *,
        owner_id: str,
        authority_id: str,
        key_id: str,
        public_key_path: str | os.PathLike[str],
        actor: ReviewActorBinding,
        now: float | None = None,
    ) -> CustodyTimestampAuthorityRecord:
        public_key = _load_public(public_key_path)
        record = CustodyTimestampAuthorityRecord.active(
            owner_id=owner_id,
            authority_id=authority_id,
            key_id=key_id,
            public_key_sha256=_public_fingerprint(public_key),
            actor=actor,
            now=time.time() if now is None else now,
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    f"SELECT * FROM {_TABLE} WHERE owner_id=? AND authority_id=? AND key_id=?",
                    (record.owner_id, record.authority_id, record.key_id),
                ).fetchone()
                if row is not None:
                    stored = self._value(row)
                    same_scope = (
                        stored.owner_id == record.owner_id
                        and stored.authority_id == record.authority_id
                        and stored.key_id == record.key_id
                        and stored.algorithm == record.algorithm
                        and stored.public_key_sha256 == record.public_key_sha256
                        and stored.registered_actor_id == record.registered_actor_id
                        and stored.registered_binding_method == record.registered_binding_method
                        and stored.registered_binding_digest == record.registered_binding_digest
                    )
                    if same_scope:
                        connection.execute("COMMIT")
                        return stored
                    raise RuntimeError("timestamp authority registration collision detected.")
                connection.execute(
                    f"INSERT INTO {_TABLE} VALUES ({','.join('?' for _ in range(16))})",
                    tuple(asdict(record).values()),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(
            owner_id=record.owner_id,
            authority_id=record.authority_id,
            key_id=record.key_id,
        )

    def get(
        self,
        *,
        owner_id: str,
        authority_id: str,
        key_id: str,
    ) -> CustodyTimestampAuthorityRecord:
        owner = normalize_owner_id(owner_id)
        authority = _identifier(authority_id, "authority_id", 200)
        selected_key = _identifier(key_id, "key_id", 200)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM {_TABLE} WHERE owner_id=? AND authority_id=? AND key_id=?",
                (owner, authority, selected_key),
            ).fetchone()
        if row is None:
            raise KeyError((owner, authority, selected_key))
        return self._value(row)

    def list(
        self,
        *,
        owner_id: str,
        state: str | None = None,
        limit: int = 100,
    ) -> tuple[CustodyTimestampAuthorityRecord, ...]:
        owner = normalize_owner_id(owner_id)
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        selected_state = None if state is None else _identifier(state, "state", 30)
        if selected_state is not None and selected_state not in _STATES:
            raise ValueError("timestamp authority state is unsupported.")
        query = f"SELECT * FROM {_TABLE} WHERE owner_id=?"
        parameters: list[Any] = [owner]
        if selected_state is not None:
            query += " AND state=?"
            parameters.append(selected_state)
        query += " ORDER BY registered_at DESC, authority_id, key_id LIMIT ?"
        parameters.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return tuple(self._value(row) for row in rows)

    def retire(
        self,
        *,
        owner_id: str,
        authority_id: str,
        key_id: str,
        confirm_key_id: str,
        actor: ReviewActorBinding,
        now: float | None = None,
    ) -> CustodyTimestampAuthorityRecord:
        selected_key = _identifier(key_id, "key_id", 200)
        if selected_key != _identifier(confirm_key_id, "confirm_key_id", 200):
            raise ValueError("timestamp authority key confirmation differs.")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    f"SELECT * FROM {_TABLE} WHERE owner_id=? AND authority_id=? AND key_id=?",
                    (
                        normalize_owner_id(owner_id),
                        _identifier(authority_id, "authority_id", 200),
                        selected_key,
                    ),
                ).fetchone()
                if row is None:
                    raise KeyError(selected_key)
                current = self._value(row)
                retired = current.retire(
                    actor=actor,
                    now=time.time() if now is None else now,
                )
                if retired != current:
                    connection.execute(
                        f"""UPDATE {_TABLE} SET state=?, retired_actor_id=?,
                        retired_binding_method=?, retired_binding_digest=?, retired_at=?,
                        record_digest=? WHERE owner_id=? AND authority_id=? AND key_id=?""",
                        (
                            retired.state,
                            retired.retired_actor_id,
                            retired.retired_binding_method,
                            retired.retired_binding_digest,
                            retired.retired_at,
                            retired.record_digest,
                            retired.owner_id,
                            retired.authority_id,
                            retired.key_id,
                        ),
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(owner_id=owner_id, authority_id=authority_id, key_id=key_id)


def issue_governed_custody_timestamp(
    *,
    registry: CustodyTimestampAuthorityRegistry,
    owner_id: str,
    authority_id: str,
    key_id: str,
    authority_private_key_path: str | os.PathLike[str],
    signed_envelope_path: str | os.PathLike[str],
    custody_signer_public_key_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    now: float | None = None,
    nonce: bytes | None = None,
) -> CustodyTimestampAttestation:
    record = registry.get(owner_id=owner_id, authority_id=authority_id, key_id=key_id)
    if record.state != "active":
        raise PermissionError("timestamp authority key is not active.")
    fingerprint = _public_fingerprint(_load_private(authority_private_key_path).public_key())
    if fingerprint != record.public_key_sha256:
        raise PermissionError("timestamp authority private key differs from registry.")
    asserted = time.time() if now is None else now
    if asserted < record.registered_at:
        raise PermissionError("timestamp assertion predates authority registration.")
    return issue_custody_timestamp_attestation(
        signed_envelope_path=signed_envelope_path,
        custody_signer_public_key_path=custody_signer_public_key_path,
        output_path=output_path,
        owner_id=owner_id,
        authority_id=authority_id,
        key_id=key_id,
        authority_private_key_path=authority_private_key_path,
        now=asserted,
        nonce=nonce,
    )


def verify_governed_custody_timestamp(
    *,
    registry: CustodyTimestampAuthorityRegistry,
    owner_id: str,
    authority_id: str,
    key_id: str,
    attestation_path: str | os.PathLike[str],
    signed_envelope_path: str | os.PathLike[str],
    custody_signer_public_key_path: str | os.PathLike[str],
    authority_public_key_path: str | os.PathLike[str],
    now: float | None = None,
    maximum_future_seconds: float = 300.0,
) -> CustodyTimestampAttestation:
    record = registry.get(owner_id=owner_id, authority_id=authority_id, key_id=key_id)
    attestation = verify_custody_timestamp_attestation(
        attestation_path=attestation_path,
        signed_envelope_path=signed_envelope_path,
        custody_signer_public_key_path=custody_signer_public_key_path,
        authority_public_key_path=authority_public_key_path,
        expected_owner_id=owner_id,
        expected_authority_id=authority_id,
        expected_key_id=key_id,
        expected_public_key_sha256=record.public_key_sha256,
        now=now,
        maximum_future_seconds=maximum_future_seconds,
    )
    if attestation.asserted_at < record.registered_at:
        raise PermissionError("timestamp assertion predates authority registration.")
    if record.retired_at is not None and attestation.asserted_at > record.retired_at:
        raise PermissionError("timestamp assertion postdates authority retirement.")
    return attestation


__all__ = [
    "CustodyTimestampAuthorityRecord",
    "CustodyTimestampAuthorityRegistry",
    "issue_governed_custody_timestamp",
    "verify_governed_custody_timestamp",
]
