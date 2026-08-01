"""Durable authoritative generation pointers with append-only history."""

from __future__ import annotations

import json
import math
import operator
import os
import sqlite3
import stat
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.security import normalize_owner_id

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_MAX_METADATA = 256
_MAX_JSON = 100_000
_MAX_LIMIT = 10_000


def _redirecting(value: os.stat_result) -> bool:
    return stat.S_ISLNK(value.st_mode) or bool(
        int(getattr(value, "st_file_attributes", 0)) & _REPARSE
    )


def _identifier(value: Any, label: str, maximum: int = 200) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip()
    if (
        not cleaned
        or len(cleaned) > maximum
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in cleaned)
    ):
        raise ValueError(f"{label} is invalid.")
    return cleaned


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip().lower()
    if len(cleaned) != 64 or any(ch not in "0123456789abcdef" for ch in cleaned):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return cleaned


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        result = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return result


def _timestamp(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("timestamp must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("timestamp must be finite.") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError("timestamp must be finite.")
    return result


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a mapping.")
    result: dict[str, Any] = {}
    try:
        for index, (raw_key, raw_value) in enumerate(value.items()):
            if index >= _MAX_METADATA:
                raise ValueError("metadata contains too many fields.")
            key = _identifier(raw_key, "metadata key")
            if raw_value is None or isinstance(raw_value, (bool, int)):
                result[key] = raw_value
            elif isinstance(raw_value, float) and math.isfinite(raw_value):
                result[key] = raw_value
            elif (
                isinstance(raw_value, str)
                and len(raw_value) <= 4000
                and "\x00" not in raw_value
            ):
                result[key] = raw_value
            else:
                raise ValueError("metadata contains an unsupported value.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("metadata is not safely iterable.") from exc
    encoded = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    if len(encoded) > _MAX_JSON:
        raise ValueError("metadata exceeds the serialized limit.")
    return result


def _decode_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or len(value) > _MAX_JSON:
        raise RuntimeError("Stored generation metadata is corrupt.")
    try:
        parsed = json.loads(
            value,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise RuntimeError("Stored generation metadata is corrupt.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Stored generation metadata is corrupt.")
    return _metadata(parsed)


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("generation database path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in rendered)
    ):
        raise ValueError("generation database path is invalid.")
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
                "generation database path could not be validated."
            ) from exc
        if _redirecting(info):
            raise ValueError("generation database path may not contain redirects.")
    return absolute


@dataclass(frozen=True)
class GenerationRecord:
    owner_id: str
    doc_id: str
    sequence: int
    state: str
    content_sha256: str
    profile_fingerprint: str
    vector_rows: int
    sparse_generation: int
    committed_at: float
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "doc_id", _identifier(self.doc_id, "doc_id"))
        object.__setattr__(
            self,
            "sequence",
            _integer(self.sequence, "sequence", 1, 2**63 - 1),
        )
        if self.state not in {"active", "deleted", "restored"}:
            raise ValueError("state is invalid.")
        object.__setattr__(
            self,
            "content_sha256",
            _digest(self.content_sha256, "content_sha256"),
        )
        object.__setattr__(
            self,
            "profile_fingerprint",
            _digest(self.profile_fingerprint, "profile_fingerprint"),
        )
        object.__setattr__(
            self,
            "vector_rows",
            _integer(self.vector_rows, "vector_rows", 0, 100_000_000),
        )
        object.__setattr__(
            self,
            "sparse_generation",
            _integer(
                self.sparse_generation,
                "sparse_generation",
                0,
                2**63 - 1,
            ),
        )
        object.__setattr__(self, "committed_at", _timestamp(self.committed_at))
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        if self.state in {"active", "restored"} and (
            self.vector_rows <= 0 or self.sparse_generation <= 0
        ):
            raise ValueError("active generations require non-empty stores.")
        if self.state == "deleted" and (
            self.vector_rows or self.sparse_generation
        ):
            raise ValueError("deleted generations must have zero store counts.")


class GenerationStore:
    """SQLite current pointers plus immutable generation history."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError(
                "generation database parent must be a regular directory."
            )
        self._parent = (parent.st_dev, parent.st_ino)
        self._lock = threading.RLock()
        self._initialize()
        self._database = self._database_identity()

    def _database_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("generation database is not a regular file.")
        return info.st_dev, info.st_ino

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or (parent.st_dev, parent.st_ino) != self._parent
        ):
            raise RuntimeError("generation database parent identity changed.")
        if self._database_identity() != self._database:
            raise RuntimeError("generation database identity changed.")

    def _connect(self) -> sqlite3.Connection:
        self._verify()
        connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
        ) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS generation_history (
                    owner_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    profile_fingerprint TEXT NOT NULL,
                    vector_rows INTEGER NOT NULL,
                    sparse_generation INTEGER NOT NULL,
                    committed_at REAL NOT NULL,
                    metadata_json TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    PRIMARY KEY (owner_id, doc_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS generation_current (
                    owner_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    PRIMARY KEY (owner_id, doc_id),
                    FOREIGN KEY (owner_id, doc_id, sequence)
                        REFERENCES generation_history(owner_id, doc_id, sequence)
                );
                """
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> GenerationRecord:
        if int(row["schema_version"]) != 1:
            raise RuntimeError("Stored generation schema is unsupported.")
        return GenerationRecord(
            row["owner_id"],
            row["doc_id"],
            row["sequence"],
            row["state"],
            row["content_sha256"],
            row["profile_fingerprint"],
            row["vector_rows"],
            row["sparse_generation"],
            row["committed_at"],
            _decode_metadata(row["metadata_json"]),
        )

    def current(
        self,
        *,
        owner_id: str,
        doc_id: str,
    ) -> GenerationRecord | None:
        owner = normalize_owner_id(owner_id)
        document = _identifier(doc_id, "doc_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT history.*
                FROM generation_current AS current
                JOIN generation_history AS history
                  ON history.owner_id = current.owner_id
                 AND history.doc_id = current.doc_id
                 AND history.sequence = current.sequence
                WHERE current.owner_id = ? AND current.doc_id = ?
                """,
                (owner, document),
            ).fetchone()
        return None if row is None else self._record(row)

    def _append(
        self,
        *,
        owner_id: str,
        doc_id: str,
        state: str,
        content_sha256: str,
        profile_fingerprint: str,
        vector_rows: int,
        sparse_generation: int,
        metadata: Mapping[str, Any] | None,
        expected_sequence: int | None,
        committed_at: float | None,
    ) -> GenerationRecord:
        owner = normalize_owner_id(owner_id)
        document = _identifier(doc_id, "doc_id")
        content = _digest(content_sha256, "content_sha256")
        profile = _digest(profile_fingerprint, "profile_fingerprint")
        rows = _integer(vector_rows, "vector_rows", 0, 100_000_000)
        sparse = _integer(
            sparse_generation,
            "sparse_generation",
            0,
            2**63 - 1,
        )
        clean_metadata = _metadata(metadata)
        timestamp = _timestamp(
            time.time() if committed_at is None else committed_at
        )
        expected = (
            None
            if expected_sequence is None
            else _integer(
                expected_sequence,
                "expected_sequence",
                0,
                2**63 - 1,
            )
        )
        GenerationRecord(
            owner,
            document,
            1,
            state,
            content,
            profile,
            rows,
            sparse,
            timestamp,
            clean_metadata,
        )
        encoded = json.dumps(
            clean_metadata,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT sequence FROM generation_current
                    WHERE owner_id = ? AND doc_id = ?
                    """,
                    (owner, document),
                ).fetchone()
                current = 0 if row is None else int(row["sequence"])
                if expected is not None and expected != current:
                    raise RuntimeError(
                        "generation sequence changed concurrently."
                    )
                sequence = current + 1
                connection.execute(
                    """
                    INSERT INTO generation_history VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        owner,
                        document,
                        sequence,
                        state,
                        content,
                        profile,
                        rows,
                        sparse,
                        timestamp,
                        encoded,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO generation_current VALUES (?, ?, ?)
                    ON CONFLICT(owner_id, doc_id)
                    DO UPDATE SET sequence = excluded.sequence
                    """,
                    (owner, document, sequence),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return GenerationRecord(
            owner,
            document,
            sequence,
            state,
            content,
            profile,
            rows,
            sparse,
            timestamp,
            clean_metadata,
        )

    def record_active(
        self,
        manifest: Any,
        *,
        expected_sequence: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        committed_at: float | None = None,
    ) -> GenerationRecord:
        return self._append(
            owner_id=getattr(manifest, "owner_id"),
            doc_id=getattr(manifest, "doc_id"),
            state="active",
            content_sha256=getattr(manifest, "content_sha256"),
            profile_fingerprint=getattr(manifest, "profile_fingerprint"),
            vector_rows=getattr(manifest, "vector_rows"),
            sparse_generation=getattr(manifest, "sparse_generation"),
            metadata=metadata,
            expected_sequence=expected_sequence,
            committed_at=committed_at,
        )

    def record_deleted(
        self,
        *,
        owner_id: str,
        doc_id: str,
        expected_sequence: int | None = None,
        prior: GenerationRecord | None = None,
        metadata: Mapping[str, Any] | None = None,
        committed_at: float | None = None,
    ) -> GenerationRecord:
        previous = prior or self.current(owner_id=owner_id, doc_id=doc_id)
        if previous is None:
            raise ValueError("Cannot record deletion without a prior generation.")
        return self._append(
            owner_id=owner_id,
            doc_id=doc_id,
            state="deleted",
            content_sha256=previous.content_sha256,
            profile_fingerprint=previous.profile_fingerprint,
            vector_rows=0,
            sparse_generation=0,
            metadata=metadata,
            expected_sequence=expected_sequence,
            committed_at=committed_at,
        )

    def restore_current(
        self,
        prior: GenerationRecord | None,
        *,
        owner_id: str,
        doc_id: str,
        expected_sequence: int | None = None,
        reason: str = "compensation",
        committed_at: float | None = None,
    ) -> GenerationRecord | None:
        owner = normalize_owner_id(owner_id)
        document = _identifier(doc_id, "doc_id")
        if prior is None:
            expected = (
                None
                if expected_sequence is None
                else _integer(
                    expected_sequence,
                    "expected_sequence",
                    0,
                    2**63 - 1,
                )
            )
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    row = connection.execute(
                        """
                        SELECT sequence FROM generation_current
                        WHERE owner_id = ? AND doc_id = ?
                        """,
                        (owner, document),
                    ).fetchone()
                    current = 0 if row is None else int(row["sequence"])
                    if expected is not None and expected != current:
                        raise RuntimeError(
                            "generation sequence changed concurrently."
                        )
                    connection.execute(
                        """
                        DELETE FROM generation_current
                        WHERE owner_id = ? AND doc_id = ?
                        """,
                        (owner, document),
                    )
                    connection.execute("COMMIT")
                except Exception:
                    connection.execute("ROLLBACK")
                    raise
            return None
        if prior.owner_id != owner or prior.doc_id != document:
            raise ValueError(
                "prior generation scope does not match restoration scope."
            )
        return self._append(
            owner_id=owner,
            doc_id=document,
            state="deleted" if prior.state == "deleted" else "restored",
            content_sha256=prior.content_sha256,
            profile_fingerprint=prior.profile_fingerprint,
            vector_rows=prior.vector_rows,
            sparse_generation=prior.sparse_generation,
            metadata={
                **dict(prior.metadata),
                "restored_from_sequence": prior.sequence,
                "reason": _identifier(reason, "reason"),
            },
            expected_sequence=expected_sequence,
            committed_at=committed_at,
        )

    def history(
        self,
        *,
        owner_id: str,
        doc_id: str | None = None,
        limit: int = 100,
    ) -> tuple[GenerationRecord, ...]:
        owner = normalize_owner_id(owner_id)
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        query = "SELECT * FROM generation_history WHERE owner_id = ?"
        params: list[Any] = [owner]
        if doc_id is not None:
            query += " AND doc_id = ?"
            params.append(_identifier(doc_id, "doc_id"))
        query += " ORDER BY committed_at DESC, sequence DESC LIMIT ?"
        params.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(self._record(row) for row in rows)

    def list_current(
        self,
        *,
        owner_id: str,
        limit: int = _MAX_LIMIT,
    ) -> tuple[GenerationRecord, ...]:
        owner = normalize_owner_id(owner_id)
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT history.*
                FROM generation_current AS current
                JOIN generation_history AS history
                  ON history.owner_id = current.owner_id
                 AND history.doc_id = current.doc_id
                 AND history.sequence = current.sequence
                WHERE current.owner_id = ?
                ORDER BY history.doc_id
                LIMIT ?
                """,
                (owner, count),
            ).fetchall()
        return tuple(self._record(row) for row in rows)


__all__ = ["GenerationRecord", "GenerationStore"]
