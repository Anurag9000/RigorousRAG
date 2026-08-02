"""Durable authorization receipts for governed semantic relation decisions."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import stat
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tools.evidence_graph_relation_policy import ReviewAuthorization
from tools.security import normalize_owner_id

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_MAX_LIMIT = 10_000
_STATES = frozenset({"authorized", "committed"})


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    selected = value.strip()
    if (
        not selected
        or len(selected) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in selected)
    ):
        raise ValueError(f"{label} is invalid.")
    return selected


def _digest(value: Any, label: str) -> str:
    selected = _identifier(value, label, 64).lower()
    if len(selected) != 64 or any(character not in "0123456789abcdef" for character in selected):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return selected


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(selected) or selected < 0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return selected


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("authorization database path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("authorization database path is invalid.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for part in (absolute, *absolute.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if _redirecting(info):
            raise ValueError("authorization database path may not contain redirects.")
    return absolute


@dataclass(frozen=True)
class ReviewAuthorizationRecord:
    authorization: ReviewAuthorization
    state: str
    prepared_at: float
    committed_at: float | None
    updated_at: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.authorization, ReviewAuthorization):
            raise ValueError("authorization must be ReviewAuthorization.")
        selected = _identifier(self.state, "state", 20)
        if selected not in _STATES:
            raise ValueError("authorization state is unsupported.")
        object.__setattr__(self, "state", selected)
        object.__setattr__(
            self,
            "prepared_at",
            _timestamp(self.prepared_at, "prepared_at"),
        )
        if self.committed_at is not None:
            object.__setattr__(
                self,
                "committed_at",
                _timestamp(self.committed_at, "committed_at"),
            )
        object.__setattr__(self, "updated_at", _timestamp(self.updated_at, "updated_at"))
        if self.updated_at < self.prepared_at:
            raise ValueError("updated_at may not precede prepared_at.")
        if self.state == "authorized" and self.committed_at is not None:
            raise ValueError("authorized receipt may not contain committed_at.")
        if self.state == "committed" and self.committed_at is None:
            raise ValueError("committed receipt requires committed_at.")
        if self.committed_at is not None and (
            self.committed_at < self.prepared_at
            or self.updated_at < self.committed_at
        ):
            raise ValueError("committed receipt timestamps are not monotonic.")
        if self.schema_version != 1:
            raise ValueError("authorization record schema is unsupported.")


class RelationReviewAuthorizationStore:
    """Append-only authorization identity with monotonic authorized→committed state."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("authorization database parent must be a regular directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("authorization database is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
        ):
            raise RuntimeError("authorization database parent identity changed.")
        if self._file_identity() != self._database_identity:
            raise RuntimeError("authorization database identity changed.")

    def _connect(self) -> sqlite3.Connection:
        self._verify()
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with sqlite3.connect(self.path, timeout=30.0, isolation_level=None) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS relation_review_authorizations (
                    decision_id TEXT PRIMARY KEY,
                    proposal_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    graph_set_key TEXT NOT NULL,
                    reviewer_id TEXT NOT NULL,
                    authorization_digest TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    prepared_at REAL NOT NULL,
                    committed_at REAL,
                    updated_at REAL NOT NULL,
                    schema_version INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS relation_review_authorization_scope
                    ON relation_review_authorizations(
                        owner_id, graph_set_key, state, updated_at, decision_id
                    );
                """
            )

    @staticmethod
    def _authorization(row: sqlite3.Row) -> ReviewAuthorization:
        try:
            value = ReviewAuthorization(**json.loads(row["payload_json"]))
        except Exception as exc:
            raise RuntimeError("stored review authorization is corrupt.") from exc
        if (
            value.decision_id != row["decision_id"]
            or value.proposal_id != row["proposal_id"]
            or value.owner_id != row["owner_id"]
            or value.graph_set_key != row["graph_set_key"]
            or value.reviewer_id != row["reviewer_id"]
            or value.authorization_digest != row["authorization_digest"]
        ):
            raise RuntimeError("stored review authorization identity is corrupt.")
        return value

    @classmethod
    def _record(cls, row: sqlite3.Row) -> ReviewAuthorizationRecord:
        if int(row["schema_version"]) != 1:
            raise RuntimeError("stored review authorization schema is unsupported.")
        return ReviewAuthorizationRecord(
            authorization=cls._authorization(row),
            state=row["state"],
            prepared_at=row["prepared_at"],
            committed_at=row["committed_at"],
            updated_at=row["updated_at"],
        )

    def prepare(
        self,
        authorization: ReviewAuthorization,
        *,
        now: float | None = None,
    ) -> ReviewAuthorizationRecord:
        if not isinstance(authorization, ReviewAuthorization):
            raise ValueError("authorization must be ReviewAuthorization.")
        timestamp = _timestamp(time.time() if now is None else now, "now")
        payload = json.dumps(
            asdict(authorization),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM relation_review_authorizations WHERE decision_id=?",
                    (authorization.decision_id,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO relation_review_authorizations VALUES (
                            ?, ?, ?, ?, ?, ?, ?, 'authorized', ?, NULL, ?, 1
                        )
                        """,
                        (
                            authorization.decision_id,
                            authorization.proposal_id,
                            authorization.owner_id,
                            authorization.graph_set_key,
                            authorization.reviewer_id,
                            authorization.authorization_digest,
                            payload,
                            timestamp,
                            timestamp,
                        ),
                    )
                else:
                    existing = self._record(row)
                    if existing.authorization.authorization_digest != authorization.authorization_digest:
                        raise RuntimeError(
                            "review authorization identity changed for an existing decision."
                        )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        result = self.get(authorization.decision_id)
        if result is None:
            raise RuntimeError("prepared review authorization disappeared.")
        return result

    def mark_committed(
        self,
        decision_id: str,
        *,
        now: float | None = None,
    ) -> ReviewAuthorizationRecord:
        selected = _digest(decision_id, "decision_id")
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM relation_review_authorizations WHERE decision_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise KeyError(selected)
                current = self._record(row)
                timestamp = max(timestamp, current.updated_at, current.prepared_at)
                if current.state == "authorized":
                    connection.execute(
                        """
                        UPDATE relation_review_authorizations
                        SET state='committed', committed_at=?, updated_at=?
                        WHERE decision_id=? AND state='authorized'
                        """,
                        (timestamp, timestamp, selected),
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        result = self.get(selected)
        if result is None or result.state != "committed":
            raise RuntimeError("review authorization commit was not durable.")
        return result

    def get(self, decision_id: str) -> ReviewAuthorizationRecord | None:
        selected = _digest(decision_id, "decision_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM relation_review_authorizations WHERE decision_id=?",
                (selected,),
            ).fetchone()
        return None if row is None else self._record(row)

    def list(
        self,
        *,
        owner_id: str,
        graph_set_key: str | None = None,
        state: str | None = None,
        limit: int = 100,
    ) -> tuple[ReviewAuthorizationRecord, ...]:
        owner = normalize_owner_id(owner_id)
        key = None if graph_set_key is None else _identifier(
            graph_set_key, "graph_set_key", 500
        )
        selected_state = None if state is None else _identifier(state, "state", 20)
        if selected_state is not None and selected_state not in _STATES:
            raise ValueError("authorization state filter is unsupported.")
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM relation_review_authorizations
                WHERE owner_id=?
                  AND (? IS NULL OR graph_set_key=?)
                  AND (? IS NULL OR state=?)
                ORDER BY updated_at, decision_id LIMIT ?
                """,
                (owner, key, key, selected_state, selected_state, count),
            ).fetchall()
        return tuple(self._record(row) for row in rows)


__all__ = [
    "RelationReviewAuthorizationStore",
    "ReviewAuthorizationRecord",
]
