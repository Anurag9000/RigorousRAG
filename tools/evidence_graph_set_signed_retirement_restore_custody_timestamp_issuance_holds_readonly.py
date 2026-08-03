"""Query-only view of custody timestamp issuance legal holds."""

from __future__ import annotations

import os
import sqlite3
import stat
from typing import Any
from urllib.parse import quote

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _integer,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_holds import (
    _MAX_LIMIT,
    _TABLE,
    CustodyTimestampIssuanceHoldStore,
    _path,
    _redirecting,
)
from tools.security import normalize_owner_id


class ReadOnlyCustodyTimestampIssuanceHoldStore:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        parent = self.path.parent.lstat()
        info = self.path.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("timestamp issuance hold parent is invalid.")
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise ValueError("timestamp issuance hold database must be initialized.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._database_identity = (int(info.st_dev), int(info.st_ino))
        with self._connect() as connection:
            row = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (_TABLE,),
            ).fetchone()
        if row is None:
            raise ValueError("timestamp issuance hold database is not initialized.")

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        info = self.path.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
            or _redirecting(info)
            or not stat.S_ISREG(info.st_mode)
            or (int(info.st_dev), int(info.st_ino)) != self._database_identity
        ):
            raise RuntimeError("timestamp issuance hold database identity changed.")

    def _connect(self) -> sqlite3.Connection:
        self._verify()
        uri = f"file:{quote(str(self.path), safe='/')}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @staticmethod
    def _value(row: sqlite3.Row):
        return CustodyTimestampIssuanceHoldStore._value(row)

    def get(self, hold_id: str):
        selected = _digest(hold_id, "hold_id")
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM {_TABLE} WHERE hold_id=?",
                (selected,),
            ).fetchone()
        if row is None:
            raise KeyError(selected)
        return self._value(row)

    def list(
        self,
        *,
        owner_id: str,
        issuance_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> tuple[Any, ...]:
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
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return tuple(self._value(row) for row in rows)

    def active_issuance_ids(self, *, owner_id: str, limit: int = _MAX_LIMIT):
        values = self.list(owner_id=owner_id, status="active", limit=limit)
        if len(values) >= limit:
            raise RuntimeError("active timestamp issuance holds reached the bounded limit.")
        return frozenset(value.issuance_id for value in values)


__all__ = ["ReadOnlyCustodyTimestampIssuanceHoldStore"]
