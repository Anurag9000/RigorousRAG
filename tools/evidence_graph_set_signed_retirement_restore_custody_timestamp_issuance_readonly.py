"""Query-only custody timestamp issuance journal."""

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
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_journal import (
    _MAX_LIMIT,
    _STATES,
    _TABLE,
    CustodyTimestampIssuanceJournal,
    _path,
    _redirecting,
)
from tools.security import normalize_owner_id


class ReadOnlyCustodyTimestampIssuanceJournal:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        parent = self.path.parent.lstat()
        info = self.path.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("timestamp issuance journal parent is invalid.")
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise ValueError("timestamp issuance journal must be initialized.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._database_identity = (int(info.st_dev), int(info.st_ino))
        with self._connect() as connection:
            row = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (_TABLE,),
            ).fetchone()
        if row is None:
            raise ValueError("timestamp issuance journal is not initialized.")

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
            raise RuntimeError("timestamp issuance journal identity changed.")

    def _connect(self) -> sqlite3.Connection:
        self._verify()
        uri = f"file:{quote(str(self.path), safe='/')}?mode=ro"
        connection = sqlite3.connect(
            uri,
            uri=True,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @staticmethod
    def _attempt(row: sqlite3.Row):
        return CustodyTimestampIssuanceJournal._attempt(row)

    def get(self, issuance_id: str):
        selected = _digest(issuance_id, "issuance_id")
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM {_TABLE} WHERE issuance_id=?",
                (selected,),
            ).fetchone()
        if row is None:
            raise KeyError(selected)
        return self._attempt(row)

    def list(
        self,
        *,
        owner_id: str,
        state: str | None = None,
        limit: int = 100,
    ) -> tuple[Any, ...]:
        owner = normalize_owner_id(owner_id)
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        selected_state = None if state is None else _identifier(state, "state", 30)
        if selected_state is not None and selected_state not in _STATES:
            raise ValueError("timestamp issuance state is unsupported.")
        query = f"SELECT * FROM {_TABLE} WHERE owner_id=?"
        parameters: list[Any] = [owner]
        if selected_state is not None:
            query += " AND state=?"
            parameters.append(selected_state)
        query += " ORDER BY created_at DESC, issuance_id DESC LIMIT ?"
        parameters.append(count)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return tuple(self._attempt(row) for row in rows)


__all__ = ["ReadOnlyCustodyTimestampIssuanceJournal"]
