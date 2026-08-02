"""SQLite read-only view of an initialized signed retirement journal."""

from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path
from typing import Any
from urllib.parse import quote

from tools.evidence_graph_set_signed_retirement_contracts import _integer
from tools.evidence_graph_set_signed_retirement_journal import (
    SignedPublicationRetirementJournal,
    _path,
    _redirecting,
)
from tools.security import normalize_owner_id


class ReadOnlySignedPublicationRetirementJournal:
    """Bounded owner-scoped listing without schema creation or journal mutation."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        parent = self.path.parent.lstat()
        info = self.path.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or _redirecting(info)
            or not stat.S_ISREG(info.st_mode)
        ):
            raise ValueError("read-only retirement database path is invalid.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._database_identity = (int(info.st_dev), int(info.st_ino))
        with self._connect() as connection:
            row = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='evidence_graph_set_signed_retirements'"
            ).fetchone()
            if row is None:
                raise ValueError("target retirement database is not initialized.")

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        info = self.path.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
        ):
            raise RuntimeError("read-only retirement parent identity changed.")
        if (
            _redirecting(info)
            or not stat.S_ISREG(info.st_mode)
            or (int(info.st_dev), int(info.st_ino)) != self._database_identity
        ):
            raise RuntimeError("read-only retirement database identity changed.")

    def _connect(self) -> sqlite3.Connection:
        if hasattr(self, "_database_identity"):
            self._verify()
        uri = "file:" + quote(str(self.path), safe="/") + "?mode=ro"
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

    def list(
        self,
        *,
        owner_id: str,
        publication_operation_id: str | None = None,
        state: str | None = None,
        limit: int = 100,
    ) -> tuple[Any, ...]:
        owner = normalize_owner_id(owner_id)
        operation = publication_operation_id
        selected_state = state
        count = _integer(limit, "limit", 1, 10_000)
        query = "SELECT * FROM evidence_graph_set_signed_retirements WHERE owner_id=?"
        params: list[Any] = [owner]
        if operation is not None:
            from tools.evidence_graph_set_signed_retirement_contracts import _digest

            operation = _digest(operation, "publication_operation_id")
            query += " AND publication_operation_id=?"
            params.append(operation)
        if selected_state is not None:
            from tools.evidence_graph_set_signed_retirement_contracts import (
                _STATES,
                _identifier,
            )

            selected_state = _identifier(selected_state, "state", 30)
            if selected_state not in _STATES:
                raise ValueError("state is unsupported.")
            query += " AND state=?"
            params.append(selected_state)
        query += " ORDER BY created_at DESC, retirement_id DESC LIMIT ?"
        params.append(count)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(
            SignedPublicationRetirementJournal._attempt(row) for row in rows
        )


__all__ = ["ReadOnlySignedPublicationRetirementJournal"]
