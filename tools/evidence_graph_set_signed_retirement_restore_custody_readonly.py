"""SQLite read-only view over signed-retirement restore custody manifests."""

from __future__ import annotations

import os
import sqlite3
import stat
from urllib.parse import quote

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _identifier,
    _integer,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_manifest import (
    SignedRetirementRestoreCustodyManifest,
    SignedRetirementRestoreCustodyStore,
    _path,
)
from tools.security import normalize_owner_id

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_LIMIT = 10_000
_TABLE = "evidence_graph_set_signed_restore_custody"


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


class ReadOnlySignedRetirementRestoreCustodyStore:
    """Identity-pinned query-only custody-manifest store."""

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
            raise ValueError("custody database must be a regular file.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._database_identity = (int(info.st_dev), int(info.st_ino))
        with self._connect() as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        if _TABLE not in tables:
            raise RuntimeError("custody database schema is not initialized.")

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
            raise RuntimeError("custody database identity changed.")

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
        query = f"SELECT * FROM {_TABLE} WHERE owner_id=?"
        parameters: list[object] = [owner]
        if selected_state is not None:
            query += " AND state=?"
            parameters.append(selected_state)
        query += " ORDER BY pre_bound_at DESC, custody_id DESC LIMIT ?"
        parameters.append(count)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return tuple(SignedRetirementRestoreCustodyStore._value(row) for row in rows)


__all__ = ["ReadOnlySignedRetirementRestoreCustodyStore"]
