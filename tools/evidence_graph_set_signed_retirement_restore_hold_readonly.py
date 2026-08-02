"""SQLite read-only view over integrity-backed restore legal holds."""

from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path
from urllib.parse import quote

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _integer,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_boundary import (
    GovernedSignedRetirementRestoreHoldStore,
)
from tools.evidence_graph_set_signed_retirement_restore_holds import _path
from tools.security import normalize_owner_id

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_LIMIT = 10_000


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


class ReadOnlySignedRetirementRestoreHoldStore:
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
            raise ValueError("restore hold database must be a regular file.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._database_identity = (int(info.st_dev), int(info.st_ino))
        with self._connect() as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        required = {
            "evidence_graph_set_signed_restore_holds",
            "evidence_graph_set_signed_restore_hold_integrity",
        }
        if not required.issubset(tables):
            raise RuntimeError("restore hold database schema is not initialized.")

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
            raise RuntimeError("restore hold database identity changed.")

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

    def active_restore_ids(
        self,
        *,
        owner_id: str,
        limit: int = _MAX_LIMIT,
    ) -> frozenset[str]:
        owner = normalize_owner_id(owner_id)
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT h.* "
                "FROM evidence_graph_set_signed_restore_holds h "
                "WHERE h.owner_id=? AND h.status='active' "
                "ORDER BY h.created_at DESC, h.hold_id DESC LIMIT ?",
                (owner, count),
            ).fetchall()
            if len(rows) >= count:
                raise RuntimeError(
                    "active restore hold list reached the bounded limit."
                )
            values: list[str] = []
            for row in rows:
                value = GovernedSignedRetirementRestoreHoldStore._verified_value(
                    connection,
                    row,
                )
                values.append(value.restore_id)
        return frozenset(values)


__all__ = ["ReadOnlySignedRetirementRestoreHoldStore"]
