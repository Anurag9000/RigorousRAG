"""SQLite query-only view over signed signer-administration reservations."""

from __future__ import annotations

import os
import sqlite3
import stat
from urllib.parse import quote

from tools.evidence_graph_set_signed_retirement_restore_contracts import _digest
from tools.evidence_graph_set_signed_retirement_restore_custody_signer_admin_use import (
    CustodySignerAdminUseStore,
    _path,
)

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_TABLE = "evidence_graph_restore_custody_signer_admin_uses"


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


class ReadOnlyCustodySignerAdminUseStore:
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
            raise ValueError("signer admin-use database must be a regular file.")
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
            raise RuntimeError("signer admin-use schema is not initialized.")

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
            raise RuntimeError("signer admin-use database identity changed.")

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

    def get(self, use_id: str):
        selected = _digest(use_id, "use_id")
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM {_TABLE} WHERE use_id=?",
                (selected,),
            ).fetchone()
        if row is None:
            raise KeyError(selected)
        return CustodySignerAdminUseStore._value(row)


__all__ = ["ReadOnlyCustodySignerAdminUseStore"]
