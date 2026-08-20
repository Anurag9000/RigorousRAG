"""Disk-backed identity, uniqueness and digest authority for canonical training data.

The ledger is intentionally ephemeral: publishers use it while constructing/verifying immutable
artifacts, then delete it before final publication. It prevents corpus-sized Python identifier
sets while preserving exact global uniqueness, scoped set membership, payload consistency,
streaming sorted SHA-256 identities, and bounded diagnostic overlap samples.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Iterator

_MAX_TEXT = 32_768
_MAX_SAMPLE = 1_000


def _text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if (
        not selected
        or len(selected) > _MAX_TEXT
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected)
    ):
        raise ValueError(f"{label} is invalid")
    return selected


def _optional_sha(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    selected = _text(value, label).lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


class SqliteIdentityLedger:
    """Exact disk-backed identity ledger with two insertion semantics.

    ``add_unique`` enforces uniqueness across every scope in one namespace. ``add_set`` permits
    the same value in different scopes but deduplicates within a scope. Both forms optionally
    bind an immutable payload SHA. For ``add_set`` the payload binding is global across scopes:
    a logical identifier may be reused by multiple splits only when it denotes exactly the same
    content everywhere.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), timeout=30.0)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA temp_store=FILE")
        self._connection.execute(
            """CREATE TABLE unique_ids (
                namespace TEXT NOT NULL,
                value TEXT NOT NULL,
                scope TEXT NOT NULL,
                payload_sha256 TEXT,
                PRIMARY KEY(namespace,value)
            ) WITHOUT ROWID"""
        )
        self._connection.execute(
            """CREATE TABLE set_ids (
                namespace TEXT NOT NULL,
                scope TEXT NOT NULL,
                value TEXT NOT NULL,
                payload_sha256 TEXT,
                PRIMARY KEY(namespace,scope,value)
            ) WITHOUT ROWID"""
        )
        self._connection.execute(
            "CREATE INDEX set_ids_value_idx ON set_ids(namespace,value,scope)"
        )
        self._connection.commit()
        self._closed = False

    def __enter__(self) -> "SqliteIdentityLedger":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _open(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("identity ledger is closed")
        return self._connection

    def add_unique(
        self,
        namespace: str,
        scope: str,
        value: str,
        *,
        payload_sha256: str | None = None,
    ) -> None:
        namespace = _text(namespace, "namespace")
        scope = _text(scope, "scope")
        value = _text(value, "identity value")
        payload = _optional_sha(payload_sha256, "payload_sha256")
        connection = self._open()
        try:
            connection.execute(
                "INSERT INTO unique_ids(namespace,value,scope,payload_sha256) VALUES(?,?,?,?)",
                (namespace, value, scope, payload),
            )
        except sqlite3.IntegrityError as exc:
            existing = connection.execute(
                "SELECT scope,payload_sha256 FROM unique_ids WHERE namespace=? AND value=?",
                (namespace, value),
            ).fetchone()
            detail = "unknown"
            if existing is not None:
                detail = f"scope={existing[0]!r}, payload_sha256={existing[1]!r}"
            raise ValueError(
                f"duplicate globally unique identity {namespace}:{value!r}; existing {detail}"
            ) from exc

    def add_set(
        self,
        namespace: str,
        scope: str,
        value: str,
        *,
        payload_sha256: str | None = None,
    ) -> bool:
        namespace = _text(namespace, "namespace")
        scope = _text(scope, "scope")
        value = _text(value, "identity value")
        payload = _optional_sha(payload_sha256, "payload_sha256")
        connection = self._open()

        # The set relation is scoped, but the logical value->payload meaning is global inside a
        # namespace. This permits one evidence/document id in multiple splits while preventing
        # split-local content drift under the same id.
        payload_rows = connection.execute(
            "SELECT DISTINCT payload_sha256 FROM set_ids WHERE namespace=? AND value=?",
            (namespace, value),
        ).fetchall()
        if payload_rows:
            existing_payloads = {row[0] for row in payload_rows}
            if existing_payloads != {payload}:
                raise ValueError(
                    f"set identity {namespace}/{value!r} was reused across scopes with different content"
                )

        existing = connection.execute(
            "SELECT payload_sha256 FROM set_ids WHERE namespace=? AND scope=? AND value=?",
            (namespace, scope, value),
        ).fetchone()
        if existing is not None:
            if existing[0] != payload:
                raise ValueError(
                    f"scoped identity {namespace}/{scope}/{value!r} was reused with different content"
                )
            return False
        connection.execute(
            "INSERT INTO set_ids(namespace,scope,value,payload_sha256) VALUES(?,?,?,?)",
            (namespace, scope, value, payload),
        )
        return True

    def commit(self) -> None:
        self._open().commit()

    def count_unique(self, namespace: str, *, scope: str | None = None) -> int:
        namespace = _text(namespace, "namespace")
        connection = self._open()
        if scope is None:
            row = connection.execute(
                "SELECT COUNT(*) FROM unique_ids WHERE namespace=?",
                (namespace,),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT COUNT(*) FROM unique_ids WHERE namespace=? AND scope=?",
                (namespace, _text(scope, "scope")),
            ).fetchone()
        return int(row[0])

    def count_set(self, namespace: str, *, scope: str | None = None) -> int:
        namespace = _text(namespace, "namespace")
        connection = self._open()
        if scope is None:
            row = connection.execute(
                "SELECT COUNT(*) FROM set_ids WHERE namespace=?",
                (namespace,),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT COUNT(*) FROM set_ids WHERE namespace=? AND scope=?",
                (namespace, _text(scope, "scope")),
            ).fetchone()
        return int(row[0])

    @staticmethod
    def _digest_cursor(cursor: sqlite3.Cursor) -> str:
        digest = hashlib.sha256()
        while True:
            rows = cursor.fetchmany(8192)
            if not rows:
                break
            for row in rows:
                digest.update(str(row[0]).encode("utf-8"))
                digest.update(b"\n")
        return digest.hexdigest()

    def digest_unique(self, namespace: str, *, scope: str | None = None) -> str:
        namespace = _text(namespace, "namespace")
        connection = self._open()
        if scope is None:
            cursor = connection.execute(
                "SELECT value FROM unique_ids WHERE namespace=? ORDER BY value",
                (namespace,),
            )
        else:
            cursor = connection.execute(
                "SELECT value FROM unique_ids WHERE namespace=? AND scope=? ORDER BY value",
                (namespace, _text(scope, "scope")),
            )
        return self._digest_cursor(cursor)

    def digest_set(self, namespace: str, *, scope: str | None = None) -> str:
        namespace = _text(namespace, "namespace")
        connection = self._open()
        if scope is None:
            cursor = connection.execute(
                "SELECT value FROM set_ids WHERE namespace=? ORDER BY value,scope",
                (namespace,),
            )
        else:
            cursor = connection.execute(
                "SELECT value FROM set_ids WHERE namespace=? AND scope=? ORDER BY value",
                (namespace, _text(scope, "scope")),
            )
        return self._digest_cursor(cursor)

    def iter_unique(self, namespace: str, *, scope: str | None = None) -> Iterator[str]:
        namespace = _text(namespace, "namespace")
        connection = self._open()
        if scope is None:
            cursor = connection.execute(
                "SELECT value FROM unique_ids WHERE namespace=? ORDER BY value",
                (namespace,),
            )
        else:
            cursor = connection.execute(
                "SELECT value FROM unique_ids WHERE namespace=? AND scope=? ORDER BY value",
                (namespace, _text(scope, "scope")),
            )
        for row in cursor:
            yield str(row[0])

    def overlap_sample(
        self,
        namespace: str,
        left_scope: str,
        right_scope: str,
        *,
        limit: int = 20,
    ) -> tuple[str, ...]:
        namespace = _text(namespace, "namespace")
        left = _text(left_scope, "left_scope")
        right = _text(right_scope, "right_scope")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_SAMPLE:
            raise ValueError(f"limit must be an integer in [1,{_MAX_SAMPLE}]")
        rows = self._open().execute(
            """SELECT a.value
               FROM set_ids AS a
               JOIN set_ids AS b
                 ON a.namespace=b.namespace AND a.value=b.value
               WHERE a.namespace=? AND a.scope=? AND b.scope=?
               ORDER BY a.value
               LIMIT ?""",
            (namespace, left, right, limit),
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def close(self) -> None:
        if self._closed:
            return
        self._connection.commit()
        self._connection.close()
        self._closed = True


__all__ = ["SqliteIdentityLedger"]