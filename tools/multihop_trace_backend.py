"""Identity-bound SQLite backend for multi-hop diagnostics."""

from __future__ import annotations

import os
import sqlite3
import stat
import threading
from pathlib import Path

_SCHEMA_VERSION = 1
_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _redirected(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        int(getattr(metadata, "st_file_attributes", 0))
        & _WINDOWS_REPARSE_POINT
    )


def _check_path(path: Path) -> None:
    for item in (path, *path.parents):
        try:
            metadata = item.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError("Multi-hop trace path could not be validated.") from exc
        if _redirected(metadata):
            raise ValueError(
                "Multi-hop trace path may not contain symbolic links or reparse points."
            )


def _identity(path: Path) -> tuple[int, int]:
    metadata = path.lstat()
    if _redirected(metadata):
        raise RuntimeError("Multi-hop trace path was redirected.")
    return int(metadata.st_dev), int(metadata.st_ino)


class MultiHopTraceBackend:
    """Own the validated database path and schema lifecycle."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        if not isinstance(path, (str, os.PathLike)):
            raise ValueError("Multi-hop trace path must be a filesystem path.")
        rendered = os.fspath(path)
        if (
            not isinstance(rendered, str)
            or not rendered
            or len(rendered) > 4_096
            or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
        ):
            raise ValueError("Multi-hop trace path is invalid.")
        candidate = Path(rendered)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        self.path = Path(os.path.abspath(candidate))
        _check_path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _check_path(self.path)
        self.lock = threading.RLock()
        self._initialize()
        self.parent_identity = _identity(self.path.parent)
        self.database_identity = _identity(self.path)

    def _initialize(self) -> None:
        with sqlite3.connect(str(self.path), timeout=30.0) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS multihop_trace_schema (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    schema_version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS multihop_runs (
                    run_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    plan_fingerprint TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    completed_at REAL NOT NULL,
                    subquestion_count INTEGER NOT NULL,
                    batch_count INTEGER NOT NULL,
                    terminal_count INTEGER NOT NULL,
                    evidence_count INTEGER NOT NULL,
                    join_count INTEGER NOT NULL,
                    terminal_evidence_count INTEGER NOT NULL,
                    abstain INTEGER NOT NULL,
                    exhausted INTEGER NOT NULL,
                    used_model INTEGER NOT NULL,
                    planner_quality REAL NOT NULL,
                    budget_limit INTEGER NOT NULL,
                    allocated_budget INTEGER NOT NULL,
                    error_hops INTEGER NOT NULL,
                    timeout_hops INTEGER NOT NULL,
                    skipped_hops INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS multihop_hops (
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    hop_id TEXT NOT NULL,
                    dependency_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    returned_evidence INTEGER NOT NULL,
                    accepted_evidence INTEGER NOT NULL,
                    error_type TEXT,
                    PRIMARY KEY(run_id, sequence),
                    FOREIGN KEY(run_id) REFERENCES multihop_runs(run_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS multihop_runs_owner_completed
                    ON multihop_runs(owner_id, completed_at DESC, run_id DESC);
                """
            )
            row = connection.execute(
                "SELECT schema_version FROM multihop_trace_schema WHERE singleton=1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO multihop_trace_schema VALUES(1, ?)",
                    (_SCHEMA_VERSION,),
                )
            elif int(row[0]) != _SCHEMA_VERSION:
                raise RuntimeError("Multi-hop trace schema version is incompatible.")

    def verify_identity(self) -> None:
        _check_path(self.path)
        try:
            parent = _identity(self.path.parent)
            database = _identity(self.path)
        except FileNotFoundError as exc:
            raise RuntimeError("Multi-hop trace database disappeared.") from exc
        if parent != self.parent_identity or database != self.database_identity:
            raise RuntimeError("Multi-hop trace database or parent was replaced.")

    def connect(self) -> sqlite3.Connection:
        self.verify_identity()
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def ping(self) -> bool:
        try:
            with self.lock, self.connect() as connection:
                row = connection.execute(
                    "SELECT schema_version FROM multihop_trace_schema "
                    "WHERE singleton=1"
                ).fetchone()
                return row is not None and int(row[0]) == _SCHEMA_VERSION
        except Exception:
            return False


__all__ = ["MultiHopTraceBackend"]
