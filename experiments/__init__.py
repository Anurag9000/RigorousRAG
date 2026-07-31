"""Deterministic experiment manifests and immutable resumable result persistence."""

from __future__ import annotations



import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ExperimentRun:
    run_id: str
    parameters: Mapping[str, Any]


def build_manifest(matrix: Mapping[str, Sequence[Any]], *, prefix: str = "run") -> list[ExperimentRun]:
    if not isinstance(matrix, Mapping) or not matrix:
        raise ValueError("matrix must be a non-empty mapping.")
    if not isinstance(prefix, str) or not prefix.strip() or len(prefix) > 100:
        raise ValueError("prefix is invalid.")
    names = sorted(matrix)
    values: list[list[Any]] = []
    for name in names:
        if not isinstance(name, str) or not name or len(name) > 200:
            raise ValueError("parameter names are invalid.")
        sequence = matrix[name]
        if isinstance(sequence, (str, bytes, bytearray)) or not 1 <= len(sequence) <= 1000:
            raise ValueError("every parameter must have 1-1000 values.")
        values.append(list(sequence))
    total = 1
    for sequence in values:
        total *= len(sequence)
        if total > 100_000:
            raise ValueError("experiment matrix exceeds 100,000 runs.")
    runs: list[ExperimentRun] = []
    for combination in itertools.product(*values):
        parameters = dict(zip(names, combination))
        payload = json.dumps(parameters, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        runs.append(ExperimentRun(f"{prefix.strip()}-{digest}", parameters))
    return runs


import json
import math
import os
import sqlite3
import stat
import threading
from pathlib import Path
from typing import Any, Mapping

_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _safe_path(value: str | os.PathLike[str]) -> Path:
    path = Path(os.path.abspath(os.fspath(value)))
    for component in (path, *path.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or bool(int(getattr(metadata, "st_file_attributes", 0)) & _WINDOWS_REPARSE_POINT):
            raise ValueError("experiment store path may not contain links or reparse points.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


class ExperimentStore:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _safe_path(path)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS results (
                    run_id TEXT PRIMARY KEY,
                    parameters_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )

    def completed(self) -> set[str]:
        with self._lock, self._connect() as connection:
            return {str(row[0]) for row in connection.execute("SELECT run_id FROM results")}

    def put(self, run_id: str, *, parameters: Mapping[str, Any], metrics: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> bool:
        if not isinstance(run_id, str) or not run_id or len(run_id) > 300 or any(ord(ch) < 32 for ch in run_id):
            raise ValueError("run_id is invalid.")
        parameter_json, metric_json, metadata_json = _json(parameters), _json(metrics), _json(metadata or {})
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO results(run_id, parameters_json, metrics_json, metadata_json) VALUES (?, ?, ?, ?)",
                (run_id, parameter_json, metric_json, metadata_json),
            )
            return cursor.rowcount == 1

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT parameters_json, metrics_json, metadata_json, created_at FROM results WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "run_id": run_id,
            "parameters": json.loads(row[0]),
            "metrics": json.loads(row[1]),
            "metadata": json.loads(row[2]),
            "created_at": row[3],
        }


__all__ = ["ExperimentRun", "ExperimentStore", "build_manifest"]
