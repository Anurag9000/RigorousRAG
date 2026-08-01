"""Transactional owner/document/generation-scoped evidence-graph storage."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tools.evidence_graph_types import EvidenceEdge, EvidenceGraphBatch, EvidenceNode
from tools.security import normalize_owner_id

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_MAX_BATCH_BYTES = 512 * 1024 * 1024
_MAX_HISTORY = 10_000
_SCHEMA_VERSION = 1


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("evidence graph database path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("evidence graph database path is invalid.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for part in (absolute, *absolute.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError("evidence graph database path could not be validated.") from exc
        if _redirecting(info):
            raise ValueError("evidence graph database path may not contain redirects.")
    return absolute


def _identifier(value: Any, label: str, maximum: int = 200) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip()
    if (
        not cleaned
        or len(cleaned) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in cleaned)
    ):
        raise ValueError(f"{label} is invalid.")
    return cleaned


def _digest(value: Any, label: str) -> str:
    cleaned = _identifier(value, label, 64).lower()
    if len(cleaned) != 64 or any(character not in "0123456789abcdef" for character in cleaned):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return cleaned


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _encoded(batch: EvidenceGraphBatch) -> str:
    payload = json.dumps(
        asdict(batch),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if not payload or len(payload.encode("utf-8")) > _MAX_BATCH_BYTES:
        raise ValueError("evidence graph batch exceeds the serialized byte limit.")
    return payload


def _decode(value: Any) -> EvidenceGraphBatch:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > _MAX_BATCH_BYTES:
        raise RuntimeError("stored evidence graph batch is corrupt.")
    try:
        raw = json.loads(
            value,
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
        if not isinstance(raw, dict) or set(raw) != {
            "owner_id",
            "doc_id",
            "generation",
            "content_sha256",
            "profile_fingerprint",
            "nodes",
            "edges",
            "created_at",
            "schema_version",
        }:
            raise ValueError("batch schema")
        raw_nodes = raw["nodes"]
        raw_edges = raw["edges"]
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            raise ValueError("batch arrays")
        nodes = tuple(EvidenceNode(**node) for node in raw_nodes)
        edges = tuple(EvidenceEdge(**edge) for edge in raw_edges)
        return EvidenceGraphBatch(
            owner_id=raw["owner_id"],
            doc_id=raw["doc_id"],
            generation=raw["generation"],
            content_sha256=raw["content_sha256"],
            profile_fingerprint=raw["profile_fingerprint"],
            nodes=nodes,
            edges=edges,
            created_at=raw["created_at"],
            schema_version=raw["schema_version"],
        )
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise RuntimeError("stored evidence graph batch is corrupt.") from exc


class EvidenceGraphStore:
    """Append-only graph generations plus one optimistic current pointer."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("evidence graph database parent must be a regular directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("evidence graph database is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
        ):
            raise RuntimeError("evidence graph database parent identity changed.")
        if self._file_identity() != self._database_identity:
            raise RuntimeError("evidence graph database identity changed.")

    def _connect(self) -> sqlite3.Connection:
        self._verify()
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with sqlite3.connect(self.path, timeout=30.0, isolation_level=None) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS graph_generations (
                    owner_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    profile_fingerprint TEXT NOT NULL,
                    graph_digest TEXT NOT NULL,
                    batch_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    schema_version INTEGER NOT NULL,
                    PRIMARY KEY(owner_id, doc_id, generation)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS graph_generation_digest
                    ON graph_generations(owner_id, doc_id, graph_digest);
                CREATE TABLE IF NOT EXISTS graph_current (
                    owner_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    graph_digest TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    schema_version INTEGER NOT NULL,
                    PRIMARY KEY(owner_id, doc_id),
                    FOREIGN KEY(owner_id, doc_id, generation)
                        REFERENCES graph_generations(owner_id, doc_id, generation)
                        ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS graph_history
                    ON graph_generations(owner_id, doc_id, generation DESC);
                """
            )

    @staticmethod
    def _row_batch(row: sqlite3.Row) -> EvidenceGraphBatch:
        if int(row["schema_version"]) != _SCHEMA_VERSION:
            raise RuntimeError("evidence graph storage schema is unsupported.")
        batch = _decode(row["batch_json"])
        if (
            batch.owner_id != row["owner_id"]
            or batch.doc_id != row["doc_id"]
            or batch.generation != int(row["generation"])
            or batch.content_sha256 != row["content_sha256"]
            or batch.profile_fingerprint != row["profile_fingerprint"]
            or batch.graph_digest != row["graph_digest"]
        ):
            raise RuntimeError("stored evidence graph row identity is corrupt.")
        return batch

    def commit(
        self,
        batch: EvidenceGraphBatch,
        *,
        make_current: bool = True,
        expected_current_generation: int | None = None,
        now: float | None = None,
    ) -> EvidenceGraphBatch:
        if not isinstance(batch, EvidenceGraphBatch):
            raise ValueError("batch must be EvidenceGraphBatch.")
        if not isinstance(make_current, bool):
            raise ValueError("make_current must be boolean.")
        if expected_current_generation is not None:
            expected_current_generation = _integer(
                expected_current_generation,
                "expected_current_generation",
                0,
                2**63 - 1,
            )
        current_time = time.time() if now is None else float(now)
        if current_time < 0:
            raise ValueError("now must be non-negative.")
        payload = _encoded(batch)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT * FROM graph_generations WHERE owner_id=? AND doc_id=? AND generation=?",
                    (batch.owner_id, batch.doc_id, batch.generation),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO graph_generations(
                            owner_id, doc_id, generation, content_sha256,
                            profile_fingerprint, graph_digest, batch_json,
                            created_at, schema_version
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                        """,
                        (
                            batch.owner_id,
                            batch.doc_id,
                            batch.generation,
                            batch.content_sha256,
                            batch.profile_fingerprint,
                            batch.graph_digest,
                            payload,
                            batch.created_at,
                        ),
                    )
                else:
                    stored = self._row_batch(existing)
                    if stored.graph_digest != batch.graph_digest:
                        raise RuntimeError("evidence graph generation identity collision detected.")
                if make_current:
                    pointer = connection.execute(
                        "SELECT generation FROM graph_current WHERE owner_id=? AND doc_id=?",
                        (batch.owner_id, batch.doc_id),
                    ).fetchone()
                    actual = 0 if pointer is None else int(pointer["generation"])
                    if expected_current_generation is not None and actual != expected_current_generation:
                        raise RuntimeError("evidence graph current generation changed concurrently.")
                    connection.execute(
                        """
                        INSERT INTO graph_current(
                            owner_id, doc_id, generation, graph_digest,
                            updated_at, schema_version
                        ) VALUES (?, ?, ?, ?, ?, 1)
                        ON CONFLICT(owner_id, doc_id) DO UPDATE SET
                            generation=excluded.generation,
                            graph_digest=excluded.graph_digest,
                            updated_at=excluded.updated_at,
                            schema_version=excluded.schema_version
                        """,
                        (
                            batch.owner_id,
                            batch.doc_id,
                            batch.generation,
                            batch.graph_digest,
                            current_time,
                        ),
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(owner_id=batch.owner_id, doc_id=batch.doc_id, generation=batch.generation)

    def get(self, *, owner_id: str, doc_id: str, generation: int) -> EvidenceGraphBatch:
        owner = normalize_owner_id(owner_id)
        document = _identifier(doc_id, "doc_id")
        sequence = _integer(generation, "generation", 1, 2**63 - 1)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM graph_generations WHERE owner_id=? AND doc_id=? AND generation=?",
                (owner, document, sequence),
            ).fetchone()
        if row is None:
            raise KeyError((owner, document, sequence))
        return self._row_batch(row)

    def current(self, *, owner_id: str, doc_id: str) -> EvidenceGraphBatch | None:
        owner = normalize_owner_id(owner_id)
        document = _identifier(doc_id, "doc_id")
        with self._lock, self._connect() as connection:
            pointer = connection.execute(
                "SELECT * FROM graph_current WHERE owner_id=? AND doc_id=?",
                (owner, document),
            ).fetchone()
            if pointer is None:
                return None
            if int(pointer["schema_version"]) != _SCHEMA_VERSION:
                raise RuntimeError("evidence graph current pointer schema is unsupported.")
            row = connection.execute(
                "SELECT * FROM graph_generations WHERE owner_id=? AND doc_id=? AND generation=?",
                (owner, document, int(pointer["generation"])),
            ).fetchone()
        if row is None:
            raise RuntimeError("evidence graph current pointer target is missing.")
        batch = self._row_batch(row)
        if batch.graph_digest != pointer["graph_digest"]:
            raise RuntimeError("evidence graph current pointer digest is corrupt.")
        return batch

    def history(
        self,
        *,
        owner_id: str,
        doc_id: str,
        limit: int = 100,
    ) -> tuple[EvidenceGraphBatch, ...]:
        owner = normalize_owner_id(owner_id)
        document = _identifier(doc_id, "doc_id")
        count = _integer(limit, "limit", 1, _MAX_HISTORY)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM graph_generations
                WHERE owner_id=? AND doc_id=?
                ORDER BY generation DESC LIMIT ?
                """,
                (owner, document, count),
            ).fetchall()
        return tuple(self._row_batch(row) for row in rows)

    def activate(
        self,
        *,
        owner_id: str,
        doc_id: str,
        generation: int,
        graph_digest: str,
        expected_current_generation: int,
        now: float | None = None,
    ) -> EvidenceGraphBatch:
        owner = normalize_owner_id(owner_id)
        document = _identifier(doc_id, "doc_id")
        sequence = _integer(generation, "generation", 1, 2**63 - 1)
        expected = _integer(expected_current_generation, "expected_current_generation", 0, 2**63 - 1)
        selected_digest = _digest(graph_digest, "graph_digest")
        current_time = time.time() if now is None else float(now)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM graph_generations WHERE owner_id=? AND doc_id=? AND generation=?",
                    (owner, document, sequence),
                ).fetchone()
                if row is None or row["graph_digest"] != selected_digest:
                    raise RuntimeError("requested evidence graph generation is unavailable.")
                pointer = connection.execute(
                    "SELECT generation FROM graph_current WHERE owner_id=? AND doc_id=?",
                    (owner, document),
                ).fetchone()
                actual = 0 if pointer is None else int(pointer["generation"])
                if actual != expected:
                    raise RuntimeError("evidence graph current generation changed concurrently.")
                connection.execute(
                    """
                    INSERT INTO graph_current(
                        owner_id, doc_id, generation, graph_digest,
                        updated_at, schema_version
                    ) VALUES (?, ?, ?, ?, ?, 1)
                    ON CONFLICT(owner_id, doc_id) DO UPDATE SET
                        generation=excluded.generation,
                        graph_digest=excluded.graph_digest,
                        updated_at=excluded.updated_at,
                        schema_version=excluded.schema_version
                    """,
                    (owner, document, sequence, selected_digest, current_time),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        value = self.current(owner_id=owner, doc_id=document)
        if value is None:
            raise RuntimeError("evidence graph activation did not publish a current graph.")
        return value

    def delete_generation(
        self,
        *,
        owner_id: str,
        doc_id: str,
        generation: int,
        confirm_graph_digest: str,
    ) -> bool:
        owner = normalize_owner_id(owner_id)
        document = _identifier(doc_id, "doc_id")
        sequence = _integer(generation, "generation", 1, 2**63 - 1)
        confirmation = _digest(confirm_graph_digest, "confirm_graph_digest")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                pointer = connection.execute(
                    "SELECT generation FROM graph_current WHERE owner_id=? AND doc_id=?",
                    (owner, document),
                ).fetchone()
                if pointer is not None and int(pointer["generation"]) == sequence:
                    raise RuntimeError("current evidence graph generation may not be deleted.")
                row = connection.execute(
                    "SELECT graph_digest FROM graph_generations WHERE owner_id=? AND doc_id=? AND generation=?",
                    (owner, document, sequence),
                ).fetchone()
                if row is None:
                    connection.execute("COMMIT")
                    return False
                if row["graph_digest"] != confirmation:
                    raise RuntimeError("evidence graph deletion confirmation is not exact.")
                connection.execute(
                    "DELETE FROM graph_generations WHERE owner_id=? AND doc_id=? AND generation=?",
                    (owner, document, sequence),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return True


__all__ = ["EvidenceGraphStore"]
