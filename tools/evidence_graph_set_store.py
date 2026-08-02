"""Transactional storage and authority checks for cross-document graph sets."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import stat
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tools.evidence_graph_sets import (
    CrossDocumentEdge,
    CrossDocumentNodeReference,
    EvidenceGraphSet,
    GraphGenerationReference,
    _digest,
    _identifier,
    _sha256,
)
from tools.security import normalize_owner_id

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_MAX_PAYLOAD_BYTES = 512 * 1024 * 1024
_MAX_HISTORY = 10_000
_SCHEMA_VERSION = 1
_UNSET = object()


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("graph set database path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("graph set database path is invalid.")
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
            raise ValueError("graph set database path could not be validated.") from exc
        if _redirecting(info):
            raise ValueError("graph set database path may not contain redirects.")
    return absolute


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return result


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _encode(value: EvidenceGraphSet) -> str:
    payload = json.dumps(
        asdict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if not payload or len(payload.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
        raise ValueError("graph set payload exceeds the byte limit.")
    return payload


def _node_ref(raw: Any) -> CrossDocumentNodeReference:
    if not isinstance(raw, dict):
        raise ValueError("node reference schema")
    return CrossDocumentNodeReference(**raw)


def _decode(payload: Any) -> EvidenceGraphSet:
    if not isinstance(payload, str) or not payload or len(payload.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
        raise RuntimeError("stored graph set payload is corrupt.")
    try:
        raw = json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
        if not isinstance(raw, dict) or set(raw) != {
            "graph_set_id",
            "owner_id",
            "graph_set_key",
            "members",
            "edges",
            "created_at",
            "schema_version",
        }:
            raise ValueError("graph set schema")
        members = tuple(GraphGenerationReference(**item) for item in raw["members"])
        edges = tuple(
            CrossDocumentEdge(
                edge_id=item["edge_id"],
                owner_id=item["owner_id"],
                graph_set_id=item["graph_set_id"],
                source=_node_ref(item["source"]),
                target=_node_ref(item["target"]),
                edge_type=item["edge_type"],
                relation_key=item["relation_key"],
                weight=item["weight"],
                metadata=item["metadata"],
            )
            for item in raw["edges"]
        )
        return EvidenceGraphSet(
            graph_set_id=raw["graph_set_id"],
            owner_id=raw["owner_id"],
            graph_set_key=raw["graph_set_key"],
            members=members,
            edges=edges,
            created_at=raw["created_at"],
            schema_version=raw["schema_version"],
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise RuntimeError("stored graph set payload is corrupt.") from exc


@dataclass(frozen=True)
class EvidenceGraphSetAuthorityReport:
    graph_set_id: str
    graph_set_digest: str
    authoritative_current: bool
    stale_member_doc_ids: tuple[str, ...]
    missing_member_doc_ids: tuple[str, ...]
    authority_digest: str


class EvidenceGraphSetAuthorityError(RuntimeError):
    """Raised when a logical current graph set contains stale member generations."""


def assess_graph_set_authority(
    graph_set: EvidenceGraphSet,
    *,
    generations: Any,
    graphs: Any,
) -> EvidenceGraphSetAuthorityReport:
    if not isinstance(graph_set, EvidenceGraphSet):
        raise ValueError("graph_set must be EvidenceGraphSet.")
    stale: list[str] = []
    missing: list[str] = []
    observations: list[dict[str, Any]] = []
    for member in graph_set.members:
        generation = generations.current(owner_id=member.owner_id, doc_id=member.doc_id)
        graph = graphs.current(owner_id=member.owner_id, doc_id=member.doc_id)
        if generation is None or graph is None:
            missing.append(member.doc_id)
            observations.append({"doc_id": member.doc_id, "status": "missing"})
            continue
        current = bool(
            getattr(generation, "sequence", None) == member.generation
            and getattr(generation, "content_sha256", None) == member.content_sha256
            and getattr(generation, "profile_fingerprint", None)
            == member.profile_fingerprint
            and getattr(graph, "generation", None) == member.generation
            and getattr(graph, "graph_digest", None) == member.graph_digest
        )
        if not current:
            stale.append(member.doc_id)
        observations.append(
            {
                "doc_id": member.doc_id,
                "status": "current" if current else "stale",
                "authoritative_sequence": getattr(generation, "sequence", None),
                "graph_generation": getattr(graph, "generation", None),
                "graph_digest": getattr(graph, "graph_digest", None),
            }
        )
    stale_values = tuple(sorted(set(stale)))
    missing_values = tuple(sorted(set(missing)))
    current = not stale_values and not missing_values
    return EvidenceGraphSetAuthorityReport(
        graph_set_id=graph_set.graph_set_id,
        graph_set_digest=graph_set.graph_set_digest,
        authoritative_current=current,
        stale_member_doc_ids=stale_values,
        missing_member_doc_ids=missing_values,
        authority_digest=_sha256(
            {
                "scope": "rigorousrag-evidence-graph-set-authority-v1",
                "graph_set_id": graph_set.graph_set_id,
                "graph_set_digest": graph_set.graph_set_digest,
                "observations": observations,
                "authoritative_current": current,
            }
        ),
    )


class EvidenceGraphSetStore:
    """Append-only graph-set versions with one optimistic logical-key pointer."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("graph set database parent must be a directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("graph set database is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
        ):
            raise RuntimeError("graph set database parent identity changed.")
        if self._file_identity() != self._database_identity:
            raise RuntimeError("graph set database identity changed.")

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
                CREATE TABLE IF NOT EXISTS evidence_graph_sets (
                    owner_id TEXT NOT NULL,
                    graph_set_key TEXT NOT NULL,
                    graph_set_id TEXT NOT NULL,
                    graph_set_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    schema_version INTEGER NOT NULL,
                    PRIMARY KEY(owner_id, graph_set_id)
                );
                CREATE TABLE IF NOT EXISTS evidence_graph_set_current (
                    owner_id TEXT NOT NULL,
                    graph_set_key TEXT NOT NULL,
                    graph_set_id TEXT NOT NULL,
                    graph_set_digest TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    schema_version INTEGER NOT NULL,
                    PRIMARY KEY(owner_id, graph_set_key),
                    FOREIGN KEY(owner_id, graph_set_id)
                        REFERENCES evidence_graph_sets(owner_id, graph_set_id)
                        ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS evidence_graph_set_history
                    ON evidence_graph_sets(owner_id, graph_set_key, created_at DESC, graph_set_id);
                """
            )

    @staticmethod
    def _value(row: sqlite3.Row) -> EvidenceGraphSet:
        if int(row["schema_version"]) != _SCHEMA_VERSION:
            raise RuntimeError("stored graph set schema is unsupported.")
        value = _decode(row["payload_json"])
        if (
            value.owner_id != row["owner_id"]
            or value.graph_set_key != row["graph_set_key"]
            or value.graph_set_id != row["graph_set_id"]
            or value.graph_set_digest != row["graph_set_digest"]
        ):
            raise RuntimeError("stored graph set row identity is corrupt.")
        return value

    def commit(
        self,
        value: EvidenceGraphSet,
        *,
        make_current: bool = True,
        expected_current_set_id: str | None | object = _UNSET,
        now: float | None = None,
    ) -> EvidenceGraphSet:
        if not isinstance(value, EvidenceGraphSet):
            raise ValueError("value must be EvidenceGraphSet.")
        if not isinstance(make_current, bool):
            raise ValueError("make_current must be boolean.")
        if expected_current_set_id is not _UNSET and expected_current_set_id is not None:
            expected_current_set_id = _digest(expected_current_set_id, "expected_current_set_id")
        timestamp = _timestamp(time.time() if now is None else now, "now")
        payload = _encode(value)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT * FROM evidence_graph_sets WHERE owner_id=? AND graph_set_id=?",
                    (value.owner_id, value.graph_set_id),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO evidence_graph_sets VALUES (?, ?, ?, ?, ?, ?, 1)
                        """,
                        (
                            value.owner_id,
                            value.graph_set_key,
                            value.graph_set_id,
                            value.graph_set_digest,
                            payload,
                            value.created_at,
                        ),
                    )
                else:
                    stored = self._value(existing)
                    if stored.graph_set_digest != value.graph_set_digest:
                        raise RuntimeError("graph set identity collision detected.")
                if make_current:
                    pointer = connection.execute(
                        "SELECT graph_set_id FROM evidence_graph_set_current "
                        "WHERE owner_id=? AND graph_set_key=?",
                        (value.owner_id, value.graph_set_key),
                    ).fetchone()
                    actual = None if pointer is None else pointer["graph_set_id"]
                    if expected_current_set_id is not _UNSET and actual != expected_current_set_id:
                        raise RuntimeError("graph set current pointer changed concurrently.")
                    connection.execute(
                        """
                        INSERT INTO evidence_graph_set_current VALUES (?, ?, ?, ?, ?, 1)
                        ON CONFLICT(owner_id, graph_set_key) DO UPDATE SET
                            graph_set_id=excluded.graph_set_id,
                            graph_set_digest=excluded.graph_set_digest,
                            updated_at=excluded.updated_at,
                            schema_version=excluded.schema_version
                        """,
                        (
                            value.owner_id,
                            value.graph_set_key,
                            value.graph_set_id,
                            value.graph_set_digest,
                            timestamp,
                        ),
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(owner_id=value.owner_id, graph_set_id=value.graph_set_id)

    def get(self, *, owner_id: str, graph_set_id: str) -> EvidenceGraphSet:
        owner = normalize_owner_id(owner_id)
        selected = _digest(graph_set_id, "graph_set_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM evidence_graph_sets WHERE owner_id=? AND graph_set_id=?",
                (owner, selected),
            ).fetchone()
        if row is None:
            raise KeyError((owner, selected))
        return self._value(row)

    def current(self, *, owner_id: str, graph_set_key: str) -> EvidenceGraphSet | None:
        owner = normalize_owner_id(owner_id)
        key = _identifier(graph_set_key, "graph_set_key", 500)
        with self._lock, self._connect() as connection:
            pointer = connection.execute(
                "SELECT * FROM evidence_graph_set_current "
                "WHERE owner_id=? AND graph_set_key=?",
                (owner, key),
            ).fetchone()
            if pointer is None:
                return None
            row = connection.execute(
                "SELECT * FROM evidence_graph_sets WHERE owner_id=? AND graph_set_id=?",
                (owner, pointer["graph_set_id"]),
            ).fetchone()
        if row is None:
            raise RuntimeError("graph set current pointer target is missing.")
        value = self._value(row)
        if value.graph_set_digest != pointer["graph_set_digest"]:
            raise RuntimeError("graph set current pointer digest is corrupt.")
        return value

    def history(
        self,
        *,
        owner_id: str,
        graph_set_key: str,
        limit: int = 100,
    ) -> tuple[EvidenceGraphSet, ...]:
        owner = normalize_owner_id(owner_id)
        key = _identifier(graph_set_key, "graph_set_key", 500)
        count = _integer(limit, "limit", 1, _MAX_HISTORY)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM evidence_graph_sets
                WHERE owner_id=? AND graph_set_key=?
                ORDER BY created_at DESC, graph_set_id DESC LIMIT ?
                """,
                (owner, key, count),
            ).fetchall()
        return tuple(self._value(row) for row in rows)

    def delete(
        self,
        *,
        owner_id: str,
        graph_set_id: str,
        confirm_graph_set_digest: str,
    ) -> bool:
        owner = normalize_owner_id(owner_id)
        selected = _digest(graph_set_id, "graph_set_id")
        confirmation = _digest(confirm_graph_set_digest, "confirm_graph_set_digest")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = connection.execute(
                    "SELECT 1 FROM evidence_graph_set_current "
                    "WHERE owner_id=? AND graph_set_id=?",
                    (owner, selected),
                ).fetchone()
                if current is not None:
                    raise RuntimeError("current graph set may not be deleted.")
                row = connection.execute(
                    "SELECT graph_set_digest FROM evidence_graph_sets "
                    "WHERE owner_id=? AND graph_set_id=?",
                    (owner, selected),
                ).fetchone()
                if row is None:
                    connection.execute("COMMIT")
                    return False
                if row["graph_set_digest"] != confirmation:
                    raise RuntimeError("graph set deletion confirmation differs.")
                connection.execute(
                    "DELETE FROM evidence_graph_sets WHERE owner_id=? AND graph_set_id=?",
                    (owner, selected),
                )
                connection.execute("COMMIT")
                return True
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def resolve_current(
        self,
        *,
        owner_id: str,
        graph_set_key: str,
        generations: Any,
        graphs: Any,
    ) -> tuple[EvidenceGraphSet, EvidenceGraphSetAuthorityReport]:
        value = self.current(owner_id=owner_id, graph_set_key=graph_set_key)
        if value is None:
            raise KeyError((owner_id, graph_set_key))
        report = assess_graph_set_authority(value, generations=generations, graphs=graphs)
        if not report.authoritative_current:
            raise EvidenceGraphSetAuthorityError(
                "current graph set contains stale or missing member generations."
            )
        return value, report


__all__ = [
    "EvidenceGraphSetAuthorityError",
    "EvidenceGraphSetAuthorityReport",
    "EvidenceGraphSetStore",
    "assess_graph_set_authority",
]
