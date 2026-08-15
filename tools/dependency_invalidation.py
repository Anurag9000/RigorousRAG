"""Durable owner-scoped dependency invalidation and bounded recomputation ledger.

Historical artifacts are immutable. Changes to sources, models, policies, schemas or
index generations mark downstream artifacts stale and create deduplicated recomputation
requests. The ledger stores identifiers/fingerprints only; it never stores source text.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.retraction_propagation import SourceStatusEvent
from tools.security import normalize_owner_id

_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_IDENTIFIER = 1000
_MAX_REASON = 5000
_MAX_DEPTH = 64
_MAX_IMPACT = 100_000
_ALLOWED_TASK_STATUS = frozenset({"queued", "claimed", "completed", "failed", "cancelled"})


def _safe_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    if len(str(absolute)) > 4096:
        raise ValueError("dependency invalidation database path is too long")
    for component in (absolute, *absolute.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        if stat.S_ISLNK(metadata.st_mode) or bool(attributes & _WINDOWS_REPARSE_POINT):
            raise RuntimeError("dependency invalidation path may not traverse symlinks/reparse points")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    return absolute


def _text(value: Any, label: str, maximum: int = _MAX_IDENTIFIER) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if not cleaned or len(cleaned) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise ValueError(f"{label} is invalid")
    return cleaned


def _sha(value: str, label: str, *, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return ""
    digest = _text(value, label, 64).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _resource_key(kind: str, resource_id: str) -> str:
    return hashlib.sha256(_canonical({"kind": kind, "resource_id": resource_id})).hexdigest()


@dataclass(frozen=True)
class DependencyRef:
    kind: str
    resource_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _text(self.kind, "kind", 64).lower())
        object.__setattr__(self, "resource_id", _text(self.resource_id, "resource_id"))

    @property
    def key(self) -> str:
        return _resource_key(self.kind, self.resource_id)


@dataclass(frozen=True)
class DependencyEdge:
    upstream: DependencyRef
    downstream: DependencyRef
    relation: str
    edge_sha256: str
    created_at: float


@dataclass(frozen=True)
class StaleArtifact:
    artifact: DependencyRef
    triggering_event_sha256: str
    reason: str
    stale_at: float
    acknowledged_at: float | None = None
    replacement_id: str = ""


@dataclass(frozen=True)
class RecomputeTask:
    task_id: str
    artifact: DependencyRef
    triggering_event_sha256: str
    reason: str
    status: str
    attempts: int
    created_at: float
    claimed_at: float | None = None
    completed_at: float | None = None
    error_type: str = ""


@dataclass(frozen=True)
class InvalidationImpact:
    triggering_event_sha256: str
    root: DependencyRef
    affected: tuple[DependencyRef, ...]
    recompute_tasks: tuple[RecomputeTask, ...]


class DependencyInvalidationStore:
    def __init__(self, path: str | Path) -> None:
        self.path = _safe_path(path)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS dependencies (
                    owner_id TEXT NOT NULL,
                    edge_sha256 CHAR(64) NOT NULL,
                    upstream_kind TEXT NOT NULL,
                    upstream_id TEXT NOT NULL,
                    upstream_key CHAR(64) NOT NULL,
                    downstream_kind TEXT NOT NULL,
                    downstream_id TEXT NOT NULL,
                    downstream_key CHAR(64) NOT NULL,
                    relation TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(owner_id, edge_sha256)
                );
                CREATE INDEX IF NOT EXISTS dependencies_upstream_idx
                  ON dependencies(owner_id, upstream_key, created_at, edge_sha256);
                CREATE INDEX IF NOT EXISTS dependencies_downstream_idx
                  ON dependencies(owner_id, downstream_key, created_at, edge_sha256);

                CREATE TABLE IF NOT EXISTS invalidation_events (
                    owner_id TEXT NOT NULL,
                    event_sha256 CHAR(64) NOT NULL,
                    root_kind TEXT NOT NULL,
                    root_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    replacement_id TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(owner_id, event_sha256)
                );

                CREATE TABLE IF NOT EXISTS stale_artifacts (
                    owner_id TEXT NOT NULL,
                    artifact_kind TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    artifact_key CHAR(64) NOT NULL,
                    event_sha256 CHAR(64) NOT NULL,
                    reason TEXT NOT NULL,
                    replacement_id TEXT NOT NULL DEFAULT '',
                    stale_at REAL NOT NULL,
                    acknowledged_at REAL,
                    PRIMARY KEY(owner_id, artifact_key, event_sha256)
                );
                CREATE INDEX IF NOT EXISTS stale_artifacts_open_idx
                  ON stale_artifacts(owner_id, acknowledged_at, stale_at, artifact_key);

                CREATE TABLE IF NOT EXISTS recompute_tasks (
                    owner_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    artifact_kind TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    artifact_key CHAR(64) NOT NULL,
                    event_sha256 CHAR(64) NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    claimed_at REAL,
                    completed_at REAL,
                    error_type TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(owner_id, task_id),
                    UNIQUE(owner_id, artifact_key, event_sha256)
                );
                CREATE INDEX IF NOT EXISTS recompute_tasks_queue_idx
                  ON recompute_tasks(owner_id, status, created_at, task_id);

                CREATE TABLE IF NOT EXISTS source_status_events (
                    owner_id TEXT NOT NULL,
                    event_sha256 CHAR(64) NOT NULL,
                    source_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    effective_at REAL NOT NULL,
                    event_source_id TEXT NOT NULL,
                    replacement_source_id TEXT NOT NULL DEFAULT '',
                    reason TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(owner_id, event_sha256)
                );
                CREATE INDEX IF NOT EXISTS source_status_source_idx
                  ON source_status_events(owner_id, source_id, effective_at, event_sha256);
                """
            )

    def register_dependency(
        self,
        owner_id: str,
        *,
        upstream: DependencyRef,
        downstream: DependencyRef,
        relation: str,
    ) -> DependencyEdge:
        owner = normalize_owner_id(owner_id)
        if not isinstance(upstream, DependencyRef) or not isinstance(downstream, DependencyRef):
            raise TypeError("upstream and downstream must be DependencyRef")
        if upstream == downstream:
            raise ValueError("dependency may not self-reference")
        relation_value = _text(relation, "relation", 128).lower()
        payload = {
            "owner_id": owner,
            "upstream": {"kind": upstream.kind, "resource_id": upstream.resource_id},
            "downstream": {"kind": downstream.kind, "resource_id": downstream.resource_id},
            "relation": relation_value,
        }
        edge_sha = hashlib.sha256(_canonical(payload)).hexdigest()
        created_at = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO dependencies
                   (owner_id,edge_sha256,upstream_kind,upstream_id,upstream_key,
                    downstream_kind,downstream_id,downstream_key,relation,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    owner, edge_sha, upstream.kind, upstream.resource_id, upstream.key,
                    downstream.kind, downstream.resource_id, downstream.key,
                    relation_value, created_at,
                ),
            )
        return DependencyEdge(upstream, downstream, relation_value, edge_sha, created_at)

    def register_dependencies(
        self,
        owner_id: str,
        *,
        downstream: DependencyRef,
        upstreams: Sequence[tuple[DependencyRef, str]],
    ) -> tuple[DependencyEdge, ...]:
        if len(upstreams) > 10_000:
            raise ValueError("dependency batch exceeds the item limit")
        return tuple(
            self.register_dependency(owner_id, upstream=upstream, downstream=downstream, relation=relation)
            for upstream, relation in upstreams
        )

    def _impact_rows(
        self,
        connection: sqlite3.Connection,
        owner: str,
        root: DependencyRef,
        *,
        max_depth: int,
        max_impact: int,
    ) -> tuple[DependencyRef, ...]:
        queue: list[tuple[DependencyRef, int]] = [(root, 0)]
        seen = {root.key}
        output: list[DependencyRef] = []
        while queue and len(output) < max_impact:
            current, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            rows = connection.execute(
                """SELECT downstream_kind,downstream_id,downstream_key
                   FROM dependencies WHERE owner_id=? AND upstream_key=?
                   ORDER BY created_at,edge_sha256 LIMIT ?""",
                (owner, current.key, max_impact - len(output)),
            ).fetchall()
            for row in rows:
                key = str(row["downstream_key"])
                if key in seen:
                    continue
                seen.add(key)
                ref = DependencyRef(str(row["downstream_kind"]), str(row["downstream_id"]))
                output.append(ref)
                queue.append((ref, depth + 1))
                if len(output) >= max_impact:
                    break
        return tuple(output)

    def invalidate(
        self,
        owner_id: str,
        *,
        root: DependencyRef,
        reason: str,
        event_type: str,
        replacement_id: str = "",
        event_sha256: str = "",
        max_depth: int = 32,
        max_impact: int = 10_000,
        recomputable_kinds: Sequence[str] = ("result", "report", "graph", "capsule", "index", "matrix"),
    ) -> InvalidationImpact:
        owner = normalize_owner_id(owner_id)
        reason_value = _text(reason, "reason", _MAX_REASON)
        event_type_value = _text(event_type, "event_type", 128).lower()
        replacement = "" if replacement_id == "" else _text(replacement_id, "replacement_id")
        if not 1 <= max_depth <= _MAX_DEPTH or not 1 <= max_impact <= _MAX_IMPACT:
            raise ValueError("invalidation traversal limits are invalid")
        allowed_recompute = frozenset(_text(item, "recomputable kind", 64).lower() for item in recomputable_kinds)
        if event_sha256:
            event_sha = _sha(event_sha256, "event_sha256")
        else:
            event_sha = hashlib.sha256(
                _canonical(
                    {
                        "owner_id": owner,
                        "root": {"kind": root.kind, "resource_id": root.resource_id},
                        "reason": reason_value,
                        "event_type": event_type_value,
                        "replacement_id": replacement,
                    }
                )
            ).hexdigest()
        created_at = time.time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                affected = self._impact_rows(
                    connection, owner, root, max_depth=max_depth, max_impact=max_impact
                )
                payload_json = json.dumps(
                    {
                        "root": {"kind": root.kind, "resource_id": root.resource_id},
                        "reason": reason_value,
                        "event_type": event_type_value,
                        "replacement_id": replacement,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                connection.execute(
                    """INSERT OR IGNORE INTO invalidation_events
                       (owner_id,event_sha256,root_kind,root_id,reason,event_type,replacement_id,created_at,payload_json)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (owner, event_sha, root.kind, root.resource_id, reason_value, event_type_value, replacement, created_at, payload_json),
                )
                tasks: list[RecomputeTask] = []
                for artifact in affected:
                    connection.execute(
                        """INSERT OR IGNORE INTO stale_artifacts
                           (owner_id,artifact_kind,artifact_id,artifact_key,event_sha256,reason,replacement_id,stale_at)
                           VALUES(?,?,?,?,?,?,?,?)""",
                        (owner, artifact.kind, artifact.resource_id, artifact.key, event_sha, reason_value, replacement, created_at),
                    )
                    if artifact.kind not in allowed_recompute:
                        continue
                    task_id = "recompute_" + hashlib.sha256(
                        _canonical({"owner_id": owner, "artifact_key": artifact.key, "event_sha256": event_sha})
                    ).hexdigest()[:40]
                    connection.execute(
                        """INSERT OR IGNORE INTO recompute_tasks
                           (owner_id,task_id,artifact_kind,artifact_id,artifact_key,event_sha256,reason,status,created_at)
                           VALUES(?,?,?,?,?,?,?,?,?)""",
                        (owner, task_id, artifact.kind, artifact.resource_id, artifact.key, event_sha, reason_value, "queued", created_at),
                    )
                    row = connection.execute(
                        "SELECT * FROM recompute_tasks WHERE owner_id=? AND artifact_key=? AND event_sha256=?",
                        (owner, artifact.key, event_sha),
                    ).fetchone()
                    tasks.append(self._task_from_row(row))
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return InvalidationImpact(event_sha, root, affected, tuple(tasks))

    def record_source_status(
        self,
        owner_id: str,
        event: SourceStatusEvent,
        *,
        invalidate_downstream: bool = True,
    ) -> InvalidationImpact | None:
        owner = normalize_owner_id(owner_id)
        if not isinstance(event, SourceStatusEvent):
            raise TypeError("event must be SourceStatusEvent")
        payload = {
            "source_id": event.source_id,
            "status": event.status,
            "effective_at": event.effective_at,
            "event_source_id": event.event_source_id,
            "replacement_source_id": event.replacement_source_id,
            "reason": event.reason,
            "event_sha256": event.event_sha256,
        }
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO source_status_events
                   (owner_id,event_sha256,source_id,status,effective_at,event_source_id,
                    replacement_source_id,reason,payload_json)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    owner, event.event_sha256, event.source_id, event.status,
                    event.effective_at, event.event_source_id, event.replacement_source_id,
                    event.reason, json.dumps(payload, sort_keys=True, separators=(",", ":")),
                ),
            )
        if not invalidate_downstream or event.status == "active":
            return None
        reason = event.reason or f"source status changed to {event.status}"
        return self.invalidate(
            owner,
            root=DependencyRef("source", event.source_id),
            reason=reason,
            event_type=f"source_{event.status}",
            replacement_id=event.replacement_source_id,
            event_sha256=event.event_sha256,
        )

    def source_status_events(self, owner_id: str, source_id: str) -> tuple[SourceStatusEvent, ...]:
        owner = normalize_owner_id(owner_id)
        source = _text(source_id, "source_id")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT payload_json FROM source_status_events
                   WHERE owner_id=? AND source_id=? ORDER BY effective_at,event_sha256""",
                (owner, source),
            ).fetchall()
        return tuple(SourceStatusEvent(**json.loads(str(row["payload_json"]))) for row in rows)

    def list_stale(
        self,
        owner_id: str,
        *,
        kind: str | None = None,
        include_acknowledged: bool = False,
        limit: int = 1000,
    ) -> tuple[StaleArtifact, ...]:
        owner = normalize_owner_id(owner_id)
        if not 1 <= limit <= 10_000:
            raise ValueError("limit is invalid")
        clauses = ["owner_id=?"]
        params: list[Any] = [owner]
        if kind is not None:
            clauses.append("artifact_kind=?")
            params.append(_text(kind, "kind", 64).lower())
        if not include_acknowledged:
            clauses.append("acknowledged_at IS NULL")
        params.append(limit)
        query = (
            "SELECT * FROM stale_artifacts WHERE " + " AND ".join(clauses)
            + " ORDER BY stale_at DESC,artifact_key LIMIT ?"
        )
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(self._stale_from_row(row) for row in rows)

    def acknowledge(self, owner_id: str, artifact: DependencyRef, *, event_sha256: str | None = None) -> int:
        owner = normalize_owner_id(owner_id)
        now = time.time()
        with self._lock, self._connect() as connection:
            if event_sha256 is None:
                cursor = connection.execute(
                    "UPDATE stale_artifacts SET acknowledged_at=? WHERE owner_id=? AND artifact_key=? AND acknowledged_at IS NULL",
                    (now, owner, artifact.key),
                )
            else:
                event = _sha(event_sha256, "event_sha256")
                cursor = connection.execute(
                    "UPDATE stale_artifacts SET acknowledged_at=? WHERE owner_id=? AND artifact_key=? AND event_sha256=? AND acknowledged_at IS NULL",
                    (now, owner, artifact.key, event),
                )
            return int(cursor.rowcount)

    def claim_recompute(self, owner_id: str, *, kinds: Sequence[str] = (), max_attempts: int = 5) -> RecomputeTask | None:
        owner = normalize_owner_id(owner_id)
        if not 1 <= max_attempts <= 100:
            raise ValueError("max_attempts is invalid")
        allowed = tuple(_text(item, "kind", 64).lower() for item in kinds)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                params: list[Any] = [owner, max_attempts]
                extra = ""
                if allowed:
                    placeholders = ",".join("?" for _ in allowed)
                    extra = f" AND artifact_kind IN ({placeholders})"
                    params.extend(allowed)
                row = connection.execute(
                    "SELECT * FROM recompute_tasks WHERE owner_id=? AND status='queued' AND attempts<?"
                    + extra + " ORDER BY created_at,task_id LIMIT 1",
                    tuple(params),
                ).fetchone()
                if row is None:
                    connection.commit()
                    return None
                now = time.time()
                cursor = connection.execute(
                    """UPDATE recompute_tasks SET status='claimed',attempts=attempts+1,claimed_at=?
                       WHERE owner_id=? AND task_id=? AND status='queued'""",
                    (now, owner, str(row["task_id"])),
                )
                if cursor.rowcount != 1:
                    connection.rollback()
                    return None
                claimed = connection.execute(
                    "SELECT * FROM recompute_tasks WHERE owner_id=? AND task_id=?",
                    (owner, str(row["task_id"])),
                ).fetchone()
                connection.commit()
                return self._task_from_row(claimed)
            except Exception:
                connection.rollback()
                raise

    def finish_recompute(
        self,
        owner_id: str,
        task_id: str,
        *,
        success: bool,
        error_type: str = "",
        acknowledge_stale: bool = True,
    ) -> RecomputeTask:
        owner = normalize_owner_id(owner_id)
        task = _text(task_id, "task_id", 256)
        error = "" if not error_type else _text(error_type, "error_type", 200)
        now = time.time()
        status = "completed" if success else "failed"
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM recompute_tasks WHERE owner_id=? AND task_id=?",
                    (owner, task),
                ).fetchone()
                if row is None:
                    raise KeyError(task)
                if str(row["status"]) != "claimed":
                    raise RuntimeError("recompute task must be claimed before completion")
                connection.execute(
                    "UPDATE recompute_tasks SET status=?,completed_at=?,error_type=? WHERE owner_id=? AND task_id=?",
                    (status, now, error, owner, task),
                )
                if success and acknowledge_stale:
                    connection.execute(
                        """UPDATE stale_artifacts SET acknowledged_at=?
                           WHERE owner_id=? AND artifact_key=? AND event_sha256=? AND acknowledged_at IS NULL""",
                        (now, owner, str(row["artifact_key"]), str(row["event_sha256"])),
                    )
                updated = connection.execute(
                    "SELECT * FROM recompute_tasks WHERE owner_id=? AND task_id=?",
                    (owner, task),
                ).fetchone()
                connection.commit()
                return self._task_from_row(updated)
            except Exception:
                connection.rollback()
                raise

    def list_recompute(self, owner_id: str, *, status: str | None = None, limit: int = 1000) -> tuple[RecomputeTask, ...]:
        owner = normalize_owner_id(owner_id)
        if not 1 <= limit <= 10_000:
            raise ValueError("limit is invalid")
        if status is None:
            query = "SELECT * FROM recompute_tasks WHERE owner_id=? ORDER BY created_at DESC,task_id LIMIT ?"
            params = (owner, limit)
        else:
            state = _text(status, "status", 32).lower()
            if state not in _ALLOWED_TASK_STATUS:
                raise ValueError("unsupported recompute status")
            query = "SELECT * FROM recompute_tasks WHERE owner_id=? AND status=? ORDER BY created_at DESC,task_id LIMIT ?"
            params = (owner, state, limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return tuple(self._task_from_row(row) for row in rows)

    @staticmethod
    def _stale_from_row(row: sqlite3.Row) -> StaleArtifact:
        return StaleArtifact(
            DependencyRef(str(row["artifact_kind"]), str(row["artifact_id"])),
            str(row["event_sha256"]),
            str(row["reason"]),
            float(row["stale_at"]),
            float(row["acknowledged_at"]) if row["acknowledged_at"] is not None else None,
            str(row["replacement_id"] or ""),
        )

    @staticmethod
    def _task_from_row(row: sqlite3.Row) -> RecomputeTask:
        return RecomputeTask(
            task_id=str(row["task_id"]),
            artifact=DependencyRef(str(row["artifact_kind"]), str(row["artifact_id"])),
            triggering_event_sha256=str(row["event_sha256"]),
            reason=str(row["reason"]),
            status=str(row["status"]),
            attempts=int(row["attempts"]),
            created_at=float(row["created_at"]),
            claimed_at=float(row["claimed_at"]) if row["claimed_at"] is not None else None,
            completed_at=float(row["completed_at"]) if row["completed_at"] is not None else None,
            error_type=str(row["error_type"] or ""),
        )


__all__ = [
    "DependencyEdge",
    "DependencyInvalidationStore",
    "DependencyRef",
    "InvalidationImpact",
    "RecomputeTask",
    "StaleArtifact",
]
