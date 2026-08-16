"""PostgreSQL dependency invalidation ledger with multi-worker safe recompute claiming."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Mapping, Sequence

from tools.dependency_invalidation import (
    DependencyEdge,
    DependencyInvalidationStore,
    DependencyRef,
    InvalidationImpact,
    RecomputeTask,
    StaleArtifact,
    _ALLOWED_TASK_STATUS,
    _MAX_DEPTH,
    _MAX_IMPACT,
    _MAX_REASON,
    _canonical,
    _sha,
    _text,
)
from tools.postgres_research_stores import _PostgresMixin, _row
from tools.retraction_propagation import SourceStatusEvent
from tools.security import normalize_owner_id
from tools.sql_control_plane import ConnectionFactory, CursorLike


class PostgresDependencyInvalidationStore(_PostgresMixin, DependencyInvalidationStore):
    def __init__(
        self,
        connection_factory: ConnectionFactory,
        *,
        schema: str = "rigorousrag",
        initialize: bool = True,
    ) -> None:
        _PostgresMixin.__init__(
            self,
            connection_factory,
            schema=schema,
            initialize=initialize,
        )

    def _initialize_postgres(self) -> None:
        schema = self.schema
        self._initialize_tables(
            (
                f"""CREATE TABLE IF NOT EXISTS {schema}.dependencies (
                    owner_id TEXT NOT NULL,
                    edge_sha256 CHAR(64) NOT NULL,
                    upstream_kind TEXT NOT NULL,
                    upstream_id TEXT NOT NULL,
                    upstream_key CHAR(64) NOT NULL,
                    downstream_kind TEXT NOT NULL,
                    downstream_id TEXT NOT NULL,
                    downstream_key CHAR(64) NOT NULL,
                    relation TEXT NOT NULL,
                    created_at DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY(owner_id,edge_sha256)
                )""",
                f"CREATE INDEX IF NOT EXISTS dependencies_upstream_idx ON {schema}.dependencies(owner_id,upstream_key,created_at,edge_sha256)",
                f"CREATE INDEX IF NOT EXISTS dependencies_downstream_idx ON {schema}.dependencies(owner_id,downstream_key,created_at,edge_sha256)",
                f"""CREATE TABLE IF NOT EXISTS {schema}.invalidation_events (
                    owner_id TEXT NOT NULL,
                    event_sha256 CHAR(64) NOT NULL,
                    root_kind TEXT NOT NULL,
                    root_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    replacement_id TEXT NOT NULL DEFAULT '',
                    created_at DOUBLE PRECISION NOT NULL,
                    payload_json JSONB NOT NULL,
                    PRIMARY KEY(owner_id,event_sha256)
                )""",
                f"""CREATE TABLE IF NOT EXISTS {schema}.stale_artifacts (
                    owner_id TEXT NOT NULL,
                    artifact_kind TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    artifact_key CHAR(64) NOT NULL,
                    event_sha256 CHAR(64) NOT NULL,
                    reason TEXT NOT NULL,
                    replacement_id TEXT NOT NULL DEFAULT '',
                    stale_at DOUBLE PRECISION NOT NULL,
                    acknowledged_at DOUBLE PRECISION,
                    PRIMARY KEY(owner_id,artifact_key,event_sha256)
                )""",
                f"CREATE INDEX IF NOT EXISTS stale_artifacts_open_idx ON {schema}.stale_artifacts(owner_id,acknowledged_at,stale_at,artifact_key)",
                f"""CREATE TABLE IF NOT EXISTS {schema}.recompute_tasks (
                    owner_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    artifact_kind TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    artifact_key CHAR(64) NOT NULL,
                    event_sha256 CHAR(64) NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at DOUBLE PRECISION NOT NULL,
                    claimed_at DOUBLE PRECISION,
                    completed_at DOUBLE PRECISION,
                    error_type TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(owner_id,task_id),
                    UNIQUE(owner_id,artifact_key,event_sha256)
                )""",
                f"CREATE INDEX IF NOT EXISTS recompute_tasks_queue_idx ON {schema}.recompute_tasks(owner_id,status,created_at,task_id)",
                f"""CREATE TABLE IF NOT EXISTS {schema}.source_status_events (
                    owner_id TEXT NOT NULL,
                    event_sha256 CHAR(64) NOT NULL,
                    source_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    effective_at DOUBLE PRECISION NOT NULL,
                    event_source_id TEXT NOT NULL,
                    replacement_source_id TEXT NOT NULL DEFAULT '',
                    reason TEXT NOT NULL DEFAULT '',
                    payload_json JSONB NOT NULL,
                    PRIMARY KEY(owner_id,event_sha256)
                )""",
                f"CREATE INDEX IF NOT EXISTS source_status_source_idx ON {schema}.source_status_events(owner_id,source_id,effective_at,event_sha256)",
            )
        )

    @staticmethod
    def _stale_from_row(row: Sequence[Any] | Mapping[str, Any]) -> StaleArtifact:
        acknowledged = _row(row, "acknowledged_at", 5)
        return StaleArtifact(
            artifact=DependencyRef(
                str(_row(row, "artifact_kind", 0)),
                str(_row(row, "artifact_id", 1)),
            ),
            triggering_event_sha256=str(_row(row, "event_sha256", 2)),
            reason=str(_row(row, "reason", 3)),
            stale_at=float(_row(row, "stale_at", 4)),
            acknowledged_at=float(acknowledged) if acknowledged is not None else None,
            replacement_id=str(_row(row, "replacement_id", 6) or ""),
        )

    @staticmethod
    def _task_from_row(row: Sequence[Any] | Mapping[str, Any]) -> RecomputeTask:
        claimed = _row(row, "claimed_at", 8)
        completed = _row(row, "completed_at", 9)
        return RecomputeTask(
            task_id=str(_row(row, "task_id", 0)),
            artifact=DependencyRef(
                str(_row(row, "artifact_kind", 1)),
                str(_row(row, "artifact_id", 2)),
            ),
            triggering_event_sha256=str(_row(row, "event_sha256", 4)),
            reason=str(_row(row, "reason", 5)),
            status=str(_row(row, "status", 6)),
            attempts=int(_row(row, "attempts", 7)),
            created_at=float(_row(row, "created_at", 3)),
            claimed_at=float(claimed) if claimed is not None else None,
            completed_at=float(completed) if completed is not None else None,
            error_type=str(_row(row, "error_type", 10) or ""),
        )

    @staticmethod
    def _task_columns() -> str:
        return (
            "task_id,artifact_kind,artifact_id,created_at,event_sha256,reason,status,"
            "attempts,claimed_at,completed_at,error_type"
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

        def operation(cursor: CursorLike) -> None:
            cursor.execute(
                f"""INSERT INTO {self.schema}.dependencies
                    (owner_id,edge_sha256,upstream_kind,upstream_id,upstream_key,
                     downstream_kind,downstream_id,downstream_key,relation,created_at)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT(owner_id,edge_sha256) DO NOTHING""",
                (
                    owner,
                    edge_sha,
                    upstream.kind,
                    upstream.resource_id,
                    upstream.key,
                    downstream.kind,
                    downstream.resource_id,
                    downstream.key,
                    relation_value,
                    created_at,
                ),
            )

        self._transaction(operation)
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
            self.register_dependency(
                owner_id,
                upstream=upstream,
                downstream=downstream,
                relation=relation,
            )
            for upstream, relation in upstreams
        )

    def _impact_rows(
        self,
        cursor: CursorLike,
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
            cursor.execute(
                f"""SELECT downstream_kind,downstream_id,downstream_key
                    FROM {self.schema}.dependencies
                    WHERE owner_id=%s AND upstream_key=%s
                    ORDER BY created_at,edge_sha256 LIMIT %s""",
                (owner, current.key, max_impact - len(output)),
            )
            for row in cursor.fetchall():
                key = str(_row(row, "downstream_key", 2))
                if key in seen:
                    continue
                seen.add(key)
                ref = DependencyRef(
                    str(_row(row, "downstream_kind", 0)),
                    str(_row(row, "downstream_id", 1)),
                )
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
        recomputable_kinds: Sequence[str] = (
            "result",
            "report",
            "graph",
            "capsule",
            "index",
            "matrix",
        ),
    ) -> InvalidationImpact:
        owner = normalize_owner_id(owner_id)
        reason_value = _text(reason, "reason", _MAX_REASON)
        event_type_value = _text(event_type, "event_type", 128).lower()
        replacement = "" if replacement_id == "" else _text(replacement_id, "replacement_id")
        if not 1 <= max_depth <= _MAX_DEPTH or not 1 <= max_impact <= _MAX_IMPACT:
            raise ValueError("invalidation traversal limits are invalid")
        allowed_recompute = frozenset(
            _text(item, "recomputable kind", 64).lower() for item in recomputable_kinds
        )
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

        def operation(cursor: CursorLike) -> InvalidationImpact:
            affected = self._impact_rows(
                cursor,
                owner,
                root,
                max_depth=max_depth,
                max_impact=max_impact,
            )
            cursor.execute(
                f"""INSERT INTO {self.schema}.invalidation_events
                    (owner_id,event_sha256,root_kind,root_id,reason,event_type,replacement_id,created_at,payload_json)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                    ON CONFLICT(owner_id,event_sha256) DO NOTHING""",
                (
                    owner,
                    event_sha,
                    root.kind,
                    root.resource_id,
                    reason_value,
                    event_type_value,
                    replacement,
                    created_at,
                    payload_json,
                ),
            )
            tasks: list[RecomputeTask] = []
            for artifact in affected:
                cursor.execute(
                    f"""INSERT INTO {self.schema}.stale_artifacts
                        (owner_id,artifact_kind,artifact_id,artifact_key,event_sha256,reason,replacement_id,stale_at)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT(owner_id,artifact_key,event_sha256) DO NOTHING""",
                    (
                        owner,
                        artifact.kind,
                        artifact.resource_id,
                        artifact.key,
                        event_sha,
                        reason_value,
                        replacement,
                        created_at,
                    ),
                )
                if artifact.kind not in allowed_recompute:
                    continue
                task_id = "recompute_" + hashlib.sha256(
                    _canonical(
                        {
                            "owner_id": owner,
                            "artifact_key": artifact.key,
                            "event_sha256": event_sha,
                        }
                    )
                ).hexdigest()[:40]
                cursor.execute(
                    f"""INSERT INTO {self.schema}.recompute_tasks
                        (owner_id,task_id,artifact_kind,artifact_id,artifact_key,event_sha256,reason,status,created_at)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,'queued',%s)
                        ON CONFLICT(owner_id,artifact_key,event_sha256) DO NOTHING""",
                    (
                        owner,
                        task_id,
                        artifact.kind,
                        artifact.resource_id,
                        artifact.key,
                        event_sha,
                        reason_value,
                        created_at,
                    ),
                )
                cursor.execute(
                    f"SELECT {self._task_columns()} FROM {self.schema}.recompute_tasks WHERE owner_id=%s AND artifact_key=%s AND event_sha256=%s",
                    (owner, artifact.key, event_sha),
                )
                row = cursor.fetchone()
                if row is None:
                    raise RuntimeError("recompute task insert disappeared")
                tasks.append(self._task_from_row(row))
            return InvalidationImpact(event_sha, root, affected, tuple(tasks))

        return self._transaction(operation)

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

        def operation(cursor: CursorLike) -> None:
            cursor.execute(
                f"""INSERT INTO {self.schema}.source_status_events
                    (owner_id,event_sha256,source_id,status,effective_at,event_source_id,replacement_source_id,reason,payload_json)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                    ON CONFLICT(owner_id,event_sha256) DO NOTHING""",
                (
                    owner,
                    event.event_sha256,
                    event.source_id,
                    event.status,
                    event.effective_at,
                    event.event_source_id,
                    event.replacement_source_id,
                    event.reason,
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                ),
            )

        self._transaction(operation)
        if not invalidate_downstream or event.status == "active":
            return None
        return self.invalidate(
            owner,
            root=DependencyRef("source", event.source_id),
            reason=event.reason or f"source status changed to {event.status}",
            event_type=f"source_{event.status}",
            replacement_id=event.replacement_source_id,
            event_sha256=event.event_sha256,
        )

    def source_status_events(
        self,
        owner_id: str,
        source_id: str,
    ) -> tuple[SourceStatusEvent, ...]:
        owner = normalize_owner_id(owner_id)
        source = _text(source_id, "source_id")

        def operation(cursor: CursorLike) -> tuple[SourceStatusEvent, ...]:
            cursor.execute(
                f"SELECT payload_json::text FROM {self.schema}.source_status_events WHERE owner_id=%s AND source_id=%s ORDER BY effective_at,event_sha256",
                (owner, source),
            )
            return tuple(
                SourceStatusEvent(**json.loads(str(_row(row, "payload_json", 0))))
                for row in cursor.fetchall()
            )

        return self._transaction(operation)

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
        clauses = ["owner_id=%s"]
        params: list[Any] = [owner]
        if kind is not None:
            clauses.append("artifact_kind=%s")
            params.append(_text(kind, "kind", 64).lower())
        if not include_acknowledged:
            clauses.append("acknowledged_at IS NULL")
        params.append(limit)

        def operation(cursor: CursorLike) -> tuple[StaleArtifact, ...]:
            cursor.execute(
                f"""SELECT artifact_kind,artifact_id,event_sha256,reason,stale_at,acknowledged_at,replacement_id
                    FROM {self.schema}.stale_artifacts WHERE {' AND '.join(clauses)}
                    ORDER BY stale_at DESC,artifact_key LIMIT %s""",
                tuple(params),
            )
            return tuple(self._stale_from_row(row) for row in cursor.fetchall())

        return self._transaction(operation)

    def acknowledge(
        self,
        owner_id: str,
        artifact: DependencyRef,
        *,
        event_sha256: str | None = None,
    ) -> int:
        owner = normalize_owner_id(owner_id)
        now = time.time()

        def operation(cursor: CursorLike) -> int:
            if event_sha256 is None:
                cursor.execute(
                    f"UPDATE {self.schema}.stale_artifacts SET acknowledged_at=%s WHERE owner_id=%s AND artifact_key=%s AND acknowledged_at IS NULL",
                    (now, owner, artifact.key),
                )
            else:
                event = _sha(event_sha256, "event_sha256")
                cursor.execute(
                    f"UPDATE {self.schema}.stale_artifacts SET acknowledged_at=%s WHERE owner_id=%s AND artifact_key=%s AND event_sha256=%s AND acknowledged_at IS NULL",
                    (now, owner, artifact.key, event),
                )
            return int(cursor.rowcount)

        return self._transaction(operation)

    def claim_recompute(
        self,
        owner_id: str,
        *,
        kinds: Sequence[str] = (),
        max_attempts: int = 5,
    ) -> RecomputeTask | None:
        owner = normalize_owner_id(owner_id)
        if not 1 <= max_attempts <= 100:
            raise ValueError("max_attempts is invalid")
        allowed = tuple(_text(item, "kind", 64).lower() for item in kinds)

        def operation(cursor: CursorLike) -> RecomputeTask | None:
            params: list[Any] = [owner, max_attempts]
            extra = ""
            if allowed:
                placeholders = ",".join("%s" for _ in allowed)
                extra = f" AND artifact_kind IN ({placeholders})"
                params.extend(allowed)
            cursor.execute(
                f"""SELECT {self._task_columns()} FROM {self.schema}.recompute_tasks
                    WHERE owner_id=%s AND status='queued' AND attempts<%s{extra}
                    ORDER BY created_at,task_id FOR UPDATE SKIP LOCKED LIMIT 1""",
                tuple(params),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            task_id = str(_row(row, "task_id", 0))
            now = time.time()
            cursor.execute(
                f"""UPDATE {self.schema}.recompute_tasks
                    SET status='claimed',attempts=attempts+1,claimed_at=%s
                    WHERE owner_id=%s AND task_id=%s AND status='queued'""",
                (now, owner, task_id),
            )
            if cursor.rowcount != 1:
                return None
            cursor.execute(
                f"SELECT {self._task_columns()} FROM {self.schema}.recompute_tasks WHERE owner_id=%s AND task_id=%s",
                (owner, task_id),
            )
            claimed = cursor.fetchone()
            if claimed is None:
                raise RuntimeError("claimed recompute task disappeared")
            return self._task_from_row(claimed)

        return self._transaction(operation)

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

        def operation(cursor: CursorLike) -> RecomputeTask:
            cursor.execute(
                f"""SELECT {self._task_columns()},artifact_key
                    FROM {self.schema}.recompute_tasks
                    WHERE owner_id=%s AND task_id=%s FOR UPDATE""",
                (owner, task),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(task)
            if str(_row(row, "status", 6)) != "claimed":
                raise RuntimeError("recompute task must be claimed before completion")
            artifact_key = str(_row(row, "artifact_key", 11))
            event_sha = str(_row(row, "event_sha256", 4))
            cursor.execute(
                f"UPDATE {self.schema}.recompute_tasks SET status=%s,completed_at=%s,error_type=%s WHERE owner_id=%s AND task_id=%s",
                (status, now, error, owner, task),
            )
            if success and acknowledge_stale:
                cursor.execute(
                    f"""UPDATE {self.schema}.stale_artifacts SET acknowledged_at=%s
                        WHERE owner_id=%s AND artifact_key=%s AND event_sha256=%s AND acknowledged_at IS NULL""",
                    (now, owner, artifact_key, event_sha),
                )
            cursor.execute(
                f"SELECT {self._task_columns()} FROM {self.schema}.recompute_tasks WHERE owner_id=%s AND task_id=%s",
                (owner, task),
            )
            updated = cursor.fetchone()
            if updated is None:
                raise RuntimeError("completed recompute task disappeared")
            return self._task_from_row(updated)

        return self._transaction(operation)

    def list_recompute(
        self,
        owner_id: str,
        *,
        status: str | None = None,
        limit: int = 1000,
    ) -> tuple[RecomputeTask, ...]:
        owner = normalize_owner_id(owner_id)
        if not 1 <= limit <= 10_000:
            raise ValueError("limit is invalid")
        params: tuple[Any, ...]
        if status is None:
            where = "owner_id=%s"
            params = (owner, limit)
        else:
            state = _text(status, "status", 32).lower()
            if state not in _ALLOWED_TASK_STATUS:
                raise ValueError("unsupported recompute status")
            where = "owner_id=%s AND status=%s"
            params = (owner, state, limit)

        def operation(cursor: CursorLike) -> tuple[RecomputeTask, ...]:
            cursor.execute(
                f"SELECT {self._task_columns()} FROM {self.schema}.recompute_tasks WHERE {where} ORDER BY created_at DESC,task_id LIMIT %s",
                params,
            )
            return tuple(self._task_from_row(row) for row in cursor.fetchall())

        return self._transaction(operation)


__all__ = ["PostgresDependencyInvalidationStore"]
