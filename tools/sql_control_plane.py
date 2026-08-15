"""PostgreSQL-backed shared control-plane persistence.

The implementation depends only on the Python DB-API protocol: applications inject a
connection factory (for example ``psycopg.connect``).  It persists immutable artifacts,
audit events, lifecycle transitions and research projects/sessions using PostgreSQL
transactions and optimistic versions. No driver is downloaded by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict
from typing import Any, Callable, Mapping, Protocol, Sequence

from tools.artifact_lineage import ArtifactRef, AuditEvent
from tools.model_lifecycle import LifecycleRecord
from tools.research_workspace import ResearchProject, ResearchSession
from tools.security import normalize_owner_id

_SCHEMA_VERSION = 1
_MAX_JSON_BYTES = 10_000_000


class CursorLike(Protocol):
    rowcount: int

    def execute(self, query: str, params: Sequence[Any] | None = None) -> Any: ...

    def fetchone(self) -> Sequence[Any] | Mapping[str, Any] | None: ...

    def fetchall(self) -> Sequence[Sequence[Any] | Mapping[str, Any]]: ...

    def close(self) -> None: ...


class ConnectionLike(Protocol):
    def cursor(self) -> CursorLike: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def close(self) -> None: ...


ConnectionFactory = Callable[[], ConnectionLike]


def _text(value: Any, label: str, maximum: int = 500, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    digest = value.strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


def _json(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("control-plane payload is not strict JSON") from exc
    if len(encoded.encode("utf-8")) > _MAX_JSON_BYTES:
        raise ValueError("control-plane JSON exceeds the size limit")
    return encoded


def _loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        raise RuntimeError("database JSON payload has unexpected type")
    return json.loads(value)


def _row_value(row: Sequence[Any] | Mapping[str, Any], key: str, index: int) -> Any:
    if isinstance(row, Mapping):
        return row[key]
    return row[index]


class PostgresControlPlane:
    """Small transactional control plane shared by governance subsystems."""

    def __init__(self, connection_factory: ConnectionFactory, *, schema: str = "rigorousrag") -> None:
        if not callable(connection_factory):
            raise TypeError("connection_factory must be callable")
        self._connect = connection_factory
        schema_name = _text(schema, "schema", 63)
        if not schema_name.replace("_", "").isalnum() or schema_name[0].isdigit():
            raise ValueError("schema must be a simple SQL identifier")
        self.schema = schema_name

    def initialize(self) -> None:
        schema = self.schema
        statements = (
            f"CREATE SCHEMA IF NOT EXISTS {schema}",
            f"""CREATE TABLE IF NOT EXISTS {schema}.schema_version (
                singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
                version INTEGER NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL
            )""",
            f"""CREATE TABLE IF NOT EXISTS {schema}.artifacts (
                owner_id TEXT NOT NULL,
                artifact_id CHAR(64) NOT NULL,
                kind TEXT NOT NULL,
                content_sha256 CHAR(64) NOT NULL,
                generation TEXT NOT NULL,
                metadata_sha256 CHAR(64) NOT NULL,
                payload JSONB NOT NULL,
                created_at DOUBLE PRECISION NOT NULL,
                PRIMARY KEY(owner_id, artifact_id)
            )""",
            f"""CREATE TABLE IF NOT EXISTS {schema}.audit_events (
                owner_id TEXT NOT NULL,
                event_id CHAR(64) NOT NULL,
                event_type TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                correlation_id TEXT NOT NULL,
                payload JSONB NOT NULL,
                occurred_at DOUBLE PRECISION NOT NULL,
                PRIMARY KEY(owner_id, event_id)
            )""",
            f"""CREATE INDEX IF NOT EXISTS audit_events_owner_time_idx
                ON {schema}.audit_events(owner_id, occurred_at, event_id)""",
            f"""CREATE TABLE IF NOT EXISTS {schema}.lifecycle_history (
                artifact_id TEXT NOT NULL,
                sequence BIGINT NOT NULL,
                kind TEXT NOT NULL,
                state TEXT NOT NULL,
                payload JSONB NOT NULL,
                created_at DOUBLE PRECISION NOT NULL,
                PRIMARY KEY(artifact_id, sequence)
            )""",
            f"""CREATE TABLE IF NOT EXISTS {schema}.lifecycle_current (
                artifact_id TEXT PRIMARY KEY,
                sequence BIGINT NOT NULL,
                state TEXT NOT NULL,
                payload JSONB NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL
            )""",
            f"""CREATE TABLE IF NOT EXISTS {schema}.research_projects (
                owner_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                fingerprint CHAR(64) NOT NULL,
                payload JSONB NOT NULL,
                created_at DOUBLE PRECISION NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL,
                version BIGINT NOT NULL DEFAULT 1,
                PRIMARY KEY(owner_id, project_id)
            )""",
            f"""CREATE TABLE IF NOT EXISTS {schema}.research_sessions (
                owner_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                fingerprint CHAR(64) NOT NULL,
                payload JSONB NOT NULL,
                created_at DOUBLE PRECISION NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL,
                version BIGINT NOT NULL DEFAULT 1,
                PRIMARY KEY(owner_id, session_id)
            )""",
            f"""CREATE INDEX IF NOT EXISTS research_sessions_project_idx
                ON {schema}.research_sessions(owner_id, project_id, updated_at)""",
        )
        connection = self._connect()
        cursor = connection.cursor()
        try:
            for statement in statements:
                cursor.execute(statement)
            cursor.execute(
                f"""INSERT INTO {schema}.schema_version(singleton, version, updated_at)
                    VALUES(TRUE, %s, %s)
                    ON CONFLICT(singleton) DO UPDATE SET version=EXCLUDED.version, updated_at=EXCLUDED.updated_at""",
                (_SCHEMA_VERSION, time.time()),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def _transaction(self, callback: Callable[[CursorLike], Any]) -> Any:
        connection = self._connect()
        cursor = connection.cursor()
        try:
            result = callback(cursor)
            connection.commit()
            return result
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def put_artifact(self, artifact: ArtifactRef) -> None:
        if not isinstance(artifact, ArtifactRef):
            raise TypeError("artifact must be ArtifactRef")
        payload = _json(asdict(artifact))

        def operation(cursor: CursorLike) -> None:
            cursor.execute(
                f"""INSERT INTO {self.schema}.artifacts
                    (owner_id, artifact_id, kind, content_sha256, generation, metadata_sha256, payload, created_at)
                    VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s)
                    ON CONFLICT(owner_id, artifact_id) DO NOTHING""",
                (
                    artifact.owner_id,
                    artifact.artifact_id,
                    artifact.kind,
                    artifact.content_sha256,
                    artifact.generation,
                    artifact.metadata_sha256,
                    payload,
                    artifact.created_at,
                ),
            )
            if cursor.rowcount == 0:
                cursor.execute(
                    f"SELECT payload::text FROM {self.schema}.artifacts WHERE owner_id=%s AND artifact_id=%s",
                    (artifact.owner_id, artifact.artifact_id),
                )
                row = cursor.fetchone()
                if row is None or _loads(_row_value(row, "payload", 0)) != json.loads(payload):
                    raise RuntimeError("artifact identity collision in SQL control plane")

        self._transaction(operation)

    def append_audit(self, event: AuditEvent) -> None:
        if not isinstance(event, AuditEvent):
            raise TypeError("event must be AuditEvent")
        payload = _json(asdict(event))

        def operation(cursor: CursorLike) -> None:
            cursor.execute(
                f"""INSERT INTO {self.schema}.audit_events
                    (owner_id,event_id,event_type,subject_id,correlation_id,payload,occurred_at)
                    VALUES(%s,%s,%s,%s,%s,%s::jsonb,%s)
                    ON CONFLICT(owner_id,event_id) DO NOTHING""",
                (
                    event.owner_id,
                    event.event_id,
                    event.event_type,
                    event.subject_id,
                    event.correlation_id,
                    payload,
                    event.occurred_at,
                ),
            )

        self._transaction(operation)

    def audit_page(
        self,
        owner_id: str,
        *,
        after: float = 0.0,
        limit: int = 200,
    ) -> tuple[Mapping[str, Any], ...]:
        owner = normalize_owner_id(owner_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit is invalid")
        timestamp = float(after)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("after is invalid")

        def operation(cursor: CursorLike) -> tuple[Mapping[str, Any], ...]:
            cursor.execute(
                f"""SELECT payload::text FROM {self.schema}.audit_events
                    WHERE owner_id=%s AND occurred_at>%s
                    ORDER BY occurred_at,event_id LIMIT %s""",
                (owner, timestamp, limit),
            )
            return tuple(_loads(_row_value(row, "payload", 0)) for row in cursor.fetchall())

        return self._transaction(operation)

    def append_lifecycle(self, record: LifecycleRecord) -> None:
        if not isinstance(record, LifecycleRecord):
            raise TypeError("record must be LifecycleRecord")
        payload = _json(asdict(record))

        def operation(cursor: CursorLike) -> None:
            cursor.execute(
                f"""INSERT INTO {self.schema}.lifecycle_history
                    (artifact_id,sequence,kind,state,payload,created_at)
                    VALUES(%s,%s,%s,%s,%s::jsonb,%s)
                    ON CONFLICT(artifact_id,sequence) DO NOTHING""",
                (
                    record.artifact.artifact_id,
                    record.sequence,
                    record.artifact.kind,
                    record.state,
                    payload,
                    record.created_at,
                ),
            )
            cursor.execute(
                f"""INSERT INTO {self.schema}.lifecycle_current
                    (artifact_id,sequence,state,payload,updated_at)
                    VALUES(%s,%s,%s,%s::jsonb,%s)
                    ON CONFLICT(artifact_id) DO UPDATE SET
                      sequence=EXCLUDED.sequence,
                      state=EXCLUDED.state,
                      payload=EXCLUDED.payload,
                      updated_at=EXCLUDED.updated_at
                    WHERE {self.schema}.lifecycle_current.sequence < EXCLUDED.sequence""",
                (
                    record.artifact.artifact_id,
                    record.sequence,
                    record.state,
                    payload,
                    time.time(),
                ),
            )

        self._transaction(operation)

    def create_project(self, project: ResearchProject) -> None:
        if not isinstance(project, ResearchProject):
            raise TypeError("project must be ResearchProject")
        payload = _json(asdict(project))
        now = time.time()

        def operation(cursor: CursorLike) -> None:
            cursor.execute(
                f"""INSERT INTO {self.schema}.research_projects
                    (owner_id,project_id,fingerprint,payload,created_at,updated_at,version)
                    VALUES(%s,%s,%s,%s::jsonb,%s,%s,1)
                    ON CONFLICT(owner_id,project_id) DO NOTHING""",
                (project.owner_id, project.project_id, project.fingerprint, payload, project.created_at, now),
            )
            if cursor.rowcount == 0:
                cursor.execute(
                    f"SELECT fingerprint FROM {self.schema}.research_projects WHERE owner_id=%s AND project_id=%s",
                    (project.owner_id, project.project_id),
                )
                row = cursor.fetchone()
                if row is None or _row_value(row, "fingerprint", 0) != project.fingerprint:
                    raise RuntimeError("research project already exists with different content")

        self._transaction(operation)

    def get_project(self, owner_id: str, project_id: str) -> Mapping[str, Any]:
        owner = normalize_owner_id(owner_id)
        project = _text(project_id, "project_id", 256)

        def operation(cursor: CursorLike) -> Mapping[str, Any]:
            cursor.execute(
                f"SELECT payload::text FROM {self.schema}.research_projects WHERE owner_id=%s AND project_id=%s",
                (owner, project),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(project)
            return _loads(_row_value(row, "payload", 0))

        return self._transaction(operation)

    def put_session(
        self,
        session: ResearchSession,
        *,
        expected_fingerprint: str | None = None,
    ) -> int:
        if not isinstance(session, ResearchSession):
            raise TypeError("session must be ResearchSession")
        payload = _json(asdict(session))
        expected = _sha(expected_fingerprint, "expected_fingerprint") if expected_fingerprint else None
        now = time.time()

        def operation(cursor: CursorLike) -> int:
            cursor.execute(
                f"""SELECT fingerprint,version FROM {self.schema}.research_sessions
                    WHERE owner_id=%s AND session_id=%s FOR UPDATE""",
                (session.owner_id, session.session_id),
            )
            row = cursor.fetchone()
            if row is None:
                if expected is not None:
                    raise RuntimeError("research session optimistic concurrency check failed")
                version = 1
                cursor.execute(
                    f"""INSERT INTO {self.schema}.research_sessions
                        (owner_id,session_id,project_id,fingerprint,payload,created_at,updated_at,version)
                        VALUES(%s,%s,%s,%s,%s::jsonb,%s,%s,%s)""",
                    (
                        session.owner_id,
                        session.session_id,
                        session.project_id,
                        session.fingerprint,
                        payload,
                        session.created_at,
                        now,
                        version,
                    ),
                )
                return version
            current_fingerprint = str(_row_value(row, "fingerprint", 0))
            current_version = int(_row_value(row, "version", 1))
            if expected is not None and current_fingerprint != expected:
                raise RuntimeError("research session optimistic concurrency check failed")
            next_version = current_version + 1
            cursor.execute(
                f"""UPDATE {self.schema}.research_sessions SET
                    project_id=%s,fingerprint=%s,payload=%s::jsonb,updated_at=%s,version=%s
                    WHERE owner_id=%s AND session_id=%s AND version=%s""",
                (
                    session.project_id,
                    session.fingerprint,
                    payload,
                    now,
                    next_version,
                    session.owner_id,
                    session.session_id,
                    current_version,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("research session concurrent update detected")
            return next_version

        return self._transaction(operation)

    def get_session(self, owner_id: str, session_id: str) -> Mapping[str, Any]:
        owner = normalize_owner_id(owner_id)
        identifier = _text(session_id, "session_id", 256)

        def operation(cursor: CursorLike) -> Mapping[str, Any]:
            cursor.execute(
                f"SELECT payload::text FROM {self.schema}.research_sessions WHERE owner_id=%s AND session_id=%s",
                (owner, identifier),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(identifier)
            return _loads(_row_value(row, "payload", 0))

        return self._transaction(operation)

    def list_projects(self, owner_id: str, *, limit: int = 200) -> tuple[Mapping[str, Any], ...]:
        owner = normalize_owner_id(owner_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit is invalid")

        def operation(cursor: CursorLike) -> tuple[Mapping[str, Any], ...]:
            cursor.execute(
                f"""SELECT payload::text FROM {self.schema}.research_projects
                    WHERE owner_id=%s ORDER BY updated_at DESC,project_id LIMIT %s""",
                (owner, limit),
            )
            return tuple(_loads(_row_value(row, "payload", 0)) for row in cursor.fetchall())

        return self._transaction(operation)

    @property
    def schema_fingerprint(self) -> str:
        payload = {
            "schema": self.schema,
            "version": _SCHEMA_VERSION,
            "tables": (
                "schema_version",
                "artifacts",
                "audit_events",
                "lifecycle_history",
                "lifecycle_current",
                "research_projects",
                "research_sessions",
            ),
        }
        return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()


__all__ = ["ConnectionFactory", "ConnectionLike", "CursorLike", "PostgresControlPlane"]
