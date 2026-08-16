"""PostgreSQL DB-API backend for immutable-version hydrology research artifacts.

Applications inject a DB-API connection factory. No driver or credential discovery occurs
here; semantics intentionally match the SQLite hydrology store.
"""
from __future__ import annotations

import json
import time
from typing import Any, Mapping, Sequence

from tools.hydrology_store import HydrologyArtifactEnvelope, HydrologyArtifactSummary, strict_json
from tools.security import normalize_owner_id
from tools.sql_control_plane import ConnectionFactory, CursorLike

_KINDS = frozenset({"topology", "engineering_package", "retrieval_plan", "evidence_projection"})


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if not cleaned or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _schema(value: str) -> str:
    cleaned = _text(value, "schema", 63)
    if not cleaned.replace("_", "").isalnum() or cleaned[0].isdigit():
        raise ValueError("schema must be a simple SQL identifier")
    return cleaned


def _kind(value: str) -> str:
    cleaned = _text(value, "kind", 64).lower()
    if cleaned not in _KINDS:
        raise ValueError("unsupported hydrology artifact kind")
    return cleaned


def _digest(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    cleaned = _text(value, label, 64).lower()
    if len(cleaned) != 64 or any(ch not in "0123456789abcdef" for ch in cleaned):
        raise ValueError(f"{label} must be SHA-256")
    return cleaned


def _row(value: Sequence[Any] | Mapping[str, Any], key: str, index: int) -> Any:
    return value[key] if isinstance(value, Mapping) else value[index]


def _payload(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        raise RuntimeError("database hydrology payload has unexpected type")
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise RuntimeError("database hydrology payload is not an object")
    return parsed


class PostgresHydrologyArtifactStore:
    def __init__(self, connection_factory: ConnectionFactory, *, schema: str = "rigorousrag", initialize: bool = False) -> None:
        if not callable(connection_factory):
            raise TypeError("connection_factory must be callable")
        self._connect = connection_factory
        self.schema = _schema(schema)
        if initialize:
            self.initialize()

    def _transaction(self, operation):
        connection = self._connect()
        cursor = connection.cursor()
        try:
            result = operation(cursor)
            connection.commit()
            return result
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def initialize(self) -> None:
        schema = self.schema
        statements = (
            f"CREATE SCHEMA IF NOT EXISTS {schema}",
            f"""CREATE TABLE IF NOT EXISTS {schema}.hydrology_artifact_versions (
                owner_id TEXT NOT NULL, project_id TEXT NOT NULL, kind TEXT NOT NULL, logical_id TEXT NOT NULL,
                version BIGINT NOT NULL, fingerprint CHAR(64) NOT NULL, schema_version INTEGER NOT NULL,
                payload JSONB NOT NULL, created_at DOUBLE PRECISION NOT NULL,
                PRIMARY KEY(owner_id, project_id, kind, logical_id, version)
            )""",
            f"CREATE INDEX IF NOT EXISTS hydrology_versions_fingerprint_idx ON {schema}.hydrology_artifact_versions(owner_id, project_id, fingerprint)",
            f"CREATE INDEX IF NOT EXISTS hydrology_versions_project_time_idx ON {schema}.hydrology_artifact_versions(owner_id, project_id, created_at DESC, kind, logical_id, version DESC)",
            f"""CREATE TABLE IF NOT EXISTS {schema}.hydrology_artifact_current (
                owner_id TEXT NOT NULL, project_id TEXT NOT NULL, kind TEXT NOT NULL, logical_id TEXT NOT NULL,
                version BIGINT NOT NULL, fingerprint CHAR(64) NOT NULL, updated_at DOUBLE PRECISION NOT NULL,
                PRIMARY KEY(owner_id, project_id, kind, logical_id),
                FOREIGN KEY(owner_id, project_id, kind, logical_id, version)
                  REFERENCES {schema}.hydrology_artifact_versions(owner_id, project_id, kind, logical_id, version)
            )""",
        )

        def operation(cursor: CursorLike) -> None:
            for statement in statements:
                cursor.execute(statement)

        self._transaction(operation)

    @staticmethod
    def _summary(row: Sequence[Any] | Mapping[str, Any], *, is_current: bool) -> HydrologyArtifactSummary:
        return HydrologyArtifactSummary(
            str(_row(row, "owner_id", 0)), str(_row(row, "project_id", 1)), str(_row(row, "kind", 2)),
            str(_row(row, "logical_id", 3)), str(_row(row, "fingerprint", 5)), int(_row(row, "version", 4)),
            float(_row(row, "created_at", 6)), is_current,
        )

    def put(self, envelope: HydrologyArtifactEnvelope, *, expected_current_fingerprint: str | None = None) -> HydrologyArtifactSummary:
        if not isinstance(envelope, HydrologyArtifactEnvelope):
            raise TypeError("envelope must be HydrologyArtifactEnvelope")
        expected = _digest(expected_current_fingerprint, "expected_current_fingerprint")
        payload, schema = strict_json(envelope.payload), self.schema

        def operation(cursor: CursorLike) -> HydrologyArtifactSummary:
            cursor.execute(
                f"SELECT version,fingerprint FROM {schema}.hydrology_artifact_current WHERE owner_id=%s AND project_id=%s AND kind=%s AND logical_id=%s FOR UPDATE",
                (envelope.owner_id, envelope.project_id, envelope.kind, envelope.logical_id),
            )
            current = cursor.fetchone()
            if current is None:
                if expected is not None:
                    raise RuntimeError("hydrology artifact optimistic concurrency check failed")
                next_version = 1
            else:
                current_version, current_fingerprint = int(_row(current, "version", 0)), str(_row(current, "fingerprint", 1))
                if expected is not None and current_fingerprint != expected:
                    raise RuntimeError("hydrology artifact optimistic concurrency check failed")
                if current_fingerprint == envelope.fingerprint:
                    cursor.execute(
                        f"SELECT owner_id,project_id,kind,logical_id,version,fingerprint,created_at FROM {schema}.hydrology_artifact_versions WHERE owner_id=%s AND project_id=%s AND kind=%s AND logical_id=%s AND version=%s",
                        (envelope.owner_id, envelope.project_id, envelope.kind, envelope.logical_id, current_version),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        raise RuntimeError("hydrology current pointer references a missing version")
                    return self._summary(row, is_current=True)
                next_version = current_version + 1
            cursor.execute(
                f"""INSERT INTO {schema}.hydrology_artifact_versions
                    (owner_id,project_id,kind,logical_id,version,fingerprint,schema_version,payload,created_at)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)""",
                (envelope.owner_id, envelope.project_id, envelope.kind, envelope.logical_id, next_version, envelope.fingerprint, envelope.schema_version, payload, envelope.created_at),
            )
            now = time.time()
            if current is None:
                cursor.execute(
                    f"INSERT INTO {schema}.hydrology_artifact_current (owner_id,project_id,kind,logical_id,version,fingerprint,updated_at) VALUES(%s,%s,%s,%s,%s,%s,%s)",
                    (envelope.owner_id, envelope.project_id, envelope.kind, envelope.logical_id, next_version, envelope.fingerprint, now),
                )
            else:
                cursor.execute(
                    f"UPDATE {schema}.hydrology_artifact_current SET version=%s,fingerprint=%s,updated_at=%s WHERE owner_id=%s AND project_id=%s AND kind=%s AND logical_id=%s AND version=%s",
                    (next_version, envelope.fingerprint, now, envelope.owner_id, envelope.project_id, envelope.kind, envelope.logical_id, int(_row(current, "version", 0))),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("hydrology artifact concurrent update detected")
            return HydrologyArtifactSummary(envelope.owner_id, envelope.project_id, envelope.kind, envelope.logical_id, envelope.fingerprint, next_version, envelope.created_at, True)

        return self._transaction(operation)

    def get(self, owner_id: str, project_id: str, kind: str, logical_id: str, *, fingerprint: str | None = None) -> HydrologyArtifactEnvelope:
        owner = normalize_owner_id(owner_id)
        project, artifact_kind, logical = _text(project_id, "project_id", 256), _kind(kind), _text(logical_id, "logical_id", 500)
        requested, schema = _digest(fingerprint, "fingerprint"), self.schema

        def operation(cursor: CursorLike) -> HydrologyArtifactEnvelope:
            if requested is None:
                cursor.execute(
                    f"""SELECT v.fingerprint,v.schema_version,v.payload::text,v.created_at FROM {schema}.hydrology_artifact_current c
                        JOIN {schema}.hydrology_artifact_versions v ON v.owner_id=c.owner_id AND v.project_id=c.project_id AND v.kind=c.kind AND v.logical_id=c.logical_id AND v.version=c.version
                        WHERE c.owner_id=%s AND c.project_id=%s AND c.kind=%s AND c.logical_id=%s""",
                    (owner, project, artifact_kind, logical),
                )
            else:
                cursor.execute(
                    f"SELECT fingerprint,schema_version,payload::text,created_at FROM {schema}.hydrology_artifact_versions WHERE owner_id=%s AND project_id=%s AND kind=%s AND logical_id=%s AND fingerprint=%s ORDER BY version DESC LIMIT 1",
                    (owner, project, artifact_kind, logical, requested),
                )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(logical)
            return HydrologyArtifactEnvelope(owner, project, artifact_kind, logical, str(_row(row, "fingerprint", 0)), _payload(_row(row, "payload", 2)), int(_row(row, "schema_version", 1)), float(_row(row, "created_at", 3)))

        return self._transaction(operation)

    def list(self, owner_id: str, project_id: str, *, kind: str | None = None, include_history: bool = False, limit: int = 200) -> tuple[HydrologyArtifactSummary, ...]:
        owner, project = normalize_owner_id(owner_id), _text(project_id, "project_id", 256)
        artifact_kind = _kind(kind) if kind is not None else None
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 5000:
            raise ValueError("limit is invalid")
        schema = self.schema

        def operation(cursor: CursorLike) -> tuple[HydrologyArtifactSummary, ...]:
            params: list[Any] = [owner, project]
            kind_clause = ""
            if artifact_kind is not None:
                kind_clause = " AND v.kind=%s"
                params.append(artifact_kind)
            if include_history:
                cursor.execute(
                    f"""SELECT v.owner_id,v.project_id,v.kind,v.logical_id,v.version,v.fingerprint,v.created_at,CASE WHEN c.version=v.version THEN TRUE ELSE FALSE END AS is_current
                        FROM {schema}.hydrology_artifact_versions v LEFT JOIN {schema}.hydrology_artifact_current c ON c.owner_id=v.owner_id AND c.project_id=v.project_id AND c.kind=v.kind AND c.logical_id=v.logical_id
                        WHERE v.owner_id=%s AND v.project_id=%s{kind_clause} ORDER BY v.created_at DESC,v.kind,v.logical_id,v.version DESC LIMIT %s""",
                    (*params, limit),
                )
                return tuple(self._summary(row, is_current=bool(_row(row, "is_current", 7))) for row in cursor.fetchall())
            cursor.execute(
                f"""SELECT v.owner_id,v.project_id,v.kind,v.logical_id,v.version,v.fingerprint,v.created_at FROM {schema}.hydrology_artifact_current c
                    JOIN {schema}.hydrology_artifact_versions v ON v.owner_id=c.owner_id AND v.project_id=c.project_id AND v.kind=c.kind AND v.logical_id=c.logical_id AND v.version=c.version
                    WHERE v.owner_id=%s AND v.project_id=%s{kind_clause} ORDER BY v.created_at DESC,v.kind,v.logical_id LIMIT %s""",
                (*params, limit),
            )
            return tuple(self._summary(row, is_current=True) for row in cursor.fetchall())

        return self._transaction(operation)


__all__ = ["PostgresHydrologyArtifactStore"]
