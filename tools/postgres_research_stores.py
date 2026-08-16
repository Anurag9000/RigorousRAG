"""PostgreSQL implementations of core research artifact and collaboration stores.

These classes intentionally subclass the SQLite reference stores so existing router and
operator type guards continue to hold. They do not import a PostgreSQL driver; callers
inject a DB-API connection factory. All identities are computed with the same canonical
helpers as the reference stores, so switching persistence backends does not change IDs.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable, Mapping, Sequence

from tools.artifact_replacements import (
    ArtifactReplacement,
    ArtifactReplacementStore,
    _canonical as _replacement_canonical,
)
from tools.dependency_invalidation import DependencyRef
from tools.models import AgentAnswer, Citation
from tools.project_acl import ProjectGrant, role_allows, role_permissions
from tools.project_acl_store import ProjectACLStore, ProjectAccessScope, _text as _acl_text
from tools.research_capsule import ResearchCapsule
from tools.research_capsule_store import (
    ResearchCapsuleStore,
    StoredResearchCapsule,
    _capsule_from_json,
    _capsule_payload,
    _canonical as _capsule_canonical,
    _sha as _capsule_sha,
    _text as _capsule_text,
)
from tools.research_report import ResearchReport
from tools.research_report_store import (
    ResearchReportStore,
    StoredResearchReport,
    _json as _report_json,
    _report_from_payload,
    _report_payload,
    _sha as _report_sha,
    _text as _report_text,
)
from tools.research_result_store import (
    ResearchResultStore,
    StoredResearchResult,
    _canonical as _result_canonical,
    _citation_identity,
    _sha as _result_sha,
    _text as _result_text,
)
from tools.security import normalize_owner_id
from tools.sql_control_plane import ConnectionFactory, CursorLike


def _schema_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("schema must be a string")
    schema = value.strip()
    if not schema or len(schema) > 63 or not schema.replace("_", "").isalnum() or schema[0].isdigit():
        raise ValueError("schema must be a simple SQL identifier")
    return schema


def _row(row: Sequence[Any] | Mapping[str, Any], key: str, index: int) -> Any:
    return row[key] if isinstance(row, Mapping) else row[index]


def _loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise RuntimeError("database JSON value has unexpected type")
    return json.loads(value)


class _PostgresMixin:
    def __init__(
        self,
        connection_factory: ConnectionFactory,
        *,
        schema: str = "rigorousrag",
        initialize: bool = True,
    ) -> None:
        if not callable(connection_factory):
            raise TypeError("connection_factory must be callable")
        self._connection_factory = connection_factory
        self.schema = _schema_name(schema)
        if initialize:
            self._initialize_postgres()

    def _transaction(self, callback: Callable[[CursorLike], Any]) -> Any:
        connection = self._connection_factory()
        cursor = connection.cursor()
        try:
            output = callback(cursor)
            connection.commit()
            return output
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def _initialize_tables(self, statements: Sequence[str]) -> None:
        def operation(cursor: CursorLike) -> None:
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
            for statement in statements:
                cursor.execute(statement)

        self._transaction(operation)


class PostgresResearchResultStore(_PostgresMixin, ResearchResultStore):
    def _initialize_postgres(self) -> None:
        schema = self.schema
        self._initialize_tables(
            (
                f"""CREATE TABLE IF NOT EXISTS {schema}.research_results (
                    owner_id TEXT NOT NULL,
                    result_id CHAR(64) NOT NULL,
                    query_sha256 CHAR(64) NOT NULL,
                    answer TEXT NOT NULL,
                    citations_json JSONB NOT NULL,
                    warnings_json JSONB NOT NULL,
                    metadata_json JSONB NOT NULL,
                    strategy TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY(owner_id,result_id)
                )""",
                f"CREATE INDEX IF NOT EXISTS research_results_owner_time_idx ON {schema}.research_results(owner_id,created_at DESC,result_id)",
                f"CREATE INDEX IF NOT EXISTS research_results_owner_query_idx ON {schema}.research_results(owner_id,query_sha256,created_at DESC)",
            )
        )

    def put(
        self,
        owner_id: str,
        *,
        query_sha256: str,
        answer: AgentAnswer,
        strategy: str,
        model: str = "",
    ) -> StoredResearchResult:
        owner = normalize_owner_id(owner_id)
        query_digest = _result_sha(query_sha256, "query_sha256")
        if not isinstance(answer, AgentAnswer):
            raise TypeError("answer must be AgentAnswer")
        final_text = _result_text(answer.answer, "answer", 100_000, allow_empty=True)
        citations = tuple(answer.citations or ())
        if len(citations) > 500 or any(not isinstance(item, Citation) for item in citations):
            raise ValueError("authoritative citations are invalid")
        citation_payload = [item.model_dump(exclude_none=True) for item in citations]
        warnings = tuple(
            _result_text(item, "warning", 5000)
            for item in (answer.warnings or ())[:100]
        )
        metadata = dict(answer.metadata or {})
        strategy_value = _result_text(strategy, "strategy", 128)
        model_value = _result_text(model, "model", 256, allow_empty=True)
        identity_payload = {
            "owner_id": owner,
            "query_sha256": query_digest,
            "answer": final_text,
            "citations": [_citation_identity(item) for item in citations],
            "warnings": warnings,
            "metadata": metadata,
            "strategy": strategy_value,
            "model": model_value,
        }
        result_id = hashlib.sha256(_result_canonical(identity_payload).encode("utf-8")).hexdigest()
        created_at = time.time()
        citations_json = _result_canonical(citation_payload)
        warnings_json = _result_canonical(list(warnings))
        metadata_json = _result_canonical(metadata)

        def operation(cursor: CursorLike) -> float:
            cursor.execute(
                f"""INSERT INTO {self.schema}.research_results
                    (owner_id,result_id,query_sha256,answer,citations_json,warnings_json,metadata_json,strategy,model,created_at)
                    VALUES(%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s)
                    ON CONFLICT(owner_id,result_id) DO NOTHING""",
                (
                    owner,
                    result_id,
                    query_digest,
                    final_text,
                    citations_json,
                    warnings_json,
                    metadata_json,
                    strategy_value,
                    model_value,
                    created_at,
                ),
            )
            cursor.execute(
                f"""SELECT answer,citations_json::text,warnings_json::text,metadata_json::text,
                           strategy,model,created_at
                    FROM {self.schema}.research_results
                    WHERE owner_id=%s AND result_id=%s""",
                (owner, result_id),
            )
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError("research result insert disappeared")
            if (
                str(_row(row, "answer", 0)) != final_text
                or _loads(_row(row, "citations_json", 1)) != citation_payload
                or _loads(_row(row, "warnings_json", 2)) != list(warnings)
                or _loads(_row(row, "metadata_json", 3)) != metadata
                or str(_row(row, "strategy", 4)) != strategy_value
                or str(_row(row, "model", 5)) != model_value
            ):
                raise RuntimeError("research result identity collision")
            return float(_row(row, "created_at", 6))

        created_at = self._transaction(operation)
        return StoredResearchResult(
            owner,
            result_id,
            query_digest,
            final_text,
            citations,
            warnings,
            metadata,
            strategy_value,
            model_value,
            created_at,
        )

    def get(self, owner_id: str, result_id: str) -> StoredResearchResult:
        owner = normalize_owner_id(owner_id)
        identifier = _result_sha(result_id, "result_id")

        def operation(cursor: CursorLike) -> StoredResearchResult:
            cursor.execute(
                f"""SELECT query_sha256,answer,citations_json::text,warnings_json::text,
                           metadata_json::text,strategy,model,created_at
                    FROM {self.schema}.research_results
                    WHERE owner_id=%s AND result_id=%s""",
                (owner, identifier),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(identifier)
            citations = tuple(Citation(**item) for item in _loads(_row(row, "citations_json", 2)))
            warnings = tuple(str(item) for item in _loads(_row(row, "warnings_json", 3)))
            metadata = _loads(_row(row, "metadata_json", 4))
            return StoredResearchResult(
                owner_id=owner,
                result_id=identifier,
                query_sha256=_result_sha(str(_row(row, "query_sha256", 0)), "query_sha256"),
                answer=str(_row(row, "answer", 1)),
                citations=citations,
                warnings=warnings,
                metadata=metadata,
                strategy=str(_row(row, "strategy", 5)),
                model=str(_row(row, "model", 6)),
                created_at=float(_row(row, "created_at", 7)),
            )

        return self._transaction(operation)

    def list(self, owner_id: str, *, limit: int = 100) -> tuple[StoredResearchResult, ...]:
        owner = normalize_owner_id(owner_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit is invalid")

        def operation(cursor: CursorLike) -> tuple[str, ...]:
            cursor.execute(
                f"SELECT result_id FROM {self.schema}.research_results WHERE owner_id=%s ORDER BY created_at DESC,result_id LIMIT %s",
                (owner, limit),
            )
            return tuple(str(_row(row, "result_id", 0)) for row in cursor.fetchall())

        return tuple(self.get(owner, identifier) for identifier in self._transaction(operation))


class PostgresResearchReportStore(_PostgresMixin, ResearchReportStore):
    def _initialize_postgres(self) -> None:
        schema = self.schema
        self._initialize_tables(
            (
                f"""CREATE TABLE IF NOT EXISTS {schema}.research_reports (
                    owner_id TEXT NOT NULL,
                    report_id CHAR(64) NOT NULL,
                    result_id CHAR(64) NOT NULL,
                    project_id TEXT NOT NULL,
                    report_fingerprint CHAR(64) NOT NULL,
                    payload_json JSONB NOT NULL,
                    created_at DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY(owner_id,report_id)
                )""",
                f"CREATE INDEX IF NOT EXISTS research_reports_owner_time_idx ON {schema}.research_reports(owner_id,created_at DESC,report_id)",
                f"CREATE INDEX IF NOT EXISTS research_reports_owner_project_idx ON {schema}.research_reports(owner_id,project_id,created_at DESC)",
                f"CREATE INDEX IF NOT EXISTS research_reports_owner_result_idx ON {schema}.research_reports(owner_id,result_id,created_at DESC)",
            )
        )

    def put(
        self,
        owner_id: str,
        *,
        result_id: str,
        project_id: str,
        report: ResearchReport,
    ) -> StoredResearchReport:
        owner = normalize_owner_id(owner_id)
        result = _report_sha(result_id, "result_id")
        project = _report_text(project_id, "project_id")
        if not isinstance(report, ResearchReport):
            raise TypeError("report must be ResearchReport")
        payload = _report_payload(report)
        payload_json = _report_json(payload)
        identity = {
            "owner_id": owner,
            "result_id": result,
            "project_id": project,
            "report_fingerprint": report.fingerprint,
        }
        report_id = hashlib.sha256(_report_json(identity).encode("utf-8")).hexdigest()
        created_at = time.time()

        def operation(cursor: CursorLike) -> float:
            cursor.execute(
                f"""INSERT INTO {self.schema}.research_reports
                    (owner_id,report_id,result_id,project_id,report_fingerprint,payload_json,created_at)
                    VALUES(%s,%s,%s,%s,%s,%s::jsonb,%s)
                    ON CONFLICT(owner_id,report_id) DO NOTHING""",
                (owner, report_id, result, project, report.fingerprint, payload_json, created_at),
            )
            cursor.execute(
                f"SELECT payload_json::text,created_at FROM {self.schema}.research_reports WHERE owner_id=%s AND report_id=%s",
                (owner, report_id),
            )
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError("research report insert disappeared")
            if _loads(_row(row, "payload_json", 0)) != payload:
                raise RuntimeError("research report identity collision")
            return float(_row(row, "created_at", 1))

        created_at = self._transaction(operation)
        return StoredResearchReport(owner, report_id, result, project, report, created_at)

    def get(self, owner_id: str, report_id: str) -> StoredResearchReport:
        owner = normalize_owner_id(owner_id)
        identifier = _report_sha(report_id, "report_id")

        def operation(cursor: CursorLike) -> StoredResearchReport:
            cursor.execute(
                f"SELECT result_id,project_id,payload_json::text,created_at FROM {self.schema}.research_reports WHERE owner_id=%s AND report_id=%s",
                (owner, identifier),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(identifier)
            report = _report_from_payload(_loads(_row(row, "payload_json", 2)))
            return StoredResearchReport(
                owner,
                identifier,
                _report_sha(str(_row(row, "result_id", 0)), "result_id"),
                str(_row(row, "project_id", 1)),
                report,
                float(_row(row, "created_at", 3)),
            )

        return self._transaction(operation)

    def list(
        self,
        owner_id: str,
        *,
        project_id: str | None = None,
        limit: int = 100,
    ) -> tuple[StoredResearchReport, ...]:
        owner = normalize_owner_id(owner_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit is invalid")
        project = _report_text(project_id, "project_id") if project_id is not None else None

        def operation(cursor: CursorLike) -> tuple[str, ...]:
            if project is None:
                cursor.execute(
                    f"SELECT report_id FROM {self.schema}.research_reports WHERE owner_id=%s ORDER BY created_at DESC,report_id LIMIT %s",
                    (owner, limit),
                )
            else:
                cursor.execute(
                    f"SELECT report_id FROM {self.schema}.research_reports WHERE owner_id=%s AND project_id=%s ORDER BY created_at DESC,report_id LIMIT %s",
                    (owner, project, limit),
                )
            return tuple(str(_row(row, "report_id", 0)) for row in cursor.fetchall())

        return tuple(self.get(owner, identifier) for identifier in self._transaction(operation))


class PostgresResearchCapsuleStore(_PostgresMixin, ResearchCapsuleStore):
    def _initialize_postgres(self) -> None:
        schema = self.schema
        self._initialize_tables(
            (
                f"""CREATE TABLE IF NOT EXISTS {schema}.research_capsules (
                    owner_id TEXT NOT NULL,
                    capsule_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    result_id CHAR(64) NOT NULL,
                    fingerprint CHAR(64) NOT NULL,
                    manifest_json JSONB NOT NULL,
                    supersedes_capsule_id TEXT NOT NULL,
                    created_at DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY(owner_id,capsule_id),
                    UNIQUE(owner_id,fingerprint)
                )""",
                f"CREATE INDEX IF NOT EXISTS research_capsules_owner_project_idx ON {schema}.research_capsules(owner_id,project_id,created_at DESC,capsule_id)",
                f"CREATE INDEX IF NOT EXISTS research_capsules_owner_result_idx ON {schema}.research_capsules(owner_id,result_id,created_at DESC,capsule_id)",
            )
        )

    @staticmethod
    def _stored(owner: str, row: Sequence[Any] | Mapping[str, Any]) -> StoredResearchCapsule:
        capsule = _capsule_from_json(str(_row(row, "manifest_json", 5)))
        capsule_id = str(_row(row, "capsule_id", 0))
        project_id = str(_row(row, "project_id", 1))
        if capsule.capsule_id != capsule_id or capsule.project_id != project_id:
            raise RuntimeError("research capsule row identity mismatch")
        fingerprint = _capsule_sha(str(_row(row, "fingerprint", 4)), "fingerprint")
        if capsule.fingerprint != fingerprint:
            raise RuntimeError("research capsule row fingerprint mismatch")
        return StoredResearchCapsule(
            owner_id=owner,
            project_id=project_id,
            session_id=str(_row(row, "session_id", 2)),
            result_id=_capsule_sha(str(_row(row, "result_id", 3)), "result_id"),
            capsule=capsule,
            supersedes_capsule_id=str(_row(row, "supersedes_capsule_id", 6)),
        )

    def put(
        self,
        owner_id: str,
        *,
        project_id: str,
        session_id: str,
        result_id: str,
        capsule: ResearchCapsule,
        supersedes_capsule_id: str = "",
    ) -> StoredResearchCapsule:
        owner = normalize_owner_id(owner_id)
        project = _capsule_text(project_id, "project_id", 256)
        session = _capsule_text(session_id, "session_id", 256)
        result = _capsule_sha(result_id, "result_id")
        supersedes = _capsule_text(
            supersedes_capsule_id, "supersedes_capsule_id", 256, allow_empty=True
        )
        if not isinstance(capsule, ResearchCapsule):
            raise TypeError("capsule must be ResearchCapsule")
        if capsule.project_id != project or capsule.run_id != result:
            raise ValueError("capsule project/run identities do not match the durable binding")
        manifest_json = _capsule_canonical(_capsule_payload(capsule))
        fingerprint = capsule.fingerprint

        def operation(cursor: CursorLike) -> StoredResearchCapsule:
            if supersedes:
                cursor.execute(
                    f"SELECT project_id FROM {self.schema}.research_capsules WHERE owner_id=%s AND capsule_id=%s",
                    (owner, supersedes),
                )
                predecessor = cursor.fetchone()
                if predecessor is None:
                    raise KeyError(supersedes)
                if str(_row(predecessor, "project_id", 0)) != project:
                    raise ValueError("a capsule may only supersede a capsule in the same project")
            cursor.execute(
                f"""SELECT capsule_id,project_id,session_id,result_id,fingerprint,
                           manifest_json::text,supersedes_capsule_id
                    FROM {self.schema}.research_capsules
                    WHERE owner_id=%s AND fingerprint=%s""",
                (owner, fingerprint),
            )
            existing = cursor.fetchone()
            if existing is not None:
                return self._stored(owner, existing)
            cursor.execute(
                f"SELECT fingerprint FROM {self.schema}.research_capsules WHERE owner_id=%s AND capsule_id=%s",
                (owner, capsule.capsule_id),
            )
            if cursor.fetchone() is not None:
                raise RuntimeError("research capsule ID collision")
            cursor.execute(
                f"""INSERT INTO {self.schema}.research_capsules
                    (owner_id,capsule_id,project_id,session_id,result_id,fingerprint,manifest_json,supersedes_capsule_id,created_at)
                    VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s)""",
                (
                    owner,
                    capsule.capsule_id,
                    project,
                    session,
                    result,
                    fingerprint,
                    manifest_json,
                    supersedes,
                    capsule.created_at,
                ),
            )
            return StoredResearchCapsule(owner, project, session, result, capsule, supersedes)

        return self._transaction(operation)

    def get(self, owner_id: str, capsule_id: str) -> StoredResearchCapsule:
        owner = normalize_owner_id(owner_id)
        identifier = _capsule_text(capsule_id, "capsule_id", 256)

        def operation(cursor: CursorLike) -> StoredResearchCapsule:
            cursor.execute(
                f"""SELECT capsule_id,project_id,session_id,result_id,fingerprint,
                           manifest_json::text,supersedes_capsule_id
                    FROM {self.schema}.research_capsules WHERE owner_id=%s AND capsule_id=%s""",
                (owner, identifier),
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(identifier)
            return self._stored(owner, row)

        return self._transaction(operation)

    def list(
        self,
        owner_id: str,
        *,
        project_id: str | None = None,
        result_id: str | None = None,
        limit: int = 100,
    ) -> tuple[StoredResearchCapsule, ...]:
        owner = normalize_owner_id(owner_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit is invalid")
        clauses = ["owner_id=%s"]
        params: list[Any] = [owner]
        if project_id is not None:
            clauses.append("project_id=%s")
            params.append(_capsule_text(project_id, "project_id", 256))
        if result_id is not None:
            clauses.append("result_id=%s")
            params.append(_capsule_sha(result_id, "result_id"))
        params.append(limit)

        def operation(cursor: CursorLike) -> tuple[StoredResearchCapsule, ...]:
            cursor.execute(
                f"""SELECT capsule_id,project_id,session_id,result_id,fingerprint,
                           manifest_json::text,supersedes_capsule_id
                    FROM {self.schema}.research_capsules WHERE {' AND '.join(clauses)}
                    ORDER BY created_at DESC,capsule_id LIMIT %s""",
                tuple(params),
            )
            return tuple(self._stored(owner, row) for row in cursor.fetchall())

        return self._transaction(operation)


class PostgresProjectACLStore(_PostgresMixin, ProjectACLStore):
    def _initialize_postgres(self) -> None:
        schema = self.schema
        self._initialize_tables(
            (
                f"""CREATE TABLE IF NOT EXISTS {schema}.project_acl_grants (
                    project_owner_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    granted_by TEXT NOT NULL,
                    granted_at DOUBLE PRECISION NOT NULL,
                    expires_at DOUBLE PRECISION,
                    revoked_at DOUBLE PRECISION,
                    grant_fingerprint CHAR(64) NOT NULL,
                    PRIMARY KEY(project_owner_id,project_id,principal_id)
                )""",
                f"CREATE INDEX IF NOT EXISTS project_acl_principal_idx ON {schema}.project_acl_grants(principal_id,revoked_at,expires_at,project_id,project_owner_id)",
                f"CREATE INDEX IF NOT EXISTS project_acl_project_idx ON {schema}.project_acl_grants(project_owner_id,project_id,revoked_at,principal_id)",
            )
        )

    @staticmethod
    def _scope(row: Sequence[Any] | Mapping[str, Any]) -> ProjectAccessScope:
        expires = _row(row, "expires_at", 6)
        return ProjectAccessScope(
            project_owner_id=str(_row(row, "project_owner_id", 0)),
            project_id=str(_row(row, "project_id", 1)),
            principal_id=str(_row(row, "principal_id", 2)),
            role=str(_row(row, "role", 3)),
            granted_by=str(_row(row, "granted_by", 4)),
            granted_at=float(_row(row, "granted_at", 5)),
            expires_at=float(expires) if expires is not None else None,
        )

    def seed_owner(self, project_owner_id: str, project_id: str) -> ProjectAccessScope:
        owner = normalize_owner_id(project_owner_id)
        project = _acl_text(project_id, "project_id")
        now = time.time()
        grant = ProjectGrant(project, owner, "owner", owner, now)

        def operation(cursor: CursorLike) -> ProjectAccessScope:
            cursor.execute(
                f"""INSERT INTO {self.schema}.project_acl_grants
                    (project_owner_id,project_id,principal_id,role,granted_by,granted_at,expires_at,revoked_at,grant_fingerprint)
                    VALUES(%s,%s,%s,'owner',%s,%s,NULL,NULL,%s)
                    ON CONFLICT(project_owner_id,project_id,principal_id) DO NOTHING""",
                (owner, project, owner, owner, now, grant.fingerprint),
            )
            cursor.execute(
                f"""SELECT project_owner_id,project_id,principal_id,role,granted_by,granted_at,expires_at,revoked_at
                    FROM {self.schema}.project_acl_grants
                    WHERE project_owner_id=%s AND project_id=%s AND principal_id=%s""",
                (owner, project, owner),
            )
            row = cursor.fetchone()
            if row is None or str(_row(row, "role", 3)) != "owner" or _row(row, "revoked_at", 7) is not None:
                raise RuntimeError("project owner ACL grant is corrupt")
            return self._scope(row)

        return self._transaction(operation)

    def permission(
        self,
        principal_id: str,
        *,
        project_owner_id: str,
        project_id: str,
        permission: str,
        now: float | None = None,
    ) -> bool:
        principal = normalize_owner_id(principal_id)
        owner = normalize_owner_id(project_owner_id)
        project = _acl_text(project_id, "project_id")
        selected_permission = _acl_text(permission, "permission", 100)
        current = time.time() if now is None else float(now)

        def operation(cursor: CursorLike) -> bool:
            cursor.execute(
                f"""SELECT role,expires_at,revoked_at FROM {self.schema}.project_acl_grants
                    WHERE project_owner_id=%s AND project_id=%s AND principal_id=%s""",
                (owner, project, principal),
            )
            row = cursor.fetchone()
            if row is None or _row(row, "revoked_at", 2) is not None:
                return False
            expires = _row(row, "expires_at", 1)
            if expires is not None and current >= float(expires):
                return False
            return role_allows(str(_row(row, "role", 0)), selected_permission)

        return self._transaction(operation)

    def require(
        self,
        principal_id: str,
        *,
        project_owner_id: str,
        project_id: str,
        permission: str,
    ) -> None:
        if not self.permission(
            principal_id,
            project_owner_id=project_owner_id,
            project_id=project_id,
            permission=permission,
        ):
            raise PermissionError("project permission is not granted")

    def grant(
        self,
        actor_id: str,
        *,
        project_owner_id: str,
        project_id: str,
        principal_id: str,
        role: str,
        expires_at: float | None = None,
    ) -> ProjectAccessScope:
        actor = normalize_owner_id(actor_id)
        owner = normalize_owner_id(project_owner_id)
        project = _acl_text(project_id, "project_id")
        principal = normalize_owner_id(principal_id)
        selected_role = _acl_text(role, "role", 32).lower()
        role_permissions(selected_role)
        self.seed_owner(owner, project)
        self.require(actor, project_owner_id=owner, project_id=project, permission="acl.manage")
        if principal == owner and selected_role != "owner":
            raise ValueError("project owner role may not be downgraded")
        now = time.time()
        expires = float(expires_at) if expires_at is not None else None
        if expires is not None and expires <= now:
            raise ValueError("ACL expiration must be in the future")
        grant = ProjectGrant(project, principal, selected_role, actor, now, expires)

        def operation(cursor: CursorLike) -> None:
            cursor.execute(
                f"""INSERT INTO {self.schema}.project_acl_grants
                    (project_owner_id,project_id,principal_id,role,granted_by,granted_at,expires_at,revoked_at,grant_fingerprint)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,NULL,%s)
                    ON CONFLICT(project_owner_id,project_id,principal_id) DO UPDATE SET
                      role=EXCLUDED.role,granted_by=EXCLUDED.granted_by,granted_at=EXCLUDED.granted_at,
                      expires_at=EXCLUDED.expires_at,revoked_at=NULL,grant_fingerprint=EXCLUDED.grant_fingerprint""",
                (owner, project, principal, selected_role, actor, now, expires, grant.fingerprint),
            )

        self._transaction(operation)
        return ProjectAccessScope(owner, project, principal, selected_role, actor, now, expires)

    def revoke(
        self,
        actor_id: str,
        *,
        project_owner_id: str,
        project_id: str,
        principal_id: str,
    ) -> bool:
        actor = normalize_owner_id(actor_id)
        owner = normalize_owner_id(project_owner_id)
        project = _acl_text(project_id, "project_id")
        principal = normalize_owner_id(principal_id)
        self.seed_owner(owner, project)
        self.require(actor, project_owner_id=owner, project_id=project, permission="acl.manage")
        if principal == owner:
            raise ValueError("project owner grant may not be revoked")

        def operation(cursor: CursorLike) -> bool:
            cursor.execute(
                f"""UPDATE {self.schema}.project_acl_grants SET revoked_at=%s
                    WHERE project_owner_id=%s AND project_id=%s AND principal_id=%s AND revoked_at IS NULL""",
                (time.time(), owner, project, principal),
            )
            return bool(cursor.rowcount)

        return self._transaction(operation)

    def grants(
        self,
        actor_id: str,
        *,
        project_owner_id: str,
        project_id: str,
        include_revoked: bool = False,
    ) -> tuple[ProjectAccessScope, ...]:
        actor = normalize_owner_id(actor_id)
        owner = normalize_owner_id(project_owner_id)
        project = _acl_text(project_id, "project_id")
        self.seed_owner(owner, project)
        self.require(actor, project_owner_id=owner, project_id=project, permission="acl.manage")

        def operation(cursor: CursorLike) -> tuple[ProjectAccessScope, ...]:
            cursor.execute(
                f"""SELECT project_owner_id,project_id,principal_id,role,granted_by,granted_at,expires_at,revoked_at
                    FROM {self.schema}.project_acl_grants WHERE project_owner_id=%s AND project_id=%s
                    {'' if include_revoked else 'AND revoked_at IS NULL'} ORDER BY principal_id""",
                (owner, project),
            )
            return tuple(self._scope(row) for row in cursor.fetchall())

        return self._transaction(operation)

    def accessible_scopes(
        self,
        principal_id: str,
        *,
        permission: str = "project.read",
        limit: int = 1000,
    ) -> tuple[ProjectAccessScope, ...]:
        principal = normalize_owner_id(principal_id)
        selected_permission = _acl_text(permission, "permission", 100)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("limit is invalid")
        now = time.time()

        def operation(cursor: CursorLike) -> tuple[ProjectAccessScope, ...]:
            cursor.execute(
                f"""SELECT project_owner_id,project_id,principal_id,role,granted_by,granted_at,expires_at,revoked_at
                    FROM {self.schema}.project_acl_grants
                    WHERE principal_id=%s AND revoked_at IS NULL AND (expires_at IS NULL OR expires_at>%s)
                    ORDER BY project_id,project_owner_id LIMIT %s""",
                (principal, now, limit),
            )
            output = []
            for row in cursor.fetchall():
                scope = self._scope(row)
                if role_allows(scope.role, selected_permission):
                    output.append(scope)
            return tuple(output)

        return self._transaction(operation)

    def resolve_project_scope(
        self,
        principal_id: str,
        project_id: str,
        *,
        permission: str = "project.read",
    ) -> ProjectAccessScope:
        principal = normalize_owner_id(principal_id)
        project = _acl_text(project_id, "project_id")
        candidates = tuple(
            scope
            for scope in self.accessible_scopes(principal, permission=permission, limit=10_000)
            if scope.project_id == project
        )
        if not candidates:
            raise PermissionError("project permission is not granted")
        if len(candidates) != 1:
            raise RuntimeError("project identifier is ambiguous across accessible owners")
        return candidates[0]


class PostgresArtifactReplacementStore(_PostgresMixin, ArtifactReplacementStore):
    def _initialize_postgres(self) -> None:
        schema = self.schema
        self._initialize_tables(
            (
                f"""CREATE TABLE IF NOT EXISTS {schema}.artifact_replacements (
                    owner_id TEXT NOT NULL,
                    old_kind TEXT NOT NULL,
                    old_id TEXT NOT NULL,
                    old_key CHAR(64) NOT NULL,
                    new_kind TEXT NOT NULL,
                    new_id TEXT NOT NULL,
                    new_key CHAR(64) NOT NULL,
                    reason TEXT NOT NULL,
                    event_sha256 CHAR(64) NOT NULL,
                    replacement_sha256 CHAR(64) NOT NULL,
                    created_at DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY(owner_id,replacement_sha256)
                )""",
                f"CREATE INDEX IF NOT EXISTS artifact_replacements_old_idx ON {schema}.artifact_replacements(owner_id,old_key,created_at DESC,replacement_sha256)",
                f"CREATE INDEX IF NOT EXISTS artifact_replacements_new_idx ON {schema}.artifact_replacements(owner_id,new_key,created_at DESC,replacement_sha256)",
            )
        )

    def put(
        self,
        owner_id: str,
        *,
        old: DependencyRef,
        new: DependencyRef,
        reason: str,
        triggering_event_sha256: str,
    ) -> ArtifactReplacement:
        owner = normalize_owner_id(owner_id)
        if not isinstance(old, DependencyRef) or not isinstance(new, DependencyRef):
            raise TypeError("old/new must be DependencyRef")
        if old.kind != new.kind:
            raise ValueError("replacement artifacts must preserve artifact kind")
        if old == new:
            raise ValueError("replacement must identify a new artifact")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 5000:
            raise ValueError("reason is invalid")
        event = triggering_event_sha256.strip().lower()
        if len(event) != 64 or any(ch not in "0123456789abcdef" for ch in event):
            raise ValueError("triggering_event_sha256 must be SHA-256")
        reason_value = reason.strip()
        payload = {
            "owner_id": owner,
            "old": {"kind": old.kind, "resource_id": old.resource_id},
            "new": {"kind": new.kind, "resource_id": new.resource_id},
            "reason": reason_value,
            "event_sha256": event,
        }
        replacement_sha = hashlib.sha256(_replacement_canonical(payload)).hexdigest()
        created_at = time.time()

        def operation(cursor: CursorLike) -> None:
            cursor.execute(
                f"""INSERT INTO {self.schema}.artifact_replacements
                    (owner_id,old_kind,old_id,old_key,new_kind,new_id,new_key,reason,event_sha256,replacement_sha256,created_at)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT(owner_id,replacement_sha256) DO NOTHING""",
                (
                    owner,
                    old.kind,
                    old.resource_id,
                    old.key,
                    new.kind,
                    new.resource_id,
                    new.key,
                    reason_value,
                    event,
                    replacement_sha,
                    created_at,
                ),
            )

        self._transaction(operation)
        return ArtifactReplacement(old, new, reason_value, event, replacement_sha, created_at)

    @staticmethod
    def _from_row(row: Sequence[Any] | Mapping[str, Any]) -> ArtifactReplacement:
        return ArtifactReplacement(
            old=DependencyRef(str(_row(row, "old_kind", 0)), str(_row(row, "old_id", 1))),
            new=DependencyRef(str(_row(row, "new_kind", 2)), str(_row(row, "new_id", 3))),
            reason=str(_row(row, "reason", 4)),
            triggering_event_sha256=str(_row(row, "event_sha256", 5)),
            replacement_sha256=str(_row(row, "replacement_sha256", 6)),
            created_at=float(_row(row, "created_at", 7)),
        )

    def latest(self, owner_id: str, old: DependencyRef) -> ArtifactReplacement | None:
        owner = normalize_owner_id(owner_id)

        def operation(cursor: CursorLike) -> ArtifactReplacement | None:
            cursor.execute(
                f"""SELECT old_kind,old_id,new_kind,new_id,reason,event_sha256,replacement_sha256,created_at
                    FROM {self.schema}.artifact_replacements WHERE owner_id=%s AND old_key=%s
                    ORDER BY created_at DESC,replacement_sha256 DESC LIMIT 1""",
                (owner, old.key),
            )
            row = cursor.fetchone()
            return None if row is None else self._from_row(row)

        return self._transaction(operation)

    def chain(
        self,
        owner_id: str,
        start: DependencyRef,
        *,
        max_depth: int = 64,
    ) -> tuple[ArtifactReplacement, ...]:
        if not 1 <= max_depth <= 256:
            raise ValueError("max_depth is invalid")
        output: list[ArtifactReplacement] = []
        current = start
        seen = {current.key}
        for _ in range(max_depth):
            replacement = self.latest(owner_id, current)
            if replacement is None:
                break
            if replacement.new.key in seen:
                raise RuntimeError("replacement lineage contains a cycle")
            output.append(replacement)
            seen.add(replacement.new.key)
            current = replacement.new
        return tuple(output)

    def current(self, owner_id: str, start: DependencyRef) -> DependencyRef:
        chain = self.chain(owner_id, start)
        return chain[-1].new if chain else start


__all__ = [
    "PostgresArtifactReplacementStore",
    "PostgresProjectACLStore",
    "PostgresResearchCapsuleStore",
    "PostgresResearchReportStore",
    "PostgresResearchResultStore",
]
