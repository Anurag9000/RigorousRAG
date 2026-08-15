"""Typed research-workspace adapter over ``PostgresControlPlane``.

The SQL control plane intentionally returns strict JSON mappings. This adapter converts
those mappings back into the canonical research dataclasses and supplies the missing
project-scoped session listing required by the live research API.
"""

from __future__ import annotations

from typing import Any, Mapping

from tools.research_workspace import CorpusBinding, ResearchProject, ResearchSession, ResearchTurn
from tools.security import normalize_owner_id
from tools.sql_control_plane import ConnectionFactory, PostgresControlPlane


def _project(value: Mapping[str, Any]) -> ResearchProject:
    return ResearchProject(
        owner_id=str(value["owner_id"]),
        project_id=str(value["project_id"]),
        title=str(value["title"]),
        research_question=str(value["research_question"]),
        corpora=tuple(CorpusBinding(**item) for item in value.get("corpora", ())),
        tags=tuple(str(item) for item in value.get("tags", ())),
        created_at=float(value["created_at"]),
        archived=bool(value.get("archived", False)),
    )


def _session(value: Mapping[str, Any]) -> ResearchSession:
    closed = value.get("closed_at")
    return ResearchSession(
        owner_id=str(value["owner_id"]),
        project_id=str(value["project_id"]),
        session_id=str(value["session_id"]),
        turns=tuple(ResearchTurn(**item) for item in value.get("turns", ())),
        created_at=float(value["created_at"]),
        closed_at=float(closed) if closed is not None else None,
    )


class PostgresResearchWorkspaceStore:
    def __init__(
        self,
        connection_factory: ConnectionFactory,
        *,
        schema: str = "rigorousrag",
        initialize: bool = True,
    ) -> None:
        self.control = PostgresControlPlane(connection_factory, schema=schema)
        if initialize:
            self.control.initialize()

    def create_project(self, project: ResearchProject) -> None:
        self.control.create_project(project)

    def get_project(self, owner_id: str, project_id: str) -> ResearchProject:
        return _project(self.control.get_project(owner_id, project_id))

    def list_projects(self, owner_id: str, *, limit: int = 200) -> tuple[ResearchProject, ...]:
        return tuple(_project(item) for item in self.control.list_projects(owner_id, limit=limit))

    def put_session(
        self,
        session: ResearchSession,
        *,
        expected_fingerprint: str | None = None,
    ) -> None:
        self.get_project(session.owner_id, session.project_id)
        self.control.put_session(session, expected_fingerprint=expected_fingerprint)

    def get_session(self, owner_id: str, session_id: str) -> ResearchSession:
        return _session(self.control.get_session(owner_id, session_id))

    def list_sessions(
        self,
        owner_id: str,
        project_id: str,
        *,
        limit: int = 200,
    ) -> tuple[ResearchSession, ...]:
        owner = normalize_owner_id(owner_id)
        project = self.get_project(owner, project_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit is invalid")
        schema = self.control.schema

        def operation(cursor):
            cursor.execute(
                f"""SELECT payload::text FROM {schema}.research_sessions
                    WHERE owner_id=%s AND project_id=%s
                    ORDER BY updated_at DESC,session_id LIMIT %s""",
                (owner, project.project_id, limit),
            )
            rows = cursor.fetchall()
            output = []
            import json
            for row in rows:
                raw = row[0] if not isinstance(row, Mapping) else row["payload"]
                output.append(raw if isinstance(raw, Mapping) else json.loads(str(raw)))
            return tuple(output)

        return tuple(_session(item) for item in self.control._transaction(operation))


__all__ = ["PostgresResearchWorkspaceStore"]
