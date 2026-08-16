"""Resolve authenticated principals to authoritative research workspace storage scopes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from tools.project_acl_store import ProjectACLStore, ProjectAccessScope
from tools.research_workspace import ResearchProject, ResearchSession
from tools.security import normalize_owner_id


class WorkspaceStore(Protocol):
    def get_project(self, owner_id: str, project_id: str) -> ResearchProject: ...
    def list_projects(self, owner_id: str, *, limit: int = 200) -> Sequence[ResearchProject]: ...
    def get_session(self, owner_id: str, session_id: str) -> ResearchSession: ...
    def list_sessions(self, owner_id: str, project_id: str, *, limit: int = 200) -> Sequence[ResearchSession]: ...


@dataclass(frozen=True)
class ProjectAccess:
    actor_id: str
    storage_owner_id: str
    project: ResearchProject
    role: str
    permissions: frozenset[str]


@dataclass(frozen=True)
class SessionAccess:
    project_access: ProjectAccess
    session: ResearchSession

    @property
    def storage_owner_id(self) -> str:
        return self.project_access.storage_owner_id


class ResearchAccessResolver:
    """Fail-closed project/session resolver for owner and collaborator access.

    Workspace storage remains partitioned by the project's immutable owner. The actor is
    never substituted into storage calls merely because they hold a collaboration grant.
    """

    def __init__(self, workspace: WorkspaceStore, acls: ProjectACLStore) -> None:
        self.workspace = workspace
        self.acls = acls

    def project(
        self,
        actor_id: str,
        project_id: str,
        *,
        permission: str = "project.read",
    ) -> ProjectAccess:
        actor = normalize_owner_id(actor_id)
        # Preserve legacy owner access and lazily seed the non-revocable owner grant.
        try:
            project = self.workspace.get_project(actor, project_id)
        except KeyError:
            project = None
        if project is not None:
            scope = self.acls.seed_owner(actor, project.project_id)
            self.acls.require(
                actor,
                project_owner_id=actor,
                project_id=project.project_id,
                permission=permission,
            )
            return ProjectAccess(actor, actor, project, scope.role, scope.permissions)

        scope = self.acls.resolve_project_scope(actor, project_id, permission=permission)
        project = self.workspace.get_project(scope.project_owner_id, scope.project_id)
        if project.owner_id != scope.project_owner_id:
            raise RuntimeError("resolved project owner does not match workspace authority")
        return ProjectAccess(actor, scope.project_owner_id, project, scope.role, scope.permissions)

    def session(
        self,
        actor_id: str,
        session_id: str,
        *,
        permission: str = "session.read",
    ) -> SessionAccess:
        actor = normalize_owner_id(actor_id)
        # Fast and backward-compatible path for sessions owned by the actor.
        try:
            session = self.workspace.get_session(actor, session_id)
        except KeyError:
            session = None
        if session is not None:
            project_access = self.project(actor, session.project_id, permission=permission)
            if project_access.storage_owner_id != session.owner_id:
                raise RuntimeError("session owner does not match resolved project authority")
            return SessionAccess(project_access, session)

        matches: list[SessionAccess] = []
        for scope in self.acls.accessible_scopes(actor, permission=permission, limit=10_000):
            if scope.project_owner_id == actor:
                continue
            try:
                candidate = self.workspace.get_session(scope.project_owner_id, session_id)
            except KeyError:
                continue
            if candidate.project_id != scope.project_id:
                continue
            project = self.workspace.get_project(scope.project_owner_id, scope.project_id)
            matches.append(
                SessionAccess(
                    ProjectAccess(actor, scope.project_owner_id, project, scope.role, scope.permissions),
                    candidate,
                )
            )
            if len(matches) > 1:
                break
        if not matches:
            raise PermissionError("session permission is not granted")
        if len(matches) != 1:
            raise RuntimeError("session identifier is ambiguous across accessible owners")
        return matches[0]

    def list_projects(
        self,
        actor_id: str,
        *,
        permission: str = "project.read",
        limit: int = 200,
    ) -> tuple[ProjectAccess, ...]:
        actor = normalize_owner_id(actor_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit is invalid")
        output: list[ProjectAccess] = []
        seen: set[tuple[str, str]] = set()
        for project in self.workspace.list_projects(actor, limit=limit):
            scope = self.acls.seed_owner(actor, project.project_id)
            if permission not in scope.permissions:
                continue
            output.append(ProjectAccess(actor, actor, project, scope.role, scope.permissions))
            seen.add((actor, project.project_id))
            if len(output) >= limit:
                return tuple(output)
        for scope in self.acls.accessible_scopes(actor, permission=permission, limit=10_000):
            key = (scope.project_owner_id, scope.project_id)
            if key in seen:
                continue
            try:
                project = self.workspace.get_project(scope.project_owner_id, scope.project_id)
            except KeyError:
                # Orphan ACL rows do not grant access to nonexistent data.
                continue
            output.append(
                ProjectAccess(actor, scope.project_owner_id, project, scope.role, scope.permissions)
            )
            seen.add(key)
            if len(output) >= limit:
                break
        return tuple(output)

    def list_sessions(
        self,
        actor_id: str,
        project_id: str,
        *,
        permission: str = "session.read",
        limit: int = 200,
    ) -> tuple[ResearchSession, ...]:
        access = self.project(actor_id, project_id, permission=permission)
        return tuple(
            self.workspace.list_sessions(
                access.storage_owner_id,
                access.project.project_id,
                limit=limit,
            )
        )


__all__ = [
    "ProjectAccess",
    "ResearchAccessResolver",
    "SessionAccess",
    "WorkspaceStore",
]
