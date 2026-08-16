"""Durable project-owner-scoped collaboration ACL storage and resolution."""

from __future__ import annotations

import os
import sqlite3
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.project_acl import ProjectGrant, role_allows, role_permissions
from tools.security import normalize_owner_id

_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _safe_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    if len(str(absolute)) > 4096:
        raise ValueError("project ACL database path is too long")
    for component in (absolute, *absolute.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        if stat.S_ISLNK(metadata.st_mode) or bool(attributes & _WINDOWS_REPARSE_POINT):
            raise RuntimeError("project ACL path may not traverse symlinks/reparse points")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    return absolute


def _text(value: Any, label: str, maximum: int = 256) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} is invalid")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if not cleaned or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise ValueError(f"{label} is invalid")
    return cleaned


@dataclass(frozen=True)
class ProjectAccessScope:
    project_owner_id: str
    project_id: str
    principal_id: str
    role: str
    granted_by: str
    granted_at: float
    expires_at: float | None

    @property
    def permissions(self) -> frozenset[str]:
        return role_permissions(self.role)


class ProjectACLStore:
    def __init__(self, path: str | Path) -> None:
        self.path = _safe_path(path)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS project_acl_grants (
                    project_owner_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    granted_by TEXT NOT NULL,
                    granted_at REAL NOT NULL,
                    expires_at REAL,
                    revoked_at REAL,
                    grant_fingerprint CHAR(64) NOT NULL,
                    PRIMARY KEY(project_owner_id, project_id, principal_id)
                );
                CREATE INDEX IF NOT EXISTS project_acl_principal_idx
                  ON project_acl_grants(principal_id, revoked_at, expires_at, project_id, project_owner_id);
                CREATE INDEX IF NOT EXISTS project_acl_project_idx
                  ON project_acl_grants(project_owner_id, project_id, revoked_at, principal_id);
                """
            )

    @staticmethod
    def _scope(row: sqlite3.Row) -> ProjectAccessScope:
        expires = row["expires_at"]
        return ProjectAccessScope(
            project_owner_id=str(row["project_owner_id"]),
            project_id=str(row["project_id"]),
            principal_id=str(row["principal_id"]),
            role=str(row["role"]),
            granted_by=str(row["granted_by"]),
            granted_at=float(row["granted_at"]),
            expires_at=float(expires) if expires is not None else None,
        )

    def seed_owner(self, project_owner_id: str, project_id: str) -> ProjectAccessScope:
        owner = normalize_owner_id(project_owner_id)
        project = _text(project_id, "project_id")
        now = time.time()
        grant = ProjectGrant(project, owner, "owner", owner, now)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """SELECT * FROM project_acl_grants
                       WHERE project_owner_id=? AND project_id=? AND principal_id=?""",
                    (owner, project, owner),
                ).fetchone()
                if row is None:
                    connection.execute(
                        """INSERT INTO project_acl_grants
                           (project_owner_id,project_id,principal_id,role,granted_by,
                            granted_at,expires_at,revoked_at,grant_fingerprint)
                           VALUES(?,?,?,?,?,?,NULL,NULL,?)""",
                        (owner, project, owner, "owner", owner, now, grant.fingerprint),
                    )
                elif str(row["role"]) != "owner" or row["revoked_at"] is not None:
                    raise RuntimeError("project owner ACL grant is corrupt")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return ProjectAccessScope(owner, project, owner, "owner", owner, now, None)

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
        project = _text(project_id, "project_id")
        selected_permission = _text(permission, "permission", 100)
        current = time.time() if now is None else float(now)
        with self._connect() as connection:
            row = connection.execute(
                """SELECT role,expires_at,revoked_at FROM project_acl_grants
                   WHERE project_owner_id=? AND project_id=? AND principal_id=?""",
                (owner, project, principal),
            ).fetchone()
        if row is None or row["revoked_at"] is not None:
            return False
        expires = row["expires_at"]
        if expires is not None and current >= float(expires):
            return False
        return role_allows(str(row["role"]), selected_permission)

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
        project = _text(project_id, "project_id")
        principal = normalize_owner_id(principal_id)
        selected_role = _text(role, "role", 32).lower()
        role_permissions(selected_role)
        self.seed_owner(owner, project)
        self.require(
            actor,
            project_owner_id=owner,
            project_id=project,
            permission="acl.manage",
        )
        if principal == owner and selected_role != "owner":
            raise ValueError("project owner role may not be downgraded")
        now = time.time()
        if expires_at is not None and float(expires_at) <= now:
            raise ValueError("ACL expiration must be in the future")
        grant = ProjectGrant(project, principal, selected_role, actor, now, expires_at)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """INSERT INTO project_acl_grants
                       (project_owner_id,project_id,principal_id,role,granted_by,
                        granted_at,expires_at,revoked_at,grant_fingerprint)
                       VALUES(?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(project_owner_id,project_id,principal_id) DO UPDATE SET
                         role=excluded.role,
                         granted_by=excluded.granted_by,
                         granted_at=excluded.granted_at,
                         expires_at=excluded.expires_at,
                         revoked_at=NULL,
                         grant_fingerprint=excluded.grant_fingerprint""",
                    (
                        owner,
                        project,
                        principal,
                        selected_role,
                        actor,
                        now,
                        float(expires_at) if expires_at is not None else None,
                        None,
                        grant.fingerprint,
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return ProjectAccessScope(owner, project, principal, selected_role, actor, now, expires_at)

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
        project = _text(project_id, "project_id")
        principal = normalize_owner_id(principal_id)
        self.seed_owner(owner, project)
        self.require(
            actor,
            project_owner_id=owner,
            project_id=project,
            permission="acl.manage",
        )
        if principal == owner:
            raise ValueError("project owner grant may not be revoked")
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """UPDATE project_acl_grants SET revoked_at=?
                   WHERE project_owner_id=? AND project_id=? AND principal_id=?
                     AND revoked_at IS NULL""",
                (time.time(), owner, project, principal),
            )
        return bool(cursor.rowcount)

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
        project = _text(project_id, "project_id")
        self.seed_owner(owner, project)
        self.require(
            actor,
            project_owner_id=owner,
            project_id=project,
            permission="acl.manage",
        )
        query = (
            "SELECT * FROM project_acl_grants WHERE project_owner_id=? AND project_id=?"
            + ("" if include_revoked else " AND revoked_at IS NULL")
            + " ORDER BY principal_id"
        )
        with self._connect() as connection:
            rows = connection.execute(query, (owner, project)).fetchall()
        return tuple(self._scope(row) for row in rows)

    def accessible_scopes(
        self,
        principal_id: str,
        *,
        permission: str = "project.read",
        limit: int = 1000,
    ) -> tuple[ProjectAccessScope, ...]:
        principal = normalize_owner_id(principal_id)
        selected_permission = _text(permission, "permission", 100)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("limit is invalid")
        now = time.time()
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM project_acl_grants
                   WHERE principal_id=? AND revoked_at IS NULL
                     AND (expires_at IS NULL OR expires_at>?)
                   ORDER BY project_id,project_owner_id LIMIT ?""",
                (principal, now, limit),
            ).fetchall()
        return tuple(
            scope
            for row in rows
            if role_allows((scope := self._scope(row)).role, selected_permission)
        )

    def resolve_project_scope(
        self,
        principal_id: str,
        project_id: str,
        *,
        permission: str = "project.read",
    ) -> ProjectAccessScope:
        principal = normalize_owner_id(principal_id)
        project = _text(project_id, "project_id")
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


__all__ = ["ProjectACLStore", "ProjectAccessScope"]
