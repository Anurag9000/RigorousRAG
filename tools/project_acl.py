"""Project collaboration ACL semantics with explicit least-privilege roles."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

from tools.security import normalize_owner_id

_ROLE_PERMISSIONS = {
    "viewer": frozenset(
        {
            "project.read",
            "session.read",
            "result.read",
            "report.read",
            "capsule.read",
        }
    ),
    "reviewer": frozenset(
        {
            "project.read",
            "session.read",
            "result.read",
            "report.read",
            "capsule.read",
            "claim.review",
        }
    ),
    "editor": frozenset(
        {
            "project.read",
            "project.write",
            "session.read",
            "session.write",
            "result.read",
            "research.execute",
            "report.read",
            "report.write",
            "capsule.read",
            "capsule.write",
            "claim.review",
        }
    ),
    "owner": frozenset(
        {
            "project.read",
            "project.write",
            "session.read",
            "session.write",
            "result.read",
            "research.execute",
            "report.read",
            "report.write",
            "capsule.read",
            "capsule.write",
            "replay.manage",
            "claim.review",
            "acl.manage",
            "project.delete",
        }
    ),
}


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} is invalid")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if not cleaned or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise ValueError(f"{label} is invalid")
    return cleaned


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def role_permissions(role: str) -> frozenset[str]:
    selected = _text(role, "role", 32).lower()
    try:
        return _ROLE_PERMISSIONS[selected]
    except KeyError as exc:
        raise ValueError("unsupported ACL role") from exc


def role_allows(role: str, permission: str) -> bool:
    selected_permission = _text(permission, "permission", 100)
    return selected_permission in role_permissions(role)


@dataclass(frozen=True)
class ProjectGrant:
    project_id: str
    principal_id: str
    role: str
    granted_by: str
    granted_at: float
    expires_at: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_id", _text(self.project_id, "project_id", 256))
        object.__setattr__(self, "principal_id", normalize_owner_id(self.principal_id))
        object.__setattr__(self, "granted_by", normalize_owner_id(self.granted_by))
        role = _text(self.role, "role", 32).lower()
        role_permissions(role)
        object.__setattr__(self, "role", role)
        granted_at = float(self.granted_at)
        object.__setattr__(self, "granted_at", granted_at)
        if self.expires_at is not None:
            expires = float(self.expires_at)
            if expires <= granted_at:
                raise ValueError("ACL expiration must follow grant time")
            object.__setattr__(self, "expires_at", expires)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


class ProjectACL:
    """In-memory policy/reference implementation used by deterministic callers."""

    def __init__(self, *, project_id: str, owner_id: str) -> None:
        self.project_id = _text(project_id, "project_id", 256)
        self.owner_id = normalize_owner_id(owner_id)
        self._grants = {
            self.owner_id: ProjectGrant(
                self.project_id,
                self.owner_id,
                "owner",
                self.owner_id,
                time.time(),
            )
        }

    def grant(
        self,
        *,
        actor_id: str,
        principal_id: str,
        role: str,
        expires_at: float | None = None,
    ) -> ProjectGrant:
        self.require(actor_id, "acl.manage")
        principal = normalize_owner_id(principal_id)
        selected_role = _text(role, "role", 32).lower()
        role_permissions(selected_role)
        if principal == self.owner_id and selected_role != "owner":
            raise ValueError("project owner role may not be downgraded")
        grant = ProjectGrant(
            self.project_id,
            principal,
            selected_role,
            normalize_owner_id(actor_id),
            time.time(),
            expires_at,
        )
        self._grants[principal] = grant
        return grant

    def revoke(self, *, actor_id: str, principal_id: str) -> None:
        self.require(actor_id, "acl.manage")
        principal = normalize_owner_id(principal_id)
        if principal == self.owner_id:
            raise ValueError("project owner grant may not be revoked")
        self._grants.pop(principal, None)

    def permissions(self, principal_id: str, *, now: float | None = None) -> frozenset[str]:
        principal = normalize_owner_id(principal_id)
        grant = self._grants.get(principal)
        if grant is None:
            return frozenset()
        current = time.time() if now is None else float(now)
        if grant.expires_at is not None and current >= grant.expires_at:
            return frozenset()
        return role_permissions(grant.role)

    def require(self, principal_id: str, permission: str) -> None:
        selected = _text(permission, "permission", 100)
        if selected not in self.permissions(principal_id):
            raise PermissionError("project permission is not granted")

    def grants(self) -> tuple[ProjectGrant, ...]:
        return tuple(sorted(self._grants.values(), key=lambda item: item.principal_id))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical([asdict(item) for item in self.grants()])).hexdigest()


__all__ = [
    "ProjectACL",
    "ProjectGrant",
    "role_allows",
    "role_permissions",
]
