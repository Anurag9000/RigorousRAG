"""Digest-only operator RBAC/ABAC authorization for high-risk control-plane actions.

Authentication is deliberately external: an identity-provider adapter verifies credentials
and emits :class:`VerifiedPrincipalAssertion`. This module binds the verified principal to
repository-owned roles/scopes, evaluates an immutable policy, emits short-lived decisions,
and persists promoted policy revisions. Raw operator names, emails, tokens and credentials
are neither required nor stored here.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_ACTIONS = frozenset(
    {
        "lifecycle.reconcile",
        "lifecycle.reindex",
        "lifecycle.adopt",
        "lifecycle.delete",
        "migration.promote",
        "migration.rollback",
        "runtime.promote",
        "runtime.rollback",
        "region.failover",
        "region.failback",
        "residency.promote",
        "graph.restore",
        "graph.cutover",
        "backup.delete",
        "audit.export",
        "legal_hold.create",
        "legal_hold.release",
        "key.rotate",
        "key.reencrypt",
        "dataset.approve",
        "model.approve",
        "experiment.promote",
    }
)
_RESOURCE_TYPES = frozenset(
    {
        "document",
        "migration",
        "runtime_stack",
        "region_authority",
        "residency_policy",
        "evidence_graph",
        "backup",
        "audit_log",
        "legal_hold",
        "key",
        "dataset",
        "model",
        "experiment",
    }
)
_HEX = frozenset("0123456789abcdef")
_MAX_DECISION_TTL_SECONDS = 3600.0


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _sha(value: Any, label: str) -> str:
    selected = _text(value, label, 64).lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _time(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative") from exc
    if not math.isfinite(selected) or selected < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return selected


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _action(value: Any) -> str:
    selected = _text(value, "action", 100).lower()
    if selected not in _ACTIONS:
        raise ValueError(f"unsupported operator action {selected!r}")
    return selected


def _resource_type(value: Any) -> str:
    selected = _text(value, "resource_type", 100).lower()
    if selected not in _RESOURCE_TYPES:
        raise ValueError(f"unsupported resource_type {selected!r}")
    return selected


@dataclass(frozen=True)
class VerifiedPrincipalAssertion:
    """Post-authentication assertion emitted only after an external verifier succeeds."""

    principal_sha256: str
    issuer_id: str
    verification_provider_id: str
    assertion_sha256: str
    authenticated_at: float
    valid_until: float
    methods: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "principal_sha256", _sha(self.principal_sha256, "principal_sha256"))
        object.__setattr__(self, "issuer_id", _text(self.issuer_id, "issuer_id", 300))
        object.__setattr__(self, "verification_provider_id", _text(self.verification_provider_id, "verification_provider_id", 300))
        object.__setattr__(self, "assertion_sha256", _sha(self.assertion_sha256, "assertion_sha256"))
        authenticated = _time(self.authenticated_at, "authenticated_at")
        valid_until = _time(self.valid_until, "valid_until")
        if valid_until <= authenticated:
            raise ValueError("valid_until must be later than authenticated_at")
        object.__setattr__(self, "authenticated_at", authenticated)
        object.__setattr__(self, "valid_until", valid_until)
        methods = tuple(sorted({_text(value, "authentication method", 100).lower() for value in self.methods}))
        if not methods:
            raise ValueError("methods must be non-empty")
        object.__setattr__(self, "methods", methods)

    @property
    def assertion_binding_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-verified-principal/v1", **asdict(self)})

    def current_at(self, now: float) -> bool:
        instant = _time(now, "now")
        return self.authenticated_at <= instant < self.valid_until


@dataclass(frozen=True)
class PrincipalRoleBinding:
    principal_sha256: str
    role_id: str
    owner_ids: tuple[str, ...] = ()
    domain_ids: tuple[str, ...] = ()
    all_owners: bool = False
    all_domains: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "principal_sha256", _sha(self.principal_sha256, "principal_sha256"))
        object.__setattr__(self, "role_id", _text(self.role_id, "role_id", 200))
        owners = tuple(sorted({_text(value, "owner_id", 500) for value in self.owner_ids}))
        domains = tuple(sorted({_text(value, "domain_id", 500) for value in self.domain_ids}))
        if not isinstance(self.all_owners, bool) or not isinstance(self.all_domains, bool):
            raise ValueError("scope flags must be boolean")
        if not self.all_owners and not owners:
            raise ValueError("binding requires owner_ids unless all_owners is true")
        if not self.all_domains and not domains:
            raise ValueError("binding requires domain_ids unless all_domains is true")
        object.__setattr__(self, "owner_ids", owners)
        object.__setattr__(self, "domain_ids", domains)

    @property
    def binding_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-principal-role-binding/v1", **asdict(self)})

    def covers(self, owner_id: str, domain_id: str) -> bool:
        owner = _text(owner_id, "owner_id")
        domain = _text(domain_id, "domain_id")
        return (self.all_owners or owner in self.owner_ids) and (self.all_domains or domain in self.domain_ids)


@dataclass(frozen=True)
class RolePermission:
    role_id: str
    actions: tuple[str, ...]
    resource_types: tuple[str, ...]
    require_mfa: bool = False
    require_reason_digest: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "role_id", _text(self.role_id, "role_id", 200))
        actions = tuple(sorted({_action(value) for value in self.actions}))
        resources = tuple(sorted({_resource_type(value) for value in self.resource_types}))
        if not actions or not resources:
            raise ValueError("role permission requires actions and resource types")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "resource_types", resources)
        if not isinstance(self.require_mfa, bool) or not isinstance(self.require_reason_digest, bool):
            raise ValueError("permission requirement flags must be boolean")

    @property
    def permission_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-role-permission/v1", **asdict(self)})


@dataclass(frozen=True)
class OperatorAuthorizationPolicy:
    policy_id: str
    bindings: tuple[PrincipalRoleBinding, ...]
    permissions: tuple[RolePermission, ...]
    mfa_methods: tuple[str, ...] = ("mfa", "webauthn", "totp")
    decision_ttl_seconds: float = 300.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id", 300))
        bindings = tuple(self.bindings)
        permissions = tuple(self.permissions)
        if not bindings or any(not isinstance(value, PrincipalRoleBinding) for value in bindings):
            raise ValueError("bindings must be a non-empty PrincipalRoleBinding sequence")
        if not permissions or any(not isinstance(value, RolePermission) for value in permissions):
            raise ValueError("permissions must be a non-empty RolePermission sequence")
        if len({value.binding_sha256 for value in bindings}) != len(bindings):
            raise ValueError("bindings contain duplicates")
        if len({value.permission_sha256 for value in permissions}) != len(permissions):
            raise ValueError("permissions contain duplicates")
        roles = {value.role_id for value in permissions}
        if any(value.role_id not in roles for value in bindings):
            raise ValueError("binding references a role without permissions")
        object.__setattr__(self, "bindings", tuple(sorted(bindings, key=lambda value: value.binding_sha256)))
        object.__setattr__(self, "permissions", tuple(sorted(permissions, key=lambda value: value.permission_sha256)))
        methods = tuple(sorted({_text(value, "mfa method", 100).lower() for value in self.mfa_methods}))
        if not methods:
            raise ValueError("mfa_methods must be non-empty")
        object.__setattr__(self, "mfa_methods", methods)
        ttl = _time(self.decision_ttl_seconds, "decision_ttl_seconds")
        if not 0.0 < ttl <= _MAX_DECISION_TTL_SECONDS:
            raise ValueError("decision_ttl_seconds is outside the supported range")
        object.__setattr__(self, "decision_ttl_seconds", ttl)

    @property
    def policy_sha256(self) -> str:
        return _digest(
            {
                "schema": "rigorousrag-operator-authorization-policy/v1",
                "policy_id": self.policy_id,
                "bindings": [asdict(value) for value in self.bindings],
                "permissions": [asdict(value) for value in self.permissions],
                "mfa_methods": self.mfa_methods,
                "decision_ttl_seconds": self.decision_ttl_seconds,
            }
        )


@dataclass(frozen=True)
class OperatorAuthorizationRequest:
    owner_id: str
    domain_id: str
    action: str
    resource_type: str
    resource_sha256: str
    reason_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", _text(self.owner_id, "owner_id"))
        object.__setattr__(self, "domain_id", _text(self.domain_id, "domain_id"))
        object.__setattr__(self, "action", _action(self.action))
        object.__setattr__(self, "resource_type", _resource_type(self.resource_type))
        object.__setattr__(self, "resource_sha256", _sha(self.resource_sha256, "resource_sha256"))
        if self.reason_sha256 is not None:
            object.__setattr__(self, "reason_sha256", _sha(self.reason_sha256, "reason_sha256"))

    @property
    def request_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-operator-authorization-request/v1", **asdict(self)})


@dataclass(frozen=True)
class OperatorAuthorizationDecision:
    request_sha256: str
    principal_sha256: str
    assertion_binding_sha256: str
    policy_sha256: str
    matched_binding_sha256: str | None
    matched_permission_sha256: str | None
    authorized: bool
    reason_codes: tuple[str, ...]
    decided_at: float
    valid_until: float
    decision_sha256: str

    def __post_init__(self) -> None:
        for name in ("request_sha256", "principal_sha256", "assertion_binding_sha256", "policy_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        for name in ("matched_binding_sha256", "matched_permission_sha256"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _sha(value, name))
        if not isinstance(self.authorized, bool):
            raise ValueError("authorized must be boolean")
        reasons = tuple(sorted({_text(value, "reason code", 200) for value in self.reason_codes}))
        if self.authorized and reasons:
            raise ValueError("authorized decision may not contain failure reasons")
        if not self.authorized and not reasons:
            raise ValueError("denied decision requires reason codes")
        if self.authorized and (self.matched_binding_sha256 is None or self.matched_permission_sha256 is None):
            raise ValueError("authorized decision requires matched binding and permission")
        object.__setattr__(self, "reason_codes", reasons)
        decided = _time(self.decided_at, "decided_at")
        valid = _time(self.valid_until, "valid_until")
        if valid <= decided:
            raise ValueError("valid_until must be later than decided_at")
        object.__setattr__(self, "decided_at", decided)
        object.__setattr__(self, "valid_until", valid)
        expected = _digest(self._payload())
        provided = _sha(self.decision_sha256, "decision_sha256")
        if expected != provided:
            raise ValueError("decision_sha256 does not match authorization decision")
        object.__setattr__(self, "decision_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-operator-authorization-decision/v1",
            "request_sha256": self.request_sha256,
            "principal_sha256": self.principal_sha256,
            "assertion_binding_sha256": self.assertion_binding_sha256,
            "policy_sha256": self.policy_sha256,
            "matched_binding_sha256": self.matched_binding_sha256,
            "matched_permission_sha256": self.matched_permission_sha256,
            "authorized": self.authorized,
            "reason_codes": self.reason_codes,
            "decided_at": self.decided_at,
            "valid_until": self.valid_until,
        }

    def current_at(self, now: float) -> bool:
        instant = _time(now, "now")
        return self.decided_at <= instant < self.valid_until


def authorize_operator_action(
    assertion: VerifiedPrincipalAssertion,
    request: OperatorAuthorizationRequest,
    *,
    policy: OperatorAuthorizationPolicy,
    now: float,
) -> OperatorAuthorizationDecision:
    if not isinstance(assertion, VerifiedPrincipalAssertion):
        raise ValueError("assertion must be VerifiedPrincipalAssertion")
    if not isinstance(request, OperatorAuthorizationRequest):
        raise ValueError("request must be OperatorAuthorizationRequest")
    if not isinstance(policy, OperatorAuthorizationPolicy):
        raise ValueError("policy must be OperatorAuthorizationPolicy")
    instant = _time(now, "now")
    reasons: list[str] = []
    matched_binding: PrincipalRoleBinding | None = None
    matched_permission: RolePermission | None = None
    if not assertion.current_at(instant):
        reasons.append("authentication_assertion_not_current")
    else:
        bindings = [
            value for value in policy.bindings
            if value.principal_sha256 == assertion.principal_sha256 and value.covers(request.owner_id, request.domain_id)
        ]
        if not bindings:
            reasons.append("principal_has_no_role_binding_for_scope")
        else:
            for binding in bindings:
                permissions = [
                    value for value in policy.permissions
                    if value.role_id == binding.role_id
                    and request.action in value.actions
                    and request.resource_type in value.resource_types
                ]
                for permission in permissions:
                    if permission.require_mfa and not set(assertion.methods).intersection(policy.mfa_methods):
                        continue
                    if permission.require_reason_digest and request.reason_sha256 is None:
                        continue
                    matched_binding, matched_permission = binding, permission
                    break
                if matched_permission is not None:
                    break
            if matched_permission is None:
                candidate_permissions = [
                    value for value in policy.permissions
                    if any(binding.role_id == value.role_id for binding in bindings)
                    and request.action in value.actions
                    and request.resource_type in value.resource_types
                ]
                if not candidate_permissions:
                    reasons.append("role_does_not_permit_action_on_resource")
                elif any(value.require_mfa for value in candidate_permissions) and not set(assertion.methods).intersection(policy.mfa_methods):
                    reasons.append("mfa_required")
                elif any(value.require_reason_digest for value in candidate_permissions) and request.reason_sha256 is None:
                    reasons.append("reason_digest_required")
                else:
                    reasons.append("no_permission_satisfied")
    authorized = not reasons and matched_binding is not None and matched_permission is not None
    valid_until = min(assertion.valid_until, instant + policy.decision_ttl_seconds)
    # Even denial receipts need a non-zero validity interval for canonical schema. A stale
    # assertion can have expired before now, so denial validity is bounded to the policy TTL.
    if valid_until <= instant:
        valid_until = instant + min(policy.decision_ttl_seconds, 1.0)
    payload = {
        "schema": "rigorousrag-operator-authorization-decision/v1",
        "request_sha256": request.request_sha256,
        "principal_sha256": assertion.principal_sha256,
        "assertion_binding_sha256": assertion.assertion_binding_sha256,
        "policy_sha256": policy.policy_sha256,
        "matched_binding_sha256": None if matched_binding is None else matched_binding.binding_sha256,
        "matched_permission_sha256": None if matched_permission is None else matched_permission.permission_sha256,
        "authorized": authorized,
        "reason_codes": tuple(sorted(set(reasons))),
        "decided_at": instant,
        "valid_until": valid_until,
    }
    return OperatorAuthorizationDecision(**payload, decision_sha256=_digest(payload))


def assert_operator_authorization(
    decision: OperatorAuthorizationDecision,
    request: OperatorAuthorizationRequest,
    *,
    policy_sha256: str,
    now: float,
) -> OperatorAuthorizationDecision:
    if not isinstance(decision, OperatorAuthorizationDecision) or not isinstance(request, OperatorAuthorizationRequest):
        raise ValueError("decision/request types are invalid")
    if not decision.authorized:
        raise RuntimeError("operator action is not authorized")
    if decision.request_sha256 != request.request_sha256:
        raise RuntimeError("operator authorization is bound to a different request")
    if decision.policy_sha256 != _sha(policy_sha256, "policy_sha256"):
        raise RuntimeError("operator authorization was issued under a stale policy")
    if not decision.current_at(now):
        raise RuntimeError("operator authorization receipt is stale or not yet valid")
    return decision


@dataclass(frozen=True)
class OperatorPolicyRecord:
    revision: int
    policy: OperatorAuthorizationPolicy
    promoted_at: float


class SQLiteOperatorAuthorizationPolicyStore:
    """CAS-promoted immutable operator authorization policy history."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS operator_policy_history (
                    revision INTEGER PRIMARY KEY,
                    policy_sha256 TEXT NOT NULL UNIQUE,
                    policy_json TEXT NOT NULL,
                    promoted_at REAL NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS operator_policy_current (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    revision INTEGER NOT NULL,
                    policy_sha256 TEXT NOT NULL
                )"""
            )

    @staticmethod
    def _encode(policy: OperatorAuthorizationPolicy) -> str:
        return _canonical(
            {
                "policy_id": policy.policy_id,
                "bindings": [asdict(value) for value in policy.bindings],
                "permissions": [asdict(value) for value in policy.permissions],
                "mfa_methods": policy.mfa_methods,
                "decision_ttl_seconds": policy.decision_ttl_seconds,
            }
        ).decode("utf-8")

    @staticmethod
    def _decode(raw: str, expected_sha256: str) -> OperatorAuthorizationPolicy:
        value = json.loads(raw)
        if not isinstance(value, Mapping):
            raise RuntimeError("persisted operator policy is invalid")
        policy = OperatorAuthorizationPolicy(
            policy_id=value["policy_id"],
            bindings=tuple(PrincipalRoleBinding(**row) for row in value["bindings"]),
            permissions=tuple(RolePermission(**row) for row in value["permissions"]),
            mfa_methods=tuple(value["mfa_methods"]),
            decision_ttl_seconds=value["decision_ttl_seconds"],
        )
        if policy.policy_sha256 != _sha(expected_sha256, "policy_sha256"):
            raise RuntimeError("persisted operator policy digest is corrupt")
        return policy

    def current(self) -> OperatorPolicyRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT h.revision,h.policy_sha256,h.policy_json,h.promoted_at
                   FROM operator_policy_current c JOIN operator_policy_history h ON h.revision=c.revision
                   WHERE c.singleton=1"""
            ).fetchone()
        if row is None:
            return None
        return OperatorPolicyRecord(int(row["revision"]), self._decode(row["policy_json"], row["policy_sha256"]), float(row["promoted_at"]))

    def promote(
        self,
        policy: OperatorAuthorizationPolicy,
        *,
        expected_revision: int | None,
        now: float,
    ) -> OperatorPolicyRecord:
        if not isinstance(policy, OperatorAuthorizationPolicy):
            raise ValueError("policy must be OperatorAuthorizationPolicy")
        if expected_revision is not None:
            _positive_int(expected_revision, "expected_revision", allow_zero=True)
        promoted_at = _time(now, "now")
        encoded = self._encode(policy)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT revision,policy_sha256 FROM operator_policy_current WHERE singleton=1").fetchone()
            if current is None:
                if expected_revision not in (None, 0):
                    raise RuntimeError("operator policy bootstrap CAS failed")
                revision = 1
            else:
                current_revision = int(current["revision"])
                if expected_revision is None or expected_revision != current_revision:
                    raise RuntimeError("operator policy promotion CAS failed")
                if current["policy_sha256"] == policy.policy_sha256:
                    row = connection.execute(
                        "SELECT policy_json,promoted_at FROM operator_policy_history WHERE revision=?",
                        (current_revision,),
                    ).fetchone()
                    if row is None:
                        raise RuntimeError("current operator policy history is missing")
                    return OperatorPolicyRecord(current_revision, self._decode(row["policy_json"], policy.policy_sha256), float(row["promoted_at"]))
                revision = current_revision + 1
            connection.execute(
                "INSERT INTO operator_policy_history(revision,policy_sha256,policy_json,promoted_at) VALUES(?,?,?,?)",
                (revision, policy.policy_sha256, encoded, promoted_at),
            )
            if current is None:
                connection.execute("INSERT INTO operator_policy_current(singleton,revision,policy_sha256) VALUES(1,?,?)", (revision, policy.policy_sha256))
            else:
                changed = connection.execute(
                    "UPDATE operator_policy_current SET revision=?,policy_sha256=? WHERE singleton=1 AND revision=?",
                    (revision, policy.policy_sha256, expected_revision),
                ).rowcount
                if changed != 1:
                    raise RuntimeError("operator policy promotion lost CAS race")
        return OperatorPolicyRecord(revision, policy, promoted_at)


__all__ = [
    "OperatorAuthorizationDecision",
    "OperatorAuthorizationPolicy",
    "OperatorAuthorizationRequest",
    "OperatorPolicyRecord",
    "PrincipalRoleBinding",
    "RolePermission",
    "SQLiteOperatorAuthorizationPolicyStore",
    "VerifiedPrincipalAssertion",
    "assert_operator_authorization",
    "authorize_operator_action",
]
