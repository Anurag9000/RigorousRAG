"""Governed authorization and read-only preflight for restore-intent retention deletion.

This module never deletes restore intents. It turns one exact retention-plan candidate
into an expiring, process-owned authorization record and can later revalidate that
candidate against current journal and durable-hold state.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _integer,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_restore_operations import (
    plan_signed_retirement_restore_retention,
)
from tools.security import normalize_owner_id

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_SCHEMA_VERSION = 1
_MAX_PATH = 4096
_MAX_LIMIT = 10_000
_STATES = frozenset({"authorized", "revoked"})
_BINDING_METHODS = frozenset(
    {"process_environment", "descriptor_file", "hmac_assertion"}
)
_DISPOSITIONS = frozenset(
    {
        "authorized_candidate_current",
        "authorization_revoked",
        "authorization_expired",
        "restore_missing",
        "restore_scope_changed",
        "durable_legal_hold_active",
        "no_longer_retention_candidate",
    }
)


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean.")
    return value


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(
            "deletion authorization database path must be a filesystem path."
        )
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("deletion authorization database path is invalid.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for part in (absolute, *absolute.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(
                "deletion authorization database path could not be validated."
            ) from exc
        if _redirecting(info):
            raise ValueError(
                "deletion authorization database path may not contain redirects."
            )
    return absolute


def deletion_policy_digest(
    *,
    minimum_age_seconds: float,
    retain_latest_per_target: int,
    include_completed: bool,
) -> str:
    return _canonical_digest(
        {
            "scope": "rigorousrag-restore-deletion-policy-v1",
            "minimum_age_seconds": _timestamp(
                minimum_age_seconds, "minimum_age_seconds"
            ),
            "retain_latest_per_target": _integer(
                retain_latest_per_target,
                "retain_latest_per_target",
                1,
                100,
            ),
            "include_completed": _boolean(
                include_completed, "include_completed"
            ),
        }
    )


def deterministic_restore_deletion_authorization_id(
    *,
    owner_id: str,
    restore_id: str,
    snapshot_digest: str,
    target_path_digest: str,
    plan_digest: str,
    policy_digest: str,
    authorization_key: str,
) -> str:
    return _canonical_digest(
        {
            "scope": "rigorousrag-restore-deletion-authorization-v1",
            "owner_id": normalize_owner_id(owner_id),
            "restore_id": _digest(restore_id, "restore_id"),
            "snapshot_digest": _digest(snapshot_digest, "snapshot_digest"),
            "target_path_digest": _digest(
                target_path_digest, "target_path_digest"
            ),
            "plan_digest": _digest(plan_digest, "plan_digest"),
            "policy_digest": _digest(policy_digest, "policy_digest"),
            "authorization_key": _identifier(
                authorization_key, "authorization_key", 200
            ),
        }
    )


@dataclass(frozen=True)
class SignedRetirementRestoreDeletionAuthorization:
    authorization_id: str
    owner_id: str
    restore_id: str
    snapshot_digest: str
    target_path_digest: str
    plan_digest: str
    policy_digest: str
    authorization_key: str
    minimum_age_seconds: float
    retain_latest_per_target: int
    include_completed: bool
    status: str
    authorized_actor_id: str
    authorized_binding_method: str
    authorized_binding_digest: str
    authorized_at: float
    expires_at: float
    revoked_actor_id: str | None = None
    revoked_binding_method: str | None = None
    revoked_binding_digest: str | None = None
    revoked_at: float | None = None
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        restore = _digest(self.restore_id, "restore_id")
        snapshot = _digest(self.snapshot_digest, "snapshot_digest")
        target = _digest(self.target_path_digest, "target_path_digest")
        plan = _digest(self.plan_digest, "plan_digest")
        policy = _digest(self.policy_digest, "policy_digest")
        key = _identifier(self.authorization_key, "authorization_key", 200)
        expected_policy = deletion_policy_digest(
            minimum_age_seconds=self.minimum_age_seconds,
            retain_latest_per_target=self.retain_latest_per_target,
            include_completed=self.include_completed,
        )
        if policy != expected_policy:
            raise ValueError("policy_digest differs from deletion policy fields.")
        expected_id = deterministic_restore_deletion_authorization_id(
            owner_id=owner,
            restore_id=restore,
            snapshot_digest=snapshot,
            target_path_digest=target,
            plan_digest=plan,
            policy_digest=policy,
            authorization_key=key,
        )
        authorization = _digest(self.authorization_id, "authorization_id")
        if authorization != expected_id:
            raise ValueError(
                "authorization_id differs from immutable deletion scope."
            )
        status = _identifier(self.status, "status", 20)
        if status not in _STATES:
            raise ValueError("deletion authorization status is unsupported.")
        authorized_actor = _identifier(
            self.authorized_actor_id, "authorized_actor_id", 200
        )
        authorized_method = _identifier(
            self.authorized_binding_method, "authorized_binding_method", 50
        )
        if authorized_method not in _BINDING_METHODS:
            raise ValueError("authorized actor binding method is unsupported.")
        authorized_binding = _digest(
            self.authorized_binding_digest, "authorized_binding_digest"
        )
        authorized_at = _timestamp(self.authorized_at, "authorized_at")
        expires_at = _timestamp(self.expires_at, "expires_at")
        if expires_at <= authorized_at:
            raise ValueError(
                "deletion authorization must expire after authorization."
            )
        release_fields = (
            self.revoked_actor_id,
            self.revoked_binding_method,
            self.revoked_binding_digest,
            self.revoked_at,
        )
        if status == "authorized":
            if any(value is not None for value in release_fields):
                raise ValueError(
                    "active authorization may not contain revoke fields."
                )
            revoked_actor = revoked_method = revoked_binding = revoked_at = None
        else:
            if any(value is None for value in release_fields):
                raise ValueError(
                    "revoked authorization requires complete revoke fields."
                )
            revoked_actor = _identifier(
                self.revoked_actor_id, "revoked_actor_id", 200
            )
            revoked_method = _identifier(
                self.revoked_binding_method, "revoked_binding_method", 50
            )
            if revoked_method not in _BINDING_METHODS:
                raise ValueError("revoked actor binding method is unsupported.")
            revoked_binding = _digest(
                self.revoked_binding_digest, "revoked_binding_digest"
            )
            revoked_at = _timestamp(self.revoked_at, "revoked_at")
            if revoked_at < authorized_at:
                raise ValueError(
                    "authorization revocation predates authorization."
                )
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("deletion authorization schema is unsupported.")
        object.__setattr__(self, "authorization_id", authorization)
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "restore_id", restore)
        object.__setattr__(self, "snapshot_digest", snapshot)
        object.__setattr__(self, "target_path_digest", target)
        object.__setattr__(self, "plan_digest", plan)
        object.__setattr__(self, "policy_digest", policy)
        object.__setattr__(self, "authorization_key", key)
        object.__setattr__(
            self,
            "minimum_age_seconds",
            _timestamp(self.minimum_age_seconds, "minimum_age_seconds"),
        )
        object.__setattr__(
            self,
            "retain_latest_per_target",
            _integer(
                self.retain_latest_per_target,
                "retain_latest_per_target",
                1,
                100,
            ),
        )
        object.__setattr__(
            self,
            "include_completed",
            _boolean(self.include_completed, "include_completed"),
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "authorized_actor_id", authorized_actor)
        object.__setattr__(self, "authorized_binding_method", authorized_method)
        object.__setattr__(self, "authorized_binding_digest", authorized_binding)
        object.__setattr__(self, "authorized_at", authorized_at)
        object.__setattr__(self, "expires_at", expires_at)
        object.__setattr__(self, "revoked_actor_id", revoked_actor)
        object.__setattr__(self, "revoked_binding_method", revoked_method)
        object.__setattr__(self, "revoked_binding_digest", revoked_binding)
        object.__setattr__(self, "revoked_at", revoked_at)

    @property
    def authorization_digest(self) -> str:
        return _canonical_digest(
            {
                "scope": "rigorousrag-restore-deletion-authorization-record-v1",
                **asdict(self),
            }
        )


@dataclass(frozen=True)
class SignedRetirementRestoreDeletionPreflight:
    authorization_id: str
    owner_id: str
    restore_id: str
    snapshot_digest: str
    target_path_digest: str
    authorization_status: str
    generated_at: float
    current_plan_digest: str | None
    durable_hold_active: bool
    retention_candidate_current: bool
    disposition: str
    eligible_for_future_deletion_executor: bool
    report_digest: str
    deletion_performed: bool = False
    journal_mutation_performed: bool = False
    restore_mutation_performed: bool = False
    source_text_returned: bool = False
    raw_paths_returned: bool = False

    def __post_init__(self) -> None:
        authorization = _digest(self.authorization_id, "authorization_id")
        owner = normalize_owner_id(self.owner_id)
        restore = _digest(self.restore_id, "restore_id")
        snapshot = _digest(self.snapshot_digest, "snapshot_digest")
        target = _digest(self.target_path_digest, "target_path_digest")
        status = _identifier(
            self.authorization_status, "authorization_status", 20
        )
        if status not in _STATES:
            raise ValueError("deletion authorization status is unsupported.")
        generated = _timestamp(self.generated_at, "generated_at")
        current_plan = (
            None
            if self.current_plan_digest is None
            else _digest(self.current_plan_digest, "current_plan_digest")
        )
        if not isinstance(self.durable_hold_active, bool) or not isinstance(
            self.retention_candidate_current, bool
        ):
            raise ValueError(
                "deletion preflight decision flags must be boolean."
            )
        disposition = _identifier(self.disposition, "disposition", 100)
        if disposition not in _DISPOSITIONS:
            raise ValueError("deletion preflight disposition is unsupported.")
        if not isinstance(self.eligible_for_future_deletion_executor, bool):
            raise ValueError("deletion preflight eligibility must be boolean.")
        if self.eligible_for_future_deletion_executor != (
            disposition == "authorized_candidate_current"
        ):
            raise ValueError(
                "deletion preflight eligibility differs from disposition."
            )
        if any(
            value is not False
            for value in (
                self.deletion_performed,
                self.journal_mutation_performed,
                self.restore_mutation_performed,
                self.source_text_returned,
                self.raw_paths_returned,
            )
        ):
            raise ValueError("deletion preflight safety flags must be false.")
        stable = {
            "scope": "rigorousrag-restore-deletion-preflight-v1",
            "authorization_id": authorization,
            "owner_id": owner,
            "restore_id": restore,
            "snapshot_digest": snapshot,
            "target_path_digest": target,
            "authorization_status": status,
            "generated_at": generated,
            "current_plan_digest": current_plan,
            "durable_hold_active": self.durable_hold_active,
            "retention_candidate_current": self.retention_candidate_current,
            "disposition": disposition,
            "eligible_for_future_deletion_executor": (
                disposition == "authorized_candidate_current"
            ),
        }
        report = _digest(self.report_digest, "report_digest")
        if report != _canonical_digest(stable):
            raise ValueError(
                "report_digest differs from deletion preflight."
            )
        object.__setattr__(self, "authorization_id", authorization)
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "restore_id", restore)
        object.__setattr__(self, "snapshot_digest", snapshot)
        object.__setattr__(self, "target_path_digest", target)
        object.__setattr__(self, "authorization_status", status)
        object.__setattr__(self, "generated_at", generated)
        object.__setattr__(self, "current_plan_digest", current_plan)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "report_digest", report)


class SignedRetirementRestoreDeletionAuthorizationStore:
    """Integrity-backed authorization history with monotonic revocation."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError(
                "deletion authorization database parent must be a directory."
            )
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError(
                "deletion authorization database is not a regular file."
            )
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
            or self._file_identity() != self._database_identity
        ):
            raise RuntimeError(
                "deletion authorization database identity changed."
            )

    def _connect(self) -> sqlite3.Connection:
        self._verify()
        connection = sqlite3.connect(
            self.path, timeout=30.0, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with sqlite3.connect(
            self.path, timeout=30.0, isolation_level=None
        ) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS signed_retirement_restore_deletion_authorizations (
                    authorization_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    restore_id TEXT NOT NULL,
                    snapshot_digest TEXT NOT NULL,
                    target_path_digest TEXT NOT NULL,
                    plan_digest TEXT NOT NULL,
                    policy_digest TEXT NOT NULL,
                    authorization_key TEXT NOT NULL,
                    minimum_age_seconds REAL NOT NULL,
                    retain_latest_per_target INTEGER NOT NULL,
                    include_completed INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    authorized_actor_id TEXT NOT NULL,
                    authorized_binding_method TEXT NOT NULL,
                    authorized_binding_digest TEXT NOT NULL,
                    authorized_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    revoked_actor_id TEXT,
                    revoked_binding_method TEXT,
                    revoked_binding_digest TEXT,
                    revoked_at REAL,
                    schema_version INTEGER NOT NULL,
                    UNIQUE(owner_id, restore_id, authorization_key)
                );
                CREATE INDEX IF NOT EXISTS restore_deletion_authorization_scope
                    ON signed_retirement_restore_deletion_authorizations(
                        owner_id, restore_id, status, authorized_at, authorization_id
                    );
                CREATE TABLE IF NOT EXISTS signed_retirement_restore_deletion_authorization_integrity (
                    authorization_id TEXT PRIMARY KEY,
                    authorization_digest TEXT NOT NULL,
                    FOREIGN KEY(authorization_id)
                        REFERENCES signed_retirement_restore_deletion_authorizations(
                            authorization_id
                        ) ON DELETE RESTRICT
                );
                """
            )

    @staticmethod
    def _value(
        row: sqlite3.Row,
    ) -> SignedRetirementRestoreDeletionAuthorization:
        try:
            include_completed_raw = row["include_completed"]
            if include_completed_raw not in (0, 1):
                raise ValueError("stored include_completed is invalid.")
            return SignedRetirementRestoreDeletionAuthorization(
                authorization_id=row["authorization_id"],
                owner_id=row["owner_id"],
                restore_id=row["restore_id"],
                snapshot_digest=row["snapshot_digest"],
                target_path_digest=row["target_path_digest"],
                plan_digest=row["plan_digest"],
                policy_digest=row["policy_digest"],
                authorization_key=row["authorization_key"],
                minimum_age_seconds=row["minimum_age_seconds"],
                retain_latest_per_target=int(row["retain_latest_per_target"]),
                include_completed=bool(include_completed_raw),
                status=row["status"],
                authorized_actor_id=row["authorized_actor_id"],
                authorized_binding_method=row["authorized_binding_method"],
                authorized_binding_digest=row["authorized_binding_digest"],
                authorized_at=row["authorized_at"],
                expires_at=row["expires_at"],
                revoked_actor_id=row["revoked_actor_id"],
                revoked_binding_method=row["revoked_binding_method"],
                revoked_binding_digest=row["revoked_binding_digest"],
                revoked_at=row["revoked_at"],
                schema_version=int(row["schema_version"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                "stored deletion authorization is corrupt."
            ) from exc

    @classmethod
    def _verified_value(
        cls, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> SignedRetirementRestoreDeletionAuthorization:
        value = cls._value(row)
        integrity = connection.execute(
            "SELECT authorization_digest FROM "
            "signed_retirement_restore_deletion_authorization_integrity "
            "WHERE authorization_id=?",
            (value.authorization_id,),
        ).fetchone()
        if integrity is None:
            raise RuntimeError(
                "deletion authorization integrity record is missing."
            )
        if _digest(
            integrity["authorization_digest"], "authorization_digest"
        ) != value.authorization_digest:
            raise RuntimeError(
                "stored deletion authorization integrity differs."
            )
        return value

    def authorize(
        self,
        *,
        owner_id: str,
        restore_id: str,
        plan_digest: str,
        plan_generated_at: float,
        authorization_key: str,
        actor: ReviewActorBinding,
        restore_journal: Any,
        hold_store: Any,
        minimum_age_seconds: float,
        retain_latest_per_target: int,
        include_completed: bool,
        expires_in_seconds: float = 24 * 60 * 60,
        now: float | None = None,
        limit: int = _MAX_LIMIT,
    ) -> SignedRetirementRestoreDeletionAuthorization:
        if not isinstance(actor, ReviewActorBinding):
            raise ValueError("actor must be ReviewActorBinding.")
        owner = normalize_owner_id(owner_id)
        restore = _digest(restore_id, "restore_id")
        expected_plan = _digest(plan_digest, "plan_digest")
        plan_time = _timestamp(plan_generated_at, "plan_generated_at")
        current = _timestamp(time.time() if now is None else now, "now")
        ttl = _timestamp(expires_in_seconds, "expires_in_seconds")
        if ttl <= 0 or ttl > 31 * 24 * 60 * 60:
            raise ValueError(
                "expires_in_seconds must be between 0 and 31 days."
            )
        if actor.expires_at is not None and actor.expires_at < current:
            raise PermissionError(
                "actor binding expired before deletion authorization."
            )
        if not callable(getattr(restore_journal, "get", None)):
            raise ValueError(
                "restore_journal lacks the required get boundary."
            )
        if not callable(getattr(hold_store, "active_restore_ids", None)):
            raise ValueError(
                "hold_store lacks the active hold boundary."
            )
        restore_value = restore_journal.get(restore)
        if restore_value.owner_id != owner:
            raise RuntimeError(
                "restore escaped deletion-authorization owner scope."
            )
        held = hold_store.active_restore_ids(owner_id=owner, limit=limit)
        if restore in held:
            raise RuntimeError(
                "durable legal hold blocks deletion authorization."
            )
        plan = plan_signed_retirement_restore_retention(
            owner_id=owner,
            journal=restore_journal,
            now=plan_time,
            minimum_age_seconds=minimum_age_seconds,
            retain_latest_per_target=retain_latest_per_target,
            include_completed=include_completed,
            held_restore_ids=held,
            limit=limit,
        )
        if plan.plan_digest != expected_plan:
            raise RuntimeError("retention plan digest differs.")
        candidates = [
            item
            for item in plan.items
            if item.restore_id == restore and item.retention_candidate
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "restore is not an authorized retention candidate."
            )
        candidate = candidates[0]
        if (
            candidate.snapshot_digest != restore_value.snapshot_digest
            or candidate.target_path_digest != restore_value.target_path_digest
        ):
            raise RuntimeError(
                "retention candidate escaped restore scope."
            )
        policy = deletion_policy_digest(
            minimum_age_seconds=minimum_age_seconds,
            retain_latest_per_target=retain_latest_per_target,
            include_completed=include_completed,
        )
        authorization_id = deterministic_restore_deletion_authorization_id(
            owner_id=owner,
            restore_id=restore,
            snapshot_digest=restore_value.snapshot_digest,
            target_path_digest=restore_value.target_path_digest,
            plan_digest=expected_plan,
            policy_digest=policy,
            authorization_key=authorization_key,
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM signed_retirement_restore_deletion_authorizations "
                    "WHERE authorization_id=?",
                    (authorization_id,),
                ).fetchone()
                if row is not None:
                    stored = self._verified_value(connection, row)
                    if (
                        stored.authorized_actor_id != actor.actor_id
                        or stored.authorized_binding_method
                        != actor.binding_method
                        or stored.authorized_binding_digest
                        != actor.binding_digest
                    ):
                        raise RuntimeError(
                            "deletion authorization identity collision detected."
                        )
                    connection.execute("COMMIT")
                    return stored
                value = SignedRetirementRestoreDeletionAuthorization(
                    authorization_id=authorization_id,
                    owner_id=owner,
                    restore_id=restore,
                    snapshot_digest=restore_value.snapshot_digest,
                    target_path_digest=restore_value.target_path_digest,
                    plan_digest=expected_plan,
                    policy_digest=policy,
                    authorization_key=authorization_key,
                    minimum_age_seconds=minimum_age_seconds,
                    retain_latest_per_target=retain_latest_per_target,
                    include_completed=include_completed,
                    status="authorized",
                    authorized_actor_id=actor.actor_id,
                    authorized_binding_method=actor.binding_method,
                    authorized_binding_digest=actor.binding_digest,
                    authorized_at=current,
                    expires_at=current + ttl,
                )
                connection.execute(
                    "INSERT INTO signed_retirement_restore_deletion_authorizations "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
                    (
                        value.authorization_id,
                        value.owner_id,
                        value.restore_id,
                        value.snapshot_digest,
                        value.target_path_digest,
                        value.plan_digest,
                        value.policy_digest,
                        value.authorization_key,
                        value.minimum_age_seconds,
                        value.retain_latest_per_target,
                        int(value.include_completed),
                        value.status,
                        value.authorized_actor_id,
                        value.authorized_binding_method,
                        value.authorized_binding_digest,
                        value.authorized_at,
                        value.expires_at,
                        None,
                        None,
                        None,
                        None,
                    ),
                )
                connection.execute(
                    "INSERT INTO "
                    "signed_retirement_restore_deletion_authorization_integrity "
                    "VALUES (?,?)",
                    (value.authorization_id, value.authorization_digest),
                )
                connection.execute("COMMIT")
                return value
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def get(
        self, authorization_id: str
    ) -> SignedRetirementRestoreDeletionAuthorization:
        selected = _digest(authorization_id, "authorization_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM signed_retirement_restore_deletion_authorizations "
                "WHERE authorization_id=?",
                (selected,),
            ).fetchone()
            if row is None:
                raise KeyError(selected)
            return self._verified_value(connection, row)

    def list(
        self,
        *,
        owner_id: str,
        restore_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> tuple[SignedRetirementRestoreDeletionAuthorization, ...]:
        owner = normalize_owner_id(owner_id)
        restore = (
            None
            if restore_id is None
            else _digest(restore_id, "restore_id")
        )
        selected_status = (
            None
            if status is None
            else _identifier(status, "status", 20)
        )
        if selected_status is not None and selected_status not in _STATES:
            raise ValueError(
                "deletion authorization status is unsupported."
            )
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        query = (
            "SELECT * FROM signed_retirement_restore_deletion_authorizations "
            "WHERE owner_id=?"
        )
        params: list[Any] = [owner]
        if restore is not None:
            query += " AND restore_id=?"
            params.append(restore)
        if selected_status is not None:
            query += " AND status=?"
            params.append(selected_status)
        query += " ORDER BY authorized_at DESC, authorization_id DESC LIMIT ?"
        params.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
            return tuple(
                self._verified_value(connection, row) for row in rows
            )

    def revoke(
        self,
        authorization_id: str,
        *,
        owner_id: str,
        confirm_authorization_id: str,
        actor: ReviewActorBinding,
        now: float | None = None,
    ) -> SignedRetirementRestoreDeletionAuthorization:
        selected = _digest(authorization_id, "authorization_id")
        if selected != _digest(
            confirm_authorization_id, "confirm_authorization_id"
        ):
            raise ValueError("authorization confirmation differs.")
        owner = normalize_owner_id(owner_id)
        if not isinstance(actor, ReviewActorBinding):
            raise ValueError("actor must be ReviewActorBinding.")
        current_time = _timestamp(time.time() if now is None else now, "now")
        if actor.expires_at is not None and actor.expires_at < current_time:
            raise PermissionError(
                "actor binding expired before authorization revocation."
            )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM signed_retirement_restore_deletion_authorizations "
                    "WHERE authorization_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise KeyError(selected)
                current = self._verified_value(connection, row)
                if current.owner_id != owner:
                    raise RuntimeError(
                        "deletion authorization escaped owner scope."
                    )
                if current.status == "revoked":
                    connection.execute("COMMIT")
                    return current
                revoked = SignedRetirementRestoreDeletionAuthorization(
                    **{
                        **asdict(current),
                        "status": "revoked",
                        "revoked_actor_id": actor.actor_id,
                        "revoked_binding_method": actor.binding_method,
                        "revoked_binding_digest": actor.binding_digest,
                        "revoked_at": current_time,
                    }
                )
                connection.execute(
                    "UPDATE signed_retirement_restore_deletion_authorizations "
                    "SET status='revoked', revoked_actor_id=?, "
                    "revoked_binding_method=?, revoked_binding_digest=?, revoked_at=? "
                    "WHERE authorization_id=? AND status='authorized'",
                    (
                        revoked.revoked_actor_id,
                        revoked.revoked_binding_method,
                        revoked.revoked_binding_digest,
                        revoked.revoked_at,
                        selected,
                    ),
                )
                connection.execute(
                    "UPDATE signed_retirement_restore_deletion_authorization_integrity "
                    "SET authorization_digest=? WHERE authorization_id=?",
                    (revoked.authorization_digest, selected),
                )
                connection.execute("COMMIT")
                return revoked
            except Exception:
                connection.execute("ROLLBACK")
                raise


def preflight_signed_retirement_restore_deletion(
    *,
    authorization: SignedRetirementRestoreDeletionAuthorization,
    restore_journal: Any,
    hold_store: Any,
    now: float | None = None,
    limit: int = _MAX_LIMIT,
) -> SignedRetirementRestoreDeletionPreflight:
    if not isinstance(
        authorization, SignedRetirementRestoreDeletionAuthorization
    ):
        raise ValueError(
            "authorization must be SignedRetirementRestoreDeletionAuthorization."
        )
    timestamp = _timestamp(time.time() if now is None else now, "now")
    current_plan_digest: str | None = None
    durable_hold_active = False
    candidate_current = False
    if authorization.status == "revoked":
        disposition = "authorization_revoked"
    elif authorization.expires_at <= timestamp:
        disposition = "authorization_expired"
    else:
        try:
            restore_value = restore_journal.get(authorization.restore_id)
        except KeyError:
            disposition = "restore_missing"
        else:
            if (
                restore_value.owner_id != authorization.owner_id
                or restore_value.snapshot_digest
                != authorization.snapshot_digest
                or restore_value.target_path_digest
                != authorization.target_path_digest
            ):
                disposition = "restore_scope_changed"
            else:
                held = hold_store.active_restore_ids(
                    owner_id=authorization.owner_id,
                    limit=limit,
                )
                durable_hold_active = authorization.restore_id in held
                if durable_hold_active:
                    disposition = "durable_legal_hold_active"
                else:
                    plan = plan_signed_retirement_restore_retention(
                        owner_id=authorization.owner_id,
                        journal=restore_journal,
                        now=timestamp,
                        minimum_age_seconds=(
                            authorization.minimum_age_seconds
                        ),
                        retain_latest_per_target=(
                            authorization.retain_latest_per_target
                        ),
                        include_completed=authorization.include_completed,
                        held_restore_ids=held,
                        limit=limit,
                    )
                    current_plan_digest = plan.plan_digest
                    candidate_current = any(
                        item.restore_id == authorization.restore_id
                        and item.retention_candidate
                        for item in plan.items
                    )
                    disposition = (
                        "authorized_candidate_current"
                        if candidate_current
                        else "no_longer_retention_candidate"
                    )
    stable = {
        "scope": "rigorousrag-restore-deletion-preflight-v1",
        "authorization_id": authorization.authorization_id,
        "owner_id": authorization.owner_id,
        "restore_id": authorization.restore_id,
        "snapshot_digest": authorization.snapshot_digest,
        "target_path_digest": authorization.target_path_digest,
        "authorization_status": authorization.status,
        "generated_at": timestamp,
        "current_plan_digest": current_plan_digest,
        "durable_hold_active": durable_hold_active,
        "retention_candidate_current": candidate_current,
        "disposition": disposition,
        "eligible_for_future_deletion_executor": (
            disposition == "authorized_candidate_current"
        ),
    }
    return SignedRetirementRestoreDeletionPreflight(
        authorization_id=authorization.authorization_id,
        owner_id=authorization.owner_id,
        restore_id=authorization.restore_id,
        snapshot_digest=authorization.snapshot_digest,
        target_path_digest=authorization.target_path_digest,
        authorization_status=authorization.status,
        generated_at=timestamp,
        current_plan_digest=current_plan_digest,
        durable_hold_active=durable_hold_active,
        retention_candidate_current=candidate_current,
        disposition=disposition,
        eligible_for_future_deletion_executor=(
            disposition == "authorized_candidate_current"
        ),
        report_digest=_canonical_digest(stable),
    )


__all__ = [
    "SignedRetirementRestoreDeletionAuthorization",
    "SignedRetirementRestoreDeletionAuthorizationStore",
    "SignedRetirementRestoreDeletionPreflight",
    "deletion_policy_digest",
    "deterministic_restore_deletion_authorization_id",
    "preflight_signed_retirement_restore_deletion",
]
