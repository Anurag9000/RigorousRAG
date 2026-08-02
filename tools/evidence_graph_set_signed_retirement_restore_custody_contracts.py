"""Integrity contracts for pre/post signed-retirement restore custody receipts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _integer,
    _timestamp,
)
from tools.security import normalize_owner_id

_SCHEMA_VERSION = 1
_METHODS = frozenset(
    {"process_environment", "descriptor_file", "hmac_assertion"}
)
_MAX_BYTES = 1024 * 1024 * 1024 * 1024


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


def _actor_fields(
    *,
    actor_id: Any,
    binding_method: Any,
    binding_digest: Any,
) -> tuple[str, str, str]:
    actor = _identifier(actor_id, "actor_id", 200)
    method = _identifier(binding_method, "binding_method", 50)
    if method not in _METHODS:
        raise ValueError("custody actor binding method is unsupported.")
    digest = _digest(binding_digest, "binding_digest")
    return actor, method, digest


@dataclass(frozen=True)
class PreRestoreBackupReceipt:
    owner_id: str
    snapshot_digest: str
    target_path_digest: str
    backup_sha256: str
    backup_size_bytes: int
    target_schema_digest: str
    backup_schema_digest: str
    target_record_count: int
    backup_record_count: int
    actor_id: str
    binding_method: str
    binding_digest: str
    created_at: float
    receipt_digest: str
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        snapshot = _digest(self.snapshot_digest, "snapshot_digest")
        target = _digest(self.target_path_digest, "target_path_digest")
        backup_sha = _digest(self.backup_sha256, "backup_sha256")
        size = _integer(
            self.backup_size_bytes,
            "backup_size_bytes",
            1,
            _MAX_BYTES,
        )
        target_schema = _digest(
            self.target_schema_digest,
            "target_schema_digest",
        )
        backup_schema = _digest(
            self.backup_schema_digest,
            "backup_schema_digest",
        )
        target_count = _integer(
            self.target_record_count,
            "target_record_count",
            0,
            10_000,
        )
        backup_count = _integer(
            self.backup_record_count,
            "backup_record_count",
            0,
            10_000,
        )
        if target_count != 0 or backup_count != 0:
            raise ValueError("pre-restore backup must represent an empty target.")
        if target_schema != backup_schema:
            raise ValueError("backup schema differs from target schema.")
        actor, method, binding = _actor_fields(
            actor_id=self.actor_id,
            binding_method=self.binding_method,
            binding_digest=self.binding_digest,
        )
        created = _timestamp(self.created_at, "created_at")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("pre-restore receipt schema is unsupported.")
        stable = {
            "scope": "rigorousrag-signed-retirement-pre-restore-backup-v1",
            "owner_id": owner,
            "snapshot_digest": snapshot,
            "target_path_digest": target,
            "backup_sha256": backup_sha,
            "backup_size_bytes": size,
            "target_schema_digest": target_schema,
            "backup_schema_digest": backup_schema,
            "target_record_count": target_count,
            "backup_record_count": backup_count,
            "actor_id": actor,
            "binding_method": method,
            "binding_digest": binding,
            "created_at": created,
            "schema_version": self.schema_version,
        }
        receipt = _digest(self.receipt_digest, "receipt_digest")
        if receipt != _canonical_digest(stable):
            raise ValueError("receipt_digest differs from pre-restore receipt.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "snapshot_digest", snapshot)
        object.__setattr__(self, "target_path_digest", target)
        object.__setattr__(self, "backup_sha256", backup_sha)
        object.__setattr__(self, "backup_size_bytes", size)
        object.__setattr__(self, "target_schema_digest", target_schema)
        object.__setattr__(self, "backup_schema_digest", backup_schema)
        object.__setattr__(self, "target_record_count", target_count)
        object.__setattr__(self, "backup_record_count", backup_count)
        object.__setattr__(self, "actor_id", actor)
        object.__setattr__(self, "binding_method", method)
        object.__setattr__(self, "binding_digest", binding)
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "receipt_digest", receipt)

    @classmethod
    def create(cls, **values: Any) -> "PreRestoreBackupReceipt":
        stable = {
            "scope": "rigorousrag-signed-retirement-pre-restore-backup-v1",
            **values,
            "schema_version": _SCHEMA_VERSION,
        }
        return cls(
            **values,
            receipt_digest=_canonical_digest(stable),
            schema_version=_SCHEMA_VERSION,
        )


@dataclass(frozen=True)
class PostRestoreComparisonReceipt:
    owner_id: str
    restore_id: str
    snapshot_digest: str
    target_path_digest: str
    pre_restore_receipt_digest: str
    backup_sha256: str
    target_verification_digest: str
    target_record_count: int
    actor_id: str
    binding_method: str
    binding_digest: str
    compared_at: float
    receipt_digest: str
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        restore = _digest(self.restore_id, "restore_id")
        snapshot = _digest(self.snapshot_digest, "snapshot_digest")
        target = _digest(self.target_path_digest, "target_path_digest")
        pre_receipt = _digest(
            self.pre_restore_receipt_digest,
            "pre_restore_receipt_digest",
        )
        backup = _digest(self.backup_sha256, "backup_sha256")
        verification = _digest(
            self.target_verification_digest,
            "target_verification_digest",
        )
        count = _integer(
            self.target_record_count,
            "target_record_count",
            1,
            10_000,
        )
        actor, method, binding = _actor_fields(
            actor_id=self.actor_id,
            binding_method=self.binding_method,
            binding_digest=self.binding_digest,
        )
        compared = _timestamp(self.compared_at, "compared_at")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("post-restore receipt schema is unsupported.")
        stable = {
            "scope": "rigorousrag-signed-retirement-post-restore-comparison-v1",
            "owner_id": owner,
            "restore_id": restore,
            "snapshot_digest": snapshot,
            "target_path_digest": target,
            "pre_restore_receipt_digest": pre_receipt,
            "backup_sha256": backup,
            "target_verification_digest": verification,
            "target_record_count": count,
            "actor_id": actor,
            "binding_method": method,
            "binding_digest": binding,
            "compared_at": compared,
            "schema_version": self.schema_version,
        }
        receipt = _digest(self.receipt_digest, "receipt_digest")
        if receipt != _canonical_digest(stable):
            raise ValueError("receipt_digest differs from post-restore receipt.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "restore_id", restore)
        object.__setattr__(self, "snapshot_digest", snapshot)
        object.__setattr__(self, "target_path_digest", target)
        object.__setattr__(self, "pre_restore_receipt_digest", pre_receipt)
        object.__setattr__(self, "backup_sha256", backup)
        object.__setattr__(self, "target_verification_digest", verification)
        object.__setattr__(self, "target_record_count", count)
        object.__setattr__(self, "actor_id", actor)
        object.__setattr__(self, "binding_method", method)
        object.__setattr__(self, "binding_digest", binding)
        object.__setattr__(self, "compared_at", compared)
        object.__setattr__(self, "receipt_digest", receipt)

    @classmethod
    def create(cls, **values: Any) -> "PostRestoreComparisonReceipt":
        stable = {
            "scope": "rigorousrag-signed-retirement-post-restore-comparison-v1",
            **values,
            "schema_version": _SCHEMA_VERSION,
        }
        return cls(
            **values,
            receipt_digest=_canonical_digest(stable),
            schema_version=_SCHEMA_VERSION,
        )


__all__ = [
    "PostRestoreComparisonReceipt",
    "PreRestoreBackupReceipt",
]
