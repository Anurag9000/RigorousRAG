"""KMS/HSM-backed envelope-encryption and key-rotation contracts.

RigorousRAG must never implement production cryptography by composing primitives inside
business code.  This module therefore defines metadata, provider protocols, key-state
transitions and rotation invariants while delegating cryptographic operations to a
reviewed KMS/HSM adapter.  There is intentionally no insecure in-memory/local production
fallback and no key material is represented by these dataclasses.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

_HEX = frozenset("0123456789abcdef")


def _identifier(value: Any, label: str, maximum: int = 4_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if not result or len(result) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{label} is invalid")
    return result


def _sha256(value: Any, label: str) -> str:
    digest = _identifier(value, label, 64).lower()
    if len(digest) != 64 or any(ch not in _HEX for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class KeyPurpose(str, Enum):
    OBJECT_ENVELOPE = "object_envelope"
    DATABASE_FIELD = "database_field"
    BACKUP_ENVELOPE = "backup_envelope"
    EXPORT_ENVELOPE = "export_envelope"
    SIGNING = "signing"


class KeyLifecycleState(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    DECRYPT_ONLY = "decrypt_only"
    DISABLED = "disabled"
    DESTROY_PENDING = "destroy_pending"
    DESTROYED = "destroyed"


@dataclass(frozen=True)
class KeyReference:
    provider: str
    key_id: str
    key_version: str
    purpose: KeyPurpose
    algorithm: str
    state: KeyLifecycleState
    created_at: datetime
    activated_at: datetime | None = None
    decrypt_only_at: datetime | None = None
    disabled_at: datetime | None = None
    destroyed_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("provider", "key_id", "key_version", "algorithm"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if not isinstance(self.purpose, KeyPurpose):
            object.__setattr__(self, "purpose", KeyPurpose(self.purpose))
        if not isinstance(self.state, KeyLifecycleState):
            object.__setattr__(self, "state", KeyLifecycleState(self.state))
        created = _utc(self.created_at, "created_at")
        object.__setattr__(self, "created_at", created)
        previous = created
        for name in ("activated_at", "decrypt_only_at", "disabled_at", "destroyed_at"):
            value = getattr(self, name)
            if value is not None:
                selected = _utc(value, name)
                if selected < previous:
                    raise ValueError("key lifecycle timestamps must be monotonic")
                object.__setattr__(self, name, selected)
                previous = selected
        if self.state == KeyLifecycleState.DESTROYED and self.destroyed_at is None:
            raise ValueError("destroyed keys require destroyed_at")

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))


@dataclass(frozen=True)
class EncryptionContext:
    tenant_id: str
    owner_id: str
    object_id: str
    generation_id: str
    purpose: KeyPurpose
    metadata_digest: str

    def __post_init__(self) -> None:
        for name in ("tenant_id", "owner_id", "object_id", "generation_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if not isinstance(self.purpose, KeyPurpose):
            object.__setattr__(self, "purpose", KeyPurpose(self.purpose))
        object.__setattr__(self, "metadata_digest", _sha256(self.metadata_digest, "metadata_digest"))

    def associated_data(self) -> Mapping[str, str]:
        return {
            "tenant_id": self.tenant_id,
            "owner_id": self.owner_id,
            "object_id": self.object_id,
            "generation_id": self.generation_id,
            "purpose": self.purpose.value,
            "metadata_digest": self.metadata_digest,
        }


@dataclass(frozen=True)
class WrappedDataKey:
    """Encrypted data-key material only; plaintext data keys are never represented here."""

    key_reference: KeyReference
    wrapped_key: bytes
    wrapping_metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.key_reference, KeyReference):
            raise ValueError("key_reference must be KeyReference")
        if not isinstance(self.wrapped_key, bytes) or not self.wrapped_key or len(self.wrapped_key) > 1_000_000:
            raise ValueError("wrapped_key must be non-empty bounded bytes")
        if not isinstance(self.wrapping_metadata, Mapping) or len(self.wrapping_metadata) > 1_000:
            raise ValueError("wrapping_metadata must be a bounded mapping")
        object.__setattr__(
            self,
            "wrapping_metadata",
            {
                _identifier(key, "wrapping metadata key", 300): _identifier(value, "wrapping metadata value", 10_000)
                for key, value in self.wrapping_metadata.items()
            },
        )

    @property
    def wrapped_key_sha256(self) -> str:
        return hashlib.sha256(self.wrapped_key).hexdigest()


@dataclass(frozen=True)
class EncryptedArtifactDescriptor:
    artifact_id: str
    ciphertext_sha256: str
    plaintext_sha256: str | None
    wrapped_data_key: WrappedDataKey
    context_digest: str
    cipher_suite: str
    encrypted_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _identifier(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "ciphertext_sha256", _sha256(self.ciphertext_sha256, "ciphertext_sha256"))
        if self.plaintext_sha256 is not None:
            object.__setattr__(self, "plaintext_sha256", _sha256(self.plaintext_sha256, "plaintext_sha256"))
        if not isinstance(self.wrapped_data_key, WrappedDataKey):
            raise ValueError("wrapped_data_key must be WrappedDataKey")
        object.__setattr__(self, "context_digest", _sha256(self.context_digest, "context_digest"))
        object.__setattr__(self, "cipher_suite", _identifier(self.cipher_suite, "cipher_suite"))
        object.__setattr__(self, "encrypted_at", _utc(self.encrypted_at, "encrypted_at"))


class KeyManagementProvider(Protocol):
    """Adapter implemented by a real KMS/HSM integration."""

    def active_key(self, *, purpose: KeyPurpose, tenant_id: str) -> KeyReference: ...

    def generate_wrapped_data_key(
        self,
        *,
        key: KeyReference,
        context: EncryptionContext,
    ) -> WrappedDataKey: ...

    def rewrap_data_key(
        self,
        wrapped: WrappedDataKey,
        *,
        destination_key: KeyReference,
        context: EncryptionContext,
    ) -> WrappedDataKey: ...

    def decrypt_data_key_for_operation(
        self,
        wrapped: WrappedDataKey,
        *,
        context: EncryptionContext,
        operation_id: str,
    ) -> Any: ...

    def schedule_key_destruction(self, key: KeyReference, *, not_before: datetime) -> KeyReference: ...

    def cancel_key_destruction(self, key: KeyReference) -> KeyReference: ...


@dataclass(frozen=True)
class KeyRotationPlan:
    plan_id: str
    purpose: KeyPurpose
    source_key: KeyReference
    destination_key: KeyReference
    artifact_ids: tuple[str, ...]
    created_at: datetime
    legal_hold_checked: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id"))
        if not isinstance(self.purpose, KeyPurpose):
            object.__setattr__(self, "purpose", KeyPurpose(self.purpose))
        if not isinstance(self.source_key, KeyReference) or not isinstance(self.destination_key, KeyReference):
            raise ValueError("source/destination keys must be KeyReference")
        if self.source_key.purpose != self.purpose or self.destination_key.purpose != self.purpose:
            raise ValueError("rotation keys must match plan purpose")
        if self.source_key.digest == self.destination_key.digest:
            raise ValueError("source and destination keys must differ")
        if self.destination_key.state != KeyLifecycleState.ACTIVE:
            raise ValueError("destination key must be active")
        if self.source_key.state not in {KeyLifecycleState.ACTIVE, KeyLifecycleState.DECRYPT_ONLY}:
            raise ValueError("source key must still be usable for rewrap/decrypt")
        if not self.artifact_ids or len(self.artifact_ids) > 10_000_000:
            raise ValueError("artifact_ids must be non-empty and bounded")
        ids = tuple(_identifier(value, "artifact_id") for value in self.artifact_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("artifact_ids must be unique")
        object.__setattr__(self, "artifact_ids", ids)
        object.__setattr__(self, "created_at", _utc(self.created_at, "created_at"))
        if not isinstance(self.legal_hold_checked, bool):
            raise ValueError("legal_hold_checked must be boolean")
        if not self.legal_hold_checked:
            raise ValueError("key rotation requires an explicit legal-hold check")

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))


@dataclass(frozen=True)
class KeyRotationRecord:
    plan_digest: str
    artifact_id: str
    old_wrapped_key_sha256: str
    new_wrapped_key_sha256: str
    source_key_digest: str
    destination_key_digest: str
    completed_at: datetime
    operator_id: str

    def __post_init__(self) -> None:
        for name in (
            "plan_digest",
            "old_wrapped_key_sha256",
            "new_wrapped_key_sha256",
            "source_key_digest",
            "destination_key_digest",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(self, "artifact_id", _identifier(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "completed_at", _utc(self.completed_at, "completed_at"))
        object.__setattr__(self, "operator_id", _identifier(self.operator_id, "operator_id"))
        if self.old_wrapped_key_sha256 == self.new_wrapped_key_sha256:
            raise ValueError("rotation record must reflect a changed wrapped data key")


def validate_rotation_completion(
    plan: KeyRotationPlan,
    records: Sequence[KeyRotationRecord],
) -> None:
    if any(not isinstance(record, KeyRotationRecord) for record in records):
        raise ValueError("records must contain KeyRotationRecord values")
    if any(record.plan_digest != plan.digest for record in records):
        raise ValueError("rotation records do not belong to the supplied plan")
    by_artifact = {record.artifact_id: record for record in records}
    if len(by_artifact) != len(records):
        raise ValueError("rotation records contain duplicate artifact ids")
    missing = sorted(set(plan.artifact_ids) - set(by_artifact))
    extra = sorted(set(by_artifact) - set(plan.artifact_ids))
    if missing or extra:
        raise ValueError(f"rotation record coverage mismatch: missing={missing[:20]} extra={extra[:20]}")
    if any(record.source_key_digest != plan.source_key.digest for record in records):
        raise ValueError("rotation record source key differs from plan")
    if any(record.destination_key_digest != plan.destination_key.digest for record in records):
        raise ValueError("rotation record destination key differs from plan")


__all__ = [
    "EncryptedArtifactDescriptor",
    "EncryptionContext",
    "KeyLifecycleState",
    "KeyManagementProvider",
    "KeyPurpose",
    "KeyReference",
    "KeyRotationPlan",
    "KeyRotationRecord",
    "WrappedDataKey",
    "canonical_digest",
    "validate_rotation_completion",
]
