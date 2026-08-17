"""Provider-native object retention, legal-hold, version-lock, and deletion contracts.

The module separates an operator's requested policy from guarantees attested by a storage
provider. A normal object delete is never labeled secure erasure. Strong guarantees such
as compliance-mode object lock, cryptographic erasure, or physical-media erasure must be
reported explicitly by an injected provider adapter.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if (not selected and not allow_empty) or len(selected) > maximum or "\x00" in selected:
        raise ValueError(f"{label} is invalid")
    return selected


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{label} is invalid")
    return parsed


def _sha(value: str, label: str, *, allow_empty: bool = False) -> str:
    selected = _text(value, label, 64, allow_empty=allow_empty).lower()
    if selected and (len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected)):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


@dataclass(frozen=True)
class ObjectRetentionPolicy:
    policy_id: str
    retain_until: float = 0.0
    legal_hold: bool = False
    version_lock_mode: str = "none"
    deletion_requirement: str = "logical_delete"
    minimum_versions: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id", 256))
        object.__setattr__(self, "retain_until", _finite(self.retain_until, "retain_until"))
        if not isinstance(self.legal_hold, bool):
            raise ValueError("legal_hold must be boolean")
        lock = _text(self.version_lock_mode, "version_lock_mode", 64).lower()
        if lock not in {"none", "governance", "compliance"}:
            raise ValueError("version_lock_mode is invalid")
        object.__setattr__(self, "version_lock_mode", lock)
        deletion = _text(self.deletion_requirement, "deletion_requirement", 64).lower()
        if deletion not in {"logical_delete", "cryptographic_erase", "physical_erase_attested"}:
            raise ValueError("deletion_requirement is invalid")
        object.__setattr__(self, "deletion_requirement", deletion)
        if isinstance(self.minimum_versions, bool) or not isinstance(self.minimum_versions, int) or not 1 <= self.minimum_versions <= 1_000_000:
            raise ValueError("minimum_versions is invalid")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True)
class ObjectVersionIdentity:
    object_key: str
    version_id: str
    content_sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_key", _text(self.object_key, "object_key", 4000))
        object.__setattr__(self, "version_id", _text(self.version_id, "version_id", 1000))
        object.__setattr__(self, "content_sha256", _sha(self.content_sha256, "content_sha256"))
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise ValueError("size_bytes is invalid")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True)
class ProviderRetentionCapabilities:
    provider_id: str
    versioning: bool
    legal_hold: bool
    governance_lock: bool
    compliance_lock: bool
    cryptographic_erase: bool
    physical_erase_attestation: bool
    provider_documentation_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", _text(self.provider_id, "provider_id", 500))
        for field in (
            "versioning",
            "legal_hold",
            "governance_lock",
            "compliance_lock",
            "cryptographic_erase",
            "physical_erase_attestation",
        ):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"{field} must be boolean")
        object.__setattr__(
            self,
            "provider_documentation_fingerprint",
            _sha(self.provider_documentation_fingerprint, "provider_documentation_fingerprint", allow_empty=True),
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True)
class ProviderRetentionStatus:
    provider_id: str
    object_version: ObjectVersionIdentity
    observed_at: float
    retention_until: float
    legal_hold_active: bool
    version_lock_mode: str
    version_count: int
    deletion_state: str
    deletion_attestation_fingerprint: str = ""
    provider_status_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", _text(self.provider_id, "provider_id", 500))
        if not isinstance(self.object_version, ObjectVersionIdentity):
            raise TypeError("object_version must be ObjectVersionIdentity")
        object.__setattr__(self, "observed_at", _finite(self.observed_at, "observed_at"))
        object.__setattr__(self, "retention_until", _finite(self.retention_until, "retention_until"))
        if not isinstance(self.legal_hold_active, bool):
            raise ValueError("legal_hold_active must be boolean")
        lock = _text(self.version_lock_mode, "version_lock_mode", 64).lower()
        if lock not in {"none", "governance", "compliance"}:
            raise ValueError("version_lock_mode is invalid")
        object.__setattr__(self, "version_lock_mode", lock)
        if isinstance(self.version_count, bool) or not isinstance(self.version_count, int) or self.version_count < 0:
            raise ValueError("version_count is invalid")
        deletion = _text(self.deletion_state, "deletion_state", 64).lower()
        if deletion not in {"present", "logical_deleted", "cryptographically_erased", "physical_erase_attested", "unknown"}:
            raise ValueError("deletion_state is invalid")
        object.__setattr__(self, "deletion_state", deletion)
        object.__setattr__(
            self,
            "deletion_attestation_fingerprint",
            _sha(self.deletion_attestation_fingerprint, "deletion_attestation_fingerprint", allow_empty=True),
        )
        computed = hashlib.sha256(
            _canonical(
                {
                    "provider_id": self.provider_id,
                    "object_version": asdict(self.object_version),
                    "observed_at": self.observed_at,
                    "retention_until": self.retention_until,
                    "legal_hold_active": self.legal_hold_active,
                    "version_lock_mode": self.version_lock_mode,
                    "version_count": self.version_count,
                    "deletion_state": self.deletion_state,
                    "deletion_attestation_fingerprint": self.deletion_attestation_fingerprint,
                }
            )
        ).hexdigest()
        supplied = self.provider_status_fingerprint.strip().lower()
        if supplied and _sha(supplied, "provider_status_fingerprint") != computed:
            raise ValueError("provider_status_fingerprint does not match status")
        object.__setattr__(self, "provider_status_fingerprint", computed)


class ObjectGovernanceProvider(Protocol):
    @property
    def capabilities(self) -> ProviderRetentionCapabilities: ...

    def apply_retention(self, object_version: ObjectVersionIdentity, policy: ObjectRetentionPolicy) -> ProviderRetentionStatus: ...

    def request_deletion(self, object_version: ObjectVersionIdentity, policy: ObjectRetentionPolicy) -> ProviderRetentionStatus: ...

    def status(self, object_version: ObjectVersionIdentity) -> ProviderRetentionStatus: ...


@dataclass(frozen=True)
class RetentionComplianceEvaluation:
    compliant: bool
    reasons: tuple[str, ...]
    policy_fingerprint: str
    provider_capabilities_fingerprint: str
    provider_status_fingerprint: str
    deletion_requirement_satisfied: bool
    fingerprint: str


def evaluate_retention_compliance(
    policy: ObjectRetentionPolicy,
    capabilities: ProviderRetentionCapabilities,
    status: ProviderRetentionStatus,
    *,
    now: float,
) -> RetentionComplianceEvaluation:
    if not isinstance(policy, ObjectRetentionPolicy) or not isinstance(capabilities, ProviderRetentionCapabilities) or not isinstance(status, ProviderRetentionStatus):
        raise TypeError("policy/capabilities/status types are invalid")
    selected_now = _finite(now, "now")
    if capabilities.provider_id != status.provider_id:
        raise ValueError("provider capabilities/status identity mismatch")
    reasons: list[str] = []
    if policy.retain_until > selected_now:
        if status.retention_until < policy.retain_until:
            reasons.append("provider_retention_deadline_is_shorter_than_policy")
        if policy.version_lock_mode == "governance" and not capabilities.governance_lock:
            reasons.append("provider_does_not_attest_governance_lock_support")
        if policy.version_lock_mode == "compliance" and not capabilities.compliance_lock:
            reasons.append("provider_does_not_attest_compliance_lock_support")
        lock_rank = {"none": 0, "governance": 1, "compliance": 2}
        if lock_rank[status.version_lock_mode] < lock_rank[policy.version_lock_mode]:
            reasons.append("observed_object_lock_mode_is_weaker_than_policy")
    if policy.legal_hold and (not capabilities.legal_hold or not status.legal_hold_active):
        reasons.append("legal_hold_requirement_is_not_satisfied")
    if policy.minimum_versions > 1:
        if not capabilities.versioning:
            reasons.append("provider_does_not_attest_versioning_support")
        if status.version_count < policy.minimum_versions:
            reasons.append("observed_version_count_is_below_policy")

    deletion_ok = False
    if policy.deletion_requirement == "logical_delete":
        deletion_ok = status.deletion_state in {"logical_deleted", "cryptographically_erased", "physical_erase_attested"}
    elif policy.deletion_requirement == "cryptographic_erase":
        deletion_ok = capabilities.cryptographic_erase and status.deletion_state in {"cryptographically_erased", "physical_erase_attested"}
    else:
        deletion_ok = (
            capabilities.physical_erase_attestation
            and status.deletion_state == "physical_erase_attested"
            and bool(status.deletion_attestation_fingerprint)
        )
    # A present object is allowed while retention/hold is active. Deletion compliance is only
    # required once deletion has been requested/observed (anything other than present).
    if status.deletion_state != "present" and not deletion_ok:
        reasons.append("observed_deletion_state_does_not_satisfy_policy")

    payload = {
        "policy": policy.fingerprint,
        "capabilities": capabilities.fingerprint,
        "status": status.provider_status_fingerprint,
        "reasons": reasons,
        "deletion_requirement_satisfied": deletion_ok,
    }
    return RetentionComplianceEvaluation(
        compliant=not reasons,
        reasons=tuple(reasons),
        policy_fingerprint=policy.fingerprint,
        provider_capabilities_fingerprint=capabilities.fingerprint,
        provider_status_fingerprint=status.provider_status_fingerprint,
        deletion_requirement_satisfied=deletion_ok,
        fingerprint=hashlib.sha256(_canonical(payload)).hexdigest(),
    )


__all__ = [
    "ObjectGovernanceProvider",
    "ObjectRetentionPolicy",
    "ObjectVersionIdentity",
    "ProviderRetentionCapabilities",
    "ProviderRetentionStatus",
    "RetentionComplianceEvaluation",
    "evaluate_retention_compliance",
]
