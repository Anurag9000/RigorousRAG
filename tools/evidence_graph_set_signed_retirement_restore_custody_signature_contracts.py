"""Validated contracts for Ed25519-signed restore custody evidence."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_export_boundary import (
    CustodyArtifactEvidence,
    RestoreChainOfCustodyManifest,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_rfc3161_contracts import (
    Rfc3161TimestampVerificationReceipt,
)
from tools.evidence_graph_set_signed_retirement_snapshot import _canonical_bytes
from tools.security import normalize_owner_id

SCHEMA_VERSION = 1
TIMESTAMP_SCHEMA_VERSION = 1
ALGORITHM = "ed25519"


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def manifest_from_dict(raw: dict[str, Any]) -> RestoreChainOfCustodyManifest:
    expected = set(RestoreChainOfCustodyManifest.__dataclass_fields__)
    if set(raw) != expected or not isinstance(raw.get("artifacts"), list):
        raise ValueError("custody manifest schema is invalid.")
    return RestoreChainOfCustodyManifest(
        **{
            **raw,
            "artifacts": tuple(
                CustodyArtifactEvidence(**value) for value in raw["artifacts"]
            ),
        }
    )


def receipt_from_dict(
    raw: dict[str, Any],
) -> Rfc3161TimestampVerificationReceipt:
    expected = set(Rfc3161TimestampVerificationReceipt.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError("RFC 3161 receipt schema is invalid.")
    return Rfc3161TimestampVerificationReceipt(**raw)


@dataclass(frozen=True)
class SignedCustodyEnvelope:
    owner_id: str
    key_id: str
    algorithm: str
    public_key_sha256: str
    manifest: RestoreChainOfCustodyManifest
    created_at: float
    signature_base64: str
    envelope_digest: str
    schema_version: int = SCHEMA_VERSION
    contains_private_key_material: bool = False
    contains_raw_paths: bool = False
    mutation_performed: bool = False

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        key_id = _identifier(self.key_id, "key_id", 200)
        algorithm = _identifier(self.algorithm, "algorithm", 30)
        if algorithm != ALGORITHM:
            raise ValueError("custody signature algorithm is unsupported.")
        fingerprint = _digest(self.public_key_sha256, "public_key_sha256")
        if not isinstance(self.manifest, RestoreChainOfCustodyManifest):
            raise ValueError("signed custody manifest is invalid.")
        if self.manifest.owner_id != owner:
            raise ValueError("signed custody owner differs from manifest owner.")
        created = _timestamp(self.created_at, "created_at")
        try:
            signature = base64.b64decode(
                self.signature_base64.encode("ascii"),
                validate=True,
            )
        except (UnicodeEncodeError, ValueError) as exc:
            raise ValueError("signature_base64 is invalid.") from exc
        if len(signature) != 64:
            raise ValueError("Ed25519 signature must contain 64 bytes.")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("signed custody envelope schema is unsupported.")
        if any(
            value is not False
            for value in (
                self.contains_private_key_material,
                self.contains_raw_paths,
                self.mutation_performed,
            )
        ):
            raise ValueError("signed custody envelope safety flags are invalid.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "key_id", key_id)
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "public_key_sha256", fingerprint)
        object.__setattr__(self, "created_at", created)
        object.__setattr__(
            self,
            "signature_base64",
            base64.b64encode(signature).decode("ascii"),
        )
        digest = _digest(self.envelope_digest, "envelope_digest")
        if digest != canonical_digest(self.digest_payload()):
            raise ValueError("envelope_digest differs from signed custody envelope.")
        object.__setattr__(self, "envelope_digest", digest)

    def signing_payload(self) -> dict[str, Any]:
        return {
            "scope": "rigorousrag-restore-custody-ed25519-signature-v1",
            "owner_id": self.owner_id,
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "public_key_sha256": self.public_key_sha256,
            "manifest": self.manifest.public_payload(),
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }

    def digest_payload(self) -> dict[str, Any]:
        return {**self.signing_payload(), "signature_base64": self.signature_base64}

    def signature_bytes(self) -> bytes:
        return base64.b64decode(self.signature_base64.encode("ascii"), validate=True)

    def canonical_export_bytes(self) -> bytes:
        return _canonical_bytes(self.public_payload()) + b"\n"

    def public_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimestampedSignedCustodyEnvelope:
    signed_envelope: SignedCustodyEnvelope
    timestamp_receipt: Rfc3161TimestampVerificationReceipt
    timestamped_subject_sha256: str
    binding_digest: str
    schema_version: int = TIMESTAMP_SCHEMA_VERSION
    contains_private_key_material: bool = False
    contains_raw_paths: bool = False
    mutation_performed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.signed_envelope, SignedCustodyEnvelope):
            raise ValueError("timestamped signed envelope is invalid.")
        if not isinstance(
            self.timestamp_receipt,
            Rfc3161TimestampVerificationReceipt,
        ):
            raise ValueError("timestamp receipt is invalid.")
        if self.timestamp_receipt.owner_id != self.signed_envelope.owner_id:
            raise ValueError("timestamp receipt owner differs from signed envelope owner.")
        subject = _digest(
            self.timestamped_subject_sha256,
            "timestamped_subject_sha256",
        )
        expected = hashlib.sha256(
            self.signed_envelope.canonical_export_bytes()
        ).hexdigest()
        if subject != expected or subject != self.timestamp_receipt.subject_sha256:
            raise ValueError("RFC 3161 receipt does not timestamp the signed envelope.")
        if self.schema_version != TIMESTAMP_SCHEMA_VERSION:
            raise ValueError("timestamped signed envelope schema is unsupported.")
        if any(
            value is not False
            for value in (
                self.contains_private_key_material,
                self.contains_raw_paths,
                self.mutation_performed,
            )
        ):
            raise ValueError("timestamped envelope safety flags are invalid.")
        object.__setattr__(self, "timestamped_subject_sha256", subject)
        binding = _digest(self.binding_digest, "binding_digest")
        if binding != canonical_digest(self.stable_payload()):
            raise ValueError("binding_digest differs from timestamped envelope.")
        object.__setattr__(self, "binding_digest", binding)

    def stable_payload(self) -> dict[str, Any]:
        return {
            "scope": "rigorousrag-restore-custody-ed25519-rfc3161-binding-v1",
            "signed_envelope": self.signed_envelope.public_payload(),
            "timestamp_receipt": self.timestamp_receipt.public_payload(),
            "timestamped_subject_sha256": self.timestamped_subject_sha256,
            "schema_version": self.schema_version,
        }

    def public_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CustodySignatureVerificationReceipt:
    owner_id: str
    key_id: str
    public_key_sha256: str
    manifest_chain_digest: str
    envelope_digest: str
    registry_record_digest: str | None
    key_state: str
    verification_time_source: str
    verification_time: float
    rfc3161_receipt_digest: str | None
    historical_retired_key_verified: bool
    verification_digest: str
    schema_version: int = 1
    signature_verified: bool = True
    manifest_integrity_verified: bool = True
    registry_scope_verified: bool = False
    trusted_timestamp_reverified: bool = False
    contains_private_key_material: bool = False
    contains_raw_paths: bool = False
    mutation_performed: bool = False

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        key_id = _identifier(self.key_id, "key_id", 200)
        for field in (
            "public_key_sha256",
            "manifest_chain_digest",
            "envelope_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        registry_digest = self.registry_record_digest
        if registry_digest is not None:
            registry_digest = _digest(registry_digest, "registry_record_digest")
        key_state = _identifier(self.key_state, "key_state", 30)
        if key_state not in {"unregistered", "active", "retired"}:
            raise ValueError("key_state is unsupported.")
        source = _identifier(
            self.verification_time_source,
            "verification_time_source",
            30,
        )
        if source not in {"current_time", "rfc3161"}:
            raise ValueError("verification_time_source is unsupported.")
        verified_at = _timestamp(self.verification_time, "verification_time")
        timestamp_digest = self.rfc3161_receipt_digest
        if timestamp_digest is not None:
            timestamp_digest = _digest(
                timestamp_digest,
                "rfc3161_receipt_digest",
            )
        if not isinstance(self.historical_retired_key_verified, bool):
            raise ValueError("historical_retired_key_verified must be boolean.")
        if self.schema_version != 1:
            raise ValueError("signature verification receipt schema is unsupported.")
        if self.signature_verified is not True or self.manifest_integrity_verified is not True:
            raise ValueError("signature verification receipt flags are invalid.")
        if self.registry_scope_verified != (registry_digest is not None):
            raise ValueError("registry_scope_verified differs from registry evidence.")
        if self.trusted_timestamp_reverified != (source == "rfc3161"):
            raise ValueError("trusted timestamp flag differs from time source.")
        if source == "rfc3161" and timestamp_digest is None:
            raise ValueError("RFC 3161 verification requires a receipt digest.")
        if self.historical_retired_key_verified != (
            key_state == "retired" and source == "rfc3161"
        ):
            raise ValueError("retired-key verification flag is inconsistent.")
        if any(
            value is not False
            for value in (
                self.contains_private_key_material,
                self.contains_raw_paths,
                self.mutation_performed,
            )
        ):
            raise ValueError("signature verification safety flags are invalid.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "key_id", key_id)
        object.__setattr__(self, "registry_record_digest", registry_digest)
        object.__setattr__(self, "key_state", key_state)
        object.__setattr__(self, "verification_time_source", source)
        object.__setattr__(self, "verification_time", verified_at)
        object.__setattr__(self, "rfc3161_receipt_digest", timestamp_digest)
        digest = _digest(self.verification_digest, "verification_digest")
        if digest != canonical_digest(self.stable_payload()):
            raise ValueError("verification_digest differs from verification receipt.")
        object.__setattr__(self, "verification_digest", digest)

    def stable_payload(self) -> dict[str, Any]:
        return {
            "scope": "rigorousrag-restore-custody-ed25519-verification-v1",
            "owner_id": self.owner_id,
            "key_id": self.key_id,
            "public_key_sha256": self.public_key_sha256,
            "manifest_chain_digest": self.manifest_chain_digest,
            "envelope_digest": self.envelope_digest,
            "registry_record_digest": self.registry_record_digest,
            "key_state": self.key_state,
            "verification_time_source": self.verification_time_source,
            "verification_time": self.verification_time,
            "rfc3161_receipt_digest": self.rfc3161_receipt_digest,
            "historical_retired_key_verified": self.historical_retired_key_verified,
            "schema_version": self.schema_version,
        }

    def public_payload(self) -> dict[str, Any]:
        return asdict(self)


def signed_envelope_from_dict(raw: dict[str, Any]) -> SignedCustodyEnvelope:
    expected = set(SignedCustodyEnvelope.__dataclass_fields__)
    if set(raw) != expected or not isinstance(raw.get("manifest"), dict):
        raise ValueError("signed custody envelope schema is invalid.")
    return SignedCustodyEnvelope(
        **{**raw, "manifest": manifest_from_dict(raw["manifest"])}
    )


def timestamped_envelope_from_dict(
    raw: dict[str, Any],
) -> TimestampedSignedCustodyEnvelope:
    expected = set(TimestampedSignedCustodyEnvelope.__dataclass_fields__)
    if (
        set(raw) != expected
        or not isinstance(raw.get("signed_envelope"), dict)
        or not isinstance(raw.get("timestamp_receipt"), dict)
    ):
        raise ValueError("timestamped signed custody schema is invalid.")
    return TimestampedSignedCustodyEnvelope(
        **{
            **raw,
            "signed_envelope": signed_envelope_from_dict(raw["signed_envelope"]),
            "timestamp_receipt": receipt_from_dict(raw["timestamp_receipt"]),
        }
    )


__all__ = [
    "ALGORITHM",
    "CustodySignatureVerificationReceipt",
    "SCHEMA_VERSION",
    "SignedCustodyEnvelope",
    "TIMESTAMP_SCHEMA_VERSION",
    "TimestampedSignedCustodyEnvelope",
    "canonical_digest",
    "signed_envelope_from_dict",
    "timestamped_envelope_from_dict",
]
