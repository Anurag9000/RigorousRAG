"""Governed Ed25519 timestamp-authority attestations for custody envelopes."""

from __future__ import annotations

import base64
import hashlib
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

from cryptography.exceptions import InvalidSignature

from tools import evidence_graph_set_signed_retirement_restore_custody_export as _export
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature import (
    SignedCustodyEnvelope,
    _canonical_signature,
    _envelope_from_dict,
    _load_private,
    _load_public,
    _public_fingerprint,
    verify_signed_restore_chain_of_custody,
)
from tools.evidence_graph_set_signed_retirement_snapshot import (
    _atomic_create,
    _canonical_bytes,
    _path,
)
from tools.security import normalize_owner_id

_SCHEMA_VERSION = 1
_MAX_FUTURE_SECONDS = 86_400


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _envelope_digest(value: SignedCustodyEnvelope) -> str:
    if not isinstance(value, SignedCustodyEnvelope):
        raise ValueError("custody envelope is invalid.")
    return _canonical_digest(value.public_payload())


@dataclass(frozen=True)
class CustodyTimestampAttestation:
    owner_id: str
    authority_id: str
    key_id: str
    algorithm: str
    public_key_sha256: str
    custody_envelope_sha256: str
    custody_manifest_digest: str
    custody_chain_digest: str
    asserted_at: float
    nonce_sha256: str
    serial: str
    signature: str
    schema_version: int = _SCHEMA_VERSION
    rfc3161_token: bool = False
    hardware_clock_proven: bool = False
    contains_private_key_material: bool = False
    mutation_performed: bool = False

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        authority = _identifier(self.authority_id, "authority_id", 200)
        key_id = _identifier(self.key_id, "key_id", 200)
        algorithm = _identifier(self.algorithm, "algorithm", 30)
        if algorithm != "ed25519":
            raise ValueError("timestamp authority algorithm is unsupported.")
        fingerprint = _digest(self.public_key_sha256, "public_key_sha256")
        envelope_digest = _digest(
            self.custody_envelope_sha256,
            "custody_envelope_sha256",
        )
        manifest_digest = _digest(
            self.custody_manifest_digest,
            "custody_manifest_digest",
        )
        chain_digest = _digest(self.custody_chain_digest, "custody_chain_digest")
        asserted = _timestamp(self.asserted_at, "asserted_at")
        nonce_digest = _digest(self.nonce_sha256, "nonce_sha256")
        serial = _digest(self.serial, "serial")
        signature = base64.b64encode(
            _canonical_signature(self.signature)
        ).decode("ascii")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("timestamp attestation schema is unsupported.")
        if any(
            value is not False
            for value in (
                self.rfc3161_token,
                self.hardware_clock_proven,
                self.contains_private_key_material,
                self.mutation_performed,
            )
        ):
            raise ValueError("timestamp attestation safety flags must be false.")
        stable = {
            "scope": "rigorousrag-restore-custody-timestamp-attestation-v1",
            "owner_id": owner,
            "authority_id": authority,
            "key_id": key_id,
            "algorithm": algorithm,
            "public_key_sha256": fingerprint,
            "custody_envelope_sha256": envelope_digest,
            "custody_manifest_digest": manifest_digest,
            "custody_chain_digest": chain_digest,
            "asserted_at": asserted,
            "nonce_sha256": nonce_digest,
            "schema_version": self.schema_version,
        }
        if serial != _canonical_digest(stable):
            raise ValueError("timestamp serial differs from attestation scope.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "authority_id", authority)
        object.__setattr__(self, "key_id", key_id)
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "public_key_sha256", fingerprint)
        object.__setattr__(self, "custody_envelope_sha256", envelope_digest)
        object.__setattr__(self, "custody_manifest_digest", manifest_digest)
        object.__setattr__(self, "custody_chain_digest", chain_digest)
        object.__setattr__(self, "asserted_at", asserted)
        object.__setattr__(self, "nonce_sha256", nonce_digest)
        object.__setattr__(self, "serial", serial)
        object.__setattr__(self, "signature", signature)

    def signing_payload(self) -> dict[str, Any]:
        return {
            "scope": "rigorousrag-restore-custody-timestamp-attestation-v1",
            "owner_id": self.owner_id,
            "authority_id": self.authority_id,
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "public_key_sha256": self.public_key_sha256,
            "custody_envelope_sha256": self.custody_envelope_sha256,
            "custody_manifest_digest": self.custody_manifest_digest,
            "custody_chain_digest": self.custody_chain_digest,
            "asserted_at": self.asserted_at,
            "nonce_sha256": self.nonce_sha256,
            "schema_version": self.schema_version,
        }

    def public_payload(self) -> dict[str, Any]:
        return asdict(self)


def _from_dict(raw: dict[str, Any]) -> CustodyTimestampAttestation:
    expected = {
        "owner_id",
        "authority_id",
        "key_id",
        "algorithm",
        "public_key_sha256",
        "custody_envelope_sha256",
        "custody_manifest_digest",
        "custody_chain_digest",
        "asserted_at",
        "nonce_sha256",
        "serial",
        "signature",
        "schema_version",
        "rfc3161_token",
        "hardware_clock_proven",
        "contains_private_key_material",
        "mutation_performed",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("timestamp attestation schema is invalid.")
    return CustodyTimestampAttestation(**raw)


def issue_custody_timestamp_attestation(
    *,
    signed_envelope_path: str | os.PathLike[str],
    custody_signer_public_key_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    owner_id: str,
    authority_id: str,
    key_id: str,
    authority_private_key_path: str | os.PathLike[str],
    now: float | None = None,
    nonce: bytes | None = None,
) -> CustodyTimestampAttestation:
    envelope = verify_signed_restore_chain_of_custody(
        envelope_path=signed_envelope_path,
        public_key_path=custody_signer_public_key_path,
    )
    owner = normalize_owner_id(owner_id)
    if envelope.manifest.owner_id != owner:
        raise PermissionError("custody envelope owner differs.")
    asserted = _timestamp(time.time() if now is None else now, "now")
    if asserted < envelope.manifest.generated_at:
        raise ValueError("timestamp assertion predates custody manifest generation.")
    selected_nonce = os.urandom(32) if nonce is None else bytes(nonce)
    if len(selected_nonce) < 16 or len(selected_nonce) > 1024:
        raise ValueError("timestamp nonce length is invalid.")
    nonce_digest = hashlib.sha256(selected_nonce).hexdigest()
    private_key = _load_private(authority_private_key_path)
    fingerprint = _public_fingerprint(private_key.public_key())
    stable = {
        "scope": "rigorousrag-restore-custody-timestamp-attestation-v1",
        "owner_id": owner,
        "authority_id": _identifier(authority_id, "authority_id", 200),
        "key_id": _identifier(key_id, "key_id", 200),
        "algorithm": "ed25519",
        "public_key_sha256": fingerprint,
        "custody_envelope_sha256": _envelope_digest(envelope),
        "custody_manifest_digest": envelope.manifest.custody_manifest_digest,
        "custody_chain_digest": envelope.manifest.chain_digest,
        "asserted_at": asserted,
        "nonce_sha256": nonce_digest,
        "schema_version": _SCHEMA_VERSION,
    }
    serial = _canonical_digest(stable)
    signature = base64.b64encode(
        private_key.sign(_canonical_bytes(stable))
    ).decode("ascii")
    attestation = CustodyTimestampAttestation(
        **{key: value for key, value in stable.items() if key != "scope"},
        serial=serial,
        signature=signature,
    )
    _atomic_create(
        _path(output_path, label="output_path"),
        _canonical_bytes(attestation.public_payload()) + b"\n",
    )
    return attestation


def verify_custody_timestamp_attestation(
    *,
    attestation_path: str | os.PathLike[str],
    signed_envelope_path: str | os.PathLike[str],
    custody_signer_public_key_path: str | os.PathLike[str],
    authority_public_key_path: str | os.PathLike[str],
    expected_owner_id: str | None = None,
    expected_authority_id: str | None = None,
    expected_key_id: str | None = None,
    expected_public_key_sha256: str | None = None,
    now: float | None = None,
    maximum_future_seconds: float = 300.0,
) -> CustodyTimestampAttestation:
    raw = _export._decode_json(attestation_path, label="timestamp_attestation")
    attestation = _from_dict(raw)
    envelope = verify_signed_restore_chain_of_custody(
        envelope_path=signed_envelope_path,
        public_key_path=custody_signer_public_key_path,
    )
    if _envelope_digest(envelope) != attestation.custody_envelope_sha256:
        raise PermissionError("timestamp attestation envelope digest differs.")
    if envelope.manifest.owner_id != attestation.owner_id:
        raise PermissionError("timestamp attestation owner differs from envelope.")
    if envelope.manifest.custody_manifest_digest != attestation.custody_manifest_digest:
        raise PermissionError("timestamp attestation manifest digest differs.")
    if envelope.manifest.chain_digest != attestation.custody_chain_digest:
        raise PermissionError("timestamp attestation chain digest differs.")
    if attestation.asserted_at < envelope.manifest.generated_at:
        raise PermissionError("timestamp attestation predates custody manifest.")
    if expected_owner_id is not None and attestation.owner_id != normalize_owner_id(
        expected_owner_id
    ):
        raise PermissionError("timestamp attestation owner differs.")
    if expected_authority_id is not None and attestation.authority_id != _identifier(
        expected_authority_id, "expected_authority_id", 200
    ):
        raise PermissionError("timestamp authority ID differs.")
    if expected_key_id is not None and attestation.key_id != _identifier(
        expected_key_id, "expected_key_id", 200
    ):
        raise PermissionError("timestamp authority key ID differs.")
    current = _timestamp(time.time() if now is None else now, "now")
    future = float(maximum_future_seconds)
    if not (0.0 <= future <= _MAX_FUTURE_SECONDS):
        raise ValueError("maximum_future_seconds is invalid.")
    if attestation.asserted_at > current + future:
        raise PermissionError("timestamp attestation is too far in the future.")
    public_key = _load_public(authority_public_key_path)
    fingerprint = _public_fingerprint(public_key)
    if fingerprint != attestation.public_key_sha256:
        raise PermissionError("timestamp authority public-key fingerprint differs.")
    if expected_public_key_sha256 is not None and fingerprint != _digest(
        expected_public_key_sha256, "expected_public_key_sha256"
    ):
        raise PermissionError("expected timestamp authority fingerprint differs.")
    try:
        public_key.verify(
            _canonical_signature(attestation.signature),
            _canonical_bytes(attestation.signing_payload()),
        )
    except InvalidSignature as exc:
        raise PermissionError("timestamp attestation signature verification failed.") from exc
    return attestation


__all__ = [
    "CustodyTimestampAttestation",
    "issue_custody_timestamp_attestation",
    "verify_custody_timestamp_attestation",
]
