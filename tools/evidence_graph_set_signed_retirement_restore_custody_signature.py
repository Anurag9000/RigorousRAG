"""Ed25519 signatures for complete external restore custody manifests.

The module is the stable compatibility facade for custody-signature consumers.
Descriptor-safe parsing and key loading live in ``custody_signature_io``; the
private helper names below remain available because timestamp, signer-registry
and operator modules were built against that original contract.
"""

from __future__ import annotations

import base64
import os

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from tools.evidence_graph_set_signed_retirement_restore_contracts import _digest
from tools.evidence_graph_set_signed_retirement_restore_custody_rfc3161_trust import (
    Rfc3161TrustRegistry,
    verify_rfc3161_timestamp_response_with_profile,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature_contracts import (
    CustodySignatureVerificationReceipt,
    SignedCustodyEnvelope,
    TimestampedSignedCustodyEnvelope,
    signed_envelope_from_dict,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature_governance import (
    sign_governed_restore_chain_of_custody,
    verify_governed_signed_restore_chain_of_custody,
    verify_governed_timestamped_signed_restore_chain_of_custody_core,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature_io import (
    bind_rfc3161_timestamp_to_signed_custody,
    load_private_key,
    load_public_key,
    load_signed_custody_envelope,
    load_timestamped_signed_custody_envelope,
    public_key_fingerprint,
    sign_restore_chain_of_custody,
    verify_signed_envelope_object,
    verify_signed_restore_chain_of_custody as _verify_signed_restore_chain_of_custody,
    verify_timestamped_signed_restore_chain_of_custody as _verify_timestamped_signed_restore_chain_of_custody,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature_keys import (
    CustodySignerKeyRegistry,
)


def _load_private(path: str | os.PathLike[str]) -> Ed25519PrivateKey:
    """Load an Ed25519 private key through the descriptor-safe I/O boundary."""

    return load_private_key(path)


def _load_public(path: str | os.PathLike[str]) -> Ed25519PublicKey:
    """Load an Ed25519 public key without exposing the tuple-based I/O API."""

    key, _fingerprint = load_public_key(path)
    return key


def _public_fingerprint(key: Ed25519PublicKey) -> str:
    """Return the canonical SHA-256 fingerprint of an Ed25519 public key."""

    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("public key is not Ed25519.")
    return public_key_fingerprint(key)


def _canonical_signature(value: str) -> bytes:
    """Decode one canonical 64-byte Ed25519 signature."""

    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError("Ed25519 signature encoding is invalid.")
    try:
        decoded = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ValueError("Ed25519 signature encoding is invalid.") from exc
    if len(decoded) != 64 or base64.b64encode(decoded).decode("ascii") != value:
        raise ValueError("Ed25519 signature encoding is not canonical.")
    return decoded


def _envelope_from_dict(raw: dict[str, object]) -> SignedCustodyEnvelope:
    """Validate a serialized envelope through the current contract parser."""

    return signed_envelope_from_dict(raw)


def verify_signed_restore_chain_of_custody(
    *,
    envelope_path: str | os.PathLike[str],
    public_key_path: str | os.PathLike[str],
    expected_key_id: str | None = None,
    expected_owner_id: str | None = None,
    expected_public_key_sha256: str | None = None,
) -> SignedCustodyEnvelope:
    """Verify a signed custody envelope and optional legacy fingerprint guard."""

    envelope = _verify_signed_restore_chain_of_custody(
        envelope_path=envelope_path,
        public_key_path=public_key_path,
        expected_key_id=expected_key_id,
        expected_owner_id=expected_owner_id,
    )
    if expected_public_key_sha256 is not None and envelope.public_key_sha256 != _digest(
        expected_public_key_sha256,
        "expected_public_key_sha256",
    ):
        raise PermissionError("expected public-key fingerprint differs.")
    return envelope


def verify_timestamped_signed_restore_chain_of_custody(
    *,
    envelope_path: str | os.PathLike[str],
    public_key_path: str | os.PathLike[str],
    expected_key_id: str | None = None,
    expected_owner_id: str | None = None,
    expected_public_key_sha256: str | None = None,
) -> TimestampedSignedCustodyEnvelope:
    """Verify a timestamped envelope and optional legacy fingerprint guard."""

    wrapped = _verify_timestamped_signed_restore_chain_of_custody(
        envelope_path=envelope_path,
        public_key_path=public_key_path,
        expected_key_id=expected_key_id,
        expected_owner_id=expected_owner_id,
    )
    if expected_public_key_sha256 is not None and (
        wrapped.signed_envelope.public_key_sha256
        != _digest(expected_public_key_sha256, "expected_public_key_sha256")
    ):
        raise PermissionError("expected public-key fingerprint differs.")
    return wrapped


def verify_governed_timestamped_signed_restore_chain_of_custody(
    *,
    registry: CustodySignerKeyRegistry,
    tsa_registry: Rfc3161TrustRegistry,
    owner_id: str,
    profile_id: str,
    timestamped_envelope_path: str | os.PathLike[str],
    request_bundle_path: str | os.PathLike[str],
    response_path: str | os.PathLike[str],
    trust_anchor_bundle_path: str | os.PathLike[str],
    untrusted_bundle_path: str | os.PathLike[str] | None = None,
    crl_bundle_path: str | os.PathLike[str] | None = None,
    openssl_binary: str = "openssl",
    timeout_seconds: int = 30,
    now: float | None = None,
    maximum_future_seconds: float = 300.0,
) -> CustodySignatureVerificationReceipt:
    return verify_governed_timestamped_signed_restore_chain_of_custody_core(
        registry=registry,
        tsa_registry=tsa_registry,
        owner_id=owner_id,
        profile_id=profile_id,
        timestamped_envelope_path=timestamped_envelope_path,
        request_bundle_path=request_bundle_path,
        response_path=response_path,
        trust_anchor_bundle_path=trust_anchor_bundle_path,
        tsa_verifier=verify_rfc3161_timestamp_response_with_profile,
        untrusted_bundle_path=untrusted_bundle_path,
        crl_bundle_path=crl_bundle_path,
        openssl_binary=openssl_binary,
        timeout_seconds=timeout_seconds,
        now=now,
        maximum_future_seconds=maximum_future_seconds,
    )


__all__ = [
    "CustodySignatureVerificationReceipt",
    "SignedCustodyEnvelope",
    "TimestampedSignedCustodyEnvelope",
    "_canonical_signature",
    "_envelope_from_dict",
    "_load_private",
    "_load_public",
    "_public_fingerprint",
    "bind_rfc3161_timestamp_to_signed_custody",
    "load_signed_custody_envelope",
    "load_timestamped_signed_custody_envelope",
    "sign_governed_restore_chain_of_custody",
    "sign_restore_chain_of_custody",
    "verify_governed_signed_restore_chain_of_custody",
    "verify_governed_timestamped_signed_restore_chain_of_custody",
    "verify_signed_envelope_object",
    "verify_signed_restore_chain_of_custody",
    "verify_timestamped_signed_restore_chain_of_custody",
]
