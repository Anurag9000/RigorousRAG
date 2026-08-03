"""Governed Ed25519 custody signing and key-validity verification."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_contracts import _timestamp
from tools.evidence_graph_set_signed_retirement_restore_custody_export_boundary import (
    verify_restore_chain_of_custody,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature_contracts import (
    CustodySignatureVerificationReceipt,
    SignedCustodyEnvelope,
    TimestampedSignedCustodyEnvelope,
    canonical_digest,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature_io import (
    load_timestamped_signed_custody_envelope,
    sign_restore_chain_of_custody,
    verify_signed_envelope_object,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature_keys import (
    CustodySignerKeyRecord,
    CustodySignerKeyRegistry,
)
from tools.security import normalize_owner_id


def sign_governed_restore_chain_of_custody(
    *,
    registry: CustodySignerKeyRegistry,
    owner_id: str,
    key_id: str,
    manifest_path: str | os.PathLike[str],
    private_key_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    now: float | None = None,
) -> tuple[SignedCustodyEnvelope, CustodySignerKeyRecord]:
    if not isinstance(registry, CustodySignerKeyRegistry):
        raise ValueError("registry must be CustodySignerKeyRegistry.")
    timestamp = _timestamp(time.time() if now is None else now, "now")
    record = registry.get(owner_id=owner_id, key_id=key_id)
    if record.state != "active" or not record.permits(
        verification_time=None,
        now=timestamp,
    ):
        raise PermissionError("signer key is not active for current signing.")
    manifest = verify_restore_chain_of_custody(manifest_path)
    if manifest.owner_id != record.owner_id:
        raise PermissionError("custody manifest owner differs from signer key owner.")
    envelope = sign_restore_chain_of_custody(
        manifest_path=manifest_path,
        output_path=output_path,
        key_id=record.key_id,
        private_key_path=private_key_path,
        expected_public_key_sha256=record.public_key_sha256,
        now=timestamp,
    )
    return envelope, record


def governed_verification_receipt(
    *,
    record: CustodySignerKeyRecord,
    signed_envelope: SignedCustodyEnvelope,
    verification_time_source: str,
    verification_time: float,
    rfc3161_receipt_digest: str | None,
) -> CustodySignatureVerificationReceipt:
    values = {
        "owner_id": record.owner_id,
        "key_id": record.key_id,
        "public_key_sha256": record.public_key_sha256,
        "manifest_chain_digest": signed_envelope.manifest.chain_digest,
        "envelope_digest": signed_envelope.envelope_digest,
        "registry_record_digest": record.record_digest,
        "key_state": record.state,
        "verification_time_source": verification_time_source,
        "verification_time": verification_time,
        "rfc3161_receipt_digest": rfc3161_receipt_digest,
        "historical_retired_key_verified": (
            record.state == "retired" and verification_time_source == "rfc3161"
        ),
        "schema_version": 1,
    }
    stable = {
        "scope": "rigorousrag-restore-custody-ed25519-verification-v1",
        **values,
    }
    return CustodySignatureVerificationReceipt(
        **values,
        verification_digest=canonical_digest(stable),
        registry_scope_verified=True,
        trusted_timestamp_reverified=verification_time_source == "rfc3161",
    )


def verify_governed_signed_restore_chain_of_custody(
    *,
    registry: CustodySignerKeyRegistry,
    owner_id: str,
    signed_envelope: SignedCustodyEnvelope,
    now: float | None = None,
) -> CustodySignatureVerificationReceipt:
    if not isinstance(registry, CustodySignerKeyRegistry):
        raise ValueError("registry must be CustodySignerKeyRegistry.")
    if not isinstance(signed_envelope, SignedCustodyEnvelope):
        raise ValueError("signed_envelope must be SignedCustodyEnvelope.")
    owner = normalize_owner_id(owner_id)
    if signed_envelope.owner_id != owner:
        raise PermissionError("signed custody owner differs from verification scope.")
    record = registry.get(owner_id=owner, key_id=signed_envelope.key_id)
    if record.public_key_sha256 != signed_envelope.public_key_sha256:
        raise PermissionError("signed custody key differs from governed registry.")
    verify_signed_envelope_object(signed_envelope, record.public_key())
    current = _timestamp(time.time() if now is None else now, "now")
    if not record.permits(verification_time=None, now=current):
        raise PermissionError("custody signature is outside governed key validity.")
    return governed_verification_receipt(
        record=record,
        signed_envelope=signed_envelope,
        verification_time_source="current_time",
        verification_time=current,
        rfc3161_receipt_digest=None,
    )


def verify_governed_timestamped_signed_restore_chain_of_custody_core(
    *,
    registry: CustodySignerKeyRegistry,
    tsa_registry: Any,
    owner_id: str,
    profile_id: str,
    timestamped_envelope_path: str | os.PathLike[str],
    request_bundle_path: str | os.PathLike[str],
    response_path: str | os.PathLike[str],
    trust_anchor_bundle_path: str | os.PathLike[str],
    tsa_verifier: Callable[..., tuple[Any, Any]],
    untrusted_bundle_path: str | os.PathLike[str] | None = None,
    crl_bundle_path: str | os.PathLike[str] | None = None,
    openssl_binary: str = "openssl",
    timeout_seconds: int = 30,
    now: float | None = None,
    maximum_future_seconds: float = 300.0,
) -> CustodySignatureVerificationReceipt:
    if not isinstance(registry, CustodySignerKeyRegistry):
        raise ValueError("registry must be CustodySignerKeyRegistry.")
    if not callable(tsa_verifier):
        raise ValueError("tsa_verifier must be callable.")
    owner = normalize_owner_id(owner_id)
    wrapped: TimestampedSignedCustodyEnvelope = (
        load_timestamped_signed_custody_envelope(timestamped_envelope_path)
    )
    envelope = wrapped.signed_envelope
    if envelope.owner_id != owner:
        raise PermissionError("signed custody owner differs from verification scope.")
    record = registry.get(owner_id=owner, key_id=envelope.key_id)
    if record.public_key_sha256 != envelope.public_key_sha256:
        raise PermissionError("signed custody key differs from governed registry.")
    verify_signed_envelope_object(envelope, record.public_key())
    receipt, _profile = tsa_verifier(
        registry=tsa_registry,
        owner_id=owner,
        profile_id=profile_id,
        request_bundle_path=request_bundle_path,
        response_path=response_path,
        trust_anchor_bundle_path=trust_anchor_bundle_path,
        output_receipt_path=None,
        untrusted_bundle_path=untrusted_bundle_path,
        crl_bundle_path=crl_bundle_path,
        openssl_binary=openssl_binary,
        timeout_seconds=timeout_seconds,
        now=now,
        maximum_future_seconds=maximum_future_seconds,
    )
    if receipt.receipt_digest != wrapped.timestamp_receipt.receipt_digest:
        raise PermissionError("governed RFC 3161 receipt differs from bound receipt.")
    if receipt.subject_sha256 != wrapped.timestamped_subject_sha256:
        raise PermissionError("governed RFC 3161 receipt differs from signed envelope.")
    current = _timestamp(time.time() if now is None else now, "now")
    if not record.permits(verification_time=receipt.generated_at_unix, now=current):
        raise PermissionError("timestamped custody signature is outside key validity.")
    return governed_verification_receipt(
        record=record,
        signed_envelope=envelope,
        verification_time_source="rfc3161",
        verification_time=receipt.generated_at_unix,
        rfc3161_receipt_digest=receipt.receipt_digest,
    )


__all__ = [
    "governed_verification_receipt",
    "sign_governed_restore_chain_of_custody",
    "verify_governed_signed_restore_chain_of_custody",
    "verify_governed_timestamped_signed_restore_chain_of_custody_core",
]
