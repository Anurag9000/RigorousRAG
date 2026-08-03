"""Ed25519 signatures for complete external restore custody manifests."""

from __future__ import annotations

import os

from tools.evidence_graph_set_signed_retirement_restore_custody_rfc3161_trust import (
    Rfc3161TrustRegistry,
    verify_rfc3161_timestamp_response_with_profile,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature_contracts import (
    CustodySignatureVerificationReceipt,
    SignedCustodyEnvelope,
    TimestampedSignedCustodyEnvelope,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature_governance import (
    sign_governed_restore_chain_of_custody,
    verify_governed_signed_restore_chain_of_custody,
    verify_governed_timestamped_signed_restore_chain_of_custody_core,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature_io import (
    bind_rfc3161_timestamp_to_signed_custody,
    load_signed_custody_envelope,
    load_timestamped_signed_custody_envelope,
    sign_restore_chain_of_custody,
    verify_signed_envelope_object,
    verify_signed_restore_chain_of_custody,
    verify_timestamped_signed_restore_chain_of_custody,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature_keys import (
    CustodySignerKeyRegistry,
)


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
