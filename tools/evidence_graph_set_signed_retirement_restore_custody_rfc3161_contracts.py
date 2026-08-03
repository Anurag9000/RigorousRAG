"""Validated contracts for RFC 3161 custody timestamp interoperability."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from asn1crypto import cms, core, tsp

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _integer,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_snapshot import _canonical_bytes
from tools.security import normalize_owner_id

SCHEMA_VERSION = 1
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_BUNDLE_BYTES = 4 * 1024 * 1024
MAX_VERIFIER_OUTPUT_BYTES = 64 * 1024
SHA256_OID = "2.16.840.1.101.3.4.2.1"
NONTERMINAL_STATUS = frozenset(
    {"waiting", "revocation_warning", "revocation_notification"}
)
GRANTED_STATUS = frozenset({"granted", "granted_with_mods"})


class Rfc3161TimeStampResp(core.Sequence):
    """RFC 3161 TimeStampResp with its token correctly marked OPTIONAL."""

    _fields = [
        ("status", tsp.PKIStatusInfo),
        ("time_stamp_token", cms.ContentInfo, {"optional": True}),
    ]


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def optional_oid(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    rendered = _identifier(value, label, 200)
    parts = rendered.split(".")
    if len(parts) < 2 or any(not part.isdigit() for part in parts):
        raise ValueError(f"{label} must be a dotted-decimal object identifier.")
    integers = [int(part) for part in parts]
    if integers[0] not in (0, 1, 2) or (integers[0] < 2 and integers[1] > 39):
        raise ValueError(f"{label} is invalid.")
    if any(part < 0 or part > (1 << 63) - 1 for part in integers):
        raise ValueError(f"{label} is invalid.")
    return ".".join(str(part) for part in integers)


def nonce_decimal(value: int | str) -> str:
    if isinstance(value, bool):
        raise ValueError("nonce must be a positive integer.")
    if isinstance(value, str):
        if not value or not value.isdigit() or (
            len(value) > 1 and value.startswith("0")
        ):
            raise ValueError("nonce must be canonical decimal.")
        selected = int(value)
    elif isinstance(value, int):
        selected = value
    else:
        raise ValueError("nonce must be a positive integer.")
    if selected <= 0 or selected.bit_length() < 64 or selected.bit_length() > 256:
        raise ValueError("nonce must contain between 64 and 256 significant bits.")
    return str(selected)


def request_der(
    *,
    subject_sha256: str,
    nonce_decimal_value: str,
    requested_policy_oid: str | None,
) -> bytes:
    payload: dict[str, Any] = {
        "version": "v1",
        "message_imprint": {
            "hash_algorithm": {"algorithm": "sha256"},
            "hashed_message": bytes.fromhex(subject_sha256),
        },
        "nonce": int(nonce_decimal_value),
        "cert_req": True,
    }
    if requested_policy_oid is not None:
        payload["req_policy"] = requested_policy_oid
    return tsp.TimeStampReq(payload).dump()


@dataclass(frozen=True)
class Rfc3161TimestampRequestBundle:
    owner_id: str
    subject_sha256: str
    subject_size_bytes: int
    hash_algorithm: str
    hash_algorithm_oid: str
    nonce_decimal: str
    nonce_sha256: str
    requested_policy_oid: str | None
    cert_req: bool
    request_der_base64: str
    request_sha256: str
    created_at: float
    bundle_digest: str
    schema_version: int = SCHEMA_VERSION
    rfc3161_request: bool = True
    trusted_time_obtained: bool = False
    contains_private_key_material: bool = False
    mutation_performed: bool = False

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        subject = _digest(self.subject_sha256, "subject_sha256")
        size = _integer(
            self.subject_size_bytes,
            "subject_size_bytes",
            1,
            MAX_INPUT_BYTES,
        )
        algorithm = _identifier(self.hash_algorithm, "hash_algorithm", 30)
        if algorithm != "sha256" or self.hash_algorithm_oid != SHA256_OID:
            raise ValueError("RFC 3161 request hash algorithm must be SHA-256.")
        nonce = nonce_decimal(self.nonce_decimal)
        nonce_digest = _digest(self.nonce_sha256, "nonce_sha256")
        if nonce_digest != hashlib.sha256(nonce.encode("ascii")).hexdigest():
            raise ValueError("nonce_sha256 differs from nonce.")
        policy = optional_oid(self.requested_policy_oid, "requested_policy_oid")
        if self.cert_req is not True:
            raise ValueError("RFC 3161 request must request the TSA certificate.")
        try:
            request = base64.b64decode(
                self.request_der_base64.encode("ascii"), validate=True
            )
        except (UnicodeEncodeError, ValueError) as exc:
            raise ValueError("request DER base64 is invalid.") from exc
        expected = request_der(
            subject_sha256=subject,
            nonce_decimal_value=nonce,
            requested_policy_oid=policy,
        )
        if request != expected:
            raise ValueError("request DER differs from request scope.")
        request_digest = _digest(self.request_sha256, "request_sha256")
        if request_digest != hashlib.sha256(request).hexdigest():
            raise ValueError("request_sha256 differs from DER.")
        created = _timestamp(self.created_at, "created_at")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("RFC 3161 request bundle schema is unsupported.")
        if self.rfc3161_request is not True:
            raise ValueError("rfc3161_request must be true.")
        if any(
            value is not False
            for value in (
                self.trusted_time_obtained,
                self.contains_private_key_material,
                self.mutation_performed,
            )
        ):
            raise ValueError("RFC 3161 request safety flags are invalid.")
        stable = {
            "scope": "rigorousrag-restore-custody-rfc3161-request-v1",
            "owner_id": owner,
            "subject_sha256": subject,
            "subject_size_bytes": size,
            "hash_algorithm": algorithm,
            "hash_algorithm_oid": SHA256_OID,
            "nonce_decimal": nonce,
            "nonce_sha256": nonce_digest,
            "requested_policy_oid": policy,
            "cert_req": True,
            "request_sha256": request_digest,
            "created_at": created,
            "schema_version": self.schema_version,
        }
        bundle_digest = _digest(self.bundle_digest, "bundle_digest")
        if bundle_digest != canonical_digest(stable):
            raise ValueError("bundle_digest differs from request bundle.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "subject_sha256", subject)
        object.__setattr__(self, "subject_size_bytes", size)
        object.__setattr__(self, "hash_algorithm", algorithm)
        object.__setattr__(self, "nonce_decimal", nonce)
        object.__setattr__(self, "nonce_sha256", nonce_digest)
        object.__setattr__(self, "requested_policy_oid", policy)
        object.__setattr__(self, "request_sha256", request_digest)
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "bundle_digest", bundle_digest)

    def request_der(self) -> bytes:
        return base64.b64decode(self.request_der_base64.encode("ascii"), validate=True)

    def public_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Rfc3161TimestampVerificationReceipt:
    owner_id: str
    request_bundle_digest: str
    request_sha256: str
    subject_sha256: str
    response_sha256: str
    token_sha256: str
    status: str
    policy_oid: str
    message_imprint_sha256: str
    nonce_sha256: str
    serial_decimal: str
    generated_at_rfc3339: str
    generated_at_unix: float
    accuracy_seconds: int | None
    accuracy_millis: int | None
    accuracy_micros: int | None
    ordering: bool
    signer_certificate_sha256: str
    signer_certificate_serial_hex: str
    signer_public_key_algorithm: str
    signature_algorithm: str
    digest_algorithm: str
    trust_anchor_bundle_sha256: str
    untrusted_bundle_sha256: str | None
    crl_bundle_sha256: str | None
    verifier_version_sha256: str
    receipt_digest: str
    schema_version: int = SCHEMA_VERSION
    rfc3161_token: bool = True
    message_imprint_verified: bool = True
    nonce_verified: bool = True
    policy_verified: bool = True
    cms_signature_verified: bool = True
    certificate_chain_verified: bool = True
    revocation_checked: bool = False
    tsa_eku_verified: bool = True
    ess_signer_binding_verified: bool = True
    independently_trusted_clock_proven: bool = False
    hardware_clock_proven: bool = False
    contains_private_key_material: bool = False
    mutation_performed: bool = False

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        digest_fields = (
            "request_bundle_digest",
            "request_sha256",
            "subject_sha256",
            "response_sha256",
            "token_sha256",
            "message_imprint_sha256",
            "nonce_sha256",
            "signer_certificate_sha256",
            "trust_anchor_bundle_sha256",
            "verifier_version_sha256",
        )
        for field in digest_fields:
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        for field in ("untrusted_bundle_sha256", "crl_bundle_sha256"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _digest(value, field))
        status = _identifier(self.status, "status", 50)
        if status not in GRANTED_STATUS:
            raise ValueError("RFC 3161 receipt status is not granted.")
        policy = optional_oid(self.policy_oid, "policy_oid")
        if policy is None:
            raise ValueError("RFC 3161 receipt requires one policy OID.")
        serial = self.serial_decimal
        if not isinstance(serial, str) or not serial.isdigit() or int(serial) <= 0:
            raise ValueError("serial_decimal is invalid.")
        generated = self.generated_at_rfc3339
        if not isinstance(generated, str) or not generated.endswith("Z"):
            raise ValueError("generated_at_rfc3339 must be UTC RFC 3339 text.")
        _timestamp(self.generated_at_unix, "generated_at_unix")
        for field in ("accuracy_seconds", "accuracy_millis", "accuracy_micros"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(
                    self, field, _integer(value, field, 0, 1_000_000)
                )
        if not isinstance(self.ordering, bool):
            raise ValueError("ordering must be boolean.")
        serial_hex = _identifier(
            self.signer_certificate_serial_hex,
            "signer_certificate_serial_hex",
            512,
        )
        if any(character not in "0123456789abcdef" for character in serial_hex):
            raise ValueError("signer_certificate_serial_hex is invalid.")
        for field in (
            "signer_public_key_algorithm",
            "signature_algorithm",
            "digest_algorithm",
        ):
            object.__setattr__(
                self, field, _identifier(getattr(self, field), field, 100)
            )
        required_true = (
            "rfc3161_token",
            "message_imprint_verified",
            "nonce_verified",
            "policy_verified",
            "cms_signature_verified",
            "certificate_chain_verified",
            "tsa_eku_verified",
            "ess_signer_binding_verified",
        )
        if any(getattr(self, field) is not True for field in required_true):
            raise ValueError("RFC 3161 verification flags must be true.")
        if self.revocation_checked != (self.crl_bundle_sha256 is not None):
            raise ValueError("revocation_checked differs from CRL evidence.")
        if any(
            value is not False
            for value in (
                self.independently_trusted_clock_proven,
                self.hardware_clock_proven,
                self.contains_private_key_material,
                self.mutation_performed,
            )
        ):
            raise ValueError("RFC 3161 receipt non-claim flags must be false.")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("RFC 3161 receipt schema is unsupported.")
        receipt_digest = _digest(self.receipt_digest, "receipt_digest")
        if receipt_digest != canonical_digest(self.stable_payload()):
            raise ValueError("receipt_digest differs from RFC 3161 receipt.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "policy_oid", policy)
        object.__setattr__(self, "serial_decimal", serial)
        object.__setattr__(self, "signer_certificate_serial_hex", serial_hex)
        object.__setattr__(self, "receipt_digest", receipt_digest)

    def stable_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("receipt_digest", None)
        for key in (
            "rfc3161_token",
            "message_imprint_verified",
            "nonce_verified",
            "policy_verified",
            "cms_signature_verified",
            "certificate_chain_verified",
            "revocation_checked",
            "tsa_eku_verified",
            "ess_signer_binding_verified",
            "independently_trusted_clock_proven",
            "hardware_clock_proven",
            "contains_private_key_material",
            "mutation_performed",
        ):
            payload.pop(key, None)
        return {
            "scope": "rigorousrag-restore-custody-rfc3161-receipt-v1",
            **payload,
        }

    def public_payload(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "GRANTED_STATUS",
    "MAX_BUNDLE_BYTES",
    "MAX_INPUT_BYTES",
    "MAX_RESPONSE_BYTES",
    "MAX_VERIFIER_OUTPUT_BYTES",
    "NONTERMINAL_STATUS",
    "Rfc3161TimeStampResp",
    "Rfc3161TimestampRequestBundle",
    "Rfc3161TimestampVerificationReceipt",
    "SCHEMA_VERSION",
    "SHA256_OID",
    "canonical_digest",
    "nonce_decimal",
    "optional_oid",
    "request_der",
]
