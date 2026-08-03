"""Descriptor-safe Ed25519 signing and offline custody verification."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import time
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_export_boundary import (
    verify_restore_chain_of_custody,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_rfc3161_io import (
    verify_rfc3161_timestamp_receipt,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature_contracts import (
    ALGORITHM,
    SCHEMA_VERSION,
    TIMESTAMP_SCHEMA_VERSION,
    SignedCustodyEnvelope,
    TimestampedSignedCustodyEnvelope,
    canonical_digest,
    signed_envelope_from_dict,
    timestamped_envelope_from_dict,
)
from tools.evidence_graph_set_signed_retirement_snapshot import (
    _atomic_create,
    _canonical_bytes,
    _pairs,
    _path,
)
from tools.security import normalize_owner_id

MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_KEY_BYTES = 64 * 1024


def read_regular(
    path: str | os.PathLike[str],
    *,
    label: str,
    maximum: int,
    require_private_permissions: bool = False,
) -> bytes:
    selected = _path(path, label=label)
    descriptor = os.open(
        selected,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file.")
        if before.st_size <= 0 or before.st_size > maximum:
            raise ValueError(f"{label} size is invalid.")
        if (
            require_private_permissions
            and os.name != "nt"
            and stat.S_IMODE(before.st_mode) & 0o077
        ):
            raise PermissionError(f"{label} permissions are too broad.")
        remaining = int(before.st_size)
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError(f"{label} changed while being read.")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(f"{label} grew while being read.")
        after = os.fstat(descriptor)
        if (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_size),
        ) != (
            int(before.st_dev),
            int(before.st_ino),
            int(before.st_size),
        ):
            raise RuntimeError(f"{label} identity changed while being read.")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_private_key(path: str | os.PathLike[str]) -> Ed25519PrivateKey:
    payload = read_regular(
        path,
        label="private_key_path",
        maximum=MAX_KEY_BYTES,
        require_private_permissions=True,
    )
    try:
        key = serialization.load_pem_private_key(payload, password=None)
    except (TypeError, ValueError) as exc:
        raise ValueError("private key must be unencrypted PEM PKCS#8.") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("private key must be Ed25519.")
    return key


def load_public_key(
    path: str | os.PathLike[str],
) -> tuple[Ed25519PublicKey, str]:
    payload = read_regular(path, label="public_key_path", maximum=MAX_KEY_BYTES)
    try:
        key = serialization.load_pem_public_key(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("public key must be PEM SubjectPublicKeyInfo.") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("public key must be Ed25519.")
    return key, public_key_fingerprint(key)


def public_key_fingerprint(key: Ed25519PublicKey) -> str:
    raw = key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def decode_json(path: str | os.PathLike[str], *, label: str) -> dict[str, Any]:
    payload = read_regular(path, label=label, maximum=MAX_INPUT_BYTES)
    try:
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{label} JSON is invalid.") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must contain a JSON object.")
    return raw


def load_signed_custody_envelope(
    path: str | os.PathLike[str],
) -> SignedCustodyEnvelope:
    return signed_envelope_from_dict(decode_json(path, label="signed_envelope_path"))


def load_timestamped_signed_custody_envelope(
    path: str | os.PathLike[str],
) -> TimestampedSignedCustodyEnvelope:
    return timestamped_envelope_from_dict(
        decode_json(path, label="timestamped_envelope_path")
    )


def verify_signed_envelope_object(
    envelope: SignedCustodyEnvelope,
    public_key: Ed25519PublicKey,
) -> SignedCustodyEnvelope:
    if not isinstance(envelope, SignedCustodyEnvelope):
        raise ValueError("envelope must be SignedCustodyEnvelope.")
    if not isinstance(public_key, Ed25519PublicKey):
        raise ValueError("public_key must be Ed25519PublicKey.")
    if public_key_fingerprint(public_key) != envelope.public_key_sha256:
        raise PermissionError("public key differs from signed custody envelope.")
    try:
        public_key.verify(
            envelope.signature_bytes(),
            _canonical_bytes(envelope.signing_payload()),
        )
    except InvalidSignature as exc:
        raise PermissionError("custody signature verification failed.") from exc
    return envelope


def sign_restore_chain_of_custody(
    *,
    manifest_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    key_id: str,
    private_key_path: str | os.PathLike[str],
    expected_public_key_sha256: str | None = None,
    now: float | None = None,
) -> SignedCustodyEnvelope:
    manifest = verify_restore_chain_of_custody(manifest_path)
    key = load_private_key(private_key_path)
    fingerprint = public_key_fingerprint(key.public_key())
    if expected_public_key_sha256 is not None and fingerprint != _digest(
        expected_public_key_sha256,
        "expected_public_key_sha256",
    ):
        raise PermissionError("private key differs from expected public key.")
    created = _timestamp(time.time() if now is None else now, "now")
    signing_payload = {
        "scope": "rigorousrag-restore-custody-ed25519-signature-v1",
        "owner_id": manifest.owner_id,
        "key_id": _identifier(key_id, "key_id", 200),
        "algorithm": ALGORITHM,
        "public_key_sha256": fingerprint,
        "manifest": manifest.public_payload(),
        "created_at": created,
        "schema_version": SCHEMA_VERSION,
    }
    signature_base64 = base64.b64encode(
        key.sign(_canonical_bytes(signing_payload))
    ).decode("ascii")
    envelope = SignedCustodyEnvelope(
        owner_id=manifest.owner_id,
        key_id=key_id,
        algorithm=ALGORITHM,
        public_key_sha256=fingerprint,
        manifest=manifest,
        created_at=created,
        signature_base64=signature_base64,
        envelope_digest=canonical_digest(
            {**signing_payload, "signature_base64": signature_base64}
        ),
    )
    _atomic_create(_path(output_path, label="output_path"), envelope.canonical_export_bytes())
    return envelope


def verify_signed_restore_chain_of_custody(
    *,
    envelope_path: str | os.PathLike[str],
    public_key_path: str | os.PathLike[str],
    expected_key_id: str | None = None,
    expected_owner_id: str | None = None,
) -> SignedCustodyEnvelope:
    envelope = load_signed_custody_envelope(envelope_path)
    if expected_key_id is not None and envelope.key_id != _identifier(
        expected_key_id,
        "expected_key_id",
        200,
    ):
        raise PermissionError("signed custody key ID differs from expectation.")
    if expected_owner_id is not None and envelope.owner_id != normalize_owner_id(
        expected_owner_id
    ):
        raise PermissionError("signed custody owner differs from expectation.")
    public_key, fingerprint = load_public_key(public_key_path)
    if fingerprint != envelope.public_key_sha256:
        raise PermissionError("public key differs from signed custody envelope.")
    return verify_signed_envelope_object(envelope, public_key)


def bind_rfc3161_timestamp_to_signed_custody(
    *,
    signed_envelope_path: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
    public_key_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    expected_key_id: str | None = None,
) -> TimestampedSignedCustodyEnvelope:
    envelope = verify_signed_restore_chain_of_custody(
        envelope_path=signed_envelope_path,
        public_key_path=public_key_path,
        expected_key_id=expected_key_id,
    )
    signed_bytes = read_regular(
        signed_envelope_path,
        label="signed_envelope_path",
        maximum=MAX_INPUT_BYTES,
    )
    if signed_bytes != envelope.canonical_export_bytes():
        raise ValueError("signed envelope file is not canonical.")
    receipt = verify_rfc3161_timestamp_receipt(receipt_path)
    subject = hashlib.sha256(signed_bytes).hexdigest()
    if receipt.subject_sha256 != subject:
        raise PermissionError("RFC 3161 receipt does not timestamp the signed envelope.")
    stable = {
        "scope": "rigorousrag-restore-custody-ed25519-rfc3161-binding-v1",
        "signed_envelope": envelope.public_payload(),
        "timestamp_receipt": receipt.public_payload(),
        "timestamped_subject_sha256": subject,
        "schema_version": TIMESTAMP_SCHEMA_VERSION,
    }
    wrapped = TimestampedSignedCustodyEnvelope(
        signed_envelope=envelope,
        timestamp_receipt=receipt,
        timestamped_subject_sha256=subject,
        binding_digest=canonical_digest(stable),
    )
    _atomic_create(
        _path(output_path, label="output_path"),
        _canonical_bytes(wrapped.public_payload()) + b"\n",
    )
    return wrapped


def verify_timestamped_signed_restore_chain_of_custody(
    *,
    envelope_path: str | os.PathLike[str],
    public_key_path: str | os.PathLike[str],
    expected_key_id: str | None = None,
    expected_owner_id: str | None = None,
) -> TimestampedSignedCustodyEnvelope:
    wrapped = load_timestamped_signed_custody_envelope(envelope_path)
    envelope = wrapped.signed_envelope
    if expected_key_id is not None and envelope.key_id != _identifier(
        expected_key_id,
        "expected_key_id",
        200,
    ):
        raise PermissionError("signed custody key ID differs from expectation.")
    if expected_owner_id is not None and envelope.owner_id != normalize_owner_id(
        expected_owner_id
    ):
        raise PermissionError("signed custody owner differs from expectation.")
    public_key, fingerprint = load_public_key(public_key_path)
    if fingerprint != envelope.public_key_sha256:
        raise PermissionError("public key differs from signed custody envelope.")
    verify_signed_envelope_object(envelope, public_key)
    return wrapped


__all__ = [
    "bind_rfc3161_timestamp_to_signed_custody",
    "load_signed_custody_envelope",
    "load_timestamped_signed_custody_envelope",
    "sign_restore_chain_of_custody",
    "verify_signed_envelope_object",
    "verify_signed_restore_chain_of_custody",
    "verify_timestamped_signed_restore_chain_of_custody",
]
