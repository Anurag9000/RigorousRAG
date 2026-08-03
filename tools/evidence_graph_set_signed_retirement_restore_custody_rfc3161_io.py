"""Descriptor-safe request bundle and receipt I/O for RFC 3161 custody evidence."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import time
from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_custody_rfc3161_contracts import (
    MAX_BUNDLE_BYTES,
    MAX_INPUT_BYTES,
    Rfc3161TimestampRequestBundle,
    Rfc3161TimestampVerificationReceipt,
    SCHEMA_VERSION,
    SHA256_OID,
    canonical_digest,
    nonce_decimal,
    optional_oid,
    request_der,
)
from tools.evidence_graph_set_signed_retirement_restore_contracts import _timestamp
from tools.evidence_graph_set_signed_retirement_snapshot import (
    _atomic_create,
    _canonical_bytes,
    _pairs,
    _path,
)
from tools.security import normalize_owner_id


def read_regular(
    path: str | os.PathLike[str],
    *,
    label: str,
    maximum: int,
) -> bytes:
    selected = _path(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(selected, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file.")
        if before.st_size <= 0 or before.st_size > maximum:
            raise ValueError(f"{label} size is invalid.")
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
            int(before.st_dev) != int(after.st_dev)
            or int(before.st_ino) != int(after.st_ino)
            or int(before.st_size) != int(after.st_size)
        ):
            raise RuntimeError(f"{label} identity changed while being read.")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def decode_json(path: str | os.PathLike[str], *, label: str) -> dict[str, Any]:
    payload = read_regular(path, label=label, maximum=MAX_BUNDLE_BYTES)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{label} JSON is invalid.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object.")
    return value


def _bundle_from_dict(raw: dict[str, Any]) -> Rfc3161TimestampRequestBundle:
    expected = set(Rfc3161TimestampRequestBundle.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError("RFC 3161 request bundle schema is invalid.")
    return Rfc3161TimestampRequestBundle(**raw)


def _receipt_from_dict(raw: dict[str, Any]) -> Rfc3161TimestampVerificationReceipt:
    expected = set(Rfc3161TimestampVerificationReceipt.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError("RFC 3161 receipt schema is invalid.")
    return Rfc3161TimestampVerificationReceipt(**raw)


def create_rfc3161_timestamp_request_bundle(
    *,
    owner_id: str,
    custody_envelope_path: str | os.PathLike[str],
    output_bundle_path: str | os.PathLike[str],
    requested_policy_oid: str | None = None,
    nonce: int | str | None = None,
    now: float | None = None,
) -> Rfc3161TimestampRequestBundle:
    owner = normalize_owner_id(owner_id)
    subject = read_regular(
        custody_envelope_path,
        label="custody_envelope_path",
        maximum=MAX_INPUT_BYTES,
    )
    subject_digest = hashlib.sha256(subject).hexdigest()
    selected_nonce = nonce_decimal(
        (int.from_bytes(os.urandom(20), "big") | (1 << 159))
        if nonce is None
        else nonce
    )
    policy = optional_oid(requested_policy_oid, "requested_policy_oid")
    created = _timestamp(time.time() if now is None else now, "now")
    der = request_der(
        subject_sha256=subject_digest,
        nonce_decimal_value=selected_nonce,
        requested_policy_oid=policy,
    )
    stable = {
        "scope": "rigorousrag-restore-custody-rfc3161-request-v1",
        "owner_id": owner,
        "subject_sha256": subject_digest,
        "subject_size_bytes": len(subject),
        "hash_algorithm": "sha256",
        "hash_algorithm_oid": SHA256_OID,
        "nonce_decimal": selected_nonce,
        "nonce_sha256": hashlib.sha256(selected_nonce.encode("ascii")).hexdigest(),
        "requested_policy_oid": policy,
        "cert_req": True,
        "request_sha256": hashlib.sha256(der).hexdigest(),
        "created_at": created,
        "schema_version": SCHEMA_VERSION,
    }
    bundle = Rfc3161TimestampRequestBundle(
        owner_id=owner,
        subject_sha256=subject_digest,
        subject_size_bytes=len(subject),
        hash_algorithm="sha256",
        hash_algorithm_oid=SHA256_OID,
        nonce_decimal=selected_nonce,
        nonce_sha256=stable["nonce_sha256"],
        requested_policy_oid=policy,
        cert_req=True,
        request_der_base64=base64.b64encode(der).decode("ascii"),
        request_sha256=stable["request_sha256"],
        created_at=created,
        bundle_digest=canonical_digest(stable),
    )
    _atomic_create(
        _path(output_bundle_path, label="output_bundle_path"),
        _canonical_bytes(bundle.public_payload()) + b"\n",
    )
    return bundle


def verify_rfc3161_timestamp_request_bundle(
    path: str | os.PathLike[str],
) -> Rfc3161TimestampRequestBundle:
    return _bundle_from_dict(decode_json(path, label="request_bundle_path"))


def emit_rfc3161_timestamp_request_der(
    *,
    request_bundle_path: str | os.PathLike[str],
    output_der_path: str | os.PathLike[str],
) -> Rfc3161TimestampRequestBundle:
    bundle = verify_rfc3161_timestamp_request_bundle(request_bundle_path)
    _atomic_create(
        _path(output_der_path, label="output_der_path"),
        bundle.request_der(),
    )
    return bundle


def verify_rfc3161_timestamp_receipt(
    path: str | os.PathLike[str],
) -> Rfc3161TimestampVerificationReceipt:
    return _receipt_from_dict(decode_json(path, label="receipt_path"))


__all__ = [
    "create_rfc3161_timestamp_request_bundle",
    "decode_json",
    "emit_rfc3161_timestamp_request_der",
    "read_regular",
    "verify_rfc3161_timestamp_receipt",
    "verify_rfc3161_timestamp_request_bundle",
]
