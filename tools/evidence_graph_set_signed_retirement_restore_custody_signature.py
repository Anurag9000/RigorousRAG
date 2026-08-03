"""Ed25519 public-key envelopes for external restore custody manifests."""

from __future__ import annotations

import base64
import hashlib
import os
import stat
from dataclasses import asdict, dataclass
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from tools import evidence_graph_set_signed_retirement_restore_custody_export as _export
from tools import evidence_graph_set_signed_retirement_restore_custody_export_integrity as _integrity
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_export_boundary import (
    RestoreChainOfCustodyManifest,
    verify_restore_chain_of_custody,
)
from tools.evidence_graph_set_signed_retirement_snapshot import (
    _atomic_create,
    _canonical_bytes,
    _path,
    _redirecting,
)

_SCHEMA_VERSION = 1
_MAX_KEY_BYTES = 1024 * 1024


def _read_key(
    path: str | os.PathLike[str],
    *,
    label: str,
    private: bool,
) -> bytes:
    selected = _path(path, label=label)
    info = selected.lstat()
    if _redirecting(info) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} must be a regular non-redirecting file.")
    if private and os.name != "nt" and stat.S_IMODE(info.st_mode) & 0o077:
        raise PermissionError("Ed25519 private-key permissions are too broad.")
    return _export._read_regular(
        selected,
        label=label,
        maximum=_MAX_KEY_BYTES,
    )


def _load_private(path: str | os.PathLike[str]) -> Ed25519PrivateKey:
    try:
        key = serialization.load_pem_private_key(
            _read_key(path, label="private_key_path", private=True),
            password=None,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Ed25519 private key is invalid or encrypted.") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("private key is not Ed25519.")
    return key


def _load_public(path: str | os.PathLike[str]) -> Ed25519PublicKey:
    try:
        key = serialization.load_pem_public_key(
            _read_key(path, label="public_key_path", private=False)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Ed25519 public key is invalid.") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("public key is not Ed25519.")
    return key


def _public_bytes(key: Ed25519PublicKey) -> bytes:
    return key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _public_fingerprint(key: Ed25519PublicKey) -> str:
    return hashlib.sha256(_public_bytes(key)).hexdigest()


def _canonical_signature(value: str) -> bytes:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError("Ed25519 signature encoding is invalid.")
    try:
        decoded = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ValueError("Ed25519 signature encoding is invalid.") from exc
    if len(decoded) != 64 or base64.b64encode(decoded).decode("ascii") != value:
        raise ValueError("Ed25519 signature encoding is not canonical.")
    return decoded


@dataclass(frozen=True)
class SignedCustodyEnvelope:
    algorithm: str
    key_id: str
    public_key_sha256: str
    manifest: RestoreChainOfCustodyManifest
    signature: str
    schema_version: int = _SCHEMA_VERSION
    contains_private_key_material: bool = False
    mutation_performed: bool = False

    def __post_init__(self) -> None:
        algorithm = _identifier(self.algorithm, "algorithm", 30)
        if algorithm != "ed25519":
            raise ValueError("signature algorithm is unsupported.")
        key_id = _identifier(self.key_id, "key_id", 200)
        fingerprint = _digest(self.public_key_sha256, "public_key_sha256")
        if not isinstance(self.manifest, RestoreChainOfCustodyManifest):
            raise ValueError("signed envelope manifest is invalid.")
        _integrity._validate_chronology(self.manifest)
        signature = base64.b64encode(
            _canonical_signature(self.signature)
        ).decode("ascii")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("signed envelope schema is unsupported.")
        if (
            self.contains_private_key_material is not False
            or self.mutation_performed is not False
        ):
            raise ValueError("signed envelope safety flags must be false.")
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "key_id", key_id)
        object.__setattr__(self, "public_key_sha256", fingerprint)
        object.__setattr__(self, "signature", signature)

    def signing_payload(self) -> dict[str, Any]:
        return {
            "scope": "rigorousrag-restore-custody-ed25519-envelope-v1",
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "public_key_sha256": self.public_key_sha256,
            "manifest": self.manifest.public_payload(),
            "schema_version": self.schema_version,
        }

    def public_payload(self) -> dict[str, Any]:
        return asdict(self)


def _envelope_from_dict(raw: dict[str, Any]) -> SignedCustodyEnvelope:
    expected = {
        "algorithm",
        "key_id",
        "public_key_sha256",
        "manifest",
        "signature",
        "schema_version",
        "contains_private_key_material",
        "mutation_performed",
    }
    if set(raw) != expected or not isinstance(raw["manifest"], dict):
        raise ValueError("signed envelope schema is invalid.")
    manifest = _export._manifest_from_dict(raw["manifest"])
    _integrity._validate_chronology(manifest)
    return SignedCustodyEnvelope(**{**raw, "manifest": manifest})


def sign_restore_chain_of_custody(
    *,
    manifest_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    key_id: str,
    private_key_path: str | os.PathLike[str],
) -> SignedCustodyEnvelope:
    manifest = verify_restore_chain_of_custody(manifest_path)
    selected_key_id = _identifier(key_id, "key_id", 200)
    private_key = _load_private(private_key_path)
    public_key = private_key.public_key()
    fingerprint = _public_fingerprint(public_key)
    unsigned = {
        "scope": "rigorousrag-restore-custody-ed25519-envelope-v1",
        "algorithm": "ed25519",
        "key_id": selected_key_id,
        "public_key_sha256": fingerprint,
        "manifest": manifest.public_payload(),
        "schema_version": _SCHEMA_VERSION,
    }
    signature = base64.b64encode(
        private_key.sign(_canonical_bytes(unsigned))
    ).decode("ascii")
    envelope = SignedCustodyEnvelope(
        algorithm="ed25519",
        key_id=selected_key_id,
        public_key_sha256=fingerprint,
        manifest=manifest,
        signature=signature,
    )
    output = _path(output_path, label="output_path")
    _atomic_create(
        output,
        _canonical_bytes(envelope.public_payload()) + b"\n",
    )
    return envelope


def verify_signed_restore_chain_of_custody(
    *,
    envelope_path: str | os.PathLike[str],
    public_key_path: str | os.PathLike[str],
    expected_key_id: str | None = None,
    expected_public_key_sha256: str | None = None,
) -> SignedCustodyEnvelope:
    raw = _export._decode_json(
        envelope_path,
        label="signed_envelope",
    )
    envelope = _envelope_from_dict(raw)
    if expected_key_id is not None and envelope.key_id != _identifier(
        expected_key_id,
        "expected_key_id",
        200,
    ):
        raise PermissionError("signed envelope key ID differs.")
    public_key = _load_public(public_key_path)
    fingerprint = _public_fingerprint(public_key)
    if fingerprint != envelope.public_key_sha256:
        raise PermissionError("signed envelope public-key fingerprint differs.")
    if expected_public_key_sha256 is not None and fingerprint != _digest(
        expected_public_key_sha256,
        "expected_public_key_sha256",
    ):
        raise PermissionError("expected public-key fingerprint differs.")
    try:
        public_key.verify(
            _canonical_signature(envelope.signature),
            _canonical_bytes(envelope.signing_payload()),
        )
    except InvalidSignature as exc:
        raise PermissionError("signed envelope verification failed.") from exc
    return envelope


__all__ = [
    "SignedCustodyEnvelope",
    "sign_restore_chain_of_custody",
    "verify_signed_restore_chain_of_custody",
]
