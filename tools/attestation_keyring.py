"""Rotation-aware external/KMS attestation signer and verifier adapters.

The module is provider-neutral by design. Deployments inject already-constructed signing
and verification callables; this code never imports cloud SDKs, discovers credentials,
or persists key material. Retired keys may remain verification-only so historical
attestations remain auditable after signing-key rotation.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from tools.manifest_attestation import ManifestSigner, ManifestVerifier

_ALLOWED_ALGORITHMS = frozenset(
    {
        "ed25519",
        "ecdsa-p256-sha256",
        "rsa-pss-sha256",
        "aws-kms-ecdsa-sha256",
        "aws-kms-rsa-pss-sha256",
        "gcp-kms-ecdsa-sha256",
        "gcp-kms-rsa-pss-sha256",
        "azure-keyvault-ecdsa-sha256",
        "azure-keyvault-rsa-pss-sha256",
        "external",
    }
)
_MAX_KEYS = 1000
_MAX_SIGNATURE_BYTES = 64 * 1024


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > maximum or "\x00" in cleaned:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _time(value: Any, label: str, *, allow_zero: bool = True) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    minimum = 0.0 if allow_zero else 1e-12
    if not math.isfinite(parsed) or parsed < minimum:
        raise ValueError(f"{label} is invalid")
    return parsed


def _algorithm(value: str) -> str:
    selected = _text(value, "algorithm", 100).lower()
    if selected not in _ALLOWED_ALGORITHMS:
        raise ValueError("unsupported attestation algorithm")
    return selected


@dataclass(frozen=True)
class AttestationKeyDescriptor:
    key_id: str
    algorithm: str
    state: str = "active"
    not_before: float = 0.0
    not_after: float = 0.0
    metadata: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_id", _text(self.key_id, "key_id", 500))
        object.__setattr__(self, "algorithm", _algorithm(self.algorithm))
        state = _text(self.state, "state", 32).lower()
        if state not in {"active", "verification_only", "disabled"}:
            raise ValueError("invalid attestation key state")
        object.__setattr__(self, "state", state)
        start = _time(self.not_before, "not_before")
        end = _time(self.not_after, "not_after")
        if end and end <= start:
            raise ValueError("not_after must be greater than not_before")
        object.__setattr__(self, "not_before", start)
        object.__setattr__(self, "not_after", end)
        details: dict[str, str] = {}
        for key, value in dict(self.metadata or {}).items():
            details[_text(key, "metadata key", 128)] = _text(value, "metadata value", 1000)
        if len(details) > 100:
            raise ValueError("too many attestation key metadata entries")
        object.__setattr__(self, "metadata", details)

    def valid_at(self, timestamp: float) -> bool:
        selected = _time(timestamp, "timestamp")
        if selected < self.not_before:
            return False
        if self.not_after and selected >= self.not_after:
            return False
        return self.state != "disabled"


@dataclass(frozen=True)
class ExternalSigningKey:
    descriptor: AttestationKeyDescriptor
    sign_callable: Callable[[bytes], bytes]

    def __post_init__(self) -> None:
        if not callable(self.sign_callable):
            raise TypeError("sign_callable must be callable")
        if self.descriptor.state != "active":
            raise ValueError("signing key descriptor must be active")


@dataclass(frozen=True)
class ExternalVerificationKey:
    descriptor: AttestationKeyDescriptor
    verify_callable: Callable[[bytes, bytes], bool]

    def __post_init__(self) -> None:
        if not callable(self.verify_callable):
            raise TypeError("verify_callable must be callable")


class ExternalManifestSigner(ManifestSigner):
    """ManifestSigner backed by one explicitly injected external signing key."""

    def __init__(self, key: ExternalSigningKey, *, clock: Callable[[], float] = time.time) -> None:
        if not isinstance(key, ExternalSigningKey):
            raise TypeError("key must be ExternalSigningKey")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._key = key
        self._clock = clock

    @property
    def key_id(self) -> str:
        return self._key.descriptor.key_id

    @property
    def algorithm(self) -> str:
        return self._key.descriptor.algorithm

    def sign(self, payload: bytes) -> bytes:
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("payload must be non-empty bytes")
        now = _time(self._clock(), "clock")
        if not self._key.descriptor.valid_at(now) or self._key.descriptor.state != "active":
            raise RuntimeError("attestation signing key is not currently active")
        signature = self._key.sign_callable(payload)
        if not isinstance(signature, bytes) or not signature or len(signature) > _MAX_SIGNATURE_BYTES:
            raise RuntimeError("external signer returned an invalid signature")
        return bytes(signature)


class RotatingManifestKeyring(ManifestVerifier):
    """Explicit signing/verification keyring with rotation and historical verification."""

    def __init__(
        self,
        *,
        signing_keys: tuple[ExternalSigningKey, ...] = (),
        verification_keys: tuple[ExternalVerificationKey, ...] = (),
        active_key_id: str = "",
        clock: Callable[[], float] = time.time,
    ) -> None:
        if len(signing_keys) + len(verification_keys) > _MAX_KEYS:
            raise ValueError("attestation keyring exceeds the key limit")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        signers: dict[str, ExternalSigningKey] = {}
        verifiers: dict[tuple[str, str], ExternalVerificationKey] = {}
        for key in signing_keys:
            if not isinstance(key, ExternalSigningKey):
                raise TypeError("signing_keys contain an invalid value")
            if key.descriptor.key_id in signers:
                raise ValueError("duplicate signing key_id")
            signers[key.descriptor.key_id] = key
        for key in verification_keys:
            if not isinstance(key, ExternalVerificationKey):
                raise TypeError("verification_keys contain an invalid value")
            identity = (key.descriptor.key_id, key.descriptor.algorithm)
            if identity in verifiers:
                raise ValueError("duplicate verification key identity")
            verifiers[identity] = key
        selected = active_key_id.strip()
        if selected and selected not in signers:
            raise ValueError("active_key_id is not present in signing_keys")
        if not selected and len(signers) == 1:
            selected = next(iter(signers))
        self._signers = signers
        self._verifiers = verifiers
        self._active_key_id = selected

    @property
    def active_key_id(self) -> str:
        return self._active_key_id

    def signer(self) -> ExternalManifestSigner:
        if not self._active_key_id:
            raise RuntimeError("no active attestation signing key is configured")
        return ExternalManifestSigner(self._signers[self._active_key_id], clock=self._clock)

    def verify(self, *, key_id: str, algorithm: str, payload: bytes, signature: bytes) -> bool:
        try:
            selected_key = _text(key_id, "key_id", 500)
            selected_algorithm = _algorithm(algorithm)
        except ValueError:
            return False
        if not isinstance(payload, bytes) or not payload:
            return False
        if not isinstance(signature, bytes) or not signature or len(signature) > _MAX_SIGNATURE_BYTES:
            return False
        key = self._verifiers.get((selected_key, selected_algorithm))
        if key is None:
            return False
        now = _time(self._clock(), "clock")
        # Historical verification remains allowed after not_after when the key is explicitly
        # retained as verification_only. Disabled keys fail closed.
        descriptor = key.descriptor
        if descriptor.state == "disabled" or now < descriptor.not_before:
            return False
        try:
            return bool(key.verify_callable(payload, signature))
        except Exception:
            return False

    def rotate(self, active_key_id: str) -> "RotatingManifestKeyring":
        selected = _text(active_key_id, "active_key_id", 500)
        if selected not in self._signers:
            raise KeyError(selected)
        return RotatingManifestKeyring(
            signing_keys=tuple(self._signers.values()),
            verification_keys=tuple(self._verifiers.values()),
            active_key_id=selected,
            clock=self._clock,
        )

    def public_metadata(self) -> Mapping[str, Any]:
        return {
            "active_key_id": self._active_key_id,
            "signing_keys": tuple(
                {
                    "key_id": item.descriptor.key_id,
                    "algorithm": item.descriptor.algorithm,
                    "state": item.descriptor.state,
                    "not_before": item.descriptor.not_before,
                    "not_after": item.descriptor.not_after,
                    "metadata": dict(item.descriptor.metadata or {}),
                }
                for item in self._signers.values()
            ),
            "verification_keys": tuple(
                {
                    "key_id": item.descriptor.key_id,
                    "algorithm": item.descriptor.algorithm,
                    "state": item.descriptor.state,
                    "not_before": item.descriptor.not_before,
                    "not_after": item.descriptor.not_after,
                    "metadata": dict(item.descriptor.metadata or {}),
                }
                for item in self._verifiers.values()
            ),
        }


__all__ = [
    "AttestationKeyDescriptor",
    "ExternalManifestSigner",
    "ExternalSigningKey",
    "ExternalVerificationKey",
    "RotatingManifestKeyring",
]
