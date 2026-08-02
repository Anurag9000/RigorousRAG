"""Strict HMAC-signed actor assertions for governed relation review."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_MAX_ASSERTION_BYTES = 16_384
_MAX_KEY_BYTES = 4_096
_MIN_KEY_BYTES = 32
_MAX_ASSERTION_LIFETIME_SECONDS = 86_400.0
_MAX_CLOCK_SKEW_SECONDS = 300.0


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    selected = value.strip()
    if (
        not selected
        or len(selected) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in selected)
    ):
        raise ValueError(f"{label} is invalid.")
    return selected


def _timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(selected) or selected < 0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return selected


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("actor assertion contains a duplicate JSON key.")
        result[key] = value
    return result


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _safe_path(value: str | os.PathLike[str], label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"{label} must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError(f"{label} is invalid.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for part in (absolute, *absolute.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(f"{label} could not be validated.") from exc
        if _redirecting(info):
            raise ValueError(f"{label} may not contain redirects.")
    return absolute


def _read_file(
    value: str | os.PathLike[str],
    *,
    label: str,
    maximum: int,
    minimum: int = 1,
) -> bytes:
    path = _safe_path(value, label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not minimum <= info.st_size <= maximum:
            raise ValueError(f"{label} is invalid or has an unsupported size.")
        payload = os.read(descriptor, maximum + 1)
        if not minimum <= len(payload) <= maximum:
            raise ValueError(f"{label} has an unsupported size.")
        return payload
    finally:
        os.close(descriptor)


def _canonical_payload(
    *,
    actor_id: str,
    issuer: str,
    issued_at: float,
    expires_at: float,
    nonce: str,
) -> bytes:
    return json.dumps(
        {
            "schema_version": 1,
            "actor_id": actor_id,
            "issuer": issuer,
            "issued_at": issued_at,
            "expires_at": expires_at,
            "nonce": nonce,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class SignedReviewActorAssertion:
    actor_id: str
    issuer: str
    issued_at: float
    expires_at: float
    nonce: str
    assertion_digest: str
    signature_digest: str
    verified_at: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "actor_id", _identifier(self.actor_id, "actor_id", 200))
        object.__setattr__(self, "issuer", _identifier(self.issuer, "issuer", 200))
        object.__setattr__(self, "issued_at", _timestamp(self.issued_at, "issued_at"))
        object.__setattr__(self, "expires_at", _timestamp(self.expires_at, "expires_at"))
        object.__setattr__(self, "verified_at", _timestamp(self.verified_at, "verified_at"))
        object.__setattr__(self, "nonce", _identifier(self.nonce, "nonce", 200))
        for name in ("assertion_digest", "signature_digest"):
            value = _identifier(getattr(self, name), name, 64).lower()
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a SHA-256 digest.")
            object.__setattr__(self, name, value)
        if self.expires_at <= self.issued_at:
            raise ValueError("actor assertion expiry must follow issuance.")
        if self.expires_at - self.issued_at > _MAX_ASSERTION_LIFETIME_SECONDS:
            raise ValueError("actor assertion lifetime exceeds the maximum.")
        if self.schema_version != 1:
            raise ValueError("actor assertion schema is unsupported.")


def sign_review_actor_assertion(
    *,
    actor_id: str,
    issuer: str,
    issued_at: float,
    expires_at: float,
    nonce: str,
    key: bytes,
) -> dict[str, Any]:
    """Create a canonical assertion payload for controlled provisioning/tests."""

    if not isinstance(key, bytes) or not _MIN_KEY_BYTES <= len(key) <= _MAX_KEY_BYTES:
        raise ValueError("actor assertion key has an unsupported size.")
    selected_actor = _identifier(actor_id, "actor_id", 200)
    selected_issuer = _identifier(issuer, "issuer", 200)
    selected_issued = _timestamp(issued_at, "issued_at")
    selected_expires = _timestamp(expires_at, "expires_at")
    selected_nonce = _identifier(nonce, "nonce", 200)
    if selected_expires <= selected_issued:
        raise ValueError("actor assertion expiry must follow issuance.")
    if selected_expires - selected_issued > _MAX_ASSERTION_LIFETIME_SECONDS:
        raise ValueError("actor assertion lifetime exceeds the maximum.")
    payload = _canonical_payload(
        actor_id=selected_actor,
        issuer=selected_issuer,
        issued_at=selected_issued,
        expires_at=selected_expires,
        nonce=selected_nonce,
    )
    return {
        "schema_version": 1,
        "actor_id": selected_actor,
        "issuer": selected_issuer,
        "issued_at": selected_issued,
        "expires_at": selected_expires,
        "nonce": selected_nonce,
        "signature": hmac.new(key, payload, hashlib.sha256).hexdigest(),
    }


def verify_review_actor_assertion(
    *,
    assertion_path: str | os.PathLike[str],
    key_path: str | os.PathLike[str],
    expected_issuer: str | None = None,
    now: float | None = None,
    clock_skew_seconds: float = 60.0,
) -> SignedReviewActorAssertion:
    raw_assertion = _read_file(
        assertion_path,
        label="actor assertion path",
        maximum=_MAX_ASSERTION_BYTES,
    )
    key = _read_file(
        key_path,
        label="actor assertion key path",
        minimum=_MIN_KEY_BYTES,
        maximum=_MAX_KEY_BYTES,
    )
    try:
        decoded = raw_assertion.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("actor assertion JSON is invalid.") from exc
    expected_fields = {
        "schema_version",
        "actor_id",
        "issuer",
        "issued_at",
        "expires_at",
        "nonce",
        "signature",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("actor assertion schema is invalid.")
    if value["schema_version"] != 1:
        raise ValueError("actor assertion schema is unsupported.")
    actor_id = _identifier(value["actor_id"], "actor_id", 200)
    issuer = _identifier(value["issuer"], "issuer", 200)
    issued_at = _timestamp(value["issued_at"], "issued_at")
    expires_at = _timestamp(value["expires_at"], "expires_at")
    nonce = _identifier(value["nonce"], "nonce", 200)
    signature = _identifier(value["signature"], "signature", 64).lower()
    if len(signature) != 64 or any(character not in "0123456789abcdef" for character in signature):
        raise ValueError("actor assertion signature must be an HMAC-SHA256 digest.")
    if expected_issuer is not None and issuer != _identifier(
        expected_issuer, "expected_issuer", 200
    ):
        raise PermissionError("actor assertion issuer differs from configuration.")
    current = _timestamp(time.time() if now is None else now, "now")
    skew = _timestamp(clock_skew_seconds, "clock_skew_seconds")
    if skew > _MAX_CLOCK_SKEW_SECONDS:
        raise ValueError("clock skew exceeds the maximum.")
    if issued_at > current + skew:
        raise PermissionError("actor assertion was issued in the future.")
    if expires_at < current - skew:
        raise PermissionError("actor assertion has expired.")
    if expires_at <= issued_at:
        raise ValueError("actor assertion expiry must follow issuance.")
    if expires_at - issued_at > _MAX_ASSERTION_LIFETIME_SECONDS:
        raise ValueError("actor assertion lifetime exceeds the maximum.")
    canonical = _canonical_payload(
        actor_id=actor_id,
        issuer=issuer,
        issued_at=issued_at,
        expires_at=expires_at,
        nonce=nonce,
    )
    expected_signature = hmac.new(key, canonical, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise PermissionError("actor assertion signature verification failed.")
    canonical_assertion = json.dumps(
        {**json.loads(canonical), "signature": signature},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return SignedReviewActorAssertion(
        actor_id=actor_id,
        issuer=issuer,
        issued_at=issued_at,
        expires_at=expires_at,
        nonce=nonce,
        assertion_digest=_digest(canonical_assertion),
        signature_digest=_digest(signature.encode("ascii")),
        verified_at=current,
    )


__all__ = [
    "SignedReviewActorAssertion",
    "sign_review_actor_assertion",
    "verify_review_actor_assertion",
]
