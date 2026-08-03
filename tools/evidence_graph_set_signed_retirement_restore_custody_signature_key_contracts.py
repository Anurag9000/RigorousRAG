"""Validated Ed25519 custody signer key contracts and descriptor-safe key loading."""

from __future__ import annotations

import base64
import hashlib
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_snapshot import _canonical_bytes
from tools.security import normalize_owner_id

SCHEMA_VERSION = 1
ALGORITHM = "ed25519"
STATES = frozenset({"active", "retired"})
MAX_KEY_BYTES = 64 * 1024
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def validated_path(value: str | os.PathLike[str], *, label: str) -> Path:
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError(f"{label} is invalid.")
    candidate = Path(rendered)
    absolute = Path(os.path.abspath(candidate if candidate.is_absolute() else Path.cwd() / candidate))
    for part in (absolute, *absolute.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(f"{label} could not be validated.") from exc
        if redirecting(info):
            raise ValueError(f"{label} may not contain redirects.")
    return absolute


def read_regular(path: str | os.PathLike[str], *, label: str, maximum: int) -> bytes:
    selected = validated_path(path, label=label)
    descriptor = os.open(
        selected,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 1 <= before.st_size <= maximum:
            raise ValueError(f"{label} must be a bounded regular file.")
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


def actor(value: ReviewActorBinding) -> ReviewActorBinding:
    if not isinstance(value, ReviewActorBinding):
        raise ValueError("actor must be ReviewActorBinding.")
    _identifier(value.actor_id, "actor_id", 200)
    _identifier(value.binding_method, "binding_method", 50)
    _digest(value.binding_digest, "binding_digest")
    _timestamp(value.loaded_at, "loaded_at")
    return value


def actor_id_digest(value: ReviewActorBinding) -> str:
    return hashlib.sha256(actor(value).actor_id.encode("utf-8")).hexdigest()


def public_key_raw(value: Ed25519PublicKey) -> bytes:
    return value.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def load_ed25519_public_key(
    path: str | os.PathLike[str],
) -> tuple[Ed25519PublicKey, str, str]:
    payload = read_regular(path, label="public_key_path", maximum=MAX_KEY_BYTES)
    try:
        key = serialization.load_pem_public_key(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("public key must be PEM SubjectPublicKeyInfo.") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError("public key must be Ed25519.")
    raw = public_key_raw(key)
    return key, base64.b64encode(raw).decode("ascii"), hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class CustodySignerKeyRecord:
    owner_id: str
    key_id: str
    algorithm: str
    public_key_raw_base64: str
    public_key_sha256: str
    valid_from: float
    valid_until: float | None
    state: str
    registered_actor_id_digest: str
    registered_binding_method: str
    registered_binding_digest: str
    registered_at: float
    retired_actor_id_digest: str | None
    retired_binding_method: str | None
    retired_binding_digest: str | None
    retired_at: float | None
    record_digest: str
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        key_id = _identifier(self.key_id, "key_id", 200)
        algorithm = _identifier(self.algorithm, "algorithm", 30)
        if algorithm != ALGORITHM:
            raise ValueError("custody signer algorithm is unsupported.")
        try:
            raw = base64.b64decode(self.public_key_raw_base64.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError) as exc:
            raise ValueError("public_key_raw_base64 is invalid.") from exc
        if len(raw) != 32:
            raise ValueError("Ed25519 public key must contain 32 raw bytes.")
        Ed25519PublicKey.from_public_bytes(raw)
        fingerprint = _digest(self.public_key_sha256, "public_key_sha256")
        if fingerprint != hashlib.sha256(raw).hexdigest():
            raise ValueError("public_key_sha256 differs from public key bytes.")
        valid_from = _timestamp(self.valid_from, "valid_from")
        valid_until = None if self.valid_until is None else _timestamp(self.valid_until, "valid_until")
        if valid_until is not None and valid_until < valid_from:
            raise ValueError("signer key validity window is reversed.")
        state = _identifier(self.state, "state", 30)
        if state not in STATES:
            raise ValueError("signer key state is unsupported.")
        registered_actor = _digest(self.registered_actor_id_digest, "registered_actor_id_digest")
        registered_method = _identifier(self.registered_binding_method, "registered_binding_method", 50)
        registered_binding = _digest(self.registered_binding_digest, "registered_binding_digest")
        registered_at = _timestamp(self.registered_at, "registered_at")
        retirement = (
            self.retired_actor_id_digest,
            self.retired_binding_method,
            self.retired_binding_digest,
            self.retired_at,
        )
        if state == "active":
            if any(value is not None for value in retirement):
                raise ValueError("active signer key may not contain retirement fields.")
            retired_actor = retired_method = retired_binding = retired_at = None
        else:
            if any(value is None for value in retirement):
                raise ValueError("retired signer key requires retirement fields.")
            retired_actor = _digest(self.retired_actor_id_digest, "retired_actor_id_digest")
            retired_method = _identifier(self.retired_binding_method, "retired_binding_method", 50)
            retired_binding = _digest(self.retired_binding_digest, "retired_binding_digest")
            retired_at = _timestamp(self.retired_at, "retired_at")
            if retired_at < registered_at:
                raise ValueError("signer key retirement predates registration.")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("signer key schema is unsupported.")
        stable = {
            "scope": "rigorousrag-custody-ed25519-key-record-v1",
            "owner_id": owner,
            "key_id": key_id,
            "algorithm": algorithm,
            "public_key_raw_base64": base64.b64encode(raw).decode("ascii"),
            "public_key_sha256": fingerprint,
            "valid_from": valid_from,
            "valid_until": valid_until,
            "state": state,
            "registered_actor_id_digest": registered_actor,
            "registered_binding_method": registered_method,
            "registered_binding_digest": registered_binding,
            "registered_at": registered_at,
            "retired_actor_id_digest": redired_actor,
            "retired_binding_method": retired_method,
            "retired_binding_digest": retired_binding,
            "retired_at": retired_at,
            "schema_version": self.schema_version,
        }
        digest = _digest(self.record_digest, "record_digest")
        if digest != canonical_digest(stable):
            raise ValueError("record_digest differs from signer key record.")
        for name, value in stable.items():
            if name != "scope":
                object.__setattr__(self, name, value)
        object.__setattr__(self, "record_digest", digest)

    @classmethod
    def active(
        cls,
        *,
        owner_id: str,
        key_id: str,
        public_key_raw_base64: str,
        public_key_sha256: str,
        valid_from: float,
        valid_until: float | None,
        actor_binding: ReviewActorBinding,
        now: float,
    ) -> "CustodySignerKeyRecord":
        selected_actor = actor(actor_binding)
        values = {
            "owner_id": normalize_owner_id(owner_id),
            "key_id": _identifier(key_id, "key_id", 200),
            "algorithm": ALGORITHM,
            "public_key_raw_base64": public_key_raw_base64,
            "public_key_sha256": _digest(public_key_sha256, "public_key_sha256"),
            "valid_from": _timestamp(valid_from, "valid_from"),
            "valid_until": None if valid_until is None else _timestamp(valid_until, "valid_until"),
            "state": "active",
            "registered_actor_id_digest": actor_id_digest(selected_actor),
            "registered_binding_method": selected_actor.binding_method,
            "registered_binding_digest": selected_actor.binding_digest,
            "registered_at": _timestamp(now, "now"),
            "retired_actor_id_digest": None,
            "retired_binding_method": None,
            "retired_binding_digest": None,
            "retired_at": None,
            "schema_version": SCHEMA_VERSION,
        }
        return cls(
            **values,
            record_digest=canonical_digest(
                {"scope": "rigorousrag-custody-ed25519-key-record-v1", **values}
            ),
        )

    def retire(self, *, actor_binding: ReviewActorBinding, now: float) -> "CustodySignerKeyRecord":
        selected_actor = actor(actor_binding)
        actor_digest = actor_id_digest(selected_actor)
        if self.state == "retired":
            if (
                self.retired_actor_id_digest != actor_digest
                or self.retired_binding_method != selected_actor.binding_method
                or self.retired_binding_digest != selected_actor.binding_digest
            ):
                raise RuntimeError("signer key already retired by another actor.")
            return self
        values = asdict(self)
        values.pop("record_digest")
        values.update(
            state="retired",
            retired_actor_id_digest=actor_digest,
            retired_binding_method=selected_actor.binding_method,
            retired_binding_digest=selected_actor.binding_digest,
            retired_at=max(_timestamp(now, "now"), self.registered_at),
        )
        return CustodySignerKeyRecord(
            **values,
            record_digest=canonical_digest(
                {"scope": "rigorousrag-custody-ed25519-key-record-v1", **values}
            ),
        )

    def public_key(self) -> Ed25519PublicKey:
        raw = base64.b64decode(self.public_key_raw_base64.encode("ascii"), validate=True)
        return Ed25519PublicKey.from_public_bytes(raw)

    def permits(self, *, verification_time: float | None, now: float) -> bool:
        current = _timestamp(now, "now")
        selected = current if verification_time is None else _timestamp(verification_time, "verification_time")
        if verification_time is None and self.state != "active":
            return False
        upper = self.valid_until
        if self.retired_at is not None:
            upper = self.retired_at if upper is None else min(upper, self.retired_at)
        return selected >= self.valid_from and (upper is None or selected <= upper)
