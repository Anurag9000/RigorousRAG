"""Process-owned and signed actor binding for governed relation review."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.evidence_graph_relation_actor_assertion import (
    verify_review_actor_assertion,
)

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_MAX_ACTOR_BYTES = 4096


def _identifier(value: Any, label: str, maximum: int = 200) -> str:
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


def _timestamp(value: Any, label: str = "actor load time") -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(selected) or selected < 0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return selected


def _sha_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _hex_digest(value: Any, label: str) -> str:
    selected = _identifier(value, label, 64).lower()
    if len(selected) != 64 or any(character not in "0123456789abcdef" for character in selected):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return selected


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _actor_path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("review actor path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("review actor path is invalid.")
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
            raise ValueError("review actor path could not be validated.") from exc
        if _redirecting(info):
            raise ValueError("review actor path may not contain redirects.")
    return absolute


def _read_actor_file(value: str | os.PathLike[str]) -> str:
    path = _actor_path(value)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= _MAX_ACTOR_BYTES:
            raise ValueError("review actor file is invalid or too large.")
        payload = os.read(descriptor, _MAX_ACTOR_BYTES + 1)
        if len(payload) > _MAX_ACTOR_BYTES:
            raise ValueError("review actor file is too large.")
        try:
            rendered = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("review actor file must contain UTF-8 text.") from exc
        return _identifier(rendered, "review actor ID")
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class ReviewActorBinding:
    actor_id: str
    binding_method: str
    binding_digest: str
    loaded_at: float
    assertion_digest: str | None = None
    issuer: str | None = None
    expires_at: float | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "actor_id", _identifier(self.actor_id, "actor_id"))
        method = _identifier(self.binding_method, "binding_method", 50)
        if method not in {
            "process_environment",
            "descriptor_file",
            "hmac_assertion",
        }:
            raise ValueError("review actor binding method is unsupported.")
        object.__setattr__(self, "binding_method", method)
        assertion_digest = self.assertion_digest
        issuer = self.issuer
        expires_at = self.expires_at
        if method == "hmac_assertion":
            if assertion_digest is None or issuer is None or expires_at is None:
                raise ValueError(
                    "signed actor binding requires assertion digest, issuer and expiry."
                )
            assertion_digest = _hex_digest(
                assertion_digest, "assertion_digest"
            )
            issuer = _identifier(issuer, "issuer", 200)
            expires_at = _timestamp(expires_at, "expires_at")
        elif any(value is not None for value in (assertion_digest, issuer, expires_at)):
            raise ValueError(
                "direct actor bindings may not contain signed assertion fields."
            )
        object.__setattr__(self, "assertion_digest", assertion_digest)
        object.__setattr__(self, "issuer", issuer)
        object.__setattr__(self, "expires_at", expires_at)
        digest = _hex_digest(self.binding_digest, "binding_digest")
        expected = _sha_digest(
            {
                "scope": "rigorousrag-review-actor-binding-v1",
                "actor_id": self.actor_id,
                "binding_method": self.binding_method,
                "assertion_digest": self.assertion_digest,
                "issuer": self.issuer,
                "expires_at": self.expires_at,
            }
        )
        if digest != expected:
            raise ValueError("binding_digest differs from the actor binding identity.")
        object.__setattr__(self, "binding_digest", digest)
        object.__setattr__(self, "loaded_at", _timestamp(self.loaded_at))
        if self.expires_at is not None and self.expires_at < self.loaded_at:
            raise ValueError("signed actor binding is expired at load time.")
        if self.schema_version != 1:
            raise ValueError("review actor binding schema is unsupported.")

    @classmethod
    def create(
        cls,
        *,
        actor_id: str,
        binding_method: str,
        loaded_at: float | None = None,
        assertion_digest: str | None = None,
        issuer: str | None = None,
        expires_at: float | None = None,
    ) -> "ReviewActorBinding":
        selected_actor = _identifier(actor_id, "actor_id")
        selected_method = _identifier(binding_method, "binding_method", 50)
        selected_loaded = _timestamp(
            time.time() if loaded_at is None else loaded_at
        )
        selected_assertion = (
            None
            if assertion_digest is None
            else _hex_digest(assertion_digest, "assertion_digest")
        )
        selected_issuer = (
            None if issuer is None else _identifier(issuer, "issuer", 200)
        )
        selected_expires = (
            None
            if expires_at is None
            else _timestamp(expires_at, "expires_at")
        )
        digest = _sha_digest(
            {
                "scope": "rigorousrag-review-actor-binding-v1",
                "actor_id": selected_actor,
                "binding_method": selected_method,
                "assertion_digest": selected_assertion,
                "issuer": selected_issuer,
                "expires_at": selected_expires,
            }
        )
        return cls(
            actor_id=selected_actor,
            binding_method=selected_method,
            binding_digest=digest,
            loaded_at=selected_loaded,
            assertion_digest=selected_assertion,
            issuer=selected_issuer,
            expires_at=selected_expires,
        )


def load_relation_review_actor(
    *,
    actor_id: str | None = None,
    path: str | os.PathLike[str] | None = None,
    assertion_path: str | os.PathLike[str] | None = None,
    key_path: str | os.PathLike[str] | None = None,
    expected_issuer: str | None = None,
    loaded_at: float | None = None,
    clock_skew_seconds: float = 60.0,
) -> ReviewActorBinding:
    """Resolve exactly one actor source; absence and ambiguity fail closed."""

    if all(value is None for value in (actor_id, path, assertion_path)):
        configured_id = os.getenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ID")
        configured_path = os.getenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ID_PATH")
        configured_assertion = os.getenv(
            "EVIDENCE_GRAPH_REVIEW_ACTOR_ASSERTION_PATH"
        )
        configured_key = os.getenv("EVIDENCE_GRAPH_REVIEW_ACTOR_HMAC_KEY_PATH")
        configured_issuer = os.getenv(
            "EVIDENCE_GRAPH_REVIEW_ACTOR_EXPECTED_ISSUER"
        )
        actor_id = configured_id if configured_id else None
        path = configured_path if configured_path else None
        assertion_path = configured_assertion if configured_assertion else None
        key_path = configured_key if configured_key else None
        expected_issuer = configured_issuer if configured_issuer else None
    sources = sum(
        value is not None for value in (actor_id, path, assertion_path)
    )
    if sources != 1:
        if sources == 0:
            raise RuntimeError("relation-review actor identity is not configured.")
        raise RuntimeError("multiple relation-review actor sources are configured.")
    current = _timestamp(time.time() if loaded_at is None else loaded_at)
    if assertion_path is not None:
        if key_path is None or expected_issuer is None:
            raise RuntimeError(
                "signed relation-review actor requires key path and expected issuer."
            )
        assertion = verify_review_actor_assertion(
            assertion_path=assertion_path,
            key_path=key_path,
            expected_issuer=expected_issuer,
            now=current,
            clock_skew_seconds=clock_skew_seconds,
        )
        return ReviewActorBinding.create(
            actor_id=assertion.actor_id,
            binding_method="hmac_assertion",
            loaded_at=assertion.verified_at,
            assertion_digest=assertion.assertion_digest,
            issuer=assertion.issuer,
            expires_at=assertion.expires_at,
        )
    if key_path is not None or expected_issuer is not None:
        raise RuntimeError(
            "actor assertion key or issuer is configured without an assertion source."
        )
    if path is not None:
        selected = _read_actor_file(path)
        method = "descriptor_file"
    else:
        selected = _identifier(actor_id, "review actor ID")
        method = "process_environment"
    return ReviewActorBinding.create(
        actor_id=selected,
        binding_method=method,
        loaded_at=current,
    )


def require_relation_review_actor(
    requested_reviewer_id: str | None,
    *,
    binding: ReviewActorBinding | None = None,
) -> ReviewActorBinding:
    selected = load_relation_review_actor() if binding is None else binding
    if not isinstance(selected, ReviewActorBinding):
        raise ValueError("binding must be ReviewActorBinding.")
    if requested_reviewer_id is not None:
        requested = _identifier(requested_reviewer_id, "reviewer_id")
        if requested != selected.actor_id:
            raise PermissionError(
                "requested reviewer ID differs from the process-owned review actor."
            )
    return selected


__all__ = [
    "ReviewActorBinding",
    "load_relation_review_actor",
    "require_relation_review_actor",
]
