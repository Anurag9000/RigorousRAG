"""Process-owned actor binding for governed semantic relation review."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _timestamp(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("actor load time must be finite and non-negative.")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("actor load time must be finite and non-negative.") from exc
    if not selected >= 0 or selected == float("inf") or selected != selected:
        raise ValueError("actor load time must be finite and non-negative.")
    return selected


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


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
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "actor_id", _identifier(self.actor_id, "actor_id"))
        method = _identifier(self.binding_method, "binding_method", 50)
        if method not in {"process_environment", "descriptor_file"}:
            raise ValueError("review actor binding method is unsupported.")
        object.__setattr__(self, "binding_method", method)
        digest = _identifier(self.binding_digest, "binding_digest", 64).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("binding_digest must be a SHA-256 digest.")
        expected = _digest(
            {
                "scope": "rigorousrag-review-actor-binding-v1",
                "actor_id": self.actor_id,
                "binding_method": self.binding_method,
            }
        )
        if digest != expected:
            raise ValueError("binding_digest differs from the actor binding identity.")
        object.__setattr__(self, "binding_digest", digest)
        object.__setattr__(self, "loaded_at", _timestamp(self.loaded_at))
        if self.schema_version != 1:
            raise ValueError("review actor binding schema is unsupported.")

    @classmethod
    def create(
        cls,
        *,
        actor_id: str,
        binding_method: str,
        loaded_at: float | None = None,
    ) -> "ReviewActorBinding":
        selected_actor = _identifier(actor_id, "actor_id")
        selected_method = _identifier(binding_method, "binding_method", 50)
        digest = _digest(
            {
                "scope": "rigorousrag-review-actor-binding-v1",
                "actor_id": selected_actor,
                "binding_method": selected_method,
            }
        )
        return cls(
            actor_id=selected_actor,
            binding_method=selected_method,
            binding_digest=digest,
            loaded_at=time.time() if loaded_at is None else loaded_at,
        )


def load_relation_review_actor(
    *,
    actor_id: str | None = None,
    path: str | os.PathLike[str] | None = None,
    loaded_at: float | None = None,
) -> ReviewActorBinding:
    """Resolve exactly one process-owned actor source; absence fails closed."""

    if actor_id is None and path is None:
        configured_id = os.getenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ID")
        configured_path = os.getenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ID_PATH")
        actor_id = configured_id if configured_id else None
        path = configured_path if configured_path else None
    if actor_id is not None and path is not None:
        raise RuntimeError("multiple relation-review actor sources are configured.")
    if actor_id is None and path is None:
        raise RuntimeError("relation-review actor identity is not configured.")
    if path is not None:
        selected = _read_actor_file(path)
        method = "descriptor_file"
    else:
        selected = _identifier(actor_id, "review actor ID")
        method = "process_environment"
    return ReviewActorBinding.create(
        actor_id=selected,
        binding_method=method,
        loaded_at=time.time() if loaded_at is None else loaded_at,
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
