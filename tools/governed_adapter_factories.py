"""Governed lazy factories for adapter-required embedding profiles.

No external model is imported or downloaded at module import time. Operators provide a
loader that receives a pinned revision and a registered adapter artifact. The wrapper
validates profile identity, row counts, vector dimensions, and finite outputs before
exposing the encoder to the existing embedding-adapter registry.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from tools.adapter_registry import AdapterRegistry, AdapterVersion
from tools.embedding_adapters import EmbeddingEncoder, register_embedding_adapter
from tools.embedding_models import EmbeddingProfile
from tools.embedding_registry import resolve_embedding_profile

_MAX_BATCH = 4_096
_MAX_PASSAGES = 100_000
_MAX_PASSAGE_CHARS = 5_000_000
_ADAPTER_PROFILES = frozenset({"instructor-base", "specter2", "bge-m3"})


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    rendered = value.strip()
    if not rendered or len(rendered) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in rendered):
        raise ValueError(f"{label} is invalid.")
    return rendered


def _checksum(value: Any) -> str:
    text = _text(value, "checksum_sha256", 64).lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError("checksum_sha256 must contain exactly 64 hexadecimal characters.")
    return text


def _batch(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _MAX_BATCH:
        raise ValueError(f"batch_size must be between 1 and {_MAX_BATCH}.")
    return value


def _passages(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not 1 <= len(values) <= _MAX_PASSAGES:
        raise ValueError("passages must be a bounded non-empty sequence.")
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("passages must contain strings.")
        text = value.strip()
        if not text or len(text) > _MAX_PASSAGE_CHARS or "\x00" in text:
            raise ValueError("passage is empty, invalid, or too long.")
        result.append(text)
    return tuple(result)


def _vector(value: Any, dimensions: int | None) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise RuntimeError("adapter returned an invalid embedding vector.")
    try:
        rows = list(itertools.islice(iter(value), (dimensions or 1_000_000) + 1))
    except Exception as exc:
        raise RuntimeError("adapter returned an invalid embedding vector.") from exc
    if not rows or (dimensions is not None and len(rows) != dimensions):
        raise RuntimeError("adapter embedding dimensions do not match the profile.")
    result: list[float] = []
    for item in rows:
        if isinstance(item, bool):
            raise RuntimeError("adapter embeddings must contain finite numbers.")
        try:
            numeric = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("adapter embeddings must contain finite numbers.") from exc
        if not math.isfinite(numeric):
            raise RuntimeError("adapter embeddings must contain finite numbers.")
        result.append(numeric)
    return tuple(result)


@dataclass(frozen=True)
class AdapterLoadRequest:
    profile_alias: str
    pinned_revision: str
    artifact_name: str
    artifact_version: str
    artifact_uri: str
    checksum_sha256: str
    allow_download: bool = False
    trust_remote_code: bool = False
    batch_size: int = 32

    def __post_init__(self) -> None:
        profile = resolve_embedding_profile(self.profile_alias, allow_compatibility=False)
        if profile.alias not in _ADAPTER_PROFILES or not profile.requires_adapter:
            raise ValueError("profile_alias must identify a built-in adapter-required profile.")
        object.__setattr__(self, "profile_alias", profile.alias)
        object.__setattr__(self, "pinned_revision", _text(self.pinned_revision, "pinned_revision", 300))
        object.__setattr__(self, "artifact_name", _text(self.artifact_name, "artifact_name", 300))
        object.__setattr__(self, "artifact_version", _text(self.artifact_version, "artifact_version", 100))
        object.__setattr__(self, "artifact_uri", _text(self.artifact_uri, "artifact_uri", 4_096))
        object.__setattr__(self, "checksum_sha256", _checksum(self.checksum_sha256))
        if not isinstance(self.allow_download, bool) or not isinstance(self.trust_remote_code, bool):
            raise ValueError("download and trust_remote_code controls must be booleans.")
        object.__setattr__(self, "batch_size", _batch(self.batch_size))


class AdapterBackend(Protocol):
    def encode_passages(self, passages: Sequence[str]) -> Sequence[Sequence[float]]: ...


AdapterLoader = Callable[[EmbeddingProfile, AdapterLoadRequest], AdapterBackend]


class GovernedAdapterEncoder:
    def __init__(self, profile: EmbeddingProfile, request: AdapterLoadRequest, backend: AdapterBackend) -> None:
        if not isinstance(profile, EmbeddingProfile) or not profile.requires_adapter:
            raise ValueError("profile must require an explicit embedding adapter.")
        if not isinstance(request, AdapterLoadRequest) or request.profile_alias != profile.alias:
            raise ValueError("request does not match the embedding profile.")
        if not callable(getattr(backend, "encode_passages", None)):
            raise ValueError("adapter backend must expose encode_passages().")
        self.profile = profile
        self.request = request
        self._backend = backend
        identity = "|".join(
            (
                profile.fingerprint,
                request.pinned_revision,
                request.artifact_name,
                request.artifact_version,
                request.checksum_sha256,
            )
        ).encode("utf-8")
        self.model_instance_id = hashlib.sha256(identity).hexdigest()

    def encode_passages(self, passages: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        bounded = _passages(passages)
        formatted = tuple(self.profile.format_passage(value) for value in bounded)
        try:
            raw = self._backend.encode_passages(formatted)
        except Exception as exc:
            raise RuntimeError("governed embedding adapter execution failed.") from exc
        if isinstance(raw, (str, bytes, bytearray)):
            raise RuntimeError("governed embedding adapter returned an invalid result.")
        try:
            rows = list(itertools.islice(iter(raw), len(formatted) + 1))
        except Exception as exc:
            raise RuntimeError("governed embedding adapter returned an invalid result.") from exc
        if len(rows) != len(formatted):
            raise RuntimeError("governed embedding adapter returned the wrong row count.")
        vectors = tuple(_vector(row, self.profile.dimensions) for row in rows)
        if self.profile.dimensions is None:
            dimension = len(vectors[0])
            if any(len(vector) != dimension for vector in vectors):
                raise RuntimeError("adapter returned inconsistent vector dimensions.")
        return vectors


def request_from_active_adapter(
    registry: AdapterRegistry,
    *,
    profile_alias: str,
    adapter_name: str,
    pinned_revision: str,
    allow_download: bool = False,
    trust_remote_code: bool = False,
    batch_size: int = 32,
) -> AdapterLoadRequest:
    if not isinstance(registry, AdapterRegistry):
        raise ValueError("registry must be AdapterRegistry.")
    name = _text(adapter_name, "adapter_name", 300)
    record = registry.active(name)
    if record is None:
        raise RuntimeError(f"no active adapter is registered for {name!r}.")
    if not isinstance(record, AdapterVersion):
        raise RuntimeError("active adapter registry record is invalid.")
    if record.kind != "embedding":
        raise ValueError("active adapter must have kind='embedding'.")
    return AdapterLoadRequest(
        profile_alias=profile_alias,
        pinned_revision=pinned_revision,
        artifact_name=record.name,
        artifact_version=record.version,
        artifact_uri=record.artifact_uri,
        checksum_sha256=record.checksum_sha256,
        allow_download=allow_download,
        trust_remote_code=trust_remote_code,
        batch_size=batch_size,
    )


def make_governed_adapter_factory(request: AdapterLoadRequest, loader: AdapterLoader) -> Callable[[EmbeddingProfile], EmbeddingEncoder]:
    if not isinstance(request, AdapterLoadRequest):
        raise ValueError("request must be AdapterLoadRequest.")
    if not callable(loader):
        raise ValueError("loader must be callable.")

    def factory(profile: EmbeddingProfile) -> EmbeddingEncoder:
        if not isinstance(profile, EmbeddingProfile) or profile.alias != request.profile_alias:
            raise RuntimeError("embedding registry requested the wrong governed adapter profile.")
        try:
            backend = loader(profile, request)
        except Exception as exc:
            raise RuntimeError("governed embedding adapter loading failed.") from exc
        return GovernedAdapterEncoder(profile, request, backend)

    return factory


def register_governed_adapter(
    request: AdapterLoadRequest,
    loader: AdapterLoader,
    *,
    replace: bool = False,
) -> None:
    register_embedding_adapter(
        request.profile_alias,
        make_governed_adapter_factory(request, loader),
        replace=replace,
    )


__all__ = [
    "AdapterBackend",
    "AdapterLoadRequest",
    "AdapterLoader",
    "GovernedAdapterEncoder",
    "make_governed_adapter_factory",
    "register_governed_adapter",
    "request_from_active_adapter",
]
