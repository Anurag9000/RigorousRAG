"""Last-mile integrity validation for artifacts referenced by the generation-scoped cache."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from orchestration.generation_scoped_cache import CacheLookup, CachedArtifact


class CachedArtifactProvider(Protocol):
    def read_bytes(self, *, artifact_id: str, max_bytes: int) -> bytes: ...


@dataclass(frozen=True)
class VerifiedCachedArtifact:
    descriptor: CachedArtifact
    content: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, CachedArtifact):
            raise ValueError("descriptor must be CachedArtifact")
        if not isinstance(self.content, bytes):
            raise ValueError("content must be bytes")
        if hashlib.sha256(self.content).hexdigest() != self.descriptor.artifact_sha256:
            raise ValueError("cached artifact content does not match descriptor SHA-256")
        if self.descriptor.size_bytes is not None and len(self.content) != self.descriptor.size_bytes:
            raise ValueError("cached artifact content size does not match descriptor")


def materialize_cache_hit(
    lookup: CacheLookup,
    *,
    provider: CachedArtifactProvider,
    max_artifact_bytes: int = 32 * 1024 * 1024,
) -> VerifiedCachedArtifact:
    if not isinstance(lookup, CacheLookup):
        raise ValueError("lookup must be CacheLookup")
    if lookup.status != "hit" or lookup.entry is None:
        raise ValueError("only a cache hit may be materialized")
    if isinstance(max_artifact_bytes, bool) or not isinstance(max_artifact_bytes, int) or max_artifact_bytes < 1:
        raise ValueError("max_artifact_bytes must be positive")
    descriptor = lookup.entry.artifact
    if descriptor.size_bytes is not None and descriptor.size_bytes > max_artifact_bytes:
        raise ValueError("cached artifact descriptor exceeds materialization size limit")
    content = provider.read_bytes(artifact_id=descriptor.artifact_id, max_bytes=max_artifact_bytes)
    if not isinstance(content, bytes):
        raise RuntimeError("cached artifact provider did not return bytes")
    if len(content) > max_artifact_bytes:
        raise RuntimeError("cached artifact provider exceeded materialization size limit")
    if hashlib.sha256(content).hexdigest() != descriptor.artifact_sha256:
        raise RuntimeError("cached artifact bytes failed SHA-256 validation")
    if descriptor.size_bytes is not None and len(content) != descriptor.size_bytes:
        raise RuntimeError("cached artifact bytes failed size validation")
    return VerifiedCachedArtifact(descriptor, content)


__all__ = ["CachedArtifactProvider", "VerifiedCachedArtifact", "materialize_cache_hit"]
