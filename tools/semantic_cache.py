"""Security-partitioned semantic cache with freshness and source-version guards."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tools.security import normalize_owner_id

_MAX_ENTRIES = 100_000
_MAX_VECTOR_DIM = 16_384
_MAX_SOURCES = 10_000


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    text = value.strip()
    if not text or len(text) > 500 or any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError(f"{label} is invalid.")
    return text


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite.") from exc
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        raise ValueError(f"{label} is invalid.")
    return parsed


def _vector(values: Sequence[Any]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not 1 <= len(values) <= _MAX_VECTOR_DIM:
        raise ValueError("embedding must be a bounded numeric sequence.")
    result = tuple(_finite(value, "embedding value") for value in values)
    norm = math.sqrt(sum(value * value for value in result))
    if norm <= 0.0:
        raise ValueError("embedding must have non-zero norm.")
    return result


def _versions(values: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(values, Mapping) or len(values) > _MAX_SOURCES:
        raise ValueError("source_versions must be a bounded mapping.")
    return {
        _identifier(key, "source ID"): _identifier(value, "source version")
        for key, value in values.items()
    }


def cosine_similarity(left: Sequence[Any], right: Sequence[Any]) -> float:
    a, b = _vector(left), _vector(right)
    if len(a) != len(b):
        raise ValueError("embeddings must have the same dimension.")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return max(-1.0, min(dot / (norm_a * norm_b), 1.0))


@dataclass(frozen=True)
class CachePartition:
    owner_id: str
    acl_fingerprint: str
    model_fingerprint: str
    policy_fingerprint: str
    corpus_generation: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        for field_name in ("acl_fingerprint", "model_fingerprint", "policy_fingerprint", "corpus_generation"):
            object.__setattr__(self, field_name, _identifier(getattr(self, field_name), field_name))


@dataclass(frozen=True)
class SemanticCacheEntry:
    entry_id: str
    partition: CachePartition
    query: str
    answer: str
    embedding: tuple[float, ...]
    source_versions: Mapping[str, str]
    created_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_id", _identifier(self.entry_id, "entry_id"))
        if not isinstance(self.partition, CachePartition):
            raise ValueError("partition must be CachePartition.")
        if not isinstance(self.query, str) or not self.query.strip() or len(self.query) > 20_000:
            raise ValueError("query must be a bounded non-empty string.")
        if not isinstance(self.answer, str) or len(self.answer) > 2_000_000:
            raise ValueError("answer must be a bounded string.")
        object.__setattr__(self, "embedding", _vector(self.embedding))
        object.__setattr__(self, "source_versions", _versions(self.source_versions))
        object.__setattr__(self, "created_at", _finite(self.created_at, "created_at", minimum=0.0))


@dataclass(frozen=True)
class CacheHit:
    entry: SemanticCacheEntry
    similarity: float
    age_seconds: float


class PartitionedSemanticCache:
    """In-memory bounded LRU cache whose matches cannot cross security partitions."""

    def __init__(self, *, max_entries: int = 10_000) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or not 1 <= max_entries <= _MAX_ENTRIES:
            raise ValueError("max_entries is invalid.")
        self._max_entries = max_entries
        self._entries: OrderedDict[str, SemanticCacheEntry] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def put(self, entry: SemanticCacheEntry) -> None:
        if not isinstance(entry, SemanticCacheEntry):
            raise ValueError("entry must be SemanticCacheEntry.")
        self._entries.pop(entry.entry_id, None)
        self._entries[entry.entry_id] = entry
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def invalidate_partition(self, partition: CachePartition) -> int:
        if not isinstance(partition, CachePartition):
            raise ValueError("partition must be CachePartition.")
        doomed = [key for key, entry in self._entries.items() if entry.partition == partition]
        for key in doomed:
            self._entries.pop(key, None)
        return len(doomed)

    def lookup(
        self,
        embedding: Sequence[Any],
        *,
        partition: CachePartition,
        current_source_versions: Mapping[str, Any],
        similarity_threshold: float = 0.92,
        max_age_seconds: float = 3600.0,
        now: float | None = None,
    ) -> CacheHit | None:
        if not isinstance(partition, CachePartition):
            raise ValueError("partition must be CachePartition.")
        query_vector = _vector(embedding)
        versions = _versions(current_source_versions)
        threshold = _finite(similarity_threshold, "similarity_threshold")
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1 and 1.")
        max_age = _finite(max_age_seconds, "max_age_seconds", minimum=0.0)
        selected_now = time.time() if now is None else _finite(now, "now", minimum=0.0)
        best: tuple[str, SemanticCacheEntry, float, float] | None = None
        for key, entry in self._entries.items():
            if entry.partition != partition or len(entry.embedding) != len(query_vector):
                continue
            age = max(0.0, selected_now - entry.created_at)
            if age > max_age:
                continue
            if any(versions.get(source_id) != version for source_id, version in entry.source_versions.items()):
                continue
            similarity = cosine_similarity(query_vector, entry.embedding)
            if similarity < threshold:
                continue
            ordering = (similarity, entry.created_at, entry.entry_id)
            if best is None or ordering > (best[2], best[1].created_at, best[1].entry_id):
                best = (key, entry, similarity, age)
        if best is None:
            return None
        key, entry, similarity, age = best
        self._entries.move_to_end(key)
        return CacheHit(entry=entry, similarity=similarity, age_seconds=age)


__all__ = [
    "CacheHit",
    "CachePartition",
    "PartitionedSemanticCache",
    "SemanticCacheEntry",
    "cosine_similarity",
]
