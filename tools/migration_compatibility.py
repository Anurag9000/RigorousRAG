"""Derived compatibility guards for embedding migrations and retrieval caches."""

from __future__ import annotations

import hashlib
import json
import math
import operator
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from tools.security import normalize_owner_id

_MAX_QUERIES = 100_000
_MAX_RANKED = 10_000


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    result = value.strip()
    if not result or len(result) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{label} is invalid.")
    return result


def _digest(value: Any, label: str) -> str:
    result = _identifier(value, label, 64).lower()
    if len(result) != 64 or any(ch not in "0123456789abcdef" for ch in result):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return result


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        result = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return result


def _unit(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be between 0 and 1.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be between 0 and 1.") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return result


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _ranked_ids(values: Sequence[str], label: str, k: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or len(values) > _MAX_RANKED:
        raise ValueError(f"{label} must be a bounded sequence.")
    normalized = tuple(_identifier(value, label) for value in values[:k])
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} may not contain duplicate IDs.")
    return normalized


@dataclass(frozen=True)
class EmbeddingNeighborhoodCase:
    query_id: str
    current_ranked_ids: tuple[str, ...]
    shadow_ranked_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _identifier(self.query_id, "query_id"))
        for name in ("current_ranked_ids", "shadow_ranked_ids"):
            values = getattr(self, name)
            if isinstance(values, (str, bytes, bytearray)) or not values or len(values) > _MAX_RANKED:
                raise ValueError(f"{name} must be a bounded non-empty sequence.")
            normalized = tuple(_identifier(value, name) for value in values)
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"{name} may not contain duplicate IDs.")
            object.__setattr__(self, name, normalized)


@dataclass(frozen=True)
class EmbeddingSpaceCompatibility:
    current_dimensions: int
    shadow_dimensions: int
    query_count: int
    top_k: int
    mean_overlap_at_k: float
    mean_rank_displacement: float
    dimension_changed: bool
    compatible: bool

    def __post_init__(self) -> None:
        for name in ("current_dimensions", "shadow_dimensions"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1, 1_000_000))
        object.__setattr__(self, "query_count", _integer(self.query_count, "query_count", 1, _MAX_QUERIES))
        object.__setattr__(self, "top_k", _integer(self.top_k, "top_k", 1, _MAX_RANKED))
        object.__setattr__(self, "mean_overlap_at_k", _unit(self.mean_overlap_at_k, "mean_overlap_at_k"))
        if isinstance(self.mean_rank_displacement, bool):
            raise ValueError("mean_rank_displacement must be non-negative and finite.")
        displacement = float(self.mean_rank_displacement)
        if not math.isfinite(displacement) or displacement < 0.0:
            raise ValueError("mean_rank_displacement must be non-negative and finite.")
        object.__setattr__(self, "mean_rank_displacement", displacement)
        if not isinstance(self.dimension_changed, bool) or not isinstance(self.compatible, bool):
            raise ValueError("compatibility flags must be booleans.")

    @property
    def report_digest(self) -> str:
        return _sha256(asdict(self))


def evaluate_embedding_space_compatibility(
    cases: Sequence[EmbeddingNeighborhoodCase],
    *,
    current_dimensions: int,
    shadow_dimensions: int,
    top_k: int = 10,
    minimum_overlap: float = 0.70,
    maximum_rank_displacement: float = 3.0,
) -> EmbeddingSpaceCompatibility:
    """Compare paired nearest-neighbor neighborhoods independent of embedding dimension."""

    if isinstance(cases, (str, bytes, bytearray)) or not cases or len(cases) > _MAX_QUERIES:
        raise ValueError("cases must be a bounded non-empty sequence.")
    values = tuple(cases)
    if any(not isinstance(item, EmbeddingNeighborhoodCase) for item in values):
        raise ValueError("every case must be EmbeddingNeighborhoodCase.")
    if len({item.query_id for item in values}) != len(values):
        raise ValueError("query IDs must be unique.")
    k = _integer(top_k, "top_k", 1, _MAX_RANKED)
    overlap_floor = _unit(minimum_overlap, "minimum_overlap")
    if isinstance(maximum_rank_displacement, bool):
        raise ValueError("maximum_rank_displacement must be non-negative and finite.")
    max_displacement = float(maximum_rank_displacement)
    if not math.isfinite(max_displacement) or max_displacement < 0.0:
        raise ValueError("maximum_rank_displacement must be non-negative and finite.")

    overlaps: list[float] = []
    displacements: list[float] = []
    for case in values:
        current = _ranked_ids(case.current_ranked_ids, "current_ranked_ids", k)
        shadow = _ranked_ids(case.shadow_ranked_ids, "shadow_ranked_ids", k)
        current_set, shadow_set = set(current), set(shadow)
        denominator = max(min(k, len(current)), 1)
        overlaps.append(len(current_set & shadow_set) / denominator)
        current_positions = {value: index for index, value in enumerate(current, start=1)}
        shadow_positions = {value: index for index, value in enumerate(shadow, start=1)}
        common = current_set & shadow_set
        if common:
            displacements.append(
                sum(abs(current_positions[item] - shadow_positions[item]) for item in common) / len(common)
            )
        else:
            displacements.append(float(k))
    mean_overlap = sum(overlaps) / len(overlaps)
    mean_displacement = sum(displacements) / len(displacements)
    current_dim = _integer(current_dimensions, "current_dimensions", 1, 1_000_000)
    shadow_dim = _integer(shadow_dimensions, "shadow_dimensions", 1, 1_000_000)
    return EmbeddingSpaceCompatibility(
        current_dimensions=current_dim,
        shadow_dimensions=shadow_dim,
        query_count=len(values),
        top_k=k,
        mean_overlap_at_k=mean_overlap,
        mean_rank_displacement=mean_displacement,
        dimension_changed=current_dim != shadow_dim,
        compatible=mean_overlap >= overlap_floor and mean_displacement <= max_displacement,
    )


@dataclass(frozen=True)
class RetrievalCacheKey:
    owner_id: str
    query_digest: str
    generation_sequence: int
    profile_fingerprint: str
    retrieval_config_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "query_digest", _digest(self.query_digest, "query_digest"))
        object.__setattr__(
            self,
            "generation_sequence",
            _integer(self.generation_sequence, "generation_sequence", 1, 2**63 - 1),
        )
        object.__setattr__(
            self,
            "profile_fingerprint",
            _digest(self.profile_fingerprint, "profile_fingerprint"),
        )
        object.__setattr__(
            self,
            "retrieval_config_digest",
            _digest(self.retrieval_config_digest, "retrieval_config_digest"),
        )

    @property
    def cache_key(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class RetrievalCacheEntry:
    key: RetrievalCacheKey
    result_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, RetrievalCacheKey):
            raise ValueError("key must be RetrievalCacheKey.")
        object.__setattr__(self, "result_digest", _digest(self.result_digest, "result_digest"))


def cache_entry_is_current(
    entry: RetrievalCacheEntry,
    *,
    owner_id: str,
    generation_sequence: int,
    profile_fingerprint: str,
    retrieval_config_digest: str,
) -> bool:
    """Fail cache reuse closed across owner/generation/profile/config cutovers."""

    if not isinstance(entry, RetrievalCacheEntry):
        raise ValueError("entry must be RetrievalCacheEntry.")
    return (
        entry.key.owner_id == normalize_owner_id(owner_id)
        and entry.key.generation_sequence
        == _integer(generation_sequence, "generation_sequence", 1, 2**63 - 1)
        and entry.key.profile_fingerprint
        == _digest(profile_fingerprint, "profile_fingerprint")
        and entry.key.retrieval_config_digest
        == _digest(retrieval_config_digest, "retrieval_config_digest")
    )


__all__ = [
    "EmbeddingNeighborhoodCase",
    "EmbeddingSpaceCompatibility",
    "RetrievalCacheEntry",
    "RetrievalCacheKey",
    "cache_entry_is_current",
    "evaluate_embedding_space_compatibility",
]
