"""Durable cutover-safe retrieval cache storing handles and digests, never snippets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tools.migration_compatibility import RetrievalCacheEntry, RetrievalCacheKey

_MAX_HITS = 1_000
_MAX_IDENTIFIER = 500
_MAX_TIME = 1.0e15


def _identifier(value: Any, label: str, maximum: int = _MAX_IDENTIFIER) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    result = value.strip()
    if (
        not result
        or len(result) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in result)
    ):
        raise ValueError(f"{label} is invalid.")
    return result


def _digest(value: Any, label: str) -> str:
    result = _identifier(value, label, 64).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return result


def _finite_nonnegative(value: Any, label: str, maximum: float = 1.0e18) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(result) or not 0.0 <= result <= maximum:
        raise ValueError(f"{label} must be finite and non-negative.")
    return result


def _rank(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("rank must be an integer.")
    if not 1 <= value <= _MAX_HITS:
        raise ValueError(f"rank must be between 1 and {_MAX_HITS}.")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _strict_json_object(value: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise RuntimeError("retrieval cache payload is corrupt.")

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = item
        return result

    try:
        parsed = json.loads(
            value,
            object_pairs_hook=hook,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("constant")),
        )
    except Exception as exc:
        raise RuntimeError("retrieval cache payload is corrupt.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("retrieval cache payload is corrupt.")
    return parsed


@dataclass(frozen=True)
class CachedRetrievalHit:
    """A cache-safe ranked handle. It intentionally contains no text/snippet field."""

    result_id: str
    source_id: str
    rank: int
    score: float
    content_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "result_id", _identifier(self.result_id, "result_id"))
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        object.__setattr__(self, "rank", _rank(self.rank))
        object.__setattr__(self, "score", _finite_nonnegative(self.score, "score"))
        object.__setattr__(
            self,
            "content_digest",
            _digest(self.content_digest, "content_digest"),
        )


@dataclass(frozen=True)
class CachedRetrievalResult:
    key: RetrievalCacheKey
    hits: tuple[CachedRetrievalHit, ...]
    created_at: float
    expires_at: float

    def __post_init__(self) -> None:
        if not isinstance(self.key, RetrievalCacheKey):
            raise ValueError("key must be RetrievalCacheKey.")
        if not isinstance(self.hits, tuple) or len(self.hits) > _MAX_HITS:
            raise ValueError("hits must be a bounded tuple.")
        if any(not isinstance(hit, CachedRetrievalHit) for hit in self.hits):
            raise ValueError("hits must contain CachedRetrievalHit values.")
        expected_ranks = tuple(range(1, len(self.hits) + 1))
        if tuple(hit.rank for hit in self.hits) != expected_ranks:
            raise ValueError("hit ranks must be contiguous and start at 1.")
        ids = [hit.result_id for hit in self.hits]
        if len(ids) != len(set(ids)):
            raise ValueError("cached result handles must be unique.")
        created = _finite_nonnegative(self.created_at, "created_at", _MAX_TIME)
        expires = _finite_nonnegative(self.expires_at, "expires_at", _MAX_TIME)
        if expires <= created:
            raise ValueError("expires_at must be greater than created_at.")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "expires_at", expires)

    @property
    def result_digest(self) -> str:
        return _sha256(
            {
                "cache_key": self.key.cache_key,
                "hits": [asdict(hit) for hit in self.hits],
                "created_at": self.created_at,
                "expires_at": self.expires_at,
            }
        )

    @property
    def compatibility_entry(self) -> RetrievalCacheEntry:
        return RetrievalCacheEntry(key=self.key, result_digest=self.result_digest)


class RetrievalCacheStore:
    """SQLite cache whose identity is cutover-bound and whose payload is handle-only."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        candidate = Path(os.fspath(path))
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        candidate.parent.mkdir(parents=True, exist_ok=True)
        self.path = candidate.absolute()
        self._lock = threading.RLock()
        try:
            self._connection = sqlite3.connect(
                str(self.path),
                check_same_thread=False,
                isolation_level=None,
            )
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS retrieval_cache (
                    cache_key TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    query_digest TEXT NOT NULL,
                    generation_sequence INTEGER NOT NULL,
                    profile_fingerprint TEXT NOT NULL,
                    retrieval_config_digest TEXT NOT NULL,
                    result_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS retrieval_cache_owner_generation "
                "ON retrieval_cache(owner_id, generation_sequence)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS retrieval_cache_expiry "
                "ON retrieval_cache(expires_at)"
            )
        except sqlite3.Error as exc:
            raise RuntimeError("retrieval cache store initialization failed.") from exc

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _payload(value: CachedRetrievalResult) -> str:
        return _canonical_json(
            {
                "hits": [asdict(hit) for hit in value.hits],
                "created_at": value.created_at,
                "expires_at": value.expires_at,
            }
        )

    @staticmethod
    def _decode(
        row: sqlite3.Row | tuple[Any, ...],
        expected_key: RetrievalCacheKey,
    ) -> CachedRetrievalResult:
        (
            cache_key,
            owner_id,
            query_digest,
            generation_sequence,
            profile_fingerprint,
            retrieval_config_digest,
            result_digest,
            payload_json,
            created_at,
            expires_at,
        ) = row
        identity = (
            cache_key,
            owner_id,
            query_digest,
            generation_sequence,
            profile_fingerprint,
            retrieval_config_digest,
        )
        expected = (
            expected_key.cache_key,
            expected_key.owner_id,
            expected_key.query_digest,
            expected_key.generation_sequence,
            expected_key.profile_fingerprint,
            expected_key.retrieval_config_digest,
        )
        if identity != expected:
            raise RuntimeError("retrieval cache identity is corrupt.")
        payload = _strict_json_object(payload_json)
        if set(payload) != {"hits", "created_at", "expires_at"}:
            raise RuntimeError("retrieval cache payload schema is corrupt.")
        raw_hits = payload["hits"]
        if not isinstance(raw_hits, list) or len(raw_hits) > _MAX_HITS:
            raise RuntimeError("retrieval cache payload is corrupt.")
        hits: list[CachedRetrievalHit] = []
        try:
            for raw in raw_hits:
                if not isinstance(raw, dict) or set(raw) != {
                    "result_id",
                    "source_id",
                    "rank",
                    "score",
                    "content_digest",
                }:
                    raise ValueError("hit schema")
                hits.append(CachedRetrievalHit(**raw))
            value = CachedRetrievalResult(
                key=expected_key,
                hits=tuple(hits),
                created_at=payload["created_at"],
                expires_at=payload["expires_at"],
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError("retrieval cache payload is corrupt.") from exc
        if created_at != value.created_at or expires_at != value.expires_at:
            raise RuntimeError("retrieval cache timing identity is corrupt.")
        if _digest(result_digest, "result_digest") != value.result_digest:
            raise RuntimeError("retrieval cache result digest is corrupt.")
        return value

    def _row(self, key: RetrievalCacheKey) -> tuple[Any, ...] | None:
        try:
            return self._connection.execute(
                """
                SELECT cache_key, owner_id, query_digest, generation_sequence,
                       profile_fingerprint, retrieval_config_digest, result_digest,
                       payload_json, created_at, expires_at
                FROM retrieval_cache
                WHERE cache_key=?
                """,
                (key.cache_key,),
            ).fetchone()
        except sqlite3.Error as exc:
            raise RuntimeError("retrieval cache read failed.") from exc

    def put(self, value: CachedRetrievalResult) -> CachedRetrievalResult:
        if not isinstance(value, CachedRetrievalResult):
            raise ValueError("value must be CachedRetrievalResult.")
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                existing = self._row(value.key)
                if existing is not None:
                    current = self._decode(existing, value.key)
                    if current.result_digest == value.result_digest:
                        self._connection.execute("COMMIT")
                        return current
                    if current.expires_at > value.created_at:
                        self._connection.execute("ROLLBACK")
                        raise RuntimeError(
                            "retrieval cache key collision requires explicit invalidation."
                        )
                    self._connection.execute(
                        "DELETE FROM retrieval_cache WHERE cache_key=?",
                        (value.key.cache_key,),
                    )
                self._connection.execute(
                    """
                    INSERT INTO retrieval_cache (
                        cache_key, owner_id, query_digest, generation_sequence,
                        profile_fingerprint, retrieval_config_digest, result_digest,
                        payload_json, created_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        value.key.cache_key,
                        value.key.owner_id,
                        value.key.query_digest,
                        value.key.generation_sequence,
                        value.key.profile_fingerprint,
                        value.key.retrieval_config_digest,
                        value.result_digest,
                        self._payload(value),
                        value.created_at,
                        value.expires_at,
                    ),
                )
                self._connection.execute("COMMIT")
                return value
            except RuntimeError:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise RuntimeError("retrieval cache write failed.") from exc

    def get(
        self,
        key: RetrievalCacheKey,
        *,
        now: float,
    ) -> CachedRetrievalResult | None:
        if not isinstance(key, RetrievalCacheKey):
            raise ValueError("key must be RetrievalCacheKey.")
        current_time = _finite_nonnegative(now, "now", _MAX_TIME)
        with self._lock:
            row = self._row(key)
            if row is None:
                return None
            value = self._decode(row, key)
            if value.expires_at <= current_time:
                try:
                    self._connection.execute(
                        "DELETE FROM retrieval_cache WHERE cache_key=?",
                        (key.cache_key,),
                    )
                except sqlite3.Error as exc:
                    raise RuntimeError("retrieval cache expiry cleanup failed.") from exc
                return None
            return value

    def invalidate_owner(self, owner_id: str) -> int:
        owner = _identifier(owner_id, "owner_id", 200)
        with self._lock:
            try:
                cursor = self._connection.execute(
                    "DELETE FROM retrieval_cache WHERE owner_id=?",
                    (owner,),
                )
            except sqlite3.Error as exc:
                raise RuntimeError("retrieval cache invalidation failed.") from exc
            return max(int(cursor.rowcount), 0)

    def invalidate_before_generation(
        self,
        *,
        owner_id: str,
        minimum_generation_sequence: int,
    ) -> int:
        owner = _identifier(owner_id, "owner_id", 200)
        if (
            isinstance(minimum_generation_sequence, bool)
            or not isinstance(minimum_generation_sequence, int)
            or not 1 <= minimum_generation_sequence <= 2**63 - 1
        ):
            raise ValueError("minimum_generation_sequence is invalid.")
        with self._lock:
            try:
                cursor = self._connection.execute(
                    """
                    DELETE FROM retrieval_cache
                    WHERE owner_id=? AND generation_sequence<?
                    """,
                    (owner, minimum_generation_sequence),
                )
            except sqlite3.Error as exc:
                raise RuntimeError("retrieval cache generation invalidation failed.") from exc
            return max(int(cursor.rowcount), 0)

    def prune_expired(self, *, now: float) -> int:
        current_time = _finite_nonnegative(now, "now", _MAX_TIME)
        with self._lock:
            try:
                cursor = self._connection.execute(
                    "DELETE FROM retrieval_cache WHERE expires_at<=?",
                    (current_time,),
                )
            except sqlite3.Error as exc:
                raise RuntimeError("retrieval cache pruning failed.") from exc
            return max(int(cursor.rowcount), 0)


__all__ = [
    "CachedRetrievalHit",
    "CachedRetrievalResult",
    "RetrievalCacheStore",
]
