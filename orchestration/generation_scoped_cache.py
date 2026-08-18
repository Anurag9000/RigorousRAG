"""Owner-scoped, dependency-addressed cache authority for RAG artifacts.

Cache keys contain no raw query/document text. They bind an opaque request SHA-256 to the
exact policies, model profile and immutable dependency generations/artifacts that produced
the result. A changed dependency therefore creates a different key automatically.
Explicit dependency revocation overrides TTL so retired/corrected artifacts cannot remain
servable through a previously valid cache entry.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _text(value: Any, label: str, maximum: int = 1000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _time(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative")
    selected = float(value)
    if not math.isfinite(selected) or selected < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return selected


@dataclass(frozen=True)
class CacheDependency:
    kind: str
    identity: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _text(self.kind, "dependency kind", 200))
        object.__setattr__(self, "identity", _text(self.identity, "dependency identity", 1000))
        object.__setattr__(self, "artifact_sha256", _sha(self.artifact_sha256, "artifact_sha256"))

    @property
    def dependency_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-cache-dependency/v1", **asdict(self)})


@dataclass(frozen=True)
class GenerationScopedCacheKey:
    owner_id: str
    operation: str
    request_sha256: str
    policy_sha256: str
    model_profile_sha256: str | None
    dependencies: tuple[CacheDependency, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", _text(self.owner_id, "owner_id"))
        object.__setattr__(self, "operation", _text(self.operation, "operation", 300))
        object.__setattr__(self, "request_sha256", _sha(self.request_sha256, "request_sha256"))
        object.__setattr__(self, "policy_sha256", _sha(self.policy_sha256, "policy_sha256"))
        if self.model_profile_sha256 is not None:
            object.__setattr__(self, "model_profile_sha256", _sha(self.model_profile_sha256, "model_profile_sha256"))
        dependencies = tuple(sorted(self.dependencies, key=lambda row: (row.kind, row.identity, row.artifact_sha256)))
        if not dependencies or any(not isinstance(row, CacheDependency) for row in dependencies):
            raise ValueError("cache key dependencies must be non-empty CacheDependency values")
        if len({(row.kind, row.identity) for row in dependencies}) != len(dependencies):
            raise ValueError("cache dependencies must be unique by kind/identity")
        object.__setattr__(self, "dependencies", dependencies)

    @property
    def key_sha256(self) -> str:
        return _digest({
            "schema": "rigorousrag-generation-scoped-cache-key/v1",
            "owner_id": self.owner_id,
            "operation": self.operation,
            "request_sha256": self.request_sha256,
            "policy_sha256": self.policy_sha256,
            "model_profile_sha256": self.model_profile_sha256,
            "dependencies": [asdict(row) for row in self.dependencies],
        })


@dataclass(frozen=True)
class CachedArtifact:
    artifact_id: str
    artifact_sha256: str
    source_receipt_sha256: str
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _text(self.artifact_id, "artifact_id", 4000))
        object.__setattr__(self, "artifact_sha256", _sha(self.artifact_sha256, "artifact_sha256"))
        object.__setattr__(self, "source_receipt_sha256", _sha(self.source_receipt_sha256, "source_receipt_sha256"))
        if self.size_bytes is not None and (isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0):
            raise ValueError("size_bytes must be non-negative when set")

    @property
    def descriptor_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-cached-artifact/v1", **asdict(self)})


@dataclass(frozen=True)
class CacheEntry:
    key: GenerationScopedCacheKey
    artifact: CachedArtifact
    created_at: float
    expires_at: float

    def __post_init__(self) -> None:
        if not isinstance(self.key, GenerationScopedCacheKey) or not isinstance(self.artifact, CachedArtifact):
            raise ValueError("cache entry key/artifact types are invalid")
        created = _time(self.created_at, "created_at")
        expires = _time(self.expires_at, "expires_at")
        if expires <= created:
            raise ValueError("expires_at must be later than created_at")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "expires_at", expires)

    @property
    def entry_sha256(self) -> str:
        return _digest({
            "schema": "rigorousrag-cache-entry/v1",
            "key_sha256": self.key.key_sha256,
            "artifact": asdict(self.artifact),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        })


@dataclass(frozen=True)
class CacheRevocation:
    owner_id: str
    dependency_sha256: str
    reason_code: str
    evidence_sha256: str
    revoked_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", _text(self.owner_id, "owner_id"))
        object.__setattr__(self, "dependency_sha256", _sha(self.dependency_sha256, "dependency_sha256"))
        object.__setattr__(self, "reason_code", _text(self.reason_code, "reason_code", 300))
        object.__setattr__(self, "evidence_sha256", _sha(self.evidence_sha256, "evidence_sha256"))
        object.__setattr__(self, "revoked_at", _time(self.revoked_at, "revoked_at"))

    @property
    def revocation_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-cache-revocation/v1", **asdict(self)})


@dataclass(frozen=True)
class CacheLookup:
    status: str
    key_sha256: str
    entry: CacheEntry | None
    blocking_dependency_sha256s: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"hit", "miss", "expired", "revoked"}:
            raise ValueError("cache lookup status is invalid")
        object.__setattr__(self, "key_sha256", _sha(self.key_sha256, "key_sha256"))
        blockers = tuple(sorted(_sha(value, "blocking dependency sha256") for value in self.blocking_dependency_sha256s))
        object.__setattr__(self, "blocking_dependency_sha256s", blockers)
        if self.status == "hit" and self.entry is None:
            raise ValueError("cache hit requires an entry")
        if self.status != "hit" and self.entry is not None:
            raise ValueError("non-hit cache lookup may not expose a servable entry")
        if self.status == "revoked" and not blockers:
            raise ValueError("revoked cache lookup requires blocking dependencies")


class SQLiteGenerationScopedCache:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS cache_entries (
                    key_sha256 TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    key_json TEXT NOT NULL,
                    artifact_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    entry_sha256 TEXT NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS cache_dependency_index (
                    key_sha256 TEXT NOT NULL,
                    dependency_sha256 TEXT NOT NULL,
                    PRIMARY KEY(key_sha256,dependency_sha256),
                    FOREIGN KEY(key_sha256) REFERENCES cache_entries(key_sha256) ON DELETE CASCADE
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS cache_revocations (
                    revocation_sha256 TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    dependency_sha256 TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    evidence_sha256 TEXT NOT NULL,
                    revoked_at REAL NOT NULL
                )"""
            )
            connection.execute("CREATE INDEX IF NOT EXISTS cache_owner_idx ON cache_entries(owner_id,key_sha256)")
            connection.execute("CREATE INDEX IF NOT EXISTS cache_revoked_dep_idx ON cache_revocations(owner_id,dependency_sha256)")

    @staticmethod
    def _key_json(key: GenerationScopedCacheKey) -> str:
        return json.dumps({"owner_id": key.owner_id, "operation": key.operation, "request_sha256": key.request_sha256, "policy_sha256": key.policy_sha256, "model_profile_sha256": key.model_profile_sha256, "dependencies": [asdict(row) for row in key.dependencies]}, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @staticmethod
    def _artifact_json(artifact: CachedArtifact) -> str:
        return json.dumps(asdict(artifact), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @staticmethod
    def _decode_entry(row: sqlite3.Row) -> CacheEntry:
        key_raw = json.loads(row["key_json"])
        key = GenerationScopedCacheKey(
            key_raw["owner_id"], key_raw["operation"], key_raw["request_sha256"], key_raw["policy_sha256"], key_raw["model_profile_sha256"], tuple(CacheDependency(**value) for value in key_raw["dependencies"])
        )
        artifact = CachedArtifact(**json.loads(row["artifact_json"]))
        entry = CacheEntry(key, artifact, float(row["created_at"]), float(row["expires_at"]))
        if entry.entry_sha256 != row["entry_sha256"] or entry.key.key_sha256 != row["key_sha256"]:
            raise RuntimeError("durable cache entry failed integrity reconstruction")
        return entry

    def put(self, entry: CacheEntry) -> str:
        if not isinstance(entry, CacheEntry):
            raise ValueError("entry must be CacheEntry")
        key_sha = entry.key.key_sha256
        key_json = self._key_json(entry.key)
        artifact_json = self._artifact_json(entry.artifact)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("SELECT * FROM cache_entries WHERE key_sha256=?", (key_sha,)).fetchone()
            if existing is not None:
                prior = self._decode_entry(existing)
                if prior != entry:
                    raise RuntimeError("cache key already maps to different immutable content")
                return prior.entry_sha256
            connection.execute(
                "INSERT INTO cache_entries(key_sha256,owner_id,key_json,artifact_json,created_at,expires_at,entry_sha256) VALUES(?,?,?,?,?,?,?)",
                (key_sha, entry.key.owner_id, key_json, artifact_json, entry.created_at, entry.expires_at, entry.entry_sha256),
            )
            connection.executemany(
                "INSERT INTO cache_dependency_index(key_sha256,dependency_sha256) VALUES(?,?)",
                [(key_sha, dependency.dependency_sha256) for dependency in entry.key.dependencies],
            )
        return entry.entry_sha256

    def revoke_dependency(self, revocation: CacheRevocation) -> str:
        if not isinstance(revocation, CacheRevocation):
            raise ValueError("revocation must be CacheRevocation")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT owner_id,dependency_sha256,reason_code,evidence_sha256,revoked_at FROM cache_revocations WHERE revocation_sha256=?", (revocation.revocation_sha256,)).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO cache_revocations(revocation_sha256,owner_id,dependency_sha256,reason_code,evidence_sha256,revoked_at) VALUES(?,?,?,?,?,?)",
                    (revocation.revocation_sha256, revocation.owner_id, revocation.dependency_sha256, revocation.reason_code, revocation.evidence_sha256, revocation.revoked_at),
                )
            else:
                reconstructed = CacheRevocation(row["owner_id"], row["dependency_sha256"], row["reason_code"], row["evidence_sha256"], float(row["revoked_at"]))
                if reconstructed != revocation:
                    raise RuntimeError("cache revocation identity collision")
        return revocation.revocation_sha256

    def lookup(self, key: GenerationScopedCacheKey, *, now: float) -> CacheLookup:
        if not isinstance(key, GenerationScopedCacheKey):
            raise ValueError("key must be GenerationScopedCacheKey")
        timestamp = _time(now, "now")
        key_sha = key.key_sha256
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM cache_entries WHERE key_sha256=? AND owner_id=?", (key_sha, key.owner_id)).fetchone()
            if row is None:
                return CacheLookup("miss", key_sha, None)
            entry = self._decode_entry(row)
            blockers = connection.execute(
                """SELECT DISTINCT r.dependency_sha256
                   FROM cache_dependency_index d
                   JOIN cache_revocations r ON r.dependency_sha256=d.dependency_sha256 AND r.owner_id=?
                   WHERE d.key_sha256=? AND r.revoked_at<=?
                   ORDER BY r.dependency_sha256""",
                (key.owner_id, key_sha, timestamp),
            ).fetchall()
        blocked = tuple(row["dependency_sha256"] for row in blockers)
        if blocked:
            return CacheLookup("revoked", key_sha, None, blocked)
        if timestamp >= entry.expires_at:
            return CacheLookup("expired", key_sha, None)
        return CacheLookup("hit", key_sha, entry)

    def prune_expired(self, *, now: float, max_entries: int = 1000) -> int:
        timestamp = _time(now, "now")
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries must be positive")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute("SELECT key_sha256 FROM cache_entries WHERE expires_at<=? ORDER BY expires_at,key_sha256 LIMIT ?", (timestamp, max_entries)).fetchall()
            keys = [row["key_sha256"] for row in rows]
            for key_sha in keys:
                connection.execute("DELETE FROM cache_dependency_index WHERE key_sha256=?", (key_sha,))
                connection.execute("DELETE FROM cache_entries WHERE key_sha256=?", (key_sha,))
        return len(keys)


__all__ = [
    "CacheDependency",
    "CacheEntry",
    "CacheLookup",
    "CacheRevocation",
    "CachedArtifact",
    "GenerationScopedCacheKey",
    "SQLiteGenerationScopedCache",
]
