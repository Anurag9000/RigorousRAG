from __future__ import annotations

import hashlib

import pytest

from orchestration.cache_artifact_validation import materialize_cache_hit
from orchestration.generation_scoped_cache import CacheDependency, CacheEntry, CacheRevocation, CachedArtifact, GenerationScopedCacheKey, SQLiteGenerationScopedCache


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def key(*, owner="alice", generation="gen-a"):
    return GenerationScopedCacheKey(
        owner_id=owner,
        operation="context_generation",
        request_sha256=sha("request"),
        policy_sha256=sha("policy"),
        model_profile_sha256=sha("model"),
        dependencies=(
            CacheDependency("document_generation", generation, sha(f"artifact:{generation}")),
            CacheDependency("retrieval_index", "index-v1", sha("index")),
        ),
    )


def entry(selected_key=None, *, content=b"cached answer", created=1.0, expires=100.0):
    selected_key = selected_key or key()
    artifact = CachedArtifact("object-1", hashlib.sha256(content).hexdigest(), sha("source-receipt"), len(content))
    return CacheEntry(selected_key, artifact, created, expires)


def test_exact_dependency_key_hits_but_generation_change_misses(tmp_path) -> None:
    cache = SQLiteGenerationScopedCache(tmp_path / "cache.sqlite3")
    original = key(generation="gen-a")
    cache.put(entry(original))
    assert cache.lookup(original, now=10.0).status == "hit"
    assert cache.lookup(key(generation="gen-b"), now=10.0).status == "miss"


def test_cache_key_is_owner_scoped_and_request_digest_only(tmp_path) -> None:
    cache = SQLiteGenerationScopedCache(tmp_path / "cache.sqlite3")
    alice = key(owner="alice")
    cache.put(entry(alice))
    assert cache.lookup(key(owner="bob"), now=10.0).status == "miss"
    assert len(alice.request_sha256) == 64


def test_expired_entry_is_not_servable(tmp_path) -> None:
    cache = SQLiteGenerationScopedCache(tmp_path / "cache.sqlite3")
    selected_key = key()
    cache.put(entry(selected_key, expires=5.0))
    lookup = cache.lookup(selected_key, now=5.0)
    assert lookup.status == "expired"
    assert lookup.entry is None


def test_dependency_revocation_overrides_unexpired_cache_hit(tmp_path) -> None:
    cache = SQLiteGenerationScopedCache(tmp_path / "cache.sqlite3")
    selected_key = key()
    cache.put(entry(selected_key, expires=100.0))
    dependency = selected_key.dependencies[0]
    revocation = CacheRevocation("alice", dependency.dependency_sha256, "source_corrected", sha("correction"), 20.0)
    cache.revoke_dependency(revocation)
    assert cache.lookup(selected_key, now=19.0).status == "hit"
    lookup = cache.lookup(selected_key, now=20.0)
    assert lookup.status == "revoked"
    assert dependency.dependency_sha256 in lookup.blocking_dependency_sha256s


def test_same_cache_key_cannot_be_rebound_to_different_artifact(tmp_path) -> None:
    cache = SQLiteGenerationScopedCache(tmp_path / "cache.sqlite3")
    selected_key = key()
    cache.put(entry(selected_key, content=b"first"))
    with pytest.raises(RuntimeError, match="different immutable content"):
        cache.put(entry(selected_key, content=b"second"))


def test_expired_pruning_is_bounded(tmp_path) -> None:
    cache = SQLiteGenerationScopedCache(tmp_path / "cache.sqlite3")
    first = key(generation="gen-a")
    second = key(generation="gen-b")
    cache.put(entry(first, expires=5.0))
    cache.put(entry(second, expires=5.0))
    assert cache.prune_expired(now=10.0, max_entries=1) == 1
    statuses = {cache.lookup(first, now=10.0).status, cache.lookup(second, now=10.0).status}
    assert statuses == {"miss", "expired"}


class Provider:
    def __init__(self, content):
        self.content = content

    def read_bytes(self, *, artifact_id, max_bytes):
        return self.content


def test_cache_hit_bytes_are_rehashed_before_serving(tmp_path) -> None:
    cache = SQLiteGenerationScopedCache(tmp_path / "cache.sqlite3")
    content = b"cached answer"
    selected_key = key()
    cache.put(entry(selected_key, content=content))
    lookup = cache.lookup(selected_key, now=10.0)
    verified = materialize_cache_hit(lookup, provider=Provider(content))
    assert verified.content == content


def test_cached_artifact_tamper_is_rejected(tmp_path) -> None:
    cache = SQLiteGenerationScopedCache(tmp_path / "cache.sqlite3")
    selected_key = key()
    cache.put(entry(selected_key, content=b"cached answer"))
    lookup = cache.lookup(selected_key, now=10.0)
    with pytest.raises(RuntimeError, match="SHA-256"):
        materialize_cache_hit(lookup, provider=Provider(b"mutated"))
