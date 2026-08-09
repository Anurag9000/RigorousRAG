from __future__ import annotations

import hashlib
import sqlite3

import pytest

from tools.migration_compatibility import RetrievalCacheKey, cache_entry_is_current
from tools.retrieval_cache import (
    CachedRetrievalHit,
    CachedRetrievalResult,
    RetrievalCacheStore,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def cache_key(
    *,
    owner: str = "alice",
    generation: int = 3,
    profile: str = "profile-a",
    config: str = "config-a",
) -> RetrievalCacheKey:
    return RetrievalCacheKey(
        owner_id=owner,
        query_digest=digest("query"),
        generation_sequence=generation,
        profile_fingerprint=digest(profile),
        retrieval_config_digest=digest(config),
    )


def cached_result(
    key: RetrievalCacheKey,
    *,
    result_id: str = "chunk-1",
    created_at: float = 10.0,
    expires_at: float = 20.0,
) -> CachedRetrievalResult:
    return CachedRetrievalResult(
        key=key,
        hits=(
            CachedRetrievalHit(
                result_id=result_id,
                source_id="doc-1",
                rank=1,
                score=0.91,
                content_digest=digest(result_id),
            ),
        ),
        created_at=created_at,
        expires_at=expires_at,
    )


def test_roundtrip_is_handle_only_and_cutover_bound(tmp_path):
    path = tmp_path / "retrieval-cache.sqlite3"
    store = RetrievalCacheStore(path)
    key = cache_key()
    value = cached_result(key)
    assert store.put(value) == value
    assert store.put(value) == value
    assert store.get(key, now=15.0) == value
    assert cache_entry_is_current(
        value.compatibility_entry,
        owner_id="alice",
        generation_sequence=3,
        profile_fingerprint=digest("profile-a"),
        retrieval_config_digest=digest("config-a"),
    )
    assert store.get(cache_key(generation=4), now=15.0) is None
    assert store.get(cache_key(profile="profile-b"), now=15.0) is None
    assert store.get(cache_key(config="config-b"), now=15.0) is None
    assert store.get(cache_key(owner="bob"), now=15.0) is None

    with sqlite3.connect(path) as connection:
        payload = connection.execute(
            "SELECT payload_json FROM retrieval_cache WHERE cache_key=?",
            (key.cache_key,),
        ).fetchone()[0]
    assert "secret snippet" not in payload
    assert "text" not in payload and "snippet" not in payload
    store.close()


def test_expiry_pruning_and_expired_key_replacement_are_deterministic(tmp_path):
    store = RetrievalCacheStore(tmp_path / "cache.sqlite3")
    key = cache_key()
    store.put(cached_result(key))
    assert store.get(key, now=20.0) is None
    replacement = cached_result(
        key,
        result_id="chunk-2",
        created_at=21.0,
        expires_at=30.0,
    )
    store.put(replacement)
    assert store.get(key, now=25.0) == replacement
    assert store.prune_expired(now=30.0) == 1
    assert store.get(key, now=30.0) is None


def test_live_key_collision_requires_explicit_invalidation(tmp_path):
    store = RetrievalCacheStore(tmp_path / "cache.sqlite3")
    key = cache_key()
    store.put(cached_result(key))
    conflicting = cached_result(
        key,
        result_id="chunk-2",
        created_at=11.0,
        expires_at=21.0,
    )
    with pytest.raises(RuntimeError, match="collision"):
        store.put(conflicting)
    assert store.invalidate_owner("alice") == 1
    assert store.put(conflicting) == conflicting


def test_generation_invalidation_is_owner_scoped(tmp_path):
    store = RetrievalCacheStore(tmp_path / "cache.sqlite3")
    old_alice = cached_result(cache_key(generation=2), result_id="alice-old")
    new_alice = cached_result(cache_key(generation=3), result_id="alice-new")
    old_bob = cached_result(cache_key(owner="bob", generation=2), result_id="bob-old")
    for value in (old_alice, new_alice, old_bob):
        store.put(value)
    assert store.invalidate_before_generation(
        owner_id="alice",
        minimum_generation_sequence=3,
    ) == 1
    assert store.get(old_alice.key, now=15.0) is None
    assert store.get(new_alice.key, now=15.0) == new_alice
    assert store.get(old_bob.key, now=15.0) == old_bob


def test_corrupt_payload_or_identity_fails_closed(tmp_path):
    path = tmp_path / "cache.sqlite3"
    store = RetrievalCacheStore(path)
    key = cache_key()
    store.put(cached_result(key))
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE retrieval_cache SET payload_json=? WHERE cache_key=?",
            ('{"hits":[],"created_at":10,"expires_at":20,"snippet":"private"}', key.cache_key),
        )
    with pytest.raises(RuntimeError, match="schema"):
        store.get(key, now=15.0)


def test_cache_models_reject_unbounded_duplicate_and_nonfinite_values():
    key = cache_key()
    hit = CachedRetrievalHit(
        result_id="chunk-1",
        source_id="doc-1",
        rank=1,
        score=0.5,
        content_digest=digest("chunk-1"),
    )
    with pytest.raises(ValueError, match="contiguous"):
        CachedRetrievalResult(
            key=key,
            hits=(
                hit,
                CachedRetrievalHit(
                    result_id="chunk-2",
                    source_id="doc-1",
                    rank=3,
                    score=0.4,
                    content_digest=digest("chunk-2"),
                ),
            ),
            created_at=1.0,
            expires_at=2.0,
        )
    with pytest.raises(ValueError, match="unique"):
        CachedRetrievalResult(
            key=key,
            hits=(
                hit,
                CachedRetrievalHit(
                    result_id="chunk-1",
                    source_id="doc-2",
                    rank=2,
                    score=0.4,
                    content_digest=digest("chunk-1-copy"),
                ),
            ),
            created_at=1.0,
            expires_at=2.0,
        )
    with pytest.raises(ValueError, match="finite"):
        CachedRetrievalHit(
            result_id="chunk-3",
            source_id="doc-1",
            rank=1,
            score=float("nan"),
            content_digest=digest("chunk-3"),
        )
    with pytest.raises(ValueError, match="invalid"):
        CachedRetrievalHit(
            result_id="bad\tcontrol",
            source_id="doc-1",
            rank=1,
            score=0.5,
            content_digest=digest("bad"),
        )
