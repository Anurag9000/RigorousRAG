from __future__ import annotations

from tools.migration_compatibility import (
    EmbeddingNeighborhoodCase,
    RetrievalCacheEntry,
    RetrievalCacheKey,
    cache_entry_is_current,
    evaluate_embedding_space_compatibility,
)

D = "a" * 64
P = "b" * 64
C = "c" * 64
R = "d" * 64


def test_embedding_space_compatibility_is_dimension_agnostic_but_neighborhood_guarded():
    report = evaluate_embedding_space_compatibility(
        [
            EmbeddingNeighborhoodCase("q1", ("a", "b", "c"), ("a", "c", "b")),
            EmbeddingNeighborhoodCase("q2", ("x", "y", "z"), ("x", "y", "z")),
        ],
        current_dimensions=384,
        shadow_dimensions=768,
        top_k=3,
        minimum_overlap=0.9,
        maximum_rank_displacement=1.0,
    )
    assert report.dimension_changed is True
    assert report.mean_overlap_at_k == 1.0
    assert report.mean_rank_displacement <= 1.0
    assert report.compatible is True
    assert len(report.report_digest) == 64


def test_embedding_space_compatibility_rejects_neighborhood_collapse():
    report = evaluate_embedding_space_compatibility(
        [EmbeddingNeighborhoodCase("q", ("a", "b", "c"), ("x", "y", "z"))],
        current_dimensions=384,
        shadow_dimensions=384,
        top_k=3,
        minimum_overlap=0.5,
        maximum_rank_displacement=2.0,
    )
    assert report.mean_overlap_at_k == 0.0
    assert report.mean_rank_displacement == 3.0
    assert report.compatible is False


def test_cache_key_binds_owner_generation_profile_and_retrieval_configuration():
    key = RetrievalCacheKey(
        owner_id="alice",
        query_digest=D,
        generation_sequence=4,
        profile_fingerprint=P,
        retrieval_config_digest=C,
    )
    entry = RetrievalCacheEntry(key=key, result_digest=R)
    assert len(key.cache_key) == 64
    assert cache_entry_is_current(
        entry,
        owner_id="alice",
        generation_sequence=4,
        profile_fingerprint=P,
        retrieval_config_digest=C,
    )
    assert not cache_entry_is_current(
        entry,
        owner_id="alice",
        generation_sequence=5,
        profile_fingerprint=P,
        retrieval_config_digest=C,
    )
    assert not cache_entry_is_current(
        entry,
        owner_id="alice",
        generation_sequence=4,
        profile_fingerprint="9" * 64,
        retrieval_config_digest=C,
    )
    assert not cache_entry_is_current(
        entry,
        owner_id="bob",
        generation_sequence=4,
        profile_fingerprint=P,
        retrieval_config_digest=C,
    )
