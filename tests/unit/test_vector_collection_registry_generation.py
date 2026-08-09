from __future__ import annotations

import hashlib

import pytest

from tools.embedding_models import EmbeddingProfile
from tools.vector_collection_registry import VectorCollectionRegistry


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def profile(alias: str) -> EmbeddingProfile:
    return EmbeddingProfile(
        alias=alias,
        model_name=f"model/{alias}",
        dimensions=384,
        max_sequence_tokens=512,
    )


def test_generation_advance_keeps_collection_and_profile_but_updates_route_generation(tmp_path):
    registry = VectorCollectionRegistry(tmp_path / "routes.sqlite3")
    collection = registry.register_collection(profile("v1"), now=1.0)
    boot = registry.bootstrap_route(
        owner_id="alice",
        doc_id="doc",
        collection_id=collection.collection_id,
        generation_sequence=7,
        now=2.0,
    )
    advanced = registry.transition_route(
        owner_id="alice",
        doc_id="doc",
        expected_revision=boot.revision,
        expected_collection_id=collection.collection_id,
        expected_profile_fingerprint=collection.profile_fingerprint,
        expected_generation_sequence=7,
        target_collection_id=collection.collection_id,
        target_generation_sequence=8,
        operation_id=digest("generation-advance"),
        action="generation_advance",
        now=3.0,
    )
    assert advanced.revision == 2
    assert advanced.collection_id == collection.collection_id
    assert advanced.profile_fingerprint == collection.profile_fingerprint
    assert advanced.generation_sequence == 8
    assert advanced.previous_collection_id == collection.collection_id
    assert registry.current_route("alice", "doc") == advanced


def test_generation_advance_rejects_collection_change_and_switch_rejects_same_collection(tmp_path):
    registry = VectorCollectionRegistry(tmp_path / "routes.sqlite3")
    first = registry.register_collection(profile("v1"), now=1.0)
    second = registry.register_collection(profile("v2"), now=1.0)
    registry.bootstrap_route(
        owner_id="alice",
        doc_id="doc",
        collection_id=first.collection_id,
        generation_sequence=1,
        now=2.0,
    )
    with pytest.raises(ValueError, match="retain"):
        registry.transition_route(
            owner_id="alice",
            doc_id="doc",
            expected_revision=1,
            expected_collection_id=first.collection_id,
            expected_profile_fingerprint=first.profile_fingerprint,
            expected_generation_sequence=1,
            target_collection_id=second.collection_id,
            target_generation_sequence=2,
            operation_id=digest("bad-refresh"),
            action="generation_advance",
            now=3.0,
        )
    with pytest.raises(ValueError, match="different"):
        registry.transition_route(
            owner_id="alice",
            doc_id="doc",
            expected_revision=1,
            expected_collection_id=first.collection_id,
            expected_profile_fingerprint=first.profile_fingerprint,
            expected_generation_sequence=1,
            target_collection_id=first.collection_id,
            target_generation_sequence=2,
            operation_id=digest("bad-switch"),
            action="switch",
            now=3.0,
        )


def test_immediate_transition_retry_is_idempotent_but_superseded_retry_is_refused(tmp_path):
    registry = VectorCollectionRegistry(tmp_path / "routes.sqlite3")
    first = registry.register_collection(profile("v1"), now=1.0)
    second = registry.register_collection(profile("v2"), now=1.0)
    boot = registry.bootstrap_route(
        owner_id="alice",
        doc_id="doc",
        collection_id=first.collection_id,
        generation_sequence=1,
        now=2.0,
    )
    operation_id = digest("switch")
    switched = registry.transition_route(
        owner_id="alice",
        doc_id="doc",
        expected_revision=boot.revision,
        expected_collection_id=first.collection_id,
        expected_profile_fingerprint=first.profile_fingerprint,
        expected_generation_sequence=1,
        target_collection_id=second.collection_id,
        target_generation_sequence=2,
        operation_id=operation_id,
        action="switch",
        now=3.0,
    )
    retry = registry.transition_route(
        owner_id="alice",
        doc_id="doc",
        expected_revision=boot.revision,
        expected_collection_id=first.collection_id,
        expected_profile_fingerprint=first.profile_fingerprint,
        expected_generation_sequence=1,
        target_collection_id=second.collection_id,
        target_generation_sequence=2,
        operation_id=operation_id,
        action="switch",
        now=4.0,
    )
    assert retry == switched
    advanced = registry.transition_route(
        owner_id="alice",
        doc_id="doc",
        expected_revision=switched.revision,
        expected_collection_id=second.collection_id,
        expected_profile_fingerprint=second.profile_fingerprint,
        expected_generation_sequence=2,
        target_collection_id=second.collection_id,
        target_generation_sequence=3,
        operation_id=digest("advance"),
        action="generation_advance",
        now=5.0,
    )
    assert advanced.revision == 3
    with pytest.raises(RuntimeError, match="superseded"):
        registry.transition_route(
            owner_id="alice",
            doc_id="doc",
            expected_revision=boot.revision,
            expected_collection_id=first.collection_id,
            expected_profile_fingerprint=first.profile_fingerprint,
            expected_generation_sequence=1,
            target_collection_id=second.collection_id,
            target_generation_sequence=2,
            operation_id=operation_id,
            action="switch",
            now=6.0,
        )
