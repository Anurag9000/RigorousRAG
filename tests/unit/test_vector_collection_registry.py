from __future__ import annotations

import hashlib
import sqlite3

import pytest

from tools.embedding_models import EmbeddingProfile
from tools.vector_collection_registry import (
    VectorCollectionRegistry,
    VectorCollectionRouter,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def profile(alias: str, model: str, dimensions: int) -> EmbeddingProfile:
    return EmbeddingProfile(
        alias=alias,
        model_name=model,
        dimensions=dimensions,
        max_sequence_tokens=512,
        language="multilingual",
        domain="general",
    )


def test_blue_green_route_switch_and_append_only_rollback(tmp_path):
    path = tmp_path / "vector-routes.sqlite3"
    registry = VectorCollectionRegistry(path)
    source = registry.register_collection(
        profile("dense-v1", "model/v1", 384),
        now=1.0,
    )
    target = registry.register_collection(
        profile("dense-v2", "model/v2", 768),
        now=2.0,
    )
    assert source.collection_name.endswith("-d384")
    assert target.collection_name.endswith("-d768")
    assert source.collection_id != target.collection_id

    boot = registry.bootstrap_route(
        owner_id="alice",
        doc_id="doc-1",
        collection_id=source.collection_id,
        generation_sequence=7,
        now=3.0,
    )
    switched = registry.transition_route(
        owner_id="alice",
        doc_id="doc-1",
        expected_revision=boot.revision,
        expected_collection_id=source.collection_id,
        expected_profile_fingerprint=source.profile_fingerprint,
        expected_generation_sequence=7,
        target_collection_id=target.collection_id,
        target_generation_sequence=8,
        operation_id=digest("cutover-op"),
        action="switch",
        now=4.0,
    )
    assert switched.revision == 2
    assert switched.collection_id == target.collection_id
    assert switched.previous_collection_id == source.collection_id
    assert registry.current_route("alice", "doc-1") == switched

    rolled_back = registry.transition_route(
        owner_id="alice",
        doc_id="doc-1",
        expected_revision=switched.revision,
        expected_collection_id=target.collection_id,
        expected_profile_fingerprint=target.profile_fingerprint,
        expected_generation_sequence=8,
        target_collection_id=source.collection_id,
        target_generation_sequence=9,
        operation_id=digest("rollback-op"),
        action="rollback",
        now=5.0,
    )
    assert rolled_back.revision == 3
    assert rolled_back.collection_id == source.collection_id
    assert rolled_back.generation_sequence == 9
    assert rolled_back.previous_collection_id == target.collection_id
    assert [item.action for item in registry.route_history("alice", "doc-1")] == [
        "rollback",
        "switch",
        "bootstrap",
    ]

    registry.close()
    reopened = VectorCollectionRegistry(path)
    assert reopened.current_route("alice", "doc-1") == rolled_back
    assert reopened.get_collection(target.collection_id) == target


def test_stale_cas_and_nonmonotonic_generation_fail_closed(tmp_path):
    registry = VectorCollectionRegistry(tmp_path / "routes.sqlite3")
    source = registry.register_collection(profile("v1", "model/v1", 384), now=1.0)
    target = registry.register_collection(profile("v2", "model/v2", 768), now=1.0)
    boot = registry.bootstrap_route(
        owner_id="alice",
        doc_id="doc",
        collection_id=source.collection_id,
        generation_sequence=1,
        now=2.0,
    )
    switched = registry.transition_route(
        owner_id="alice",
        doc_id="doc",
        expected_revision=1,
        expected_collection_id=source.collection_id,
        expected_profile_fingerprint=source.profile_fingerprint,
        expected_generation_sequence=1,
        target_collection_id=target.collection_id,
        target_generation_sequence=2,
        operation_id=digest("op-1"),
        action="switch",
        now=3.0,
    )
    assert switched.revision == 2
    with pytest.raises(RuntimeError, match="compare-and-swap"):
        registry.transition_route(
            owner_id="alice",
            doc_id="doc",
            expected_revision=1,
            expected_collection_id=source.collection_id,
            expected_profile_fingerprint=source.profile_fingerprint,
            expected_generation_sequence=1,
            target_collection_id=target.collection_id,
            target_generation_sequence=2,
            operation_id=digest("stale-op"),
            action="switch",
            now=4.0,
        )
    with pytest.raises(ValueError, match="advance monotonically"):
        registry.transition_route(
            owner_id="alice",
            doc_id="doc",
            expected_revision=2,
            expected_collection_id=target.collection_id,
            expected_profile_fingerprint=target.profile_fingerprint,
            expected_generation_sequence=2,
            target_collection_id=source.collection_id,
            target_generation_sequence=2,
            operation_id=digest("bad-generation"),
            action="rollback",
            now=4.0,
        )


def test_collection_retirement_requires_exact_confirmation_and_no_current_routes(tmp_path):
    registry = VectorCollectionRegistry(tmp_path / "routes.sqlite3")
    source = registry.register_collection(profile("v1", "model/v1", 384), now=1.0)
    target = registry.register_collection(profile("v2", "model/v2", 768), now=1.0)
    registry.bootstrap_route(
        owner_id="alice",
        doc_id="doc",
        collection_id=source.collection_id,
        generation_sequence=1,
        now=2.0,
    )
    with pytest.raises(ValueError, match="confirmation"):
        registry.retire_collection(
            target.collection_id,
            confirm_collection_id=source.collection_id,
            now=3.0,
        )
    with pytest.raises(RuntimeError, match="still reference"):
        registry.retire_collection(
            source.collection_id,
            confirm_collection_id=source.collection_id,
            now=3.0,
        )
    retired = registry.retire_collection(
        target.collection_id,
        confirm_collection_id=target.collection_id,
        now=3.0,
    )
    assert retired.state == "retired"
    assert retired.retired_at == 3.0
    with pytest.raises(RuntimeError, match="not ready"):
        registry.transition_route(
            owner_id="alice",
            doc_id="doc",
            expected_revision=1,
            expected_collection_id=source.collection_id,
            expected_profile_fingerprint=source.profile_fingerprint,
            expected_generation_sequence=1,
            target_collection_id=target.collection_id,
            target_generation_sequence=2,
            operation_id=digest("retired-target"),
            action="switch",
            now=4.0,
        )


def test_owner_document_isolation_and_current_route_listing(tmp_path):
    registry = VectorCollectionRegistry(tmp_path / "routes.sqlite3")
    collection = registry.register_collection(profile("v1", "model/v1", 384), now=1.0)
    registry.bootstrap_route(
        owner_id="alice",
        doc_id="a",
        collection_id=collection.collection_id,
        generation_sequence=1,
        now=2.0,
    )
    registry.bootstrap_route(
        owner_id="alice",
        doc_id="b",
        collection_id=collection.collection_id,
        generation_sequence=2,
        now=2.0,
    )
    registry.bootstrap_route(
        owner_id="bob",
        doc_id="a",
        collection_id=collection.collection_id,
        generation_sequence=1,
        now=2.0,
    )
    assert [route.doc_id for route in registry.current_routes("alice")] == ["a", "b"]
    assert [route.doc_id for route in registry.current_routes("bob")] == ["a"]
    assert registry.current_route("alice", "a").owner_id == "alice"
    assert registry.current_route("bob", "a").owner_id == "bob"


def test_router_resolves_new_physical_collection_after_cutover(tmp_path):
    registry = VectorCollectionRegistry(tmp_path / "routes.sqlite3")
    source = registry.register_collection(profile("v1", "model/v1", 384), now=1.0)
    target = registry.register_collection(profile("v2", "model/v2", 768), now=1.0)
    boot = registry.bootstrap_route(
        owner_id="alice",
        doc_id="doc",
        collection_id=source.collection_id,
        generation_sequence=1,
        now=2.0,
    )
    created: list[str] = []

    class Layer:
        def __init__(self, collection_name):
            self.collection_name = collection_name

        def query(self, query_text, **kwargs):
            return {
                "collection": self.collection_name,
                "query": query_text,
                "owner": kwargs["owner_id"],
                "doc": kwargs["doc_id"],
            }

    def factory(collection):
        created.append(collection.collection_id)
        return Layer(collection.collection_name)

    router = VectorCollectionRouter(registry, factory)
    first = router.query_document("alpha", owner_id="alice", doc_id="doc")
    assert first["collection"] == source.collection_name
    assert created == [source.collection_id]
    registry.transition_route(
        owner_id="alice",
        doc_id="doc",
        expected_revision=boot.revision,
        expected_collection_id=source.collection_id,
        expected_profile_fingerprint=source.profile_fingerprint,
        expected_generation_sequence=1,
        target_collection_id=target.collection_id,
        target_generation_sequence=2,
        operation_id=digest("switch"),
        action="switch",
        now=3.0,
    )
    second = router.query_document("beta", owner_id="alice", doc_id="doc")
    assert second["collection"] == target.collection_name
    assert created == [source.collection_id, target.collection_id]
    router.query_document("gamma", owner_id="alice", doc_id="doc")
    assert created == [source.collection_id, target.collection_id]


def test_corrupt_collection_record_fails_closed(tmp_path):
    path = tmp_path / "routes.sqlite3"
    registry = VectorCollectionRegistry(path)
    collection = registry.register_collection(profile("v1", "model/v1", 384), now=1.0)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE vector_collections SET dimensions=385 WHERE collection_id=?",
            (collection.collection_id,),
        )
    with pytest.raises(RuntimeError, match="corrupt"):
        registry.get_collection(collection.collection_id)


def test_registry_requires_explicit_dimensions_and_safe_transition_controls(tmp_path):
    registry = VectorCollectionRegistry(tmp_path / "routes.sqlite3")
    with pytest.raises(ValueError, match="explicit dimensions"):
        registry.register_collection(
            EmbeddingProfile(
                alias="unknown-dim",
                model_name="model/unknown",
                dimensions=None,
                max_sequence_tokens=512,
            ),
            now=1.0,
        )
    source = registry.register_collection(profile("v1", "model/v1", 384), now=1.0)
    registry.bootstrap_route(
        owner_id="alice",
        doc_id="doc",
        collection_id=source.collection_id,
        generation_sequence=1,
        now=2.0,
    )
    with pytest.raises(ValueError, match="switch or rollback"):
        registry.transition_route(
            owner_id="alice",
            doc_id="doc",
            expected_revision=1,
            expected_collection_id=source.collection_id,
            expected_profile_fingerprint=source.profile_fingerprint,
            expected_generation_sequence=1,
            target_collection_id=source.collection_id,
            target_generation_sequence=2,
            operation_id=digest("bad-action"),
            action="delete",
            now=3.0,
        )
