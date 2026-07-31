import json

import pytest

from tools.embedding_registry import (
    BUILTIN_PROFILES,
    EmbeddingProfile,
    EmbeddingProfileRegistry,
    compatibility_profile,
    load_operator_profiles,
    resolve_embedding_profile,
)


def operator_definition(**overrides):
    value = {
        "model_name": "lab/biomedical-retriever",
        "dimensions": 768,
        "max_sequence_tokens": 512,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "normalize_embeddings": True,
        "language": "English",
        "domain": "biomedical",
        "modes": ["dense"],
        "license": "internal",
        "source_url": "",
        "notes": "",
        "schema_version": 1,
        "requires_adapter": False,
    }
    value.update(overrides)
    return value


def test_builtin_profiles_have_stable_unique_fingerprints_and_expected_contracts():
    assert set(BUILTIN_PROFILES) == {
        "minilm-l6-v2",
        "e5-base-v2",
        "bge-base-en-v1.5",
        "gte-base",
        "instructor-base",
        "specter2",
        "bge-m3",
    }
    fingerprints = {profile.fingerprint for profile in BUILTIN_PROFILES.values()}
    assert len(fingerprints) == len(BUILTIN_PROFILES)
    assert all(len(value) == 64 for value in fingerprints)
    assert BUILTIN_PROFILES["e5-base-v2"].format_query("question") == "query: question"
    assert BUILTIN_PROFILES["e5-base-v2"].format_passage("evidence") == "passage: evidence"
    assert BUILTIN_PROFILES["bge-m3"].modes == ("dense", "sparse", "multi-vector")


def test_fingerprint_changes_for_every_index_compatibility_dimension():
    base = EmbeddingProfile(
        alias="custom",
        model_name="lab/model",
        dimensions=768,
        max_sequence_tokens=512,
    )
    changed = EmbeddingProfile(
        alias="custom",
        model_name="lab/model",
        dimensions=1024,
        max_sequence_tokens=512,
    )
    instruction_changed = EmbeddingProfile(
        alias="custom",
        model_name="lab/model",
        dimensions=768,
        max_sequence_tokens=512,
        query_prefix="query: ",
    )
    assert len({base.fingerprint, changed.fingerprint, instruction_changed.fingerprint}) == 3


def test_registry_resolves_alias_model_name_and_historical_default():
    registry = EmbeddingProfileRegistry()
    expected = registry.get("minilm-l6-v2")
    assert registry.resolve("all-MiniLM-L6-v2") == expected
    assert registry.resolve("sentence-transformers/all-MiniLM-L6-v2") == expected
    assert resolve_embedding_profile("minilm-l6-v2", registry=registry) == expected


def test_unknown_models_get_explicit_incomplete_compatibility_profiles():
    profile = compatibility_profile("lab/unknown-model")
    assert profile.model_name == "lab/unknown-model"
    assert profile.dimensions is None
    assert profile.max_sequence_tokens is None
    assert profile.license == "operator-review-required"
    assert profile.alias.startswith("compat-")
    registry = EmbeddingProfileRegistry()
    assert registry.resolve("lab/unknown-model").fingerprint == profile.fingerprint
    with pytest.raises(KeyError):
        registry.resolve("lab/unknown-model", allow_compatibility=False)


def test_operator_profiles_are_strict_and_can_override_builtin_aliases():
    raw = json.dumps({"minilm-l6-v2": operator_definition(model_name="lab/replacement")})
    registry = EmbeddingProfileRegistry(operator_json=raw)
    assert registry.get("minilm-l6-v2").model_name == "lab/replacement"
    assert registry.get("minilm-l6-v2").fingerprint != BUILTIN_PROFILES["minilm-l6-v2"].fingerprint


def test_operator_json_rejects_duplicate_unknown_missing_and_nonstandard_values():
    duplicate = '{"lab":{"model_name":"a/b","model_name":"c/d"}}'
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        load_operator_profiles(duplicate)

    unknown = operator_definition(extra="bad")
    with pytest.raises(ValueError, match="unknown fields"):
        load_operator_profiles(json.dumps({"lab": unknown}))

    missing = operator_definition()
    del missing["dimensions"]
    with pytest.raises(ValueError, match="missing fields"):
        load_operator_profiles(json.dumps({"lab": missing}))

    with pytest.raises(ValueError, match="Non-standard JSON"):
        load_operator_profiles('{"lab":{"dimensions":NaN}}')


def test_profile_validation_rejects_fractional_numeric_boolean_duplicate_modes_and_padding():
    for invalid in (
        operator_definition(dimensions=1.5),
        operator_definition(dimensions=True),
        operator_definition(max_sequence_tokens=0),
        operator_definition(modes=["dense", "dense"]),
        operator_definition(modes=["unknown"]),
    ):
        with pytest.raises(ValueError):
            load_operator_profiles(json.dumps({"lab": invalid}))
    with pytest.raises(ValueError, match="whitespace"):
        load_operator_profiles(json.dumps({" lab": operator_definition()}))
    with pytest.raises(ValueError, match="whitespace"):
        load_operator_profiles(json.dumps({"lab": operator_definition(model_name=" lab/model")}))


def test_operator_profile_is_resolvable_by_its_model_name():
    registry = EmbeddingProfileRegistry(
        operator_json=json.dumps({"lab-biomedical": operator_definition()})
    )
    assert registry.resolve("lab/biomedical-retriever").alias == "lab-biomedical"
