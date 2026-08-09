import hashlib

import pytest

from tools.embedding_models import EmbeddingProfile
from tools.model_artifacts import (
    GovernedLateInteractionAdapter,
    GovernedMultilingualDenseEncoder,
    GovernedSparseExpansionAdapter,
    ModelArtifactRegistry,
    ModelArtifactSpec,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def spec(kind: str, *, dimensions=None, model="org/model", revision="refs/tags/v1.2.3") -> ModelArtifactSpec:
    return ModelArtifactSpec(
        kind=kind,
        model_id=model,
        revision=revision,
        config_sha256=digest("config"),
        weights_sha256=digest("weights"),
        tokenizer_sha256=digest("tokenizer"),
        output_dimensions=dimensions,
        languages=("en", "hi"),
        license_id="apache-2.0",
    )


def test_artifact_fingerprint_binds_revision_weights_config_shape_and_languages():
    first = spec("colbert", dimensions=128)
    assert first.artifact_fingerprint == spec("colbert", dimensions=128).artifact_fingerprint
    changed_revision = ModelArtifactSpec(**{**first.__dict__, "revision": "refs/tags/v1.2.4"})
    changed_weights = ModelArtifactSpec(**{**first.__dict__, "weights_sha256": digest("other-weights")})
    changed_shape = ModelArtifactSpec(**{**first.__dict__, "output_dimensions": 256})
    assert len({
        first.artifact_fingerprint,
        changed_revision.artifact_fingerprint,
        changed_weights.artifact_fingerprint,
        changed_shape.artifact_fingerprint,
    }) == 4


def test_floating_revisions_are_rejected_but_commit_tag_digest_and_version_are_allowed():
    for revision in ("main", "master", "latest", "HEAD", "trunk", "refs/heads/release"):
        with pytest.raises(ValueError, match="revision"):
            spec("splade", revision=revision)
    assert spec("splade", revision="refs/tags/v2.0.1").revision == "refs/tags/v2.0.1"
    assert spec("splade", revision="abcdef1234567890").revision == "abcdef1234567890"
    assert spec("splade", revision=f"sha256:{digest('artifact')}").revision.startswith("sha256:")
    assert spec("splade", revision="v1").revision == "v1"
    assert spec("splade", revision="release-2026.08").revision == "release-2026.08"


def test_registry_is_idempotent_and_kind_checked():
    registry = ModelArtifactRegistry()
    item = spec("splade")
    fingerprint = registry.register(item)
    assert registry.register(item) == fingerprint
    assert registry.require(fingerprint, kind="splade") == item
    with pytest.raises(RuntimeError, match="kind"):
        registry.require(fingerprint, kind="colbert")
    assert registry.list(kind="splade") == (item,)


def test_governed_splade_adapter_bounds_terms_and_hides_private_model_errors():
    item = spec("splade")
    adapter = GovernedSparseExpansionAdapter(item, lambda text: {text: 1.5, "zero": 0.0})
    assert adapter.query_weights("term") == {"term": 1.5}
    assert adapter.artifact_fingerprint == item.artifact_fingerprint
    broken = GovernedSparseExpansionAdapter(
        item,
        lambda _text: (_ for _ in ()).throw(RuntimeError("secret provider detail")),
    )
    with pytest.raises(RuntimeError, match="sparse model execution failed") as captured:
        broken.document_weights("document")
    assert "secret provider detail" not in str(captured.value)


def test_governed_colbert_adapter_enforces_declared_dimensions_and_finite_vectors():
    item = spec("colbert", dimensions=3)
    adapter = GovernedLateInteractionAdapter(
        item,
        lambda _text: ((1.0, 0.5, 0.25), (0.2, 0.3, 0.4)),
    )
    assert len(adapter.query_vectors("query")) == 2
    bad = GovernedLateInteractionAdapter(item, lambda _text: ((1.0, 2.0),))
    with pytest.raises(RuntimeError, match="dimensions"):
        bad.document_vectors("document")


def test_multilingual_dense_encoder_is_profile_and_artifact_bound():
    profile = EmbeddingProfile(
        alias="multi-v1",
        model_name="org/model",
        dimensions=3,
        max_sequence_tokens=512,
        language="multilingual",
        modes=("dense",),
        requires_adapter=True,
    )
    item = spec("multilingual_dense", dimensions=3)
    captured = []

    def infer(passages):
        captured.extend(passages)
        return [(1.0, 0.5, 0.25) for _ in passages]

    encoder = GovernedMultilingualDenseEncoder(profile, item, infer)
    assert encoder.encode_passages(("hello", "नमस्ते")) == (
        (1.0, 0.5, 0.25),
        (1.0, 0.5, 0.25),
    )
    assert captured == ["hello", "नमस्ते"]
    assert encoder.artifact_fingerprint == item.artifact_fingerprint
    wrong_model = spec("multilingual_dense", dimensions=3, model="org/other")
    with pytest.raises(ValueError, match="model_id"):
        GovernedMultilingualDenseEncoder(profile, wrong_model, infer)


def test_artifact_spec_rejects_malformed_digests_and_zero_vectors():
    with pytest.raises(ValueError, match="SHA-256"):
        ModelArtifactSpec(
            kind="splade",
            model_id="org/model",
            revision="refs/tags/v1.0.0",
            config_sha256="not-a-digest",
            weights_sha256=digest("weights"),
            tokenizer_sha256=digest("tokenizer"),
        )
    adapter = GovernedLateInteractionAdapter(
        spec("colbert", dimensions=3),
        lambda _text: ((0.0, 0.0, 0.0),),
    )
    with pytest.raises(RuntimeError, match="all zero"):
        adapter.query_vectors("query")
