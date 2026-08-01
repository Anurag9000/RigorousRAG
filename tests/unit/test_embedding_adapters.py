from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.embedding_adapters import (
    SentenceTransformerEncoder,
    clear_embedding_adapters,
    create_embedding_encoder,
    register_embedding_adapter,
    unregister_embedding_adapter,
)
from tools.embedding_registry import resolve_embedding_profile


class FakeModel:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def encode(self, passages, **kwargs):
        self.calls.append((list(passages), kwargs))
        return self.rows


class FakeEncoder:
    def __init__(self, profile):
        self.profile = profile

    def encode_passages(self, passages):
        return tuple((0.0,) * (self.profile.dimensions or 2) for _ in passages)


def setup_function():
    clear_embedding_adapters()


def teardown_function():
    clear_embedding_adapters()


def test_sentence_transformer_adapter_applies_profile_passage_contract():
    profile = resolve_embedding_profile("e5-base-v2", allow_compatibility=False)
    model = FakeModel([[0.0] * profile.dimensions, [1.0] * profile.dimensions])
    encoder = SentenceTransformerEncoder(profile, model=model, batch_size=2)
    vectors = encoder.encode_passages(("first", "second"))
    assert len(vectors) == 2
    assert len(vectors[0]) == profile.dimensions
    passages, kwargs = model.calls[0]
    assert passages == ["passage: first", "passage: second"]
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["convert_to_numpy"] is True
    assert kwargs["show_progress_bar"] is False
    assert kwargs["batch_size"] == 2


def test_default_adapter_rejects_wrong_rows_dimensions_and_nonfinite_values():
    profile = resolve_embedding_profile("minilm-l6-v2", allow_compatibility=False)
    with pytest.raises(RuntimeError, match="wrong row count"):
        SentenceTransformerEncoder(profile, model=FakeModel([])).encode_passages(("one",))
    with pytest.raises(ValueError, match="dimensions"):
        SentenceTransformerEncoder(
            profile,
            model=FakeModel([[0.0, 1.0]]),
        ).encode_passages(("one",))
    with pytest.raises(ValueError, match="finite"):
        SentenceTransformerEncoder(
            profile,
            model=FakeModel([[float("nan")] * profile.dimensions]),
        ).encode_passages(("one",))


def test_adapter_required_profile_fails_until_explicit_registration():
    profile = resolve_embedding_profile("bge-m3", allow_compatibility=False)
    with pytest.raises(RuntimeError, match="explicitly registered"):
        create_embedding_encoder(profile)
    register_embedding_adapter(profile.alias, lambda selected: FakeEncoder(selected))
    encoder = create_embedding_encoder(profile)
    assert encoder.profile == profile
    assert len(encoder.encode_passages(("one",))[0]) == profile.dimensions
    assert unregister_embedding_adapter(profile.alias) is True
    assert unregister_embedding_adapter(profile.alias) is False


def test_duplicate_registration_and_incompatible_factory_fail_closed():
    profile = resolve_embedding_profile("bge-m3", allow_compatibility=False)
    register_embedding_adapter(profile.alias, lambda selected: FakeEncoder(selected))
    with pytest.raises(ValueError, match="already registered"):
        register_embedding_adapter(profile.alias, lambda selected: FakeEncoder(selected))
    register_embedding_adapter(
        profile.alias,
        lambda selected: SimpleNamespace(
            profile=resolve_embedding_profile("minilm-l6-v2", allow_compatibility=False),
            encode_passages=lambda passages: (),
        ),
        replace=True,
    )
    with pytest.raises(RuntimeError, match="incompatible profile"):
        create_embedding_encoder(profile)


def test_passage_and_batch_bounds_reject_boolean_and_hostile_inputs():
    profile = resolve_embedding_profile("minilm-l6-v2", allow_compatibility=False)
    with pytest.raises(ValueError, match="batch_size"):
        SentenceTransformerEncoder(profile, model=FakeModel([]), batch_size=True)
    encoder = SentenceTransformerEncoder(
        profile,
        model=FakeModel([[0.0] * profile.dimensions]),
    )
    with pytest.raises(ValueError, match="sequence"):
        encoder.encode_passages("not-a-sequence")

    class Hostile:
        def __iter__(self):
            yield "one"
            raise RuntimeError("private detail")

    with pytest.raises(ValueError, match="safely iterable"):
        encoder.encode_passages(Hostile())
