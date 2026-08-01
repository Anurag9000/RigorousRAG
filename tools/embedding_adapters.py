"""Explicit bounded embedding encoders for governed migration execution."""

from __future__ import annotations

import itertools
import math
import threading
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol

from tools.embedding_models import EmbeddingProfile
from tools.embedding_registry import resolve_embedding_profile

_MAX_PASSAGES = 100_000
_MAX_PASSAGE_CHARS = 5_000_000
_MAX_DIMENSIONS = 1_000_000
_MAX_BATCH_SIZE = 4_096


class EmbeddingEncoder(Protocol):
    profile: EmbeddingProfile

    def encode_passages(
        self,
        passages: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]: ...


EncoderFactory = Callable[[EmbeddingProfile], EmbeddingEncoder]
_FACTORIES: dict[str, EncoderFactory] = {}
_FACTORY_LOCK = threading.RLock()


def _passages(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("passages must be a sequence of strings.")
    result: list[str] = []
    try:
        for value in itertools.islice(iter(values), _MAX_PASSAGES + 1):
            if len(result) >= _MAX_PASSAGES:
                raise ValueError("passages exceed the item limit.")
            if not isinstance(value, str):
                raise ValueError("every passage must be a string.")
            cleaned = value.strip()
            if (
                not cleaned
                or len(cleaned) > _MAX_PASSAGE_CHARS
                or "\x00" in cleaned
            ):
                raise ValueError("a passage is empty, invalid, or too long.")
            result.append(cleaned)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("passages are not safely iterable.") from exc
    if not result:
        raise ValueError("at least one passage is required.")
    return tuple(result)


def _vector(value: Any, *, dimensions: int | None) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError("embedding vectors must be numeric sequences.")
    try:
        raw = list(itertools.islice(iter(value), _MAX_DIMENSIONS + 1))
    except Exception as exc:
        raise ValueError("embedding vector is not safely iterable.") from exc
    if not raw or len(raw) > _MAX_DIMENSIONS:
        raise ValueError("embedding vector is empty or exceeds the dimension limit.")
    if dimensions is not None and len(raw) != dimensions:
        raise ValueError("embedding vector dimensions do not match the profile.")
    result: list[float] = []
    for item in raw:
        if isinstance(item, bool):
            raise ValueError("embedding vector values must be finite numbers.")
        try:
            numeric = float(item)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("embedding vector values must be finite numbers.") from exc
        if not math.isfinite(numeric):
            raise ValueError("embedding vector values must be finite numbers.")
        result.append(numeric)
    return tuple(result)


class SentenceTransformerEncoder:
    """Default adapter for profiles supported by plain SentenceTransformer encode."""

    def __init__(
        self,
        profile: EmbeddingProfile,
        *,
        model: Any = None,
        batch_size: int = 32,
    ) -> None:
        if not isinstance(profile, EmbeddingProfile):
            raise ValueError("profile must be an EmbeddingProfile.")
        if profile.requires_adapter:
            raise ValueError(
                "the selected embedding profile requires an explicit adapter."
            )
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError("batch_size must be an integer.")
        if not 1 <= batch_size <= _MAX_BATCH_SIZE:
            raise ValueError(
                f"batch_size must be between 1 and {_MAX_BATCH_SIZE}."
            )
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:
                raise RuntimeError(
                    "sentence-transformers is unavailable for this profile."
                ) from exc
            try:
                model = SentenceTransformer(profile.model_name)
            except Exception as exc:
                raise RuntimeError("embedding model initialization failed.") from exc
        encode = getattr(model, "encode", None)
        if not callable(encode):
            raise ValueError("embedding model must expose encode().")
        self.profile = profile
        self._model = model
        self._batch_size = batch_size

    def encode_passages(
        self,
        passages: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        bounded = _passages(passages)
        formatted = [self.profile.format_passage(value) for value in bounded]
        try:
            output = self._model.encode(
                formatted,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.profile.normalize_embeddings,
            )
        except Exception as exc:
            raise RuntimeError("embedding model execution failed.") from exc
        if isinstance(output, (str, bytes, bytearray)):
            raise RuntimeError("embedding model returned an invalid result.")
        try:
            rows = list(itertools.islice(iter(output), len(formatted) + 1))
        except Exception as exc:
            raise RuntimeError("embedding model returned an invalid result.") from exc
        if len(rows) != len(formatted):
            raise RuntimeError("embedding model returned the wrong row count.")
        vectors = tuple(
            _vector(row, dimensions=self.profile.dimensions) for row in rows
        )
        if self.profile.dimensions is None:
            dimensions = len(vectors[0])
            if any(len(vector) != dimensions for vector in vectors):
                raise RuntimeError(
                    "embedding model returned inconsistent vector dimensions."
                )
        return vectors


def register_embedding_adapter(
    profile_alias: str,
    factory: EncoderFactory,
    *,
    replace: bool = False,
) -> None:
    profile = resolve_embedding_profile(
        profile_alias,
        allow_compatibility=False,
    )
    if not callable(factory):
        raise ValueError("factory must be callable.")
    if not isinstance(replace, bool):
        raise ValueError("replace must be a boolean.")
    with _FACTORY_LOCK:
        if profile.alias in _FACTORIES and not replace:
            raise ValueError("an adapter is already registered for this profile.")
        _FACTORIES[profile.alias] = factory


def unregister_embedding_adapter(profile_alias: str) -> bool:
    profile = resolve_embedding_profile(
        profile_alias,
        allow_compatibility=False,
    )
    with _FACTORY_LOCK:
        return _FACTORIES.pop(profile.alias, None) is not None


def clear_embedding_adapters() -> None:
    with _FACTORY_LOCK:
        _FACTORIES.clear()


def create_embedding_encoder(
    profile: EmbeddingProfile | str,
) -> EmbeddingEncoder:
    selected = (
        profile
        if isinstance(profile, EmbeddingProfile)
        else resolve_embedding_profile(profile, allow_compatibility=False)
    )
    if not isinstance(selected, EmbeddingProfile):
        raise ValueError("profile must resolve to an EmbeddingProfile.")
    with _FACTORY_LOCK:
        factory = _FACTORIES.get(selected.alias)
    if factory is None:
        if selected.requires_adapter:
            raise RuntimeError(
                "the selected profile requires an explicitly registered adapter."
            )
        return SentenceTransformerEncoder(selected)
    encoder = factory(selected)
    if getattr(encoder, "profile", None) != selected:
        raise RuntimeError("embedding adapter returned an incompatible profile.")
    if not callable(getattr(encoder, "encode_passages", None)):
        raise RuntimeError("embedding adapter does not expose encode_passages().")
    return encoder


__all__ = [
    "EmbeddingEncoder",
    "SentenceTransformerEncoder",
    "clear_embedding_adapters",
    "create_embedding_encoder",
    "register_embedding_adapter",
    "unregister_embedding_adapter",
]
