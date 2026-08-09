"""Immutable model-artifact governance and injected retrieval adapters.

This module never downloads or initializes a model. Production callers provide an
inference callable already bound to the exact artifact described by
``ModelArtifactSpec``. The spec fingerprints model identity, non-floating revision,
configuration, weights, tokenizer, output shape, language declaration and license so
an alias cannot silently drift between indexing, evaluation and serving.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import operator
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from tools.embedding_models import EmbeddingProfile

_MODEL_KINDS = {
    "dense",
    "multilingual_dense",
    "splade",
    "colbert",
    "multivector",
    "reranker",
    "image_text",
    "table_chart",
}
_FLOATING_REVISIONS = {"main", "master", "latest", "head", "trunk", "default"}
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")
_MAX_TERMS = 100_000
_MAX_VECTORS = 512
_MAX_DIMENSIONS = 16_384
_MAX_BATCH = 10_000
_MAX_TEXT = 5_000_000


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a bounded string.")
    selected = value.strip()
    if (
        not selected
        or len(selected) > maximum
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected)
    ):
        raise ValueError(f"{label} is invalid.")
    return selected


def _immutable_revision(value: Any) -> str:
    revision = _identifier(value, "revision", 200)
    lowered = revision.lower()
    if lowered in _FLOATING_REVISIONS:
        raise ValueError("revision must identify a non-floating model artifact.")
    if lowered.startswith("refs/heads/"):
        raise ValueError("revision may not be a mutable branch reference.")
    if lowered.startswith("refs/tags/"):
        tag = revision[len("refs/tags/") :].strip()
        if not tag:
            raise ValueError("revision tag reference is invalid.")
        return revision
    if lowered.startswith("sha256:"):
        selected = lowered[len("sha256:") :]
        if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
            raise ValueError("sha256 model revision is invalid.")
        return revision
    if _COMMIT_RE.fullmatch(revision):
        return revision.lower()
    # A version/release label is descriptive only; exact byte-level digests below are
    # the immutable anchor. Requiring at least one digit excludes vague aliases such
    # as "dev" or "stable" while accepting common labels such as v1 or release-2026.
    if any(character.isdigit() for character in revision):
        return revision
    raise ValueError("revision must be a commit, tag, digest, or explicit versioned release.")


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return selected


def _exact_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        selected = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= selected <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return selected


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite.")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite.") from exc
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite.")
    return selected


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("model input must be a string.")
    selected = value.strip()
    if not selected or len(selected) > _MAX_TEXT or "\x00" in selected:
        raise ValueError("model input is empty, invalid, or too long.")
    return selected


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ModelArtifactSpec:
    kind: str
    model_id: str
    revision: str
    config_sha256: str
    weights_sha256: str
    tokenizer_sha256: str
    output_dimensions: int | None = None
    languages: tuple[str, ...] = ("und",)
    license_id: str = "unknown"

    def __post_init__(self) -> None:
        kind = _identifier(self.kind, "kind", 64).lower()
        if kind not in _MODEL_KINDS:
            raise ValueError("model artifact kind is unsupported.")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "model_id", _identifier(self.model_id, "model_id", 500))
        object.__setattr__(self, "revision", _immutable_revision(self.revision))
        for name in ("config_sha256", "weights_sha256", "tokenizer_sha256"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.output_dimensions is not None:
            object.__setattr__(
                self,
                "output_dimensions",
                _exact_int(self.output_dimensions, "output_dimensions", 1, _MAX_DIMENSIONS),
            )
        if isinstance(self.languages, (str, bytes, bytearray)):
            raise ValueError("languages must be a bounded sequence.")
        languages = tuple(
            sorted({_identifier(value, "language", 32).lower() for value in self.languages})
        )
        if not languages or len(languages) > 128:
            raise ValueError("languages must contain between 1 and 128 entries.")
        object.__setattr__(self, "languages", languages)
        object.__setattr__(self, "license_id", _identifier(self.license_id, "license_id", 200))

    @property
    def artifact_fingerprint(self) -> str:
        return _canonical_digest({"contract": "rigorousrag-model-artifact-v1", **asdict(self)})


class ModelArtifactRegistry:
    """Thread-safe immutable-by-fingerprint registry for verified artifacts."""

    def __init__(self) -> None:
        self._items: dict[str, ModelArtifactSpec] = {}
        self._lock = threading.RLock()

    def register(self, spec: ModelArtifactSpec) -> str:
        if not isinstance(spec, ModelArtifactSpec):
            raise ValueError("spec must be ModelArtifactSpec.")
        fingerprint = spec.artifact_fingerprint
        with self._lock:
            existing = self._items.get(fingerprint)
            if existing is not None and existing != spec:
                raise RuntimeError("model artifact fingerprint collision.")
            self._items[fingerprint] = spec
        return fingerprint

    def get(self, fingerprint: str) -> ModelArtifactSpec | None:
        selected = _digest(fingerprint, "fingerprint")
        with self._lock:
            return self._items.get(selected)

    def require(self, fingerprint: str, *, kind: str | None = None) -> ModelArtifactSpec:
        selected = self.get(fingerprint)
        if selected is None:
            raise KeyError(_digest(fingerprint, "fingerprint"))
        if kind is not None and selected.kind != _identifier(kind, "kind", 64).lower():
            raise RuntimeError("model artifact kind does not match the required adapter.")
        return selected

    def list(self, *, kind: str | None = None) -> tuple[ModelArtifactSpec, ...]:
        selected_kind = None if kind is None else _identifier(kind, "kind", 64).lower()
        with self._lock:
            values = tuple(self._items.values())
        return tuple(
            sorted(
                (item for item in values if selected_kind is None or item.kind == selected_kind),
                key=lambda item: item.artifact_fingerprint,
            )
        )


def _sparse_output(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or len(value) > _MAX_TERMS:
        raise RuntimeError("sparse model returned an invalid bounded mapping.")
    result: dict[str, float] = {}
    for term, raw in value.items():
        if not isinstance(term, str) or not term or len(term) > 500:
            raise RuntimeError("sparse model returned an invalid term.")
        try:
            score = _finite(raw, "sparse model weight")
        except ValueError as exc:
            raise RuntimeError("sparse model returned an invalid weight.") from exc
        if not 0.0 <= score <= 1_000_000.0:
            raise RuntimeError("sparse model weight is outside the bounded range.")
        if score > 0.0:
            result[term] = score
    return result


def _vector(value: Any, *, dimensions: int | None) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise RuntimeError("model returned an invalid vector.")
    try:
        raw = list(itertools.islice(iter(value), _MAX_DIMENSIONS + 1))
    except Exception as exc:
        raise RuntimeError("model returned an invalid vector.") from exc
    if not raw or len(raw) > _MAX_DIMENSIONS:
        raise RuntimeError("model vector is empty or exceeds the dimension limit.")
    if dimensions is not None and len(raw) != dimensions:
        raise RuntimeError("model vector dimensions do not match the governed artifact.")
    try:
        result = tuple(_finite(item, "model vector value") for item in raw)
    except ValueError as exc:
        raise RuntimeError("model returned a non-finite vector value.") from exc
    if not any(item != 0.0 for item in result):
        raise RuntimeError("model vector may not be all zero.")
    return result


def _vectors(value: Any, *, dimensions: int | None) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise RuntimeError("model returned an invalid vector sequence.")
    try:
        raw = list(itertools.islice(iter(value), _MAX_VECTORS + 1))
    except Exception as exc:
        raise RuntimeError("model returned an invalid vector sequence.") from exc
    if not raw or len(raw) > _MAX_VECTORS:
        raise RuntimeError("model vector sequence is empty or exceeds the item limit.")
    result = tuple(_vector(item, dimensions=dimensions) for item in raw)
    width = len(result[0])
    if any(len(item) != width for item in result):
        raise RuntimeError("model vectors have inconsistent dimensions.")
    return result


class GovernedSparseExpansionAdapter:
    def __init__(self, spec: ModelArtifactSpec, infer: Callable[[str], Mapping[str, Any]]) -> None:
        if not isinstance(spec, ModelArtifactSpec) or spec.kind != "splade":
            raise ValueError("SPLADE adapter requires a splade ModelArtifactSpec.")
        if not callable(infer):
            raise ValueError("infer must be callable.")
        self.spec = spec
        self.artifact_fingerprint = spec.artifact_fingerprint
        self._infer = infer

    def _weights(self, text: str) -> Mapping[str, float]:
        bounded = _text(text)
        try:
            value = self._infer(bounded)
        except Exception as exc:
            raise RuntimeError("sparse model execution failed.") from exc
        return _sparse_output(value)

    def query_weights(self, query: str) -> Mapping[str, float]:
        return self._weights(query)

    def document_weights(self, text: str) -> Mapping[str, float]:
        return self._weights(text)


class GovernedLateInteractionAdapter:
    def __init__(self, spec: ModelArtifactSpec, infer: Callable[[str], Any]) -> None:
        if not isinstance(spec, ModelArtifactSpec) or spec.kind != "colbert":
            raise ValueError("late-interaction adapter requires a colbert ModelArtifactSpec.")
        if spec.output_dimensions is None:
            raise ValueError("ColBERT artifact must declare output_dimensions.")
        if not callable(infer):
            raise ValueError("infer must be callable.")
        self.spec = spec
        self.artifact_fingerprint = spec.artifact_fingerprint
        self._infer = infer

    def _encode(self, text: str) -> Sequence[Sequence[float]]:
        bounded = _text(text)
        try:
            value = self._infer(bounded)
        except Exception as exc:
            raise RuntimeError("late-interaction model execution failed.") from exc
        return _vectors(value, dimensions=self.spec.output_dimensions)

    def query_vectors(self, query: str) -> Sequence[Sequence[float]]:
        return self._encode(query)

    def document_vectors(self, text: str) -> Sequence[Sequence[float]]:
        return self._encode(text)


class GovernedMultilingualDenseEncoder:
    """Injected dense encoder compatible with ``EmbeddingEncoder`` without downloads."""

    def __init__(
        self,
        profile: EmbeddingProfile,
        spec: ModelArtifactSpec,
        infer: Callable[[Sequence[str]], Any],
    ) -> None:
        if not isinstance(profile, EmbeddingProfile):
            raise ValueError("profile must be EmbeddingProfile.")
        if not isinstance(spec, ModelArtifactSpec) or spec.kind != "multilingual_dense":
            raise ValueError("dense adapter requires a multilingual_dense ModelArtifactSpec.")
        if spec.model_id != profile.model_name:
            raise ValueError("artifact model_id must exactly match the embedding profile model_name.")
        if (
            profile.dimensions is not None
            and spec.output_dimensions is not None
            and profile.dimensions != spec.output_dimensions
        ):
            raise ValueError("artifact dimensions do not match the embedding profile.")
        if not callable(infer):
            raise ValueError("infer must be callable.")
        self.profile = profile
        self.spec = spec
        self.artifact_fingerprint = spec.artifact_fingerprint
        self._infer = infer

    def encode_passages(self, passages: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if isinstance(passages, (str, bytes, bytearray)):
            raise ValueError("passages must be a bounded sequence of strings.")
        try:
            selected = tuple(itertools.islice(iter(passages), _MAX_BATCH + 1))
        except Exception as exc:
            raise ValueError("passages must be a bounded sequence of strings.") from exc
        if not selected or len(selected) > _MAX_BATCH:
            raise ValueError("passages are empty or exceed the batch limit.")
        bounded = tuple(_text(item) for item in selected)
        formatted = tuple(self.profile.format_passage(item) for item in bounded)
        try:
            raw = list(itertools.islice(iter(self._infer(formatted)), len(formatted) + 1))
        except Exception as exc:
            raise RuntimeError("multilingual dense model execution failed.") from exc
        if len(raw) != len(formatted):
            raise RuntimeError("multilingual dense model returned the wrong row count.")
        dimensions = self.profile.dimensions or self.spec.output_dimensions
        rows = tuple(_vector(item, dimensions=dimensions) for item in raw)
        width = len(rows[0])
        if any(len(item) != width for item in rows):
            raise RuntimeError("multilingual dense model returned inconsistent dimensions.")
        return rows


__all__ = [
    "GovernedLateInteractionAdapter",
    "GovernedMultilingualDenseEncoder",
    "GovernedSparseExpansionAdapter",
    "ModelArtifactRegistry",
    "ModelArtifactSpec",
]
