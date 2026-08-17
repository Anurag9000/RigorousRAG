"""Revision-pinned embedding-model profiles without hidden downloads or model claims.

This module deliberately distinguishes a *model-family proposal* from a runnable model
artifact.  A deployment must provide the exact provider/model/revision, output dimension,
pooling/normalization contract, license decision and artifact/tokenizer digests before a
profile is promotable.  The repository therefore can name useful research families such
as SPECTER2, BGE-M3 and INSTRUCTOR without inventing mutable revisions or silently
fetching weights.

Actual inference is injected through ``EmbeddingProvider``. Importing this module never
loads weights, opens a network connection, or executes a model.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

_HEX = frozenset("0123456789abcdef")
_MAX_DIMENSION = 1_000_000
_MAX_BATCH = 100_000


def _identifier(value: Any, label: str, maximum: int = 2_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if not result or len(result) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{label} is invalid")
    return result


def _text(value: Any, label: str, maximum: int = 100_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if not result or len(result) > maximum or "\x00" in result:
        raise ValueError(f"{label} is empty or too long")
    return result


def _sha256(value: Any, label: str) -> str:
    digest = _identifier(value, label, 64).lower()
    if len(digest) != 64 or any(ch not in _HEX for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class PoolingStrategy(str, Enum):
    CLS = "cls"
    MEAN = "mean"
    MAX = "max"
    LAST_TOKEN = "last_token"
    MODEL_NATIVE = "model_native"
    PROVIDER_NATIVE = "provider_native"


class VectorNormalization(str, Enum):
    NONE = "none"
    L2 = "l2"
    MODEL_NATIVE = "model_native"


class ModelPurpose(str, Enum):
    GENERAL_RETRIEVAL = "general_retrieval"
    MULTILINGUAL_RETRIEVAL = "multilingual_retrieval"
    SCIENTIFIC_RETRIEVAL = "scientific_retrieval"
    INSTRUCTION_TUNED_RETRIEVAL = "instruction_tuned_retrieval"
    MULTI_VECTOR_RETRIEVAL = "multi_vector_retrieval"


@dataclass(frozen=True)
class ModelFamilyProposal:
    family_name: str
    purposes: tuple[ModelPurpose, ...]
    rationale: str
    exact_revision_required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "family_name", _identifier(self.family_name, "family_name"))
        if not self.purposes or any(not isinstance(value, ModelPurpose) for value in self.purposes):
            raise ValueError("purposes must contain ModelPurpose values")
        object.__setattr__(self, "rationale", _text(self.rationale, "rationale", 20_000))
        if not isinstance(self.exact_revision_required, bool):
            raise ValueError("exact_revision_required must be boolean")


@dataclass(frozen=True)
class LicenseDecision:
    identifier: str
    allowed_for_intended_use: bool
    evidence: str
    reviewer: str
    decision_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "identifier", _identifier(self.identifier, "license identifier"))
        if not isinstance(self.allowed_for_intended_use, bool):
            raise ValueError("allowed_for_intended_use must be boolean")
        object.__setattr__(self, "evidence", _text(self.evidence, "license evidence"))
        object.__setattr__(self, "reviewer", _identifier(self.reviewer, "license reviewer"))
        if self.decision_digest is not None:
            object.__setattr__(self, "decision_digest", _sha256(self.decision_digest, "decision_digest"))


@dataclass(frozen=True)
class EmbeddingInstructionTemplate:
    query_prefix: str = ""
    document_prefix: str = ""
    query_template: str = "{text}"
    document_template: str = "{text}"

    def __post_init__(self) -> None:
        for name in ("query_prefix", "document_prefix", "query_template", "document_template"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) > 20_000 or "\x00" in value:
                raise ValueError(f"{name} is invalid")
        if "{text}" not in self.query_template or "{text}" not in self.document_template:
            raise ValueError("query/document templates must contain {text}")

    def render_query(self, text: str) -> str:
        selected = _text(text, "query text")
        return self.query_prefix + self.query_template.format(text=selected)

    def render_document(self, text: str) -> str:
        selected = _text(text, "document text")
        return self.document_prefix + self.document_template.format(text=selected)


@dataclass(frozen=True)
class EmbeddingModelProfile:
    profile_id: str
    provider: str
    model_id: str
    exact_revision: str
    purpose: ModelPurpose
    output_dimension: int
    pooling: PoolingStrategy
    normalization: VectorNormalization
    max_input_tokens: int
    artifact_sha256: str
    tokenizer_sha256: str
    license_decision: LicenseDecision
    instructions: EmbeddingInstructionTemplate = EmbeddingInstructionTemplate()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("profile_id", "provider", "model_id", "exact_revision"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name, 4_000))
        placeholders = {"latest", "main", "master", "head", "unknown", "tbd", "todo", "none", "n/a"}
        if self.exact_revision.casefold() in placeholders:
            raise ValueError("exact_revision may not be a moving or unknown placeholder")
        if not isinstance(self.purpose, ModelPurpose):
            object.__setattr__(self, "purpose", ModelPurpose(self.purpose))
        if isinstance(self.output_dimension, bool) or not isinstance(self.output_dimension, int) or not 1 <= self.output_dimension <= _MAX_DIMENSION:
            raise ValueError("output_dimension is invalid")
        if not isinstance(self.pooling, PoolingStrategy):
            object.__setattr__(self, "pooling", PoolingStrategy(self.pooling))
        if not isinstance(self.normalization, VectorNormalization):
            object.__setattr__(self, "normalization", VectorNormalization(self.normalization))
        if isinstance(self.max_input_tokens, bool) or not isinstance(self.max_input_tokens, int) or not 1 <= self.max_input_tokens <= 10_000_000:
            raise ValueError("max_input_tokens is invalid")
        object.__setattr__(self, "artifact_sha256", _sha256(self.artifact_sha256, "artifact_sha256"))
        object.__setattr__(self, "tokenizer_sha256", _sha256(self.tokenizer_sha256, "tokenizer_sha256"))
        if not isinstance(self.license_decision, LicenseDecision):
            raise ValueError("license_decision must be LicenseDecision")
        if not isinstance(self.instructions, EmbeddingInstructionTemplate):
            raise ValueError("instructions must be EmbeddingInstructionTemplate")
        if not isinstance(self.metadata, Mapping) or len(self.metadata) > 2_000:
            raise ValueError("metadata must be a bounded mapping")
        object.__setattr__(
            self,
            "metadata",
            {
                _identifier(key, "metadata key", 300): _text(value, "metadata value", 20_000)
                for key, value in self.metadata.items()
            },
        )

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))

    def assert_promotable(self) -> None:
        if not self.license_decision.allowed_for_intended_use:
            raise ValueError("model profile is not licensed for the intended use")
        if not self.artifact_sha256 or not self.tokenizer_sha256:
            raise ValueError("promotable profiles require pinned artifact and tokenizer digests")


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: tuple[tuple[float, ...], ...]
    profile_digest: str
    input_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_digest", _sha256(self.profile_digest, "profile_digest"))
        object.__setattr__(
            self,
            "input_digests",
            tuple(_sha256(value, "input_digest") for value in self.input_digests),
        )
        if len(self.vectors) != len(self.input_digests) or len(self.vectors) > _MAX_BATCH:
            raise ValueError("vectors and input digests must be aligned and bounded")
        cleaned: list[tuple[float, ...]] = []
        for vector in self.vectors:
            cleaned.append(tuple(_finite(value, "embedding component") for value in vector))
        object.__setattr__(self, "vectors", tuple(cleaned))

    def validate_against(self, profile: EmbeddingModelProfile) -> None:
        if self.profile_digest != profile.digest:
            raise ValueError("embedding batch profile digest does not match profile")
        if any(len(vector) != profile.output_dimension for vector in self.vectors):
            raise ValueError("embedding vector dimension does not match profile")
        if profile.normalization == VectorNormalization.L2:
            for vector in self.vectors:
                norm = math.sqrt(sum(value * value for value in vector))
                if abs(norm - 1.0) > 1e-4:
                    raise ValueError("profile requires L2-normalized vectors")


class EmbeddingProvider(Protocol):
    """Injected runtime adapter. Implementations must not mutate the supplied profile."""

    def embed_queries(self, texts: Sequence[str], *, profile: EmbeddingModelProfile) -> EmbeddingBatch: ...

    def embed_documents(self, texts: Sequence[str], *, profile: EmbeddingModelProfile) -> EmbeddingBatch: ...


def text_digest(text: str) -> str:
    return hashlib.sha256(_text(text, "text").encode("utf-8")).hexdigest()


def validate_provider_batch(
    texts: Sequence[str],
    batch: EmbeddingBatch,
    *,
    profile: EmbeddingModelProfile,
) -> None:
    if len(texts) != len(batch.vectors):
        raise ValueError("provider returned a different batch cardinality")
    expected = tuple(text_digest(text) for text in texts)
    if batch.input_digests != expected:
        raise ValueError("provider input digests do not match the requested text order")
    batch.validate_against(profile)


RECOMMENDED_MODEL_FAMILY_PROPOSALS: tuple[ModelFamilyProposal, ...] = (
    ModelFamilyProposal(
        "SPECTER2 family",
        (ModelPurpose.SCIENTIFIC_RETRIEVAL,),
        "Candidate family for scientific-paper representation experiments; exact revision and task adapter must be governed before use.",
    ),
    ModelFamilyProposal(
        "BGE-M3 family",
        (ModelPurpose.MULTILINGUAL_RETRIEVAL, ModelPurpose.GENERAL_RETRIEVAL),
        "Candidate family for multilingual and hybrid-retrieval experiments; exact runtime capabilities must be verified from the pinned artifact.",
    ),
    ModelFamilyProposal(
        "INSTRUCTOR family",
        (ModelPurpose.INSTRUCTION_TUNED_RETRIEVAL,),
        "Candidate family for instruction-conditioned query/document embedding experiments; instructions are profile data, never hidden defaults.",
    ),
)


__all__ = [
    "EmbeddingBatch",
    "EmbeddingInstructionTemplate",
    "EmbeddingModelProfile",
    "EmbeddingProvider",
    "LicenseDecision",
    "ModelFamilyProposal",
    "ModelPurpose",
    "PoolingStrategy",
    "RECOMMENDED_MODEL_FAMILY_PROPOSALS",
    "VectorNormalization",
    "canonical_digest",
    "text_digest",
    "validate_provider_batch",
]
