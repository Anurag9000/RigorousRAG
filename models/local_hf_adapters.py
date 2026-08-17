"""Concrete local-only Hugging Face adapters for RigorousRAG learned retrieval.

All loaders require operator-provided local artifact directories and verify deterministic
file-tree digests before loading.  They never fall back to a Hub/network lookup.
Implemented adapters cover dense embeddings, SPLADE sparse expansion, ColBERT token
embeddings/MaxSim and cross-encoder reranking.  Exact model family/revision/license facts
remain governed by :mod:`models.governed_embedding_profiles` rather than hardcoded here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from models.governed_embedding_profiles import (
    EmbeddingBatch,
    EmbeddingModelProfile,
    PoolingStrategy as GovernedPooling,
    VectorNormalization,
    text_digest,
)
from training.model_architectures import (
    ColBERTConfig,
    ColBERTEncoder,
    CrossEncoderReranker,
    DenseEncoder,
    EncoderConfig,
    PoolingStrategy,
    SpladeConfig,
    SpladeEncoder,
)

_HEX = frozenset("0123456789abcdef")
_MAX_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024 * 1024


def _sha256(value: str, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _require_torch_transformers() -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional inference dependency.
        raise RuntimeError("local learned adapters require optional torch + transformers dependencies") from exc
    return torch, AutoTokenizer


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def artifact_tree_digest(root: str | Path) -> str:
    """Digest a regular-file tree by relative path, size and each file's SHA-256."""

    selected = Path(root).expanduser().resolve(strict=True)
    if not selected.is_dir() or selected.is_symlink():
        raise ValueError("model artifact root must be a non-symlink directory")
    records: list[dict[str, Any]] = []
    total = 0
    for path in sorted(selected.rglob("*"), key=lambda value: value.relative_to(selected).as_posix()):
        if path.is_symlink():
            raise ValueError("model artifact tree may not contain symlinks")
        if not path.is_file():
            continue
        relative = path.relative_to(selected).as_posix()
        size = path.stat().st_size
        total += size
        if total > _MAX_ARTIFACT_BYTES:
            raise ValueError("model artifact tree exceeds safety bound")
        records.append({"path": relative, "size": size, "sha256": _sha256_file(path)})
    if not records:
        raise ValueError("model artifact tree contains no regular files")
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class LocalArtifactBinding:
    model_root: str
    model_tree_sha256: str
    tokenizer_root: str
    tokenizer_tree_sha256: str
    declared_revision: str

    def __post_init__(self) -> None:
        model_root = Path(self.model_root).expanduser().resolve(strict=True)
        tokenizer_root = Path(self.tokenizer_root).expanduser().resolve(strict=True)
        if not model_root.is_dir() or model_root.is_symlink() or not tokenizer_root.is_dir() or tokenizer_root.is_symlink():
            raise ValueError("model/tokenizer roots must be non-symlink directories")
        object.__setattr__(self, "model_root", str(model_root))
        object.__setattr__(self, "tokenizer_root", str(tokenizer_root))
        object.__setattr__(self, "model_tree_sha256", _sha256(self.model_tree_sha256, "model_tree_sha256"))
        object.__setattr__(self, "tokenizer_tree_sha256", _sha256(self.tokenizer_tree_sha256, "tokenizer_tree_sha256"))
        revision = str(self.declared_revision).strip()
        if not revision:
            raise ValueError("declared_revision is required")
        object.__setattr__(self, "declared_revision", revision)

    def verify(self, profile: EmbeddingModelProfile | None = None) -> None:
        if artifact_tree_digest(self.model_root) != self.model_tree_sha256:
            raise RuntimeError("local model artifact tree digest mismatch")
        if artifact_tree_digest(self.tokenizer_root) != self.tokenizer_tree_sha256:
            raise RuntimeError("local tokenizer artifact tree digest mismatch")
        if profile is not None:
            if self.declared_revision != profile.exact_revision:
                raise ValueError("local artifact declared revision differs from governed profile")
            if self.model_tree_sha256 != profile.artifact_sha256:
                raise ValueError("local model tree digest differs from governed profile")
            if self.tokenizer_tree_sha256 != profile.tokenizer_sha256:
                raise ValueError("local tokenizer tree digest differs from governed profile")


def _pooling(profile: EmbeddingModelProfile) -> PoolingStrategy:
    mapping = {
        GovernedPooling.CLS: PoolingStrategy.CLS,
        GovernedPooling.MEAN: PoolingStrategy.MEAN,
        GovernedPooling.LAST_TOKEN: PoolingStrategy.LAST_TOKEN,
    }
    if profile.pooling not in mapping:
        raise ValueError(
            "this concrete HF dense adapter requires explicit cls/mean/last_token pooling; "
            "model-native/provider-native profiles need a family-specific reviewed adapter"
        )
    return mapping[profile.pooling]


def _device(torch_module: Any, requested: str) -> Any:
    selected = requested.strip().lower()
    if selected == "auto":
        selected = "cuda" if torch_module.cuda.is_available() else "cpu"
    device = torch_module.device(selected)
    if device.type == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA requested for local adapter but unavailable")
    return device


def _batches(values: Sequence[str], batch_size: int) -> Sequence[Sequence[str]]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


class HFDenseEmbeddingProvider:
    """Governed dense embedding provider using a verified local HF snapshot."""

    def __init__(
        self,
        profile: EmbeddingModelProfile,
        binding: LocalArtifactBinding,
        *,
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        profile.assert_promotable()
        binding.verify(profile)
        self.profile = profile
        self.binding = binding
        self.device_name = device
        self.batch_size = batch_size
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        torch, AutoTokenizer = _require_torch_transformers()
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.binding.tokenizer_root,
                local_files_only=True,
                trust_remote_code=False,
            )
        if self._model is None:
            self._model = DenseEncoder.from_local_pretrained(
                self.binding.model_root,
                config=EncoderConfig(
                    pooling=_pooling(self.profile),
                    projection_dim=None,
                    normalize=self.profile.normalization == VectorNormalization.L2,
                ),
                local_files_only=True,
                trust_remote_code=False,
            )
            if self._model.output_dim != self.profile.output_dimension:
                raise RuntimeError("loaded dense model output dimension differs from governed profile")
            self._model.to(_device(torch, self.device_name))
            self._model.eval()
        return torch, self._tokenizer, self._model

    def _embed(self, texts: Sequence[str], *, query: bool) -> EmbeddingBatch:
        torch, tokenizer, model = self._load()
        if not texts:
            return EmbeddingBatch((), self.profile.digest, ())
        rendered = [
            self.profile.instructions.render_query(text) if query else self.profile.instructions.render_document(text)
            for text in texts
        ]
        vectors: list[tuple[float, ...]] = []
        device = _device(torch, self.device_name)
        for batch in _batches(rendered, self.batch_size):
            tokens = tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=self.profile.max_input_tokens,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            with torch.inference_mode():
                output = model(**tokens).detach().cpu().float()
            vectors.extend(tuple(float(value) for value in row.tolist()) for row in output)
        result = EmbeddingBatch(
            vectors=tuple(vectors),
            profile_digest=self.profile.digest,
            input_digests=tuple(text_digest(text) for text in texts),
        )
        result.validate_against(self.profile)
        return result

    def embed_queries(self, texts: Sequence[str], *, profile: EmbeddingModelProfile) -> EmbeddingBatch:
        if profile.digest != self.profile.digest:
            raise ValueError("requested profile differs from provider's governed profile")
        return self._embed(texts, query=True)

    def embed_documents(self, texts: Sequence[str], *, profile: EmbeddingModelProfile) -> EmbeddingBatch:
        if profile.digest != self.profile.digest:
            raise ValueError("requested profile differs from provider's governed profile")
        return self._embed(texts, query=False)


@dataclass(frozen=True)
class SparseVectorBatch:
    token_weights: tuple[Mapping[int, float], ...]
    artifact_digest: str
    input_digests: tuple[str, ...]


class LocalHFSpladeProvider:
    def __init__(
        self,
        binding: LocalArtifactBinding,
        *,
        artifact_digest: str,
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 512,
        activation_threshold: float = 0.0,
        special_token_ids: Sequence[int] = (),
    ) -> None:
        binding.verify()
        self.binding = binding
        self.artifact_digest = _sha256(artifact_digest, "artifact_digest")
        self.device_name = device
        self.batch_size = batch_size
        self.max_length = int(max_length)
        self.activation_threshold = float(activation_threshold)
        self.special_token_ids = tuple(int(value) for value in special_token_ids)
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        torch, AutoTokenizer = _require_torch_transformers()
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.binding.tokenizer_root,
                local_files_only=True,
                trust_remote_code=False,
            )
        if self._model is None:
            self._model = SpladeEncoder.from_local_pretrained(
                self.binding.model_root,
                config=SpladeConfig(mask_special_token_ids=self.special_token_ids),
                local_files_only=True,
                trust_remote_code=False,
            )
            self._model.to(_device(torch, self.device_name))
            self._model.eval()
        return torch, self._tokenizer, self._model

    def encode(self, texts: Sequence[str]) -> SparseVectorBatch:
        torch, tokenizer, model = self._load()
        device = _device(torch, self.device_name)
        output_rows: list[Mapping[int, float]] = []
        for batch in _batches(list(texts), self.batch_size):
            tokens = tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            with torch.inference_mode():
                weights = model(**tokens).detach().cpu().float()
            for row in weights:
                active = torch.nonzero(row > self.activation_threshold, as_tuple=False).squeeze(-1)
                output_rows.append({int(index): float(row[index]) for index in active.tolist()})
        return SparseVectorBatch(tuple(output_rows), self.artifact_digest, tuple(text_digest(text) for text in texts))


@dataclass(frozen=True)
class LateInteractionItem:
    token_vectors: tuple[tuple[float, ...], ...]
    mask: tuple[bool, ...]


@dataclass(frozen=True)
class LateInteractionBatch:
    items: tuple[LateInteractionItem, ...]
    artifact_digest: str
    input_digests: tuple[str, ...]


class LocalHFColBERTProvider:
    def __init__(
        self,
        binding: LocalArtifactBinding,
        *,
        artifact_digest: str,
        projection_dim: int = 128,
        device: str = "auto",
        max_length: int = 512,
        special_token_ids: Sequence[int] = (),
    ) -> None:
        binding.verify()
        self.binding = binding
        self.artifact_digest = _sha256(artifact_digest, "artifact_digest")
        self.projection_dim = int(projection_dim)
        self.device_name = device
        self.max_length = int(max_length)
        self.special_token_ids = tuple(int(value) for value in special_token_ids)
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        torch, AutoTokenizer = _require_torch_transformers()
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.binding.tokenizer_root,
                local_files_only=True,
                trust_remote_code=False,
            )
        if self._model is None:
            self._model = ColBERTEncoder.from_local_pretrained(
                self.binding.model_root,
                config=ColBERTConfig(
                    projection_dim=self.projection_dim,
                    exclude_token_ids=self.special_token_ids,
                ),
                local_files_only=True,
                trust_remote_code=False,
            )
            self._model.to(_device(torch, self.device_name))
            self._model.eval()
        return torch, self._tokenizer, self._model

    def encode(self, texts: Sequence[str]) -> LateInteractionBatch:
        torch, tokenizer, model = self._load()
        if not texts:
            return LateInteractionBatch((), self.artifact_digest, ())
        device = _device(torch, self.device_name)
        tokens = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.inference_mode():
            embeddings, mask = model(**tokens)
        embeddings = embeddings.detach().cpu().float()
        mask = mask.detach().cpu().bool()
        items: list[LateInteractionItem] = []
        for row, row_mask in zip(embeddings, mask):
            items.append(
                LateInteractionItem(
                    token_vectors=tuple(tuple(float(value) for value in vector.tolist()) for vector in row),
                    mask=tuple(bool(value) for value in row_mask.tolist()),
                )
            )
        return LateInteractionBatch(tuple(items), self.artifact_digest, tuple(text_digest(text) for text in texts))


class LocalHFCrossEncoderProvider:
    def __init__(
        self,
        binding: LocalArtifactBinding,
        *,
        artifact_digest: str,
        device: str = "auto",
        max_length: int = 512,
        score_index: int = 0,
    ) -> None:
        binding.verify()
        self.binding = binding
        self.artifact_digest = _sha256(artifact_digest, "artifact_digest")
        self.device_name = device
        self.max_length = int(max_length)
        self.score_index = int(score_index)
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def _load(self) -> tuple[Any, Any, Any]:
        torch, AutoTokenizer = _require_torch_transformers()
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.binding.tokenizer_root,
                local_files_only=True,
                trust_remote_code=False,
            )
        if self._model is None:
            self._model = CrossEncoderReranker.from_local_pretrained(
                self.binding.model_root,
                score_index=self.score_index,
                local_files_only=True,
                trust_remote_code=False,
            )
            self._model.to(_device(torch, self.device_name))
            self._model.eval()
        return torch, self._tokenizer, self._model

    def score(self, query: str, documents: Sequence[str]) -> tuple[float, ...]:
        torch, tokenizer, model = self._load()
        if not documents:
            return ()
        device = _device(torch, self.device_name)
        tokens = tokenizer(
            [query] * len(documents),
            list(documents),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.inference_mode():
            scores = model(**tokens).detach().cpu().float()
        return tuple(float(value) for value in scores.tolist())


__all__ = [
    "HFDenseEmbeddingProvider",
    "LateInteractionBatch",
    "LateInteractionItem",
    "LocalArtifactBinding",
    "LocalHFColBERTProvider",
    "LocalHFCrossEncoderProvider",
    "LocalHFSpladeProvider",
    "SparseVectorBatch",
    "artifact_tree_digest",
]
