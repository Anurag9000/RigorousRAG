"""Executable neural architectures for learned retrieval and reranking.

This module contains the actual tensor-level model implementations that the earlier
framework-neutral contracts intentionally omitted.  It is still safe for source-only
work: importing it does not download weights, and every Hugging Face loader defaults to
``local_files_only=True`` and ``trust_remote_code=False``.

Architectures implemented here:

* tied or untied dense bi-encoder with CLS/mean/last-token pooling and projection;
* SPLADE-style masked-LM sparse expansion ``max(log(1 + relu(logits)))``;
* uniCOIL-style contextual token weighting and vocabulary aggregation;
* ColBERT-style projected token embeddings and MaxSim scoring;
* cross-encoder scalar reranker; and
* grouped listwise reranker over cross-encoder pair scores.

The classes require PyTorch only when instantiated.  Transformers is required only when
calling the ``from_local_pretrained`` constructors.  There is no implicit network path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

try:  # Optional training dependency; production runtime need not install torch.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - dependency absence is handled at instantiation.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


class _UnavailableModule:
    def __init__(self, *_: Any, **__: Any) -> None:
        _require_torch()


_ModuleBase = nn.Module if nn is not None else _UnavailableModule


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError(
            "learned retrieval training requires the optional PyTorch dependency; "
            "install training dependencies explicitly before executing this module"
        )


def _require_transformers() -> tuple[Any, Any, Any]:
    try:
        from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification
    except Exception as exc:  # pragma: no cover - optional dependency boundary.
        raise RuntimeError(
            "local pretrained loading requires transformers; install the optional training dependencies"
        ) from exc
    return AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification


def _hidden_size(module: Any) -> int:
    config = getattr(module, "config", None)
    for name in ("hidden_size", "d_model", "dim", "n_embd"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("encoder config does not expose a supported hidden-size field")


def _vocab_size(module: Any) -> int:
    config = getattr(module, "config", None)
    value = getattr(config, "vocab_size", None)
    if not isinstance(value, int) or value <= 0:
        raise ValueError("model config does not expose a positive vocab_size")
    return value


def _last_hidden_state(outputs: Any) -> Any:
    value = getattr(outputs, "last_hidden_state", None)
    if value is not None:
        return value
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    raise ValueError("encoder output does not contain last_hidden_state")


def _masked_mean(hidden: Any, attention_mask: Any) -> Any:
    mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / denominator


def _last_token(hidden: Any, attention_mask: Any) -> Any:
    lengths = attention_mask.to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1
    batch = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch, lengths]


class PoolingStrategy(str, Enum):
    CLS = "cls"
    MEAN = "mean"
    LAST_TOKEN = "last_token"


@dataclass(frozen=True)
class EncoderConfig:
    pooling: PoolingStrategy = PoolingStrategy.MEAN
    projection_dim: int | None = None
    normalize: bool = True
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.pooling, PoolingStrategy):
            object.__setattr__(self, "pooling", PoolingStrategy(self.pooling))
        if self.projection_dim is not None and (
            isinstance(self.projection_dim, bool)
            or not isinstance(self.projection_dim, int)
            or self.projection_dim <= 0
        ):
            raise ValueError("projection_dim must be a positive integer")
        if not isinstance(self.normalize, bool):
            raise ValueError("normalize must be boolean")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise ValueError("dropout must be numeric")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0,1)")


class DenseEncoder(_ModuleBase):
    """Transformer encoder + pooling + optional projection + optional L2 normalization."""

    def __init__(self, backbone: Any, config: EncoderConfig = EncoderConfig()) -> None:
        _require_torch()
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise ValueError("backbone must be a torch.nn.Module")
        self.backbone = backbone
        self.encoder_config = config
        hidden = _hidden_size(backbone)
        output_dim = config.projection_dim or hidden
        self.dropout = nn.Dropout(float(config.dropout))
        self.projection = nn.Identity() if output_dim == hidden else nn.Linear(hidden, output_dim, bias=False)
        self.output_dim = output_dim

    @classmethod
    def from_local_pretrained(
        cls,
        model_name_or_path: str,
        *,
        config: EncoderConfig = EncoderConfig(),
        revision: str | None = None,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        **model_kwargs: Any,
    ) -> "DenseEncoder":
        AutoModel, _, _ = _require_transformers()
        backbone = AutoModel.from_pretrained(
            model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )
        return cls(backbone, config)

    def forward(self, input_ids: Any, attention_mask: Any, **encoder_kwargs: Any) -> Any:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **encoder_kwargs)
        hidden = _last_hidden_state(outputs)
        if self.encoder_config.pooling == PoolingStrategy.CLS:
            pooled = hidden[:, 0]
        elif self.encoder_config.pooling == PoolingStrategy.LAST_TOKEN:
            pooled = _last_token(hidden, attention_mask)
        else:
            pooled = _masked_mean(hidden, attention_mask)
        pooled = self.projection(self.dropout(pooled))
        if self.encoder_config.normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled


class DenseBiEncoder(_ModuleBase):
    """Query/document bi-encoder supporting tied or independent towers."""

    def __init__(self, query_encoder: DenseEncoder, document_encoder: DenseEncoder | None = None) -> None:
        _require_torch()
        super().__init__()
        if not isinstance(query_encoder, DenseEncoder):
            raise ValueError("query_encoder must be DenseEncoder")
        if document_encoder is not None and not isinstance(document_encoder, DenseEncoder):
            raise ValueError("document_encoder must be DenseEncoder or None")
        self.query_encoder = query_encoder
        self.document_encoder = document_encoder or query_encoder
        if self.query_encoder.output_dim != self.document_encoder.output_dim:
            raise ValueError("query and document encoder dimensions must match")
        self.output_dim = self.query_encoder.output_dim

    @classmethod
    def from_local_pretrained(
        cls,
        model_name_or_path: str,
        *,
        config: EncoderConfig = EncoderConfig(),
        revision: str | None = None,
        untied_document_model_name_or_path: str | None = None,
        untied_document_revision: str | None = None,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        **model_kwargs: Any,
    ) -> "DenseBiEncoder":
        query = DenseEncoder.from_local_pretrained(
            model_name_or_path,
            config=config,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )
        document = None
        if untied_document_model_name_or_path is not None:
            document = DenseEncoder.from_local_pretrained(
                untied_document_model_name_or_path,
                config=config,
                revision=untied_document_revision,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
        return cls(query, document)

    def encode_queries(self, batch: Mapping[str, Any]) -> Any:
        return self.query_encoder(**dict(batch))

    def encode_documents(self, batch: Mapping[str, Any]) -> Any:
        return self.document_encoder(**dict(batch))

    def score_matrix(self, query_batch: Mapping[str, Any], document_batch: Mapping[str, Any]) -> Any:
        queries = self.encode_queries(query_batch)
        documents = self.encode_documents(document_batch)
        return queries @ documents.transpose(0, 1)


@dataclass(frozen=True)
class SpladeConfig:
    aggregation: str = "max"
    mask_special_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.aggregation not in {"max", "sum"}:
            raise ValueError("SPLADE aggregation must be max or sum")
        for token_id in self.mask_special_token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                raise ValueError("special token ids must be non-negative integers")


class SpladeEncoder(_ModuleBase):
    """SPLADE-style sparse vocabulary expansion over masked-LM logits."""

    def __init__(self, masked_lm: Any, config: SpladeConfig = SpladeConfig()) -> None:
        _require_torch()
        super().__init__()
        if not isinstance(masked_lm, nn.Module):
            raise ValueError("masked_lm must be a torch.nn.Module")
        self.masked_lm = masked_lm
        self.splade_config = config
        self.vocab_size = _vocab_size(masked_lm)

    @classmethod
    def from_local_pretrained(
        cls,
        model_name_or_path: str,
        *,
        config: SpladeConfig = SpladeConfig(),
        revision: str | None = None,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        **model_kwargs: Any,
    ) -> "SpladeEncoder":
        _, AutoModelForMaskedLM, _ = _require_transformers()
        model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )
        return cls(model, config)

    def forward(self, input_ids: Any, attention_mask: Any, **model_kwargs: Any) -> Any:
        outputs = self.masked_lm(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs)
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise ValueError("masked-LM output does not expose logits")
        activations = torch.log1p(torch.relu(logits))
        token_mask = attention_mask.to(dtype=activations.dtype).unsqueeze(-1)
        activations = activations * token_mask
        if self.splade_config.mask_special_token_ids:
            allowed = torch.ones(self.vocab_size, dtype=activations.dtype, device=activations.device)
            special = torch.tensor(self.splade_config.mask_special_token_ids, dtype=torch.long, device=activations.device)
            special = special[(special >= 0) & (special < self.vocab_size)]
            allowed.index_fill_(0, special, 0.0)
            activations = activations * allowed.view(1, 1, -1)
        if self.splade_config.aggregation == "sum":
            return activations.sum(dim=1)
        return activations.max(dim=1).values

    @staticmethod
    def score_matrix(query_weights: Any, document_weights: Any) -> Any:
        return query_weights @ document_weights.transpose(0, 1)


@dataclass(frozen=True)
class UniCOILConfig:
    aggregation: str = "max"
    nonnegative: bool = True
    exclude_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.aggregation not in {"max", "sum"}:
            raise ValueError("uniCOIL aggregation must be max or sum")
        if not isinstance(self.nonnegative, bool):
            raise ValueError("nonnegative must be boolean")
        for token_id in self.exclude_token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                raise ValueError("exclude token ids must be non-negative integers")


class UniCOILEncoder(_ModuleBase):
    """Contextual token weighting with aggregation into vocabulary coordinates."""

    def __init__(self, backbone: Any, config: UniCOILConfig = UniCOILConfig()) -> None:
        _require_torch()
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise ValueError("backbone must be a torch.nn.Module")
        self.backbone = backbone
        self.unicoil_config = config
        self.vocab_size = _vocab_size(backbone)
        self.token_scorer = nn.Linear(_hidden_size(backbone), 1)

    @classmethod
    def from_local_pretrained(
        cls,
        model_name_or_path: str,
        *,
        config: UniCOILConfig = UniCOILConfig(),
        revision: str | None = None,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        **model_kwargs: Any,
    ) -> "UniCOILEncoder":
        AutoModel, _, _ = _require_transformers()
        backbone = AutoModel.from_pretrained(
            model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )
        return cls(backbone, config)

    def forward(self, input_ids: Any, attention_mask: Any, **model_kwargs: Any) -> Any:
        hidden = _last_hidden_state(
            self.backbone(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs)
        )
        token_weights = self.token_scorer(hidden).squeeze(-1)
        if self.unicoil_config.nonnegative:
            token_weights = torch.relu(token_weights)
        valid = attention_mask.to(dtype=torch.bool)
        if self.unicoil_config.exclude_token_ids:
            for token_id in self.unicoil_config.exclude_token_ids:
                valid = valid & input_ids.ne(token_id)
        weights = torch.where(valid, token_weights, torch.zeros_like(token_weights))
        batch_size = input_ids.size(0)
        if self.unicoil_config.aggregation == "sum":
            output = torch.zeros(batch_size, self.vocab_size, dtype=weights.dtype, device=weights.device)
            output.scatter_add_(1, input_ids.clamp(0, self.vocab_size - 1), weights)
            return output
        # Max aggregation, implemented without relying on scatter_reduce availability.
        output = torch.zeros(batch_size, self.vocab_size, dtype=weights.dtype, device=weights.device)
        for row in range(batch_size):
            row_ids = input_ids[row]
            row_weights = weights[row]
            unique_ids = torch.unique(row_ids[valid[row]])
            for token_id in unique_ids:
                positions = (row_ids == token_id) & valid[row]
                output[row, token_id] = row_weights[positions].max()
        return output

    @staticmethod
    def score_matrix(query_weights: Any, document_weights: Any) -> Any:
        return query_weights @ document_weights.transpose(0, 1)


@dataclass(frozen=True)
class ColBERTConfig:
    projection_dim: int = 128
    dropout: float = 0.0
    normalize: bool = True
    exclude_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.projection_dim, bool) or not isinstance(self.projection_dim, int) or self.projection_dim <= 0:
            raise ValueError("projection_dim must be a positive integer")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if not isinstance(self.normalize, bool):
            raise ValueError("normalize must be boolean")


class ColBERTEncoder(_ModuleBase):
    """Projected token encoder plus exact MaxSim late interaction."""

    def __init__(self, backbone: Any, config: ColBERTConfig = ColBERTConfig()) -> None:
        _require_torch()
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise ValueError("backbone must be a torch.nn.Module")
        self.backbone = backbone
        self.colbert_config = config
        self.projection = nn.Linear(_hidden_size(backbone), config.projection_dim, bias=False)
        self.dropout = nn.Dropout(float(config.dropout))
        self.output_dim = config.projection_dim

    @classmethod
    def from_local_pretrained(
        cls,
        model_name_or_path: str,
        *,
        config: ColBERTConfig = ColBERTConfig(),
        revision: str | None = None,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        **model_kwargs: Any,
    ) -> "ColBERTEncoder":
        AutoModel, _, _ = _require_transformers()
        backbone = AutoModel.from_pretrained(
            model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )
        return cls(backbone, config)

    def forward(self, input_ids: Any, attention_mask: Any, **model_kwargs: Any) -> tuple[Any, Any]:
        hidden = _last_hidden_state(
            self.backbone(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs)
        )
        embeddings = self.projection(self.dropout(hidden))
        if self.colbert_config.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        mask = attention_mask.to(dtype=torch.bool)
        for token_id in self.colbert_config.exclude_token_ids:
            mask = mask & input_ids.ne(token_id)
        return embeddings, mask

    @staticmethod
    def pairwise_maxsim(
        query_embeddings: Any,
        query_mask: Any,
        document_embeddings: Any,
        document_mask: Any,
    ) -> Any:
        """Score aligned query/document batches with ColBERT MaxSim."""

        similarity = torch.einsum("bqd,bkd->bqk", query_embeddings, document_embeddings)
        similarity = similarity.masked_fill(~document_mask[:, None, :], torch.finfo(similarity.dtype).min)
        maxima = similarity.max(dim=-1).values
        maxima = torch.where(query_mask, maxima, torch.zeros_like(maxima))
        return maxima.sum(dim=-1)

    @staticmethod
    def score_matrix(
        query_embeddings: Any,
        query_mask: Any,
        document_embeddings: Any,
        document_mask: Any,
    ) -> Any:
        """All-pairs query/document MaxSim matrix, shaped ``[Q, D]``."""

        similarity = torch.einsum("qte,dse->qdts", query_embeddings, document_embeddings)
        similarity = similarity.masked_fill(~document_mask[None, :, None, :], torch.finfo(similarity.dtype).min)
        maxima = similarity.max(dim=-1).values
        maxima = torch.where(query_mask[:, None, :], maxima, torch.zeros_like(maxima))
        return maxima.sum(dim=-1)


class CrossEncoderReranker(_ModuleBase):
    """Scalar sequence-pair reranker backed by a sequence-classification model."""

    def __init__(self, model: Any, *, score_index: int = 0) -> None:
        _require_torch()
        super().__init__()
        if not isinstance(model, nn.Module):
            raise ValueError("model must be a torch.nn.Module")
        if isinstance(score_index, bool) or not isinstance(score_index, int) or score_index < 0:
            raise ValueError("score_index must be a non-negative integer")
        self.model = model
        self.score_index = score_index

    @classmethod
    def from_local_pretrained(
        cls,
        model_name_or_path: str,
        *,
        revision: str | None = None,
        score_index: int = 0,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        num_labels: int = 1,
        **model_kwargs: Any,
    ) -> "CrossEncoderReranker":
        _, _, AutoModelForSequenceClassification = _require_transformers()
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            num_labels=num_labels,
            **model_kwargs,
        )
        return cls(model, score_index=score_index)

    def forward(self, **pair_batch: Any) -> Any:
        outputs = self.model(**pair_batch)
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise ValueError("cross-encoder output does not expose logits")
        if logits.ndim == 1:
            return logits
        if logits.size(-1) == 1:
            return logits.squeeze(-1)
        if self.score_index >= logits.size(-1):
            raise ValueError("score_index is outside the model logits dimension")
        return logits[:, self.score_index]


class ListwiseReranker(_ModuleBase):
    """Cross-encoder scorer with explicit grouped-list reconstruction."""

    def __init__(self, cross_encoder: CrossEncoderReranker) -> None:
        _require_torch()
        super().__init__()
        if not isinstance(cross_encoder, CrossEncoderReranker):
            raise ValueError("cross_encoder must be CrossEncoderReranker")
        self.cross_encoder = cross_encoder

    def forward(self, *, group_sizes: Sequence[int], **pair_batch: Any) -> tuple[Any, ...]:
        scores = self.cross_encoder(**pair_batch)
        if not group_sizes:
            raise ValueError("group_sizes must be non-empty")
        total = 0
        for size in group_sizes:
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError("group sizes must be positive integers")
            total += size
        if total != scores.size(0):
            raise ValueError("sum(group_sizes) must equal number of pair scores")
        result = []
        offset = 0
        for size in group_sizes:
            result.append(scores[offset : offset + size])
            offset += size
        return tuple(result)


__all__ = [
    "ColBERTConfig",
    "ColBERTEncoder",
    "CrossEncoderReranker",
    "DenseBiEncoder",
    "DenseEncoder",
    "EncoderConfig",
    "ListwiseReranker",
    "PoolingStrategy",
    "SpladeConfig",
    "SpladeEncoder",
    "UniCOILConfig",
    "UniCOILEncoder",
]
