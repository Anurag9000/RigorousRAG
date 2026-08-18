"""Authoritative variable-length cache alignment for advanced RAG batches."""
from __future__ import annotations

from typing import Any, Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from training.advanced_rag_data import (
    DynamicCollatorConfig,
    DynamicRagEpisodeStep,
    GroundedCollatorConfig,
    GroundedGenerationExample,
    RetrieverBatchBuilder,
    TensorCacheProvider,
)
from training.advanced_rag_strict import StrictDynamicRagEpisodeCollator
from training.grounded_supervision_pipeline import CompleteGroundedGenerationCollator
from training.seq2seq_grounded import Seq2SeqGroundedGenerationCollator


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("final advanced RAG collation requires optional PyTorch")


def _teacher_batch(examples: Sequence[GroundedGenerationExample], cache: TensorCacheProvider, labels: Any) -> Any:
    _require_torch()
    tensors = []
    vocabulary = None
    for row, example in enumerate(examples):
        if example.teacher_cache_key is None:
            raise ValueError(f"example {example.example_id} lacks teacher_cache_key")
        value = cache.get(example.teacher_cache_key).get("teacher_token_logits")
        if not torch.is_tensor(value):
            raise ValueError("teacher cache entry lacks teacher_token_logits tensor")
        tensor = value.detach().cpu()
        if tensor.ndim == 3 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 2 or tensor.size(0) < 1 or tensor.size(1) < 2:
            raise ValueError("teacher cache tensor must have shape [T,V]")
        if vocabulary is None:
            vocabulary = int(tensor.size(1))
        elif vocabulary != int(tensor.size(1)):
            raise ValueError("teacher cache vocabulary width differs across batch")
        active = torch.nonzero(labels[row].ne(-100), as_tuple=False).flatten()
        if active.numel() and int(active.max().item()) >= tensor.size(0):
            raise ValueError("teacher cache is shorter than a supervised target position")
        tensors.append(tensor)
    if vocabulary is None:
        raise ValueError("teacher batch is empty")
    result = torch.zeros((len(examples), labels.size(1), vocabulary), dtype=tensors[0].dtype)
    for row, tensor in enumerate(tensors):
        count = min(labels.size(1), tensor.size(0))
        result[row, :count] = tensor[:count]
    return result


class FinalCausalGroundedCollator(CompleteGroundedGenerationCollator):
    """Causal grounded collation with variable-length teacher-cache alignment."""
    def __init__(self, tokenizer: Any, config: GroundedCollatorConfig = GroundedCollatorConfig(), *, teacher_cache: TensorCacheProvider | None = None, reference_cache: TensorCacheProvider | None = None, retriever_batch_builder: RetrieverBatchBuilder | None = None) -> None:
        super().__init__(tokenizer, config, teacher_cache=None, reference_cache=reference_cache, retriever_batch_builder=retriever_batch_builder)
        self.final_teacher_cache = teacher_cache

    def __call__(self, examples: Sequence[GroundedGenerationExample]) -> dict[str, Any]:
        batch = super().__call__(examples)
        if self.final_teacher_cache is not None:
            batch["teacher_token_logits"] = _teacher_batch(examples, self.final_teacher_cache, batch["labels"])
        return batch


class FinalSeq2SeqGroundedCollator(Seq2SeqGroundedGenerationCollator):
    """Encoder-decoder grounded collation with variable-length teacher-cache alignment."""
    def __init__(self, tokenizer: Any, config: GroundedCollatorConfig = GroundedCollatorConfig(), *, teacher_cache: TensorCacheProvider | None = None, reference_cache: TensorCacheProvider | None = None, retriever_batch_builder: RetrieverBatchBuilder | None = None) -> None:
        super().__init__(tokenizer, config, teacher_cache=None, reference_cache=reference_cache, retriever_batch_builder=retriever_batch_builder)
        self.final_teacher_cache = teacher_cache

    def __call__(self, examples: Sequence[GroundedGenerationExample]) -> dict[str, Any]:
        batch = super().__call__(examples)
        if self.final_teacher_cache is not None:
            batch["teacher_token_logits"] = _teacher_batch(examples, self.final_teacher_cache, batch["labels"])
        return batch


class FinalDynamicRagEpisodeCollator(StrictDynamicRagEpisodeCollator):
    """Dynamic collation that aligns canonical unpadded hidden-state cache entries."""
    def __init__(self, tokenizer: Any, architecture: Any, config: DynamicCollatorConfig = DynamicCollatorConfig(), *, hidden_state_cache: TensorCacheProvider | None = None) -> None:
        super().__init__(tokenizer, architecture, config, hidden_state_cache=None)
        self.final_hidden_state_cache = hidden_state_cache

    def __call__(self, examples: Sequence[DynamicRagEpisodeStep]) -> dict[str, Any]:
        _require_torch()
        batch = super().__call__(examples)
        if self.final_hidden_state_cache is None:
            return batch
        if "attention_mask" not in batch:
            encoded = self.tokenizer(
                [example.context for example in examples], padding=True, truncation=True,
                max_length=self.config.context_max_length, pad_to_multiple_of=self.config.pad_to_multiple_of,
                return_tensors="pt", add_special_tokens=True,
            )
            attention = encoded["attention_mask"].to(dtype=torch.bool)
            batch["attention_mask"] = attention
            batch["need_valid_mask"] = attention
            batch["need_target_mask"] = torch.zeros_like(attention, dtype=torch.float32)
        attention = batch["attention_mask"].to(dtype=torch.bool)
        sequence = attention.size(1)
        width = self.architecture.context_hidden_size
        token_hidden = torch.zeros((len(examples), sequence, width), dtype=torch.float32)
        state_hidden = torch.zeros((len(examples), width), dtype=torch.float32)
        for row, example in enumerate(examples):
            if example.hidden_state_cache_key is None:
                raise ValueError(f"dynamic step {example.episode_id}:{example.step_id} lacks hidden_state_cache_key")
            cached = self.final_hidden_state_cache.get(example.hidden_state_cache_key)
            tokens = cached.get("token_hidden"); state = cached.get("state_hidden"); cached_mask = cached.get("attention_mask")
            if not torch.is_tensor(tokens) or not torch.is_tensor(state):
                raise ValueError("hidden-state cache entry requires token_hidden and state_hidden tensors")
            tokens = tokens.detach().cpu(); state = state.detach().cpu()
            if tokens.ndim == 3 and tokens.size(0) == 1:
                tokens = tokens.squeeze(0)
            if state.ndim == 2 and state.size(0) == 1:
                state = state.squeeze(0)
            if tokens.ndim != 2 or state.ndim != 1 or tokens.size(1) != width or state.numel() != width:
                raise ValueError("cached hidden-state shapes differ from DynamicPolicyArchitecture")
            valid = int(attention[row].long().sum().item())
            if tokens.size(0) < valid:
                raise ValueError("cached token hidden states are shorter than current tokenization")
            if cached_mask is not None:
                if not torch.is_tensor(cached_mask):
                    raise ValueError("cached attention_mask must be a tensor")
                mask = cached_mask.detach().cpu().reshape(-1)
                if mask.numel() < valid or int(mask[:valid].long().sum().item()) != valid:
                    raise ValueError("cached attention mask does not cover current visible tokens")
            token_hidden[row, :valid] = tokens[:valid].to(dtype=torch.float32)
            state_hidden[row] = state.to(dtype=torch.float32)
        batch["token_hidden"] = token_hidden
        batch["state_hidden"] = state_hidden
        return batch


__all__ = ["FinalCausalGroundedCollator", "FinalDynamicRagEpisodeCollator", "FinalSeq2SeqGroundedCollator"]
