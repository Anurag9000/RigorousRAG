"""Concrete cached preference and generator-retriever coupling for grounded RAG training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from training.advanced_rag_data import GroundedGenerationExample, TensorCacheProvider
from training.advanced_rag_strict import StrictGroundedGenerationCollator


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("grounded supervision pipeline requires optional PyTorch")


class CompleteGroundedGenerationCollator(StrictGroundedGenerationCollator):
    """Strict grounded collation plus immutable reference-policy cache fallback.

    Reference scores may live directly in governed JSONL or in a separate content-bound
    safetensor cache keyed by ``example_id``. Keeping both paths explicit supports small
    hand-annotated sets as well as large offline reference-policy materializations.
    """
    def __init__(self, *args: Any, reference_cache: TensorCacheProvider | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reference_cache = reference_cache

    def __call__(self, examples: Sequence[GroundedGenerationExample]) -> dict[str, Any]:
        _require_torch()
        batch = super().__call__(examples)
        has_preference = any(item.chosen_answer is not None for item in examples)
        if not has_preference:
            return batch
        if "reference_chosen_log_prob" in batch:
            return batch
        if self.reference_cache is None:
            return batch
        chosen, rejected = [], []
        for example in examples:
            cached = self.reference_cache.get(example.example_id)
            if "reference_chosen_log_prob" not in cached or "reference_rejected_log_prob" not in cached:
                raise ValueError("reference cache entry lacks chosen/rejected sequence log probabilities")
            chosen_value = cached["reference_chosen_log_prob"]
            rejected_value = cached["reference_rejected_log_prob"]
            if not torch.is_tensor(chosen_value) or not torch.is_tensor(rejected_value) or chosen_value.numel() != 1 or rejected_value.numel() != 1:
                raise ValueError("cached reference log probabilities must be scalar tensors")
            chosen.append(float(chosen_value.item())); rejected.append(float(rejected_value.item()))
        batch["reference_chosen_log_prob"] = torch.tensor(chosen, dtype=torch.float32)
        batch["reference_rejected_log_prob"] = torch.tensor(rejected, dtype=torch.float32)
        return batch


@dataclass(frozen=True)
class RetrieverCouplingConfig:
    evidence_limit: int = 16
    pair_max_length: int = 512
    positive_label_index: int = 0
    pad_to_multiple_of: int | None = 8

    def __post_init__(self) -> None:
        for name in ("evidence_limit", "pair_max_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if isinstance(self.positive_label_index, bool) or not isinstance(self.positive_label_index, int) or self.positive_label_index < 0:
            raise ValueError("positive_label_index must be non-negative")


class CachedDocumentUtilityRetrieverBatchBuilder:
    """Build [B,D,L] query/evidence pairs and generator-derived document utilities.

    Each cache entry is keyed by ``example.retriever_cache_key`` (or ``example_id`` as a
    deterministic fallback) and contains ``document_lm_log_likelihood`` aligned to the
    example's evidence order. The cache identity separately binds the generator/tokenizer/
    dataset that produced those scores.
    """
    def __init__(self, tokenizer: Any, utility_cache: TensorCacheProvider, config: RetrieverCouplingConfig = RetrieverCouplingConfig()) -> None:
        self.tokenizer, self.utility_cache, self.config = tokenizer, utility_cache, config

    def __call__(self, examples: Sequence[GroundedGenerationExample]) -> Mapping[str, Any]:
        _require_torch()
        if not examples:
            raise ValueError("retriever coupling requires a non-empty batch")
        candidate_count = max(min(len(example.evidence), self.config.evidence_limit) for example in examples)
        prompts, documents, owner_rows = [], [], []
        utility = torch.zeros((len(examples), candidate_count), dtype=torch.float32)
        candidate_mask = torch.zeros((len(examples), candidate_count), dtype=torch.bool)
        for row, example in enumerate(examples):
            selected = example.evidence[: self.config.evidence_limit]
            cache_key = example.retriever_cache_key or example.example_id
            cached = self.utility_cache.get(cache_key)
            scores = cached.get("document_lm_log_likelihood")
            if not torch.is_tensor(scores) or scores.ndim != 1 or scores.numel() < len(selected):
                raise ValueError("document-utility cache must contain a 1-D score for every selected evidence item")
            utility[row, :len(selected)] = scores[:len(selected)].detach().float().cpu()
            candidate_mask[row, :len(selected)] = True
            for evidence in selected:
                prompts.append(example.prompt); documents.append(evidence.text); owner_rows.append(row)
        encoded = self.tokenizer(
            prompts,
            documents,
            padding=True,
            truncation=True,
            max_length=self.config.pair_max_length,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
            return_tensors="pt",
        )
        sequence = encoded["input_ids"].size(1)
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        model_inputs: dict[str, Any] = {}
        for key, tensor in encoded.items():
            if not torch.is_tensor(tensor) or tensor.ndim != 2:
                continue
            fill = int(pad_id) if key == "input_ids" else 0
            shaped = torch.full((len(examples), candidate_count, sequence), fill, dtype=tensor.dtype)
            positions = [0] * len(examples)
            for source_row, owner in enumerate(owner_rows):
                column = positions[owner]; positions[owner] += 1
                shaped[owner, column] = tensor[source_row]
            model_inputs[key] = shaped
        return {
            "model_inputs": model_inputs,
            "document_lm_log_likelihood": utility,
            "retriever_candidate_mask": candidate_mask,
        }


if nn is not None:
    class PairwiseCandidateRetriever(nn.Module):
        """Adapt a local sequence-classification model to RigorousRAG [B,D] retriever logits."""
        def __init__(self, pair_model: nn.Module, *, positive_label_index: int = 0) -> None:
            super().__init__()
            if not isinstance(pair_model, nn.Module):
                raise ValueError("pair_model must be nn.Module")
            if isinstance(positive_label_index, bool) or not isinstance(positive_label_index, int) or positive_label_index < 0:
                raise ValueError("positive_label_index must be non-negative")
            self.pair_model = pair_model
            self.positive_label_index = positive_label_index

        def forward(self, **inputs: Any) -> Mapping[str, Any]:
            if "input_ids" not in inputs or inputs["input_ids"].ndim != 3:
                raise ValueError("pairwise retriever input_ids must have shape [B,D,L]")
            batch, documents, length = inputs["input_ids"].shape
            flattened = {key: value.reshape(batch * documents, length) if torch.is_tensor(value) and value.ndim == 3 else value for key, value in inputs.items()}
            output = self.pair_model(**flattened, return_dict=True)
            logits = output["logits"] if isinstance(output, Mapping) else getattr(output, "logits", None)
            if logits is None:
                raise ValueError("pair model does not expose logits")
            if logits.ndim == 1:
                scores = logits
            elif logits.ndim == 2 and logits.size(1) == 1:
                scores = logits[:, 0]
            elif logits.ndim == 2:
                if self.positive_label_index >= logits.size(1):
                    raise ValueError("positive_label_index exceeds pair-model label dimension")
                scores = logits[:, self.positive_label_index]
            else:
                raise ValueError("pair-model logits must have shape [N], [N,1], or [N,C]")
            return {"logits": scores.reshape(batch, documents)}
else:
    class PairwiseCandidateRetriever:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


__all__ = [
    "CachedDocumentUtilityRetrieverBatchBuilder",
    "CompleteGroundedGenerationCollator",
    "PairwiseCandidateRetriever",
    "RetrieverCouplingConfig",
]
