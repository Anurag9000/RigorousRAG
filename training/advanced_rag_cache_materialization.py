"""Canonical per-example cache materialization for advanced RAG supervision.

These helpers standardize the tensor shapes consumed by the authoritative collators. They
execute only when explicitly called with admitted providers; importing the module performs no
model inference, retrieval or training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from orchestration.reference_dynamic_features import GeneratorHiddenStateAdapter
from training.advanced_rag_data import DynamicRagEpisodeStep, GroundedGenerationExample
from training.advanced_rag_supervision import GroundedSupervisionMaterializer, SafetensorSupervisionCache


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("advanced RAG cache materialization requires optional PyTorch")


def _tensor(value: Any, label: str) -> Any:
    _require_torch()
    if not torch.is_tensor(value):
        raise ValueError(f"{label} must be a tensor")
    return value.detach().cpu().contiguous()


@dataclass(frozen=True)
class CacheMaterializationReceipt:
    cache_identity_sha256: str
    record_count: int
    entry_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.cache_identity_sha256) != 64:
            raise ValueError("cache_identity_sha256 must be SHA-256")
        if isinstance(self.record_count, bool) or not isinstance(self.record_count, int) or self.record_count < 0:
            raise ValueError("record_count must be non-negative")
        if self.record_count != len(self.entry_sha256) or any(len(value) != 64 for value in self.entry_sha256):
            raise ValueError("entry digests must contain one SHA-256 per record")


def materialize_reference_policy_cache(examples: Sequence[GroundedGenerationExample], *, materializer: GroundedSupervisionMaterializer, cache: SafetensorSupervisionCache) -> CacheMaterializationReceipt:
    _require_torch()
    if not examples:
        raise ValueError("reference cache materialization requires examples")
    values = materializer.preference_log_probs(examples)
    digests = []
    for example, (chosen, rejected) in zip(examples, values):
        digests.append(cache.put(example.example_id, {
            "reference_chosen_log_prob": torch.tensor(float(chosen), dtype=torch.float32),
            "reference_rejected_log_prob": torch.tensor(float(rejected), dtype=torch.float32),
        }))
    return CacheMaterializationReceipt(cache.identity.digest, len(examples), tuple(digests))


def materialize_teacher_logit_cache(examples: Sequence[GroundedGenerationExample], *, materializer: GroundedSupervisionMaterializer, cache: SafetensorSupervisionCache) -> CacheMaterializationReceipt:
    """Persist one unpadded ``[T,V]`` teacher-logit tensor per grounded example."""
    _require_torch()
    if not examples:
        raise ValueError("teacher cache materialization requires examples")
    values = materializer.teacher_logits(examples)
    digests = []
    for example, raw in zip(examples, values):
        tensor = _tensor(raw, "teacher logits")
        if tensor.ndim == 3 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 2 or tensor.size(0) < 1 or tensor.size(1) < 2:
            raise ValueError("teacher provider must produce one unpadded [T,V] tensor per example")
        key = example.teacher_cache_key or example.example_id
        digests.append(cache.put(key, {"teacher_token_logits": tensor}))
    return CacheMaterializationReceipt(cache.identity.digest, len(examples), tuple(digests))


def materialize_document_utility_cache(examples: Sequence[GroundedGenerationExample], *, materializer: GroundedSupervisionMaterializer, cache: SafetensorSupervisionCache) -> CacheMaterializationReceipt:
    """Persist one measured generator document-utility vector ``[D]`` per example."""
    _require_torch()
    if not examples:
        raise ValueError("document utility cache materialization requires examples")
    values = materializer.document_utilities(examples)
    digests = []
    for example, raw in zip(examples, values):
        tensor = _tensor(raw, "document utility")
        if tensor.ndim == 2 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 1 or tensor.numel() < len(example.evidence):
            raise ValueError("document utility provider must produce at least one score per evidence item")
        key = example.retriever_cache_key or example.example_id
        digests.append(cache.put(key, {"document_lm_log_likelihood": tensor}))
    return CacheMaterializationReceipt(cache.identity.digest, len(examples), tuple(digests))


def materialize_generator_hidden_state_cache(steps: Sequence[DynamicRagEpisodeStep], *, adapter: GeneratorHiddenStateAdapter, cache: SafetensorSupervisionCache) -> CacheMaterializationReceipt:
    """Persist canonical unpadded ``[T,H]`` token states and ``[H]`` state vectors."""
    _require_torch()
    if not steps:
        raise ValueError("hidden-state cache materialization requires dynamic episode steps")
    digests = []
    for step in steps:
        encoded = adapter.encode([step.context])
        token_hidden = _tensor(encoded["token_hidden"], "token_hidden")
        state_hidden = _tensor(encoded["state_hidden"], "state_hidden")
        attention_mask = _tensor(encoded["attention_mask"], "attention_mask")
        if token_hidden.ndim != 3 or token_hidden.size(0) != 1 or state_hidden.ndim != 2 or state_hidden.size(0) != 1 or attention_mask.ndim != 2 or attention_mask.size(0) != 1:
            raise ValueError("generator hidden-state adapter must return batch-one [1,T,H]/[1,H]/[1,T] tensors")
        valid = int(attention_mask[0].long().sum().item())
        if valid <= 0 or valid > token_hidden.size(1):
            raise ValueError("hidden-state adapter returned an invalid visible-token count")
        key = step.hidden_state_cache_key or f"{step.episode_id}:{step.step_id}"
        digests.append(cache.put(key, {
            "token_hidden": token_hidden[0, :valid],
            "state_hidden": state_hidden[0],
            "attention_mask": attention_mask[0, :valid],
        }))
    return CacheMaterializationReceipt(cache.identity.digest, len(steps), tuple(digests))


__all__ = [
    "CacheMaterializationReceipt",
    "materialize_document_utility_cache",
    "materialize_generator_hidden_state_cache",
    "materialize_reference_policy_cache",
    "materialize_teacher_logit_cache",
]
