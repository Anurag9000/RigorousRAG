"""Executable PyTorch model composition for grounded and dynamic RAG training.

This module supplies architecture glue, not pretrained weights. It composes an injected
causal/seq2seq language model with grounded auxiliary heads and the dynamic retrieval
policy/value/information-need models. No model download or device allocation occurs on
import; exact admitted model artifacts are injected by callers.
"""
from __future__ import annotations

from typing import Any, Mapping

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from training.dynamic_retrieval_policy import DynamicPolicyArchitecture, DynamicRetrievalController, InformationNeedSelector
from training.grounded_generation import GroundedAuxiliaryHeads, GroundedGenerationArchitectureConfig


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("advanced RAG model composition requires optional PyTorch")


def _tensor_from_output(output: Any, key: str) -> Any | None:
    if isinstance(output, Mapping):
        value = output.get(key)
        if value is not None:
            return value
    return getattr(output, key, None)


def _lm_logits_and_hidden(output: Any) -> tuple[Any, Any]:
    logits = _tensor_from_output(output, "logits")
    if logits is None:
        raise ValueError("base language model output does not expose logits")
    hidden = _tensor_from_output(output, "last_hidden_state")
    if hidden is None:
        hidden_states = _tensor_from_output(output, "hidden_states")
        if hidden_states is not None and len(hidden_states) > 0:
            hidden = hidden_states[-1]
    if hidden is None:
        decoder_hidden_states = _tensor_from_output(output, "decoder_hidden_states")
        if decoder_hidden_states is not None and len(decoder_hidden_states) > 0:
            hidden = decoder_hidden_states[-1]
    if hidden is None:
        raise ValueError("base language model output does not expose a final hidden state")
    if logits.ndim != 3 or hidden.ndim != 3 or logits.shape[:2] != hidden.shape[:2]:
        raise ValueError("language-model logits and final hidden state must align on [B,T]")
    return logits, hidden


def _gather_positions(hidden: Any, indices: Any, *, label: str) -> Any:
    _require_torch()
    if hidden.ndim != 3 or indices.ndim != 2 or hidden.size(0) != indices.size(0):
        raise ValueError(f"{label} indices must have shape [B,N]")
    if torch.any(indices < 0) or torch.any(indices >= hidden.size(1)):
        raise ValueError(f"{label} index is outside sequence length")
    gather = indices.long().unsqueeze(-1).expand(-1, -1, hidden.size(-1))
    return hidden.gather(1, gather)


def _masked_mean(hidden: Any, mask: Any | None) -> Any:
    """Mean-pool visible tokens; all-padding rows intentionally map to the zero vector."""
    _require_torch()
    if hidden.ndim != 3:
        raise ValueError("hidden states must have shape [B,T,H]")
    if mask is None:
        return hidden.mean(dim=1)
    if mask.shape != hidden.shape[:2]:
        raise ValueError("attention mask must align with hidden states")
    weights = mask.to(device=hidden.device, dtype=hidden.dtype)
    denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (hidden * weights.unsqueeze(-1)).sum(dim=1) / denominator


if nn is not None:

    class GroundedGeneratorTrainingModule(nn.Module):
        """Injected base LM plus claim/evidence grounded auxiliary heads.

        Evidence inputs use ``[B,E,L]``. Variable evidence counts are represented by
        all-zero attention masks on padded evidence slots. Those slots are safely passed
        through the base model with one temporary visible pad position, pooled back to zero,
        and masked out of citation logits so padding can never win citation attribution.
        """
        def __init__(self, *, base_model: nn.Module, config: GroundedGenerationArchitectureConfig, retriever_model: nn.Module | None = None) -> None:
            super().__init__()
            if not isinstance(base_model, nn.Module):
                raise ValueError("base_model must be an nn.Module")
            if not isinstance(config, GroundedGenerationArchitectureConfig):
                raise ValueError("config must be GroundedGenerationArchitectureConfig")
            if retriever_model is not None and not isinstance(retriever_model, nn.Module):
                raise ValueError("retriever_model must be an nn.Module or None")
            self.base_model = base_model
            self.auxiliary = GroundedAuxiliaryHeads(config)
            self.retriever_model = retriever_model
            self.config = config

        def _run_lm(self, **inputs: Any) -> tuple[Any, Any]:
            selected = dict(inputs)
            selected["output_hidden_states"] = True
            selected["return_dict"] = True
            return _lm_logits_and_hidden(self.base_model(**selected))

        def forward(self, *, input_ids: Any, attention_mask: Any | None, claim_token_indices: Any, generation_token_index: Any, evidence_input_ids: Any, evidence_attention_mask: Any | None = None, chosen_inputs: Mapping[str, Any] | None = None, rejected_inputs: Mapping[str, Any] | None = None, retriever_inputs: Mapping[str, Any] | None = None, **extra_lm_inputs: Any) -> Mapping[str, Any]:
            answer_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, **extra_lm_inputs}
            token_logits, hidden = self._run_lm(**answer_inputs)
            if hidden.size(-1) != self.config.hidden_size:
                raise ValueError("base LM hidden width does not match grounded architecture")
            claim_hidden = _gather_positions(hidden, claim_token_indices, label="claim token")
            if generation_token_index.ndim != 1 or generation_token_index.size(0) != hidden.size(0):
                raise ValueError("generation_token_index must have shape [B]")
            generation_hidden = _gather_positions(hidden, generation_token_index.unsqueeze(1), label="generation token").squeeze(1)

            if evidence_input_ids.ndim != 3:
                raise ValueError("evidence_input_ids must have shape [B,E,L]")
            batch, evidence_count, sequence = evidence_input_ids.shape
            if batch != hidden.size(0) or evidence_count < 1:
                raise ValueError("evidence inputs must align with answer batch and contain evidence")
            flat_ids = evidence_input_ids.reshape(batch * evidence_count, sequence)
            flat_mask = None
            slot_mask = torch.ones((batch, evidence_count), dtype=torch.bool, device=evidence_input_ids.device)
            safe_mask = None
            if evidence_attention_mask is not None:
                if evidence_attention_mask.shape != evidence_input_ids.shape:
                    raise ValueError("evidence_attention_mask must match evidence_input_ids")
                flat_mask = evidence_attention_mask.reshape(batch * evidence_count, sequence)
                slot_mask = evidence_attention_mask.to(dtype=torch.bool).any(dim=-1)
                safe_mask = flat_mask.clone()
                empty = ~safe_mask.to(dtype=torch.bool).any(dim=-1)
                if torch.any(empty):
                    safe_mask[empty, 0] = 1
            _, evidence_hidden_tokens = self._run_lm(input_ids=flat_ids, attention_mask=safe_mask if safe_mask is not None else flat_mask)
            evidence_hidden = _masked_mean(evidence_hidden_tokens, flat_mask).reshape(batch, evidence_count, -1)
            auxiliary = dict(self.auxiliary(claim_hidden, evidence_hidden, generation_hidden))
            citation_logits = auxiliary["citation_logits"]
            auxiliary["citation_logits"] = citation_logits.masked_fill(~slot_mask.unsqueeze(1), torch.finfo(citation_logits.dtype).min)
            auxiliary["evidence_slot_mask"] = slot_mask
            auxiliary["token_logits"] = token_logits

            if (chosen_inputs is None) != (rejected_inputs is None):
                raise ValueError("chosen_inputs and rejected_inputs must be supplied together")
            if chosen_inputs is not None:
                chosen_logits, _ = self._run_lm(**dict(chosen_inputs))
                rejected_logits, _ = self._run_lm(**dict(rejected_inputs or {}))
                auxiliary["chosen_logits"] = chosen_logits
                auxiliary["rejected_logits"] = rejected_logits

            if retriever_inputs is not None:
                if self.retriever_model is None:
                    raise ValueError("retriever_inputs were supplied but no retriever_model is configured")
                retriever_output = self.retriever_model(**dict(retriever_inputs))
                if isinstance(retriever_output, Mapping):
                    retriever_logits = retriever_output.get("logits")
                else:
                    retriever_logits = getattr(retriever_output, "logits", retriever_output)
                if retriever_logits is None or getattr(retriever_logits, "ndim", 0) != 2:
                    raise ValueError("retriever_model must expose [B,D] logits")
                auxiliary["retriever_logits"] = retriever_logits
            return auxiliary


    class DynamicRagPolicyModel(nn.Module):
        """Policy/value controller plus information-need token selector."""
        def __init__(self, config: DynamicPolicyArchitecture) -> None:
            super().__init__()
            if not isinstance(config, DynamicPolicyArchitecture):
                raise ValueError("config must be DynamicPolicyArchitecture")
            self.config = config
            self.controller = DynamicRetrievalController(config)
            self.need_selector = InformationNeedSelector(config)

        def forward(self, *, features: Any, token_hidden: Any | None = None, state_hidden: Any | None = None, attention_mask: Any | None = None) -> Mapping[str, Any]:
            result = dict(self.controller(features))
            if token_hidden is None and state_hidden is None:
                return result
            if token_hidden is None or state_hidden is None:
                raise ValueError("token_hidden and state_hidden must be supplied together")
            result["need_logits"] = self.need_selector(token_hidden, state_hidden, attention_mask)
            return result

else:
    class GroundedGeneratorTrainingModule:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()

    class DynamicRagPolicyModel:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


__all__ = ["DynamicRagPolicyModel", "GroundedGeneratorTrainingModule"]
