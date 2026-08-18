"""Complete encoder-decoder grounded-RAG model and collator.

The original grounded wrapper is intentionally causal-LM friendly. This module provides the
corresponding source-complete seq2seq path: encoder prompts, decoder-target supervision,
decoder-state claim attribution, encoder-derived evidence representations, preference pairs,
teacher/reference caches and optional generator-retriever coupling. No model is loaded or
executed on import.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from training.advanced_rag_data import GroundedCollatorConfig, GroundedGenerationExample, RetrieverBatchBuilder, TensorCacheProvider
from training.grounded_generation import GroundedAuxiliaryHeads, GroundedGenerationArchitectureConfig, ReflectionAction


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("seq2seq grounded training requires optional PyTorch")


def _tokenize_offsets(tokenizer: Any, texts: Sequence[str], config: GroundedCollatorConfig) -> dict[str, Any]:
    encoded = dict(tokenizer(
        list(texts), padding=True, truncation=True, max_length=config.sequence_max_length,
        pad_to_multiple_of=config.pad_to_multiple_of, return_offsets_mapping=True,
        return_tensors="pt", add_special_tokens=True,
    ))
    if "input_ids" not in encoded or "attention_mask" not in encoded or "offset_mapping" not in encoded:
        raise ValueError("seq2seq tokenizer must expose input_ids, attention_mask and offset_mapping")
    return encoded


def _positions_for_span(offsets: Sequence[Sequence[int]], start: int, end: int) -> list[int]:
    return [index for index, pair in enumerate(offsets) if int(pair[1]) > int(pair[0]) and int(pair[0]) < end and int(pair[1]) > start]


def _decoder_labels(input_ids: Any, attention_mask: Any, ignore_index: int) -> Any:
    _require_torch()
    labels = input_ids.clone()
    labels = labels.masked_fill(~attention_mask.to(dtype=torch.bool), int(ignore_index))
    return labels


def _output_value(output: Any, key: str) -> Any | None:
    if isinstance(output, Mapping):
        return output.get(key)
    return getattr(output, key, None)


def _decoder_logits_hidden(output: Any) -> tuple[Any, Any]:
    logits = _output_value(output, "logits")
    hidden_states = _output_value(output, "decoder_hidden_states")
    if logits is None or hidden_states is None or not hidden_states:
        raise ValueError("seq2seq model must expose logits and decoder_hidden_states")
    hidden = hidden_states[-1]
    if logits.ndim != 3 or hidden.ndim != 3 or logits.shape[:2] != hidden.shape[:2]:
        raise ValueError("seq2seq decoder logits/hidden states must align on [B,T]")
    return logits, hidden


def _gather(hidden: Any, indices: Any, label: str) -> Any:
    _require_torch()
    if hidden.ndim != 3 or indices.ndim != 2 or hidden.size(0) != indices.size(0):
        raise ValueError(f"{label} indices must have shape [B,N]")
    if torch.any(indices < 0) or torch.any(indices >= hidden.size(1)):
        raise ValueError(f"{label} index lies outside decoder sequence")
    gather = indices.long().unsqueeze(-1).expand(-1, -1, hidden.size(-1))
    return hidden.gather(1, gather)


def _masked_mean(hidden: Any, mask: Any) -> Any:
    _require_torch()
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2]:
        raise ValueError("encoder hidden states and mask must align")
    weights = mask.to(device=hidden.device, dtype=hidden.dtype)
    denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (hidden * weights.unsqueeze(-1)).sum(dim=1) / denominator


class Seq2SeqGroundedGenerationCollator:
    """Build direct decoder-target grounded supervision for encoder-decoder models."""
    def __init__(self, tokenizer: Any, config: GroundedCollatorConfig = GroundedCollatorConfig(), *, teacher_cache: TensorCacheProvider | None = None, reference_cache: TensorCacheProvider | None = None, retriever_batch_builder: RetrieverBatchBuilder | None = None) -> None:
        self.tokenizer, self.config = tokenizer, config
        self.teacher_cache, self.reference_cache = teacher_cache, reference_cache
        self.retriever_batch_builder = retriever_batch_builder
        self._calls = 0

    def state_dict(self) -> dict[str, Any]:
        return {"schema": "rigorousrag-seq2seq-grounded-collator-state/v1", "calls": self._calls, "config": asdict(self.config)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != "rigorousrag-seq2seq-grounded-collator-state/v1" or state.get("config") != asdict(self.config):
            raise ValueError("seq2seq grounded collator checkpoint is incompatible")
        calls = state.get("calls")
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
            raise ValueError("seq2seq grounded collator call counter is invalid")
        self._calls = calls

    def _targets(self, answers: Sequence[str]) -> tuple[dict[str, Any], Any, list[list[list[int]]]]:
        encoded = _tokenize_offsets(self.tokenizer, answers, self.config)
        offsets = encoded.pop("offset_mapping").tolist()
        labels = _decoder_labels(encoded["input_ids"], encoded["attention_mask"], self.config.ignore_index)
        return encoded, labels, offsets

    def _preference_inputs(self, prompts: Sequence[str], answers: Sequence[str]) -> tuple[dict[str, Any], Any]:
        encoder = self.tokenizer(list(prompts), padding=True, truncation=True, max_length=self.config.sequence_max_length, pad_to_multiple_of=self.config.pad_to_multiple_of, return_tensors="pt", add_special_tokens=True)
        targets, labels, _ = self._targets(answers)
        return {"input_ids": encoder["input_ids"], "attention_mask": encoder["attention_mask"], "labels": labels}, labels

    def __call__(self, examples: Sequence[GroundedGenerationExample]) -> dict[str, Any]:
        _require_torch()
        if not examples or any(not isinstance(item, GroundedGenerationExample) for item in examples):
            raise ValueError("seq2seq grounded collator requires GroundedGenerationExample values")
        self._calls += 1
        prompts = [item.prompt for item in examples]
        answers = [item.answer for item in examples]
        encoder = self.tokenizer(prompts, padding=True, truncation=True, max_length=self.config.sequence_max_length, pad_to_multiple_of=self.config.pad_to_multiple_of, return_tensors="pt", add_special_tokens=True)
        target_encoded, labels, offsets = self._targets(answers)

        selected_evidence = [item.evidence[: self.config.evidence_limit] for item in examples]
        max_evidence = max(len(group) for group in selected_evidence)
        max_claims = max(1, min(self.config.claim_limit, max((len(item.claims) for item in examples), default=0)))
        claim_rows: list[list[int]] = []
        claim_masks: list[list[bool]] = []
        citation_rows: list[list[int]] = []
        support_rows: list[list[float]] = []
        contradiction_rows: list[list[float]] = []
        unsupported_rows: list[list[bool]] = []
        generation_indices: list[int] = []
        for row, example in enumerate(examples):
            valid = [index for index, pair in enumerate(offsets[row]) if bool(target_encoded["attention_mask"][row, index].item()) and (int(pair[1]) > int(pair[0]) or not example.answer)]
            if not valid:
                valid = [index for index in range(target_encoded["attention_mask"].size(1)) if bool(target_encoded["attention_mask"][row, index].item())]
            if not valid:
                raise ValueError("seq2seq target was entirely truncated")
            generation_index = max(valid)
            generation_indices.append(generation_index)
            evidence_index = {item.evidence_id: index for index, item in enumerate(selected_evidence[row])}
            cpos: list[int] = []
            cmask: list[bool] = []
            ctgt: list[int] = []
            sup: list[float] = []
            con: list[float] = []
            for claim in example.claims[:max_claims]:
                positions = _positions_for_span(offsets[row], claim.span.start, claim.span.end)
                if not positions:
                    raise ValueError("seq2seq claim annotation was truncated or has no decoder-token overlap")
                cpos.append(positions[-1]); cmask.append(True)
                available = sorted(evidence_index[eid] for eid in claim.evidence_ids if eid in evidence_index)
                ctgt.append(available[0] if available else self.config.ignore_index)
                sup.append(1.0 if claim.supported else 0.0); con.append(1.0 if claim.contradicted else 0.0)
            while len(cpos) < max_claims:
                cpos.append(generation_index); cmask.append(False); ctgt.append(self.config.ignore_index); sup.append(0.0); con.append(0.0)
            claim_rows.append(cpos); claim_masks.append(cmask); citation_rows.append(ctgt); support_rows.append(sup); contradiction_rows.append(con)
            unsupported = [False] * len(offsets[row])
            for span in example.unsupported_spans:
                positions = _positions_for_span(offsets[row], span.start, span.end)
                if not positions:
                    raise ValueError("seq2seq unsupported span was truncated")
                for position in positions:
                    unsupported[position] = True
            unsupported_rows.append(unsupported)

        flat_evidence = [item.text for group in selected_evidence for item in group]
        evidence_encoded = self.tokenizer(flat_evidence, padding=True, truncation=True, max_length=self.config.evidence_max_length, pad_to_multiple_of=self.config.pad_to_multiple_of, return_tensors="pt", add_special_tokens=True)
        evidence_length = evidence_encoded["input_ids"].size(1)
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        evidence_ids = torch.full((len(examples), max_evidence, evidence_length), int(pad_id), dtype=evidence_encoded["input_ids"].dtype)
        evidence_mask = torch.zeros((len(examples), max_evidence, evidence_length), dtype=evidence_encoded["attention_mask"].dtype)
        cursor = 0
        for row, group in enumerate(selected_evidence):
            count = len(group)
            evidence_ids[row, :count] = evidence_encoded["input_ids"][cursor:cursor + count]
            evidence_mask[row, :count] = evidence_encoded["attention_mask"][cursor:cursor + count]
            cursor += count

        model_inputs: dict[str, Any] = {
            "input_ids": encoder["input_ids"],
            "attention_mask": encoder["attention_mask"],
            "labels": labels,
            "claim_token_indices": torch.tensor(claim_rows, dtype=torch.long),
            "generation_token_index": torch.tensor(generation_indices, dtype=torch.long),
            "evidence_input_ids": evidence_ids,
            "evidence_attention_mask": evidence_mask,
        }
        batch: dict[str, Any] = {
            "example_ids": tuple(item.example_id for item in examples),
            "model_inputs": model_inputs,
            "labels": labels,
            "citation_targets": torch.tensor(citation_rows, dtype=torch.long),
            "support_targets": torch.tensor(support_rows, dtype=torch.float32),
            "contradiction_targets": torch.tensor(contradiction_rows, dtype=torch.float32),
            "claim_mask": torch.tensor(claim_masks, dtype=torch.bool),
            "abstention_targets": torch.tensor([1.0 if item.abstain else 0.0 for item in examples], dtype=torch.float32),
            "reflection_targets": torch.tensor([list(ReflectionAction).index(item.reflection_action) for item in examples], dtype=torch.long),
            "unsupported_token_mask": torch.tensor(unsupported_rows, dtype=torch.bool),
            "unsupported_target_alignment": "direct",
        }

        if any(item.chosen_answer is not None for item in examples):
            if not all(item.chosen_answer is not None and item.rejected_answer is not None for item in examples):
                raise ValueError("preference batches may not mix annotated and unannotated examples")
            chosen_inputs, chosen_labels = self._preference_inputs(prompts, [str(item.chosen_answer) for item in examples])
            rejected_inputs, rejected_labels = self._preference_inputs(prompts, [str(item.rejected_answer) for item in examples])
            model_inputs["chosen_inputs"] = chosen_inputs; model_inputs["rejected_inputs"] = rejected_inputs
            batch["chosen_labels"] = chosen_labels; batch["rejected_labels"] = rejected_labels
            if all(item.reference_chosen_log_prob is not None for item in examples):
                batch["reference_chosen_log_prob"] = torch.tensor([float(item.reference_chosen_log_prob) for item in examples], dtype=torch.float32)
                batch["reference_rejected_log_prob"] = torch.tensor([float(item.reference_rejected_log_prob) for item in examples], dtype=torch.float32)
            elif self.reference_cache is not None:
                chosen, rejected = [], []
                for item in examples:
                    cached = self.reference_cache.get(item.example_id)
                    c = cached.get("reference_chosen_log_prob"); r = cached.get("reference_rejected_log_prob")
                    if not torch.is_tensor(c) or not torch.is_tensor(r) or c.numel() != 1 or r.numel() != 1:
                        raise ValueError("seq2seq reference cache values must be scalar tensors")
                    chosen.append(float(c.item())); rejected.append(float(r.item()))
                batch["reference_chosen_log_prob"] = torch.tensor(chosen, dtype=torch.float32)
                batch["reference_rejected_log_prob"] = torch.tensor(rejected, dtype=torch.float32)

        if self.teacher_cache is not None:
            keys = [item.teacher_cache_key for item in examples]
            if all(key is not None for key in keys):
                teacher = torch.stack([self.teacher_cache.get(str(key))["teacher_token_logits"] for key in keys])
                if teacher.ndim != 3 or tuple(teacher.shape[:2]) != tuple(labels.shape):
                    raise ValueError("seq2seq teacher logits must align with decoder targets [B,T,V]")
                batch["teacher_token_logits"] = teacher

        if self.retriever_batch_builder is not None:
            retriever = dict(self.retriever_batch_builder(examples))
            if "model_inputs" not in retriever or "document_lm_log_likelihood" not in retriever:
                raise ValueError("retriever batch builder must return model_inputs and document_lm_log_likelihood")
            model_inputs["retriever_inputs"] = retriever["model_inputs"]
            batch["document_lm_log_likelihood"] = retriever["document_lm_log_likelihood"]
            if "retriever_candidate_mask" in retriever:
                batch["retriever_candidate_mask"] = retriever["retriever_candidate_mask"]
        return batch


if nn is not None:
    class Seq2SeqGroundedGeneratorTrainingModule(nn.Module):
        """Encoder-decoder LM plus grounded auxiliary heads and optional retriever."""
        def __init__(self, *, base_model: nn.Module, config: GroundedGenerationArchitectureConfig, retriever_model: nn.Module | None = None) -> None:
            super().__init__()
            if not isinstance(base_model, nn.Module):
                raise ValueError("base_model must be nn.Module")
            if not hasattr(base_model, "get_encoder"):
                raise ValueError("seq2seq grounded model requires base_model.get_encoder()")
            self.base_model, self.config, self.retriever_model = base_model, config, retriever_model
            self.auxiliary = GroundedAuxiliaryHeads(config)

        def _forward_decoder(self, **inputs: Any) -> tuple[Any, Any]:
            selected = dict(inputs); selected["output_hidden_states"] = True; selected["return_dict"] = True
            return _decoder_logits_hidden(self.base_model(**selected))

        def _encode_evidence(self, evidence_input_ids: Any, evidence_attention_mask: Any) -> tuple[Any, Any]:
            batch, evidence_count, length = evidence_input_ids.shape
            flat_ids = evidence_input_ids.reshape(batch * evidence_count, length)
            flat_mask = evidence_attention_mask.reshape(batch * evidence_count, length)
            slot_mask = evidence_attention_mask.to(dtype=torch.bool).any(dim=-1)
            safe_mask = flat_mask.clone()
            empty = ~safe_mask.to(dtype=torch.bool).any(dim=-1)
            if torch.any(empty):
                safe_mask[empty, 0] = 1
            encoder = self.base_model.get_encoder()
            output = encoder(input_ids=flat_ids, attention_mask=safe_mask, return_dict=True)
            hidden = _output_value(output, "last_hidden_state")
            if hidden is None:
                raise ValueError("seq2seq encoder does not expose last_hidden_state")
            pooled = _masked_mean(hidden, flat_mask).reshape(batch, evidence_count, -1)
            return pooled, slot_mask

        def forward(self, *, input_ids: Any, attention_mask: Any, labels: Any, claim_token_indices: Any, generation_token_index: Any, evidence_input_ids: Any, evidence_attention_mask: Any, chosen_inputs: Mapping[str, Any] | None = None, rejected_inputs: Mapping[str, Any] | None = None, retriever_inputs: Mapping[str, Any] | None = None, **_: Any) -> Mapping[str, Any]:
            token_logits, decoder_hidden = self._forward_decoder(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            if decoder_hidden.size(-1) != self.config.hidden_size:
                raise ValueError("seq2seq decoder hidden width differs from grounded architecture")
            claim_hidden = _gather(decoder_hidden, claim_token_indices, "claim token")
            if generation_token_index.ndim != 1 or generation_token_index.size(0) != decoder_hidden.size(0):
                raise ValueError("generation_token_index must have shape [B]")
            generation_hidden = _gather(decoder_hidden, generation_token_index.unsqueeze(1), "generation token").squeeze(1)
            evidence_hidden, slot_mask = self._encode_evidence(evidence_input_ids, evidence_attention_mask)
            auxiliary = dict(self.auxiliary(claim_hidden, evidence_hidden, generation_hidden))
            citation_logits = auxiliary["citation_logits"]
            auxiliary["citation_logits"] = citation_logits.masked_fill(~slot_mask.unsqueeze(1), torch.finfo(citation_logits.dtype).min)
            auxiliary["evidence_slot_mask"] = slot_mask
            auxiliary["token_logits"] = token_logits
            if (chosen_inputs is None) != (rejected_inputs is None):
                raise ValueError("chosen_inputs and rejected_inputs must be supplied together")
            if chosen_inputs is not None:
                chosen_logits, _ = self._forward_decoder(**dict(chosen_inputs))
                rejected_logits, _ = self._forward_decoder(**dict(rejected_inputs or {}))
                auxiliary["chosen_logits"] = chosen_logits; auxiliary["rejected_logits"] = rejected_logits
            if retriever_inputs is not None:
                if self.retriever_model is None:
                    raise ValueError("retriever_inputs supplied without retriever_model")
                output = self.retriever_model(**dict(retriever_inputs))
                logits = output.get("logits") if isinstance(output, Mapping) else getattr(output, "logits", output)
                if logits is None or getattr(logits, "ndim", 0) != 2:
                    raise ValueError("retriever model must expose [B,D] logits")
                auxiliary["retriever_logits"] = logits
            return auxiliary
else:
    class Seq2SeqGroundedGeneratorTrainingModule:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


__all__ = ["Seq2SeqGroundedGenerationCollator", "Seq2SeqGroundedGeneratorTrainingModule"]
