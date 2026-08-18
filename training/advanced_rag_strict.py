"""Strict collation and mathematically aligned steps for advanced RAG training.

The generic research primitives intentionally expose low-level tensor contracts. This module
is the authoritative turnkey-training layer and tightens the cross-module contracts found
while composing complete data→model→loss paths:

* current-policy/behavior-policy importance ratios are computed from logged probabilities;
* cached teacher/hidden-state tensors are shape checked against the exact batch; and
* unsupported-answer annotations are gathered at their actual target-token probabilities
  instead of being confused with an impractical full-vocabulary mask.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from training.advanced_rag_data import DynamicRagEpisodeCollator, DynamicRagEpisodeStep, GroundedGenerationCollator, GroundedGenerationExample
from training.advanced_rag_steps import DynamicPolicyStepConfig, GroundedStepConfig
from training.dynamic_retrieval_policy import action_cost_expectations, action_imitation_loss, dynamic_policy_objective, information_need_bce_loss, offpolicy_policy_gradient_loss, retrieval_value_loss
from training.grounded_generation import (
    binary_supervision_loss,
    dpo_grounded_preference_loss,
    grounded_generation_objective,
    lm_supervised_retriever_kl,
    masked_token_nll,
    reflection_action_loss,
    sequence_log_prob,
    teacher_token_distillation_kl,
)
from training.torch_engine import StepResult


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("strict advanced RAG training requires optional PyTorch")


def _metric(value: Any) -> float:
    return float(value.detach().float().cpu())


def _extract(outputs: Mapping[str, Any], key: str, *, required: bool) -> Any | None:
    value = outputs.get(key)
    if value is None and required:
        raise ValueError(f"training model output is missing required key: {key}")
    return value


def _safe_citation_pointer_loss(citation_logits: Any, targets: Any, *, ignore_index: int) -> Any:
    _require_torch()
    if citation_logits.ndim != 3 or targets.ndim != 2 or citation_logits.shape[:2] != targets.shape:
        raise ValueError("citation logits/targets must have shapes [B,C,E] and [B,C]")
    active = targets.ne(ignore_index)
    if not bool(active.any().detach().item()):
        return citation_logits.sum() * 0.0
    return F.cross_entropy(citation_logits.reshape(-1, citation_logits.size(-1)), targets.reshape(-1).long(), ignore_index=ignore_index)


def unsupported_target_token_unlikelihood(token_logits: Any, labels: Any, unsupported_token_mask: Any, *, alignment: str, ignore_index: int, eps: float = 1e-6) -> Any:
    """Penalize the probability of the exact annotated unsupported target tokens.

    Causal collation stores unsupported spans at their observed token positions while its
    labels are next-token shifted one position left. Seq2seq collation uses direct decoder
    target alignment. The alignment flag makes that distinction explicit without allocating
    a ``[B,T,V]`` one-hot mask.
    """
    _require_torch()
    if token_logits.ndim != 3 or labels.ndim != 2 or unsupported_token_mask.ndim != 2 or token_logits.shape[:2] != labels.shape or labels.shape != unsupported_token_mask.shape:
        raise ValueError("unsupported target supervision requires aligned [B,T,V]/[B,T]/[B,T] tensors")
    observed = unsupported_token_mask.to(dtype=torch.bool)
    if alignment == "causal_next_token":
        selected = torch.zeros_like(observed)
        if selected.size(1) > 1:
            selected[:, :-1] = observed[:, 1:]
    elif alignment == "direct":
        selected = observed
    else:
        raise ValueError("unsupported_target_alignment must be causal_next_token or direct")
    selected = selected & labels.ne(ignore_index)
    if not bool(selected.any().detach().item()):
        return token_logits.sum() * 0.0
    safe_labels = labels.long().masked_fill(~selected, 0)
    probability = F.softmax(token_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    epsilon = float(eps)
    if not 0.0 < epsilon < 0.1:
        raise ValueError("eps must lie in (0,0.1)")
    return (-torch.log(torch.clamp(1.0 - probability, min=epsilon, max=1.0)))[selected].mean()


class StrictGroundedGenerationCollator(GroundedGenerationCollator):
    """Grounded collator with exact cache/batch compatibility checks."""
    def __call__(self, examples: Sequence[GroundedGenerationExample]) -> dict[str, Any]:
        batch = super().__call__(examples)
        batch["unsupported_target_alignment"] = "causal_next_token"
        labels = batch["labels"]
        teacher = batch.get("teacher_token_logits")
        if teacher is not None:
            if getattr(teacher, "ndim", 0) != 3:
                raise ValueError("teacher_token_logits must have shape [B,T,V]")
            if tuple(teacher.shape[:2]) != tuple(labels.shape):
                raise ValueError("teacher logits must align exactly with the tokenized student batch [B,T]")
            if teacher.size(-1) <= 1:
                raise ValueError("teacher logits require a non-trivial vocabulary dimension")
        document_utility = batch.get("document_lm_log_likelihood")
        if document_utility is not None:
            if getattr(document_utility, "ndim", 0) != 2 or document_utility.size(0) != labels.size(0):
                raise ValueError("document_lm_log_likelihood must have shape [B,D]")
            candidate_mask = batch.get("retriever_candidate_mask")
            if candidate_mask is not None and tuple(candidate_mask.shape) != tuple(document_utility.shape):
                raise ValueError("retriever_candidate_mask must match document_lm_log_likelihood [B,D]")
        return batch


class StrictGroundedGenerationStep:
    """Grounded step with target-aligned unsupported-content unlikelihood."""
    def __init__(self, config: GroundedStepConfig) -> None:
        self.config = config

    def __call__(self, model: Any, batch: Mapping[str, Any]) -> StepResult:
        _require_torch()
        if "model_inputs" not in batch or not isinstance(batch["model_inputs"], Mapping):
            raise ValueError("grounded batch requires model_inputs mapping")
        outputs = model(**batch["model_inputs"])
        if not isinstance(outputs, Mapping):
            raise ValueError("grounded training model must return a mapping")
        weights = self.config.objective
        losses: dict[str, Any | None] = {name: None for name in (
            "token_nll", "citation", "support", "contradiction", "abstention", "reflection",
            "unsupported_unlikelihood", "preference", "teacher_distillation", "retriever_coupling",
        )}
        token_logits = _extract(outputs, "token_logits", required=weights.token_nll > 0.0 or weights.unsupported_unlikelihood > 0.0 or weights.teacher_distillation > 0.0)
        if weights.token_nll > 0.0:
            losses["token_nll"] = masked_token_nll(token_logits, batch["labels"], ignore_index=self.config.ignore_index)
        if weights.citation > 0.0:
            losses["citation"] = _safe_citation_pointer_loss(_extract(outputs, "citation_logits", required=True), batch["citation_targets"], ignore_index=self.config.ignore_index)
        if weights.support > 0.0:
            losses["support"] = binary_supervision_loss(_extract(outputs, "support_logits", required=True), batch["support_targets"], mask=batch.get("claim_mask"))
        if weights.contradiction > 0.0:
            losses["contradiction"] = binary_supervision_loss(_extract(outputs, "contradiction_logits", required=True), batch["contradiction_targets"], mask=batch.get("claim_mask"))
        if weights.abstention > 0.0:
            losses["abstention"] = binary_supervision_loss(_extract(outputs, "abstention_logits", required=True), batch["abstention_targets"])
        if weights.reflection > 0.0:
            losses["reflection"] = reflection_action_loss(_extract(outputs, "reflection_logits", required=True), batch["reflection_targets"], ignore_index=self.config.ignore_index)
        if weights.unsupported_unlikelihood > 0.0:
            losses["unsupported_unlikelihood"] = unsupported_target_token_unlikelihood(
                token_logits,
                batch["labels"],
                batch["unsupported_token_mask"],
                alignment=str(batch.get("unsupported_target_alignment", "causal_next_token")),
                ignore_index=self.config.ignore_index,
            )
        if weights.preference > 0.0:
            chosen_log_prob = sequence_log_prob(_extract(outputs, "chosen_logits", required=True), batch["chosen_labels"], ignore_index=self.config.ignore_index)
            rejected_log_prob = sequence_log_prob(_extract(outputs, "rejected_logits", required=True), batch["rejected_labels"], ignore_index=self.config.ignore_index)
            losses["preference"] = dpo_grounded_preference_loss(chosen_log_prob, rejected_log_prob, batch["reference_chosen_log_prob"], batch["reference_rejected_log_prob"], beta=self.config.dpo_beta)
        if weights.teacher_distillation > 0.0:
            losses["teacher_distillation"] = teacher_token_distillation_kl(token_logits, batch["teacher_token_logits"], temperature=self.config.distillation_temperature)
        if weights.retriever_coupling > 0.0:
            losses["retriever_coupling"] = lm_supervised_retriever_kl(
                _extract(outputs, "retriever_logits", required=True),
                batch["document_lm_log_likelihood"],
                temperature=self.config.retriever_temperature,
                candidate_mask=batch.get("retriever_candidate_mask"),
            )
        breakdown = grounded_generation_objective(weights=weights, **losses)
        metrics = {f"grounded_{name}": _metric(value) for name, value in losses.items() if value is not None}
        metrics["grounded_total"] = _metric(breakdown.total)
        return StepResult(breakdown.total, metrics)


class StrictDynamicRagEpisodeCollator(DynamicRagEpisodeCollator):
    """Dynamic episode collator preserving logged behavior-policy probabilities."""
    def __call__(self, examples: Sequence[DynamicRagEpisodeStep]) -> dict[str, Any]:
        _require_torch()
        batch = super().__call__(examples)
        batch.pop("importance_ratio", None)
        have_behavior = [item.behavior_action_probability is not None for item in examples]
        if any(have_behavior) and not all(have_behavior):
            raise ValueError("a dynamic batch may not mix logged and unlogged behavior probabilities")
        if all(have_behavior):
            batch["behavior_action_probability"] = torch.tensor([float(item.behavior_action_probability) for item in examples], dtype=torch.float32)
        token_hidden = batch.get("token_hidden")
        state_hidden = batch.get("state_hidden")
        if (token_hidden is None) != (state_hidden is None):
            raise ValueError("cached token_hidden and state_hidden must be supplied together")
        if token_hidden is not None:
            expected_hidden = self.architecture.context_hidden_size
            if token_hidden.ndim != 3 or state_hidden.ndim != 2:
                raise ValueError("cached dynamic hidden states must have shapes [B,T,H] and [B,H]")
            if token_hidden.size(0) != len(examples) or state_hidden.size(0) != len(examples):
                raise ValueError("cached hidden-state batch dimension does not match examples")
            if token_hidden.size(-1) != expected_hidden or state_hidden.size(-1) != expected_hidden:
                raise ValueError("cached hidden-state width differs from DynamicPolicyArchitecture.context_hidden_size")
            attention_mask = batch.get("attention_mask")
            if attention_mask is None or tuple(attention_mask.shape) != tuple(token_hidden.shape[:2]):
                raise ValueError("cached token_hidden sequence length must match tokenized attention_mask")
        return batch


class StrictDynamicRetrievalPolicyStep:
    """Dynamic policy step with mathematically correct logged-policy importance weighting."""
    def __init__(self, config: DynamicPolicyStepConfig, *, actions: tuple[Any, ...]) -> None:
        self.config = config
        self.actions = tuple(actions)
        if not self.actions:
            raise ValueError("dynamic policy action vocabulary is required")

    def __call__(self, model: Any, batch: Mapping[str, Any]) -> StepResult:
        _require_torch()
        outputs = model(features=batch["features"], token_hidden=batch.get("token_hidden"), state_hidden=batch.get("state_hidden"), attention_mask=batch.get("attention_mask"))
        if not isinstance(outputs, Mapping):
            raise ValueError("dynamic policy model must return a mapping")
        objective = self.config.objective
        action_logits = _extract(outputs, "action_logits", required=(objective.action_weight > 0.0 or objective.policy_gradient_weight > 0.0 or objective.retrieval_cost_weight > 0.0 or objective.verification_cost_weight > 0.0 or objective.abstention_cost_weight > 0.0 or objective.entropy_bonus_weight > 0.0))
        losses: dict[str, Any | None] = {name: None for name in ("action", "need_selection", "value", "policy_gradient", "retrieval_cost", "verification_cost", "abstention_cost", "entropy")}
        if objective.action_weight > 0.0:
            losses["action"] = action_imitation_loss(action_logits, batch["action_targets"], class_weights=batch.get("action_class_weights"), ignore_index=self.config.ignore_index)
        if objective.need_selection_weight > 0.0:
            losses["need_selection"] = information_need_bce_loss(_extract(outputs, "need_logits", required=True), batch["need_target_mask"], valid_mask=batch.get("need_valid_mask"))
        if objective.value_weight > 0.0:
            losses["value"] = retrieval_value_loss(_extract(outputs, "retrieval_value", required=True), batch["realized_retrieval_gain"], huber_delta=self.config.huber_delta)
        importance_ratio = None
        if objective.policy_gradient_weight > 0.0:
            if action_logits is None:
                raise ValueError("policy-gradient objective requires action_logits")
            action_indices = batch["logged_action_indices"].long()
            if action_indices.ndim != 1 or action_indices.numel() != action_logits.size(0):
                raise ValueError("logged_action_indices must contain one action per row")
            if torch.any(action_indices < 0) or torch.any(action_indices >= action_logits.size(1)):
                raise ValueError("logged action index is outside action vocabulary")
            log_probability = F.log_softmax(action_logits, dim=-1)
            selected_log_probability = log_probability.gather(1, action_indices.unsqueeze(1)).squeeze(1)
            behavior = batch.get("behavior_action_probability")
            if behavior is not None:
                behavior = behavior.to(device=action_logits.device, dtype=action_logits.dtype)
                if behavior.ndim != 1 or behavior.numel() != action_logits.size(0) or torch.any(~torch.isfinite(behavior)) or torch.any(behavior <= 0.0) or torch.any(behavior > 1.0):
                    raise ValueError("behavior_action_probability must provide one finite probability in (0,1] per row")
                importance_ratio = (selected_log_probability.exp().detach() / behavior).clamp_max(self.config.max_importance_ratio)
            elif batch.get("importance_ratio") is not None:
                importance_ratio = batch["importance_ratio"].to(device=action_logits.device, dtype=action_logits.dtype)
            losses["policy_gradient"] = offpolicy_policy_gradient_loss(selected_log_probability, batch["advantage"], importance_ratio=importance_ratio, max_importance_ratio=self.config.max_importance_ratio)
        if any(weight > 0.0 for weight in (objective.retrieval_cost_weight, objective.verification_cost_weight, objective.abstention_cost_weight, objective.entropy_bonus_weight)):
            costs = action_cost_expectations(action_logits, actions=self.actions)
            for name in ("retrieval_cost", "verification_cost", "abstention_cost", "entropy"):
                losses[name] = costs[name]
        breakdown = dynamic_policy_objective(objective=objective, **losses)
        metrics = {f"dynamic_{name}": _metric(value) for name, value in losses.items() if value is not None}
        if importance_ratio is not None:
            metrics["dynamic_importance_ratio_mean"] = _metric(importance_ratio.mean())
            metrics["dynamic_importance_ratio_max"] = _metric(importance_ratio.max())
        metrics["dynamic_total"] = _metric(breakdown.total)
        return StepResult(breakdown.total, metrics)


__all__ = [
    "StrictDynamicRagEpisodeCollator",
    "StrictDynamicRetrievalPolicyStep",
    "StrictGroundedGenerationCollator",
    "StrictGroundedGenerationStep",
    "unsupported_target_token_unlikelihood",
]
