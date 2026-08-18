"""Strict collation and off-policy steps for advanced RAG training.

This module tightens two source-level contracts discovered while composing the complete
training path:

* logged behavior probabilities are converted into a current-policy importance ratio inside
  the differentiable policy step instead of being silently treated as one; and
* cached teacher/hidden-state tensors are shape-checked against the exact tokenized batch and
  architecture before they can enter optimization.

The original data/step modules remain reusable primitives; turnkey runners use these strict
adapters as the authoritative training path.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from training.advanced_rag_data import (
    DynamicRagEpisodeCollator,
    DynamicRagEpisodeStep,
    GroundedGenerationCollator,
    GroundedGenerationExample,
)
from training.advanced_rag_steps import DynamicPolicyStepConfig
from training.dynamic_retrieval_policy import (
    action_cost_expectations,
    action_imitation_loss,
    dynamic_policy_objective,
    information_need_bce_loss,
    offpolicy_policy_gradient_loss,
    retrieval_value_loss,
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


class StrictGroundedGenerationCollator(GroundedGenerationCollator):
    """Grounded collator with exact cache/batch compatibility checks."""

    def __call__(self, examples: Sequence[GroundedGenerationExample]) -> dict[str, Any]:
        batch = super().__call__(examples)
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


class StrictDynamicRagEpisodeCollator(DynamicRagEpisodeCollator):
    """Dynamic episode collator preserving logged behavior-policy probabilities."""

    def __call__(self, examples: Sequence[DynamicRagEpisodeStep]) -> dict[str, Any]:
        _require_torch()
        batch = super().__call__(examples)
        # The base collator predates the full off-policy composition and emitted an all-one
        # placeholder. Never let that placeholder reach the authoritative training path.
        batch.pop("importance_ratio", None)
        have_behavior = [item.behavior_action_probability is not None for item in examples]
        if any(have_behavior) and not all(have_behavior):
            raise ValueError("a dynamic batch may not mix logged and unlogged behavior probabilities")
        if all(have_behavior):
            batch["behavior_action_probability"] = torch.tensor(
                [float(item.behavior_action_probability) for item in examples], dtype=torch.float32
            )

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
        outputs = model(
            features=batch["features"],
            token_hidden=batch.get("token_hidden"),
            state_hidden=batch.get("state_hidden"),
            attention_mask=batch.get("attention_mask"),
        )
        if not isinstance(outputs, Mapping):
            raise ValueError("dynamic policy model must return a mapping")
        objective = self.config.objective
        action_logits = _extract(
            outputs,
            "action_logits",
            required=(
                objective.action_weight > 0.0
                or objective.policy_gradient_weight > 0.0
                or objective.retrieval_cost_weight > 0.0
                or objective.verification_cost_weight > 0.0
                or objective.abstention_cost_weight > 0.0
                or objective.entropy_bonus_weight > 0.0
            ),
        )
        losses: dict[str, Any | None] = {
            "action": None,
            "need_selection": None,
            "value": None,
            "policy_gradient": None,
            "retrieval_cost": None,
            "verification_cost": None,
            "abstention_cost": None,
            "entropy": None,
        }
        if objective.action_weight > 0.0:
            losses["action"] = action_imitation_loss(
                action_logits,
                batch["action_targets"],
                class_weights=batch.get("action_class_weights"),
                ignore_index=self.config.ignore_index,
            )
        if objective.need_selection_weight > 0.0:
            losses["need_selection"] = information_need_bce_loss(
                _extract(outputs, "need_logits", required=True),
                batch["need_target_mask"],
                valid_mask=batch.get("need_valid_mask"),
            )
        if objective.value_weight > 0.0:
            losses["value"] = retrieval_value_loss(
                _extract(outputs, "retrieval_value", required=True),
                batch["realized_retrieval_gain"],
                huber_delta=self.config.huber_delta,
            )
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
                if behavior.ndim != 1 or behavior.numel() != action_logits.size(0):
                    raise ValueError("behavior_action_probability must contain one probability per row")
                if torch.any(~torch.isfinite(behavior)) or torch.any(behavior <= 0.0) or torch.any(behavior > 1.0):
                    raise ValueError("behavior_action_probability must be finite and lie in (0,1]")
                current_probability = selected_log_probability.exp()
                # Treat the IS correction as a sampled weight, not an additional gradient path.
                importance_ratio = (current_probability.detach() / behavior).clamp_max(self.config.max_importance_ratio)
            elif batch.get("importance_ratio") is not None:
                # Explicit precomputed ratios remain supported for governed offline datasets.
                importance_ratio = batch["importance_ratio"].to(device=action_logits.device, dtype=action_logits.dtype)
            losses["policy_gradient"] = offpolicy_policy_gradient_loss(
                selected_log_probability,
                batch["advantage"],
                importance_ratio=importance_ratio,
                max_importance_ratio=self.config.max_importance_ratio,
            )

        if any(
            weight > 0.0
            for weight in (
                objective.retrieval_cost_weight,
                objective.verification_cost_weight,
                objective.abstention_cost_weight,
                objective.entropy_bonus_weight,
            )
        ):
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
]
