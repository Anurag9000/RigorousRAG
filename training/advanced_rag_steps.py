"""Executable step adapters for advanced grounded and dynamic RAG training.

The generic :mod:`training.torch_engine` already owns device placement, AMP, DDP,
optimizers, schedulers, gradient accumulation/clipping, early stopping and exact
checkpoint/resume.  This module provides task-specific ``BatchStep`` adapters and immutable
plan-to-engine mappings for the newer grounded-generator and generation-time retrieval
policy families.

Nothing runs on import.  No dataset/model is downloaded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional training dependency.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from training.dynamic_retrieval_policy import (
    DynamicPolicyObjective,
    DynamicPolicyTrainingPlan,
    action_cost_expectations,
    action_imitation_loss,
    dynamic_policy_objective,
    information_need_bce_loss,
    offpolicy_policy_gradient_loss,
    retrieval_value_loss,
)
from training.grounded_generation import (
    GroundedObjectiveWeights,
    GroundedTrainingPlan,
    binary_supervision_loss,
    citation_pointer_loss,
    dpo_grounded_preference_loss,
    grounded_generation_objective,
    lm_supervised_retriever_kl,
    masked_token_nll,
    reflection_action_loss,
    sequence_log_prob,
    teacher_token_distillation_kl,
    unsupported_mass_unlikelihood,
)
from training.torch_engine import StepResult, TrainerConfig, TrainingStageSpec


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("advanced RAG training steps require the optional PyTorch dependency")


def _metric(value: Any) -> float:
    return float(value.detach().float().cpu())


def _extract(outputs: Mapping[str, Any], key: str, *, required: bool) -> Any | None:
    value = outputs.get(key)
    if value is None and required:
        raise ValueError(f"training model output is missing required key: {key}")
    return value


@dataclass(frozen=True)
class GroundedStepConfig:
    objective: GroundedObjectiveWeights
    ignore_index: int = -100
    dpo_beta: float = 0.1
    distillation_temperature: float = 1.0
    retriever_temperature: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.objective, GroundedObjectiveWeights):
            raise ValueError("objective must be GroundedObjectiveWeights")
        if isinstance(self.ignore_index, bool) or not isinstance(self.ignore_index, int):
            raise ValueError("ignore_index must be an integer")
        for name in ("dpo_beta", "distillation_temperature", "retriever_temperature"):
            value = float(getattr(self, name))
            if not value > 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)


class GroundedGenerationStep:
    """One differentiable grounded-generator training step.

    Model contract
    --------------
    ``model(**batch['model_inputs'])`` returns a mapping.  Required keys are determined by
    nonzero objective weights and use the following names:

    ``token_logits``
        [B,T,V] language-model logits.
    ``citation_logits``
        [B,C,E] claim-to-evidence logits.
    ``support_logits`` / ``contradiction_logits``
        [B,C] claim-level binary logits.
    ``abstention_logits``
        [B] answer-level binary logits.
    ``reflection_logits``
        [B,A] closed reflection/action logits.
    ``chosen_logits`` / ``rejected_logits``
        [B,T,V] policy logits for preference pairs.
    ``retriever_logits``
        [B,D] document selection logits for LM-supervised retriever coupling.

    Teacher/reference quantities remain caller-supplied batch data and are detached by the
    corresponding losses.  Evidence/citation identities are never inferred by this step.
    """

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
        losses: dict[str, Any | None] = {
            "token_nll": None,
            "citation": None,
            "support": None,
            "contradiction": None,
            "abstention": None,
            "reflection": None,
            "unsupported_unlikelihood": None,
            "preference": None,
            "teacher_distillation": None,
            "retriever_coupling": None,
        }

        token_logits = _extract(outputs, "token_logits", required=weights.token_nll > 0.0 or weights.unsupported_unlikelihood > 0.0 or weights.teacher_distillation > 0.0)
        if weights.token_nll > 0.0:
            if "labels" not in batch:
                raise ValueError("token_nll objective requires labels")
            losses["token_nll"] = masked_token_nll(token_logits, batch["labels"], ignore_index=self.config.ignore_index)
        if weights.citation > 0.0:
            losses["citation"] = citation_pointer_loss(
                _extract(outputs, "citation_logits", required=True),
                batch["citation_targets"],
                ignore_index=self.config.ignore_index,
            )
        if weights.support > 0.0:
            losses["support"] = binary_supervision_loss(
                _extract(outputs, "support_logits", required=True),
                batch["support_targets"],
                mask=batch.get("claim_mask"),
            )
        if weights.contradiction > 0.0:
            losses["contradiction"] = binary_supervision_loss(
                _extract(outputs, "contradiction_logits", required=True),
                batch["contradiction_targets"],
                mask=batch.get("claim_mask"),
            )
        if weights.abstention > 0.0:
            losses["abstention"] = binary_supervision_loss(
                _extract(outputs, "abstention_logits", required=True),
                batch["abstention_targets"],
            )
        if weights.reflection > 0.0:
            losses["reflection"] = reflection_action_loss(
                _extract(outputs, "reflection_logits", required=True),
                batch["reflection_targets"],
                ignore_index=self.config.ignore_index,
            )
        if weights.unsupported_unlikelihood > 0.0:
            losses["unsupported_unlikelihood"] = unsupported_mass_unlikelihood(
                token_logits,
                batch["unsupported_token_mask"],
            )
        if weights.preference > 0.0:
            chosen_logits = _extract(outputs, "chosen_logits", required=True)
            rejected_logits = _extract(outputs, "rejected_logits", required=True)
            chosen_log_prob = sequence_log_prob(chosen_logits, batch["chosen_labels"], ignore_index=self.config.ignore_index)
            rejected_log_prob = sequence_log_prob(rejected_logits, batch["rejected_labels"], ignore_index=self.config.ignore_index)
            reference_chosen = batch["reference_chosen_log_prob"]
            reference_rejected = batch["reference_rejected_log_prob"]
            losses["preference"] = dpo_grounded_preference_loss(
                chosen_log_prob,
                rejected_log_prob,
                reference_chosen,
                reference_rejected,
                beta=self.config.dpo_beta,
            )
        if weights.teacher_distillation > 0.0:
            losses["teacher_distillation"] = teacher_token_distillation_kl(
                token_logits,
                batch["teacher_token_logits"],
                temperature=self.config.distillation_temperature,
            )
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


@dataclass(frozen=True)
class DynamicPolicyStepConfig:
    objective: DynamicPolicyObjective
    huber_delta: float = 1.0
    max_importance_ratio: float = 10.0
    ignore_index: int = -100

    def __post_init__(self) -> None:
        if not isinstance(self.objective, DynamicPolicyObjective):
            raise ValueError("objective must be DynamicPolicyObjective")
        for name in ("huber_delta", "max_importance_ratio"):
            value = float(getattr(self, name))
            if not value > 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if isinstance(self.ignore_index, bool) or not isinstance(self.ignore_index, int):
            raise ValueError("ignore_index must be an integer")


class DynamicRetrievalPolicyStep:
    """One training step for retrieve/continue/verify/abstain/stop policy learning.

    ``model(features=..., token_hidden=..., state_hidden=..., attention_mask=...)`` must
    return a mapping containing ``action_logits`` and, when weighted, ``retrieval_value`` and
    ``need_logits``.  A combined module can internally compose
    ``DynamicRetrievalController`` and ``InformationNeedSelector``.
    """

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
        action_logits = _extract(outputs, "action_logits", required=objective.action_weight > 0.0 or objective.policy_gradient_weight > 0.0 or objective.retrieval_cost_weight > 0.0 or objective.verification_cost_weight > 0.0 or objective.abstention_cost_weight > 0.0 or objective.entropy_bonus_weight > 0.0)
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
        if objective.policy_gradient_weight > 0.0:
            if action_logits is None:
                raise ValueError("policy-gradient objective requires action_logits")
            log_probability = F.log_softmax(action_logits, dim=-1)
            action_indices = batch["logged_action_indices"].long()
            if action_indices.ndim != 1 or action_indices.numel() != action_logits.size(0):
                raise ValueError("logged_action_indices must contain one action per row")
            if torch.any(action_indices < 0) or torch.any(action_indices >= action_logits.size(1)):
                raise ValueError("logged action index is outside action vocabulary")
            selected_log_probability = log_probability.gather(1, action_indices.unsqueeze(1)).squeeze(1)
            losses["policy_gradient"] = offpolicy_policy_gradient_loss(
                selected_log_probability,
                batch["advantage"],
                importance_ratio=batch.get("importance_ratio"),
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
        metrics["dynamic_total"] = _metric(breakdown.total)
        return StepResult(breakdown.total, metrics)


def grounded_plan_to_trainer_config(
    plan: GroundedTrainingPlan,
    *,
    device: str = "auto",
    precision: str = "fp32",
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float | None = 1.0,
    seed: int = 0,
    deterministic_algorithms: bool = False,
    ddp: bool = False,
    weight_decay: float = 0.01,
    scheduler: str = "linear",
    warmup_steps: int = 0,
) -> TrainerConfig:
    """Map an immutable grounded plan onto the generic checkpointed trainer."""

    if not isinstance(plan, GroundedTrainingPlan):
        raise ValueError("plan must be GroundedTrainingPlan")
    stages = tuple(
        TrainingStageSpec(
            name=stage.name,
            max_optimizer_steps=stage.max_steps,
            learning_rate=stage.learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            scheduler=scheduler,
            checkpoint_every_steps=stage.checkpoint_every_steps,
        )
        for stage in plan.stages
    )
    return TrainerConfig(
        run_id=plan.run_id,
        source_commit=plan.source_commit,
        dataset_manifest_digest=plan.dataset_manifest_sha256,
        model_architecture=f"grounded_generation:{plan.plan_sha256}",
        stages=stages,
        device=device,
        precision=precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        seed=seed,
        deterministic_algorithms=deterministic_algorithms,
        ddp=ddp,
    )


def dynamic_plan_to_trainer_config(
    plan: DynamicPolicyTrainingPlan,
    *,
    device: str = "auto",
    precision: str = "fp32",
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float | None = 1.0,
    seed: int = 0,
    deterministic_algorithms: bool = False,
    ddp: bool = False,
    weight_decay: float = 0.01,
    scheduler: str = "linear",
    warmup_steps: int = 0,
) -> TrainerConfig:
    """Map a dynamic retrieval policy plan onto the generic checkpointed trainer."""

    if not isinstance(plan, DynamicPolicyTrainingPlan):
        raise ValueError("plan must be DynamicPolicyTrainingPlan")
    stages = tuple(
        TrainingStageSpec(
            name=stage.name,
            max_optimizer_steps=stage.max_steps,
            learning_rate=stage.learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            scheduler=scheduler,
            checkpoint_every_steps=stage.checkpoint_every_steps,
        )
        for stage in plan.stages
    )
    return TrainerConfig(
        run_id=plan.run_id,
        source_commit=plan.source_commit,
        dataset_manifest_digest=plan.dataset_manifest_sha256,
        model_architecture=f"dynamic_retrieval_policy:{plan.plan_sha256}",
        stages=stages,
        device=device,
        precision=precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        seed=seed,
        deterministic_algorithms=deterministic_algorithms,
        ddp=ddp,
    )


__all__ = [
    "DynamicPolicyStepConfig",
    "DynamicRetrievalPolicyStep",
    "GroundedGenerationStep",
    "GroundedStepConfig",
    "dynamic_plan_to_trainer_config",
    "grounded_plan_to_trainer_config",
]
