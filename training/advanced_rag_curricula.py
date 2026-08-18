"""Canonical staged curricula for grounded-generator and dynamic-RAG policy learning.

The builders encode the repository's recommended methodology as immutable training plans plus
explicit parameter trainability policies. Dataset/model identities are supplied by the
operator; no mutable external name is embedded in the curriculum.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from training.advanced_rag_runner import ParameterTrainabilityPolicy
from training.dynamic_retrieval_policy import (
    DynamicPolicyArchitecture,
    DynamicPolicyObjective,
    DynamicPolicyStage,
    DynamicPolicyTrainingPlan,
    DynamicRetrievalBudget,
    DynamicRetrievalStageKind,
)
from training.grounded_generation import (
    GroundedGenerationArchitectureConfig,
    GroundedObjectiveWeights,
    GroundedStageKind,
    GroundedTrainingPlan,
    GroundedTrainingStage,
)


@dataclass(frozen=True)
class CurriculumStageHyperparameters:
    max_steps: int
    checkpoint_every_steps: int
    learning_rate: float

    def __post_init__(self) -> None:
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if isinstance(self.checkpoint_every_steps, bool) or not isinstance(self.checkpoint_every_steps, int) or not 1 <= self.checkpoint_every_steps <= self.max_steps:
            raise ValueError("checkpoint_every_steps must lie in [1,max_steps]")
        if isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float)) or not 0.0 < float(self.learning_rate) <= 10.0:
            raise ValueError("learning_rate must be positive and bounded")
        object.__setattr__(self, "learning_rate", float(self.learning_rate))


@dataclass(frozen=True)
class GroundedCurriculumHyperparameters:
    supervised: CurriculumStageHyperparameters = CurriculumStageHyperparameters(10_000, 1_000, 2e-5)
    attribution: CurriculumStageHyperparameters = CurriculumStageHyperparameters(6_000, 500, 1e-4)
    grounding: CurriculumStageHyperparameters = CurriculumStageHyperparameters(8_000, 500, 1e-4)
    reflection: CurriculumStageHyperparameters = CurriculumStageHyperparameters(5_000, 500, 1e-4)
    retriever_coupling: CurriculumStageHyperparameters = CurriculumStageHyperparameters(6_000, 500, 5e-5)
    preference: CurriculumStageHyperparameters = CurriculumStageHyperparameters(5_000, 500, 5e-6)
    joint: CurriculumStageHyperparameters = CurriculumStageHyperparameters(8_000, 500, 1e-5)


@dataclass(frozen=True)
class DynamicCurriculumHyperparameters:
    imitation: CurriculumStageHyperparameters = CurriculumStageHyperparameters(8_000, 500, 3e-4)
    need_selection: CurriculumStageHyperparameters = CurriculumStageHyperparameters(6_000, 500, 2e-4)
    value: CurriculumStageHyperparameters = CurriculumStageHyperparameters(6_000, 500, 2e-4)
    off_policy: CurriculumStageHyperparameters = CurriculumStageHyperparameters(8_000, 500, 1e-4)
    cost_aware: CurriculumStageHyperparameters = CurriculumStageHyperparameters(5_000, 500, 1e-4)
    joint: CurriculumStageHyperparameters = CurriculumStageHyperparameters(8_000, 500, 5e-5)


def _grounded_stage(name: str, kind: GroundedStageKind, hyper: CurriculumStageHyperparameters, objective: GroundedObjectiveWeights) -> GroundedTrainingStage:
    return GroundedTrainingStage(name=name, kind=kind, max_steps=hyper.max_steps, checkpoint_every_steps=hyper.checkpoint_every_steps, learning_rate=hyper.learning_rate, objective=objective)


def _dynamic_stage(name: str, kind: DynamicRetrievalStageKind, hyper: CurriculumStageHyperparameters, objective: DynamicPolicyObjective) -> DynamicPolicyStage:
    return DynamicPolicyStage(name=name, kind=kind, max_steps=hyper.max_steps, checkpoint_every_steps=hyper.checkpoint_every_steps, learning_rate=hyper.learning_rate, objective=objective)


def build_grounded_curriculum(
    *,
    run_id: str,
    architecture: GroundedGenerationArchitectureConfig,
    base_model_sha256: str,
    tokenizer_sha256: str,
    dataset_manifest_sha256: str,
    source_commit: str,
    retriever_stack_sha256: str | None = None,
    teacher_model_sha256: str | None = None,
    include_preference: bool = True,
    hyperparameters: GroundedCurriculumHyperparameters = GroundedCurriculumHyperparameters(),
) -> tuple[GroundedTrainingPlan, Mapping[str, ParameterTrainabilityPolicy]]:
    """Build the canonical SFT→attribution→grounding→reflection→coupling→preference→joint plan."""
    stages = [
        _grounded_stage("supervised_generation", GroundedStageKind.SUPERVISED, hyperparameters.supervised, GroundedObjectiveWeights(token_nll=1.0)),
        _grounded_stage("citation_attribution", GroundedStageKind.ATTRIBUTION, hyperparameters.attribution, GroundedObjectiveWeights(token_nll=0.20, citation=1.0)),
        _grounded_stage("semantic_grounding", GroundedStageKind.GROUNDING, hyperparameters.grounding, GroundedObjectiveWeights(token_nll=0.20, support=1.0, contradiction=1.0, unsupported_unlikelihood=0.50)),
        _grounded_stage("reflection_abstention", GroundedStageKind.REFLECTION, hyperparameters.reflection, GroundedObjectiveWeights(token_nll=0.10, abstention=1.0, reflection=1.0)),
    ]
    trainability: dict[str, ParameterTrainabilityPolicy] = {
        "supervised_generation": ParameterTrainabilityPolicy(("base_model",)),
        "citation_attribution": ParameterTrainabilityPolicy(("auxiliary",)),
        "semantic_grounding": ParameterTrainabilityPolicy(("auxiliary",)),
        "reflection_abstention": ParameterTrainabilityPolicy(("auxiliary",)),
    }
    if retriever_stack_sha256 is not None:
        stages.append(_grounded_stage("generator_retriever_coupling", GroundedStageKind.RETRIEVER_COUPLING, hyperparameters.retriever_coupling, GroundedObjectiveWeights(token_nll=0.10, retriever_coupling=1.0)))
        trainability["generator_retriever_coupling"] = ParameterTrainabilityPolicy(("retriever_model",))
    if include_preference:
        distillation = 0.25 if teacher_model_sha256 is not None else 0.0
        stages.append(_grounded_stage("grounded_preference", GroundedStageKind.PREFERENCE, hyperparameters.preference, GroundedObjectiveWeights(token_nll=0.10, preference=1.0, teacher_distillation=distillation)))
        trainability["grounded_preference"] = ParameterTrainabilityPolicy(("base_model",))
    stages.append(
        _grounded_stage(
            "joint_grounded_rag",
            GroundedStageKind.JOINT,
            hyperparameters.joint,
            GroundedObjectiveWeights(
                token_nll=1.0,
                citation=0.75,
                support=0.50,
                contradiction=0.50,
                abstention=0.25,
                reflection=0.25,
                unsupported_unlikelihood=0.25,
                preference=0.25 if include_preference else 0.0,
                teacher_distillation=0.10 if teacher_model_sha256 is not None else 0.0,
                retriever_coupling=0.25 if retriever_stack_sha256 is not None else 0.0,
            ),
        )
    )
    trainability["joint_grounded_rag"] = ParameterTrainabilityPolicy(())
    plan = GroundedTrainingPlan(
        run_id=run_id,
        architecture=architecture,
        base_model_sha256=base_model_sha256,
        tokenizer_sha256=tokenizer_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        source_commit=source_commit,
        stages=tuple(stages),
        retriever_stack_sha256=retriever_stack_sha256,
        teacher_model_sha256=teacher_model_sha256,
    )
    return plan, trainability


def build_dynamic_curriculum(
    *,
    run_id: str,
    architecture: DynamicPolicyArchitecture,
    budget: DynamicRetrievalBudget,
    dataset_manifest_sha256: str,
    base_generator_sha256: str,
    retrieval_stack_sha256: str,
    source_commit: str,
    include_need_selection: bool = True,
    hyperparameters: DynamicCurriculumHyperparameters = DynamicCurriculumHyperparameters(),
) -> tuple[DynamicPolicyTrainingPlan, Mapping[str, ParameterTrainabilityPolicy]]:
    """Build imitation→need-selection→value→off-policy→cost-aware→joint policy learning."""
    stages = [
        _dynamic_stage("action_imitation", DynamicRetrievalStageKind.IMITATION, hyperparameters.imitation, DynamicPolicyObjective(action_weight=1.0)),
    ]
    trainability: dict[str, ParameterTrainabilityPolicy] = {
        "action_imitation": ParameterTrainabilityPolicy(("controller",)),
    }
    if include_need_selection:
        stages.append(_dynamic_stage("information_need_selection", DynamicRetrievalStageKind.NEED_SELECTION, hyperparameters.need_selection, DynamicPolicyObjective(action_weight=0.0, need_selection_weight=1.0)))
        trainability["information_need_selection"] = ParameterTrainabilityPolicy(("need_selector",))
    stages.extend([
        _dynamic_stage("retrieval_value", DynamicRetrievalStageKind.VALUE, hyperparameters.value, DynamicPolicyObjective(action_weight=0.0, value_weight=1.0)),
        _dynamic_stage("off_policy_control", DynamicRetrievalStageKind.OFF_POLICY, hyperparameters.off_policy, DynamicPolicyObjective(action_weight=0.10, value_weight=0.25, policy_gradient_weight=1.0)),
        _dynamic_stage("cost_aware_control", DynamicRetrievalStageKind.JOINT, hyperparameters.cost_aware, DynamicPolicyObjective(action_weight=0.25, value_weight=0.25, policy_gradient_weight=1.0, retrieval_cost_weight=0.20, verification_cost_weight=0.10, abstention_cost_weight=0.05, entropy_bonus_weight=0.01)),
    ])
    trainability["retrieval_value"] = ParameterTrainabilityPolicy(("controller",))
    trainability["off_policy_control"] = ParameterTrainabilityPolicy(("controller",))
    trainability["cost_aware_control"] = ParameterTrainabilityPolicy(("controller",))
    stages.append(
        _dynamic_stage(
            "joint_dynamic_rag",
            DynamicRetrievalStageKind.JOINT,
            hyperparameters.joint,
            DynamicPolicyObjective(
                action_weight=0.50,
                need_selection_weight=0.25 if include_need_selection else 0.0,
                value_weight=0.25,
                policy_gradient_weight=1.0,
                retrieval_cost_weight=0.15,
                verification_cost_weight=0.10,
                abstention_cost_weight=0.05,
                entropy_bonus_weight=0.01,
            ),
        )
    )
    trainability["joint_dynamic_rag"] = ParameterTrainabilityPolicy(())
    plan = DynamicPolicyTrainingPlan(
        run_id=run_id,
        architecture=architecture,
        budget=budget,
        dataset_manifest_sha256=dataset_manifest_sha256,
        base_generator_sha256=base_generator_sha256,
        retrieval_stack_sha256=retrieval_stack_sha256,
        source_commit=source_commit,
        stages=tuple(stages),
    )
    return plan, trainability


__all__ = [
    "CurriculumStageHyperparameters",
    "DynamicCurriculumHyperparameters",
    "GroundedCurriculumHyperparameters",
    "build_dynamic_curriculum",
    "build_grounded_curriculum",
]
