"""Curriculum-aware data preflight for authoritative advanced RAG training."""
from __future__ import annotations

from typing import Any

from training.advanced_rag_authoritative_data import LegalDynamicRagEpisodeStep, StancedGroundedClaimAnnotation
from training.dynamic_retrieval_policy import DynamicPolicyTrainingPlan
from training.grounded_generation import GroundedTrainingPlan


def _sha_marker(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    selected = value.strip().lower()
    return len(selected) == 64 and all(ch in "0123456789abcdef" for ch in selected)


def preflight_dynamic_dataset(dataset: Any, plan: DynamicPolicyTrainingPlan) -> None:
    if not isinstance(plan, DynamicPolicyTrainingPlan):
        raise ValueError("plan must be DynamicPolicyTrainingPlan")
    needs_need = any(stage.objective.need_selection_weight > 0.0 for stage in plan.stages)
    needs_value = any(stage.objective.value_weight > 0.0 for stage in plan.stages)
    needs_pg = any(stage.objective.policy_gradient_weight > 0.0 for stage in plan.stages)
    for index in range(len(dataset)):
        step = dataset[index]
        if not isinstance(step, LegalDynamicRagEpisodeStep):
            raise ValueError("authoritative dynamic dataset must contain LegalDynamicRagEpisodeStep records")
        if step.action not in step.valid_actions:
            raise ValueError(f"dynamic step {step.episode_id}:{step.step_id} logs an illegal action")
        architecture_actions = set(plan.architecture.actions)
        if step.action not in architecture_actions:
            raise ValueError(f"dynamic step {step.episode_id}:{step.step_id} action is absent from architecture")
        if needs_need:
            if step.hidden_state_cache_key is None:
                raise ValueError(f"dynamic step {step.episode_id}:{step.step_id} lacks hidden_state_cache_key")
            # Non-empty spans are self-evidently annotated. An empty span set must be an
            # explicit negative, proven by the preparation provider marker rather than an
            # accidental omission.
            if not step.need_spans and not _sha_marker(step.metadata.get("need_annotation_provider_sha256")):
                raise ValueError(f"dynamic step {step.episode_id}:{step.step_id} has an unproven empty need-selection label")
        if needs_value and step.value_target is None:
            raise ValueError(f"dynamic step {step.episode_id}:{step.step_id} lacks explicit GAE return value_target")
        if needs_pg and (step.advantage is None or step.behavior_action_probability is None):
            raise ValueError(f"dynamic step {step.episode_id}:{step.step_id} lacks advantage/behavior probability for off-policy learning")


def preflight_grounded_dataset(dataset: Any, plan: GroundedTrainingPlan) -> None:
    if not isinstance(plan, GroundedTrainingPlan):
        raise ValueError("plan must be GroundedTrainingPlan")
    needs_claims = any(stage.objective.citation > 0.0 or stage.objective.support > 0.0 or stage.objective.contradiction > 0.0 for stage in plan.stages)
    needs_preference = any(stage.objective.preference > 0.0 for stage in plan.stages)
    needs_teacher = any(stage.objective.teacher_distillation > 0.0 for stage in plan.stages)
    needs_retriever = any(stage.objective.retriever_coupling > 0.0 for stage in plan.stages)
    for index in range(len(dataset)):
        example = dataset[index]
        if needs_claims and not example.claims:
            raise ValueError(f"grounded example {example.example_id} lacks claim supervision")
        if needs_preference and (example.chosen_answer is None or example.rejected_answer is None):
            raise ValueError(f"grounded example {example.example_id} lacks chosen/rejected preference pair")
        if needs_teacher and example.teacher_cache_key is None:
            raise ValueError(f"grounded example {example.example_id} lacks teacher_cache_key")
        if needs_retriever and example.retriever_cache_key is None:
            raise ValueError(f"grounded example {example.example_id} lacks retriever_cache_key")
        for claim in example.claims:
            if isinstance(claim, StancedGroundedClaimAnnotation):
                known = {record.evidence_id for record in example.evidence}
                unknown = (set(claim.supporting_evidence_ids) | set(claim.contradicting_evidence_ids)) - known
                if unknown:
                    raise ValueError(f"grounded example {example.example_id} claim references unknown stanced evidence")


__all__ = ["preflight_dynamic_dataset", "preflight_grounded_dataset"]
