"""Early cache-key coverage checks for authoritative advanced RAG runs.

Exact cache content is already sealed into training identity. This module additionally proves
that every record-dependent cache lookup the active curriculum will require is present before
model/optimizer construction, turning late collator failures into deterministic preflight
errors.
"""
from __future__ import annotations

from typing import Any

from training.advanced_rag_authoritative_data import LegalDynamicRagEpisodeStep
from training.dynamic_retrieval_policy import DynamicPolicyTrainingPlan
from training.grounded_generation import GroundedTrainingPlan


def preflight_grounded_cache_coverage(
    dataset: Any,
    plan: GroundedTrainingPlan,
    *,
    teacher_cache: Any | None,
    reference_cache: Any | None,
    retriever_batch_builder: Any | None,
) -> None:
    needs_teacher = any(stage.objective.teacher_distillation > 0.0 for stage in plan.stages)
    needs_preference = any(stage.objective.preference > 0.0 for stage in plan.stages)
    needs_retriever = any(stage.objective.retriever_coupling > 0.0 for stage in plan.stages)
    utility_cache = getattr(retriever_batch_builder, "utility_cache", None) if retriever_batch_builder is not None else None
    for index in range(len(dataset)):
        example = dataset[index]
        if needs_teacher:
            if teacher_cache is None or example.teacher_cache_key is None:
                raise ValueError(f"grounded example {example.example_id} lacks configured teacher supervision")
            teacher_cache.get(example.teacher_cache_key)
        if needs_preference and example.reference_chosen_log_prob is None:
            if reference_cache is None:
                raise ValueError(f"grounded example {example.example_id} lacks reference-policy supervision")
            reference_cache.get(example.example_id)
        if needs_retriever:
            if utility_cache is None:
                raise ValueError("retriever-coupling curriculum lacks a utility cache")
            utility_cache.get(example.retriever_cache_key or example.example_id)


def preflight_dynamic_cache_coverage(dataset: Any, plan: DynamicPolicyTrainingPlan, *, hidden_state_cache: Any | None) -> None:
    needs_hidden = any(stage.objective.need_selection_weight > 0.0 for stage in plan.stages)
    if not needs_hidden:
        return
    if hidden_state_cache is None:
        raise ValueError("need-selection curriculum lacks a hidden-state cache")
    for index in range(len(dataset)):
        step = dataset[index]
        if not isinstance(step, LegalDynamicRagEpisodeStep):
            raise ValueError("authoritative dynamic dataset contains a non-legal record type")
        if step.hidden_state_cache_key is None:
            raise ValueError(f"dynamic step {step.episode_id}:{step.step_id} lacks hidden_state_cache_key")
        hidden_state_cache.get(step.hidden_state_cache_key)


__all__ = ["preflight_dynamic_cache_coverage", "preflight_grounded_cache_coverage"]
