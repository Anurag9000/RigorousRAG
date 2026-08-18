"""Authoritative grounded objective step with padding-safe teacher distillation."""
from __future__ import annotations

from typing import Any, Mapping

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from training.advanced_rag_steps import GroundedStepConfig
from training.advanced_rag_strict import _extract, _metric, _safe_citation_pointer_loss, unsupported_target_token_unlikelihood
from training.grounded_generation import (
    binary_supervision_loss,
    dpo_grounded_preference_loss,
    grounded_generation_objective,
    lm_supervised_retriever_kl,
    masked_token_nll,
    reflection_action_loss,
    sequence_log_prob,
)
from training.torch_engine import StepResult


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("authoritative grounded objective requires optional PyTorch")


def masked_teacher_token_distillation_kl(student_logits: Any, teacher_logits: Any, labels: Any, *, ignore_index: int = -100, temperature: float = 1.0) -> Any:
    """Distill only positions that belong to supervised target tokens.

    Per-example teacher caches are padded to the current batch length by the authoritative
    collators. Ignored/padded/prompt-only positions therefore contribute exactly zero rather
    than inducing an artificial uniform-target loss.
    """
    _require_torch()
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 3 or labels.ndim != 2 or student_logits.shape[:2] != labels.shape:
        raise ValueError("student/teacher logits and labels must align as [B,T,V]/[B,T]")
    selected_temperature = float(temperature)
    if not 0.0 < selected_temperature <= 1000.0:
        raise ValueError("temperature must be positive and bounded")
    mask = labels.ne(ignore_index)
    if not bool(mask.any().detach().item()):
        return student_logits.sum() * 0.0
    student_log = F.log_softmax(student_logits / selected_temperature, dim=-1)
    teacher_probability = F.softmax(teacher_logits.detach() / selected_temperature, dim=-1)
    token_kl = F.kl_div(student_log, teacher_probability, reduction="none").sum(dim=-1)
    return token_kl[mask].mean() * selected_temperature * selected_temperature


class AuthoritativeGroundedGenerationStep:
    """Final grounded train/eval step used by configuration-driven launches."""
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
            chosen = sequence_log_prob(_extract(outputs, "chosen_logits", required=True), batch["chosen_labels"], ignore_index=self.config.ignore_index)
            rejected = sequence_log_prob(_extract(outputs, "rejected_logits", required=True), batch["rejected_labels"], ignore_index=self.config.ignore_index)
            losses["preference"] = dpo_grounded_preference_loss(chosen, rejected, batch["reference_chosen_log_prob"], batch["reference_rejected_log_prob"], beta=self.config.dpo_beta)
        if weights.teacher_distillation > 0.0:
            losses["teacher_distillation"] = masked_teacher_token_distillation_kl(
                token_logits,
                batch["teacher_token_logits"],
                batch["labels"],
                ignore_index=self.config.ignore_index,
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


__all__ = ["AuthoritativeGroundedGenerationStep", "masked_teacher_token_distillation_kl"]
