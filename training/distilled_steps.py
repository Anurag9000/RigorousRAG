"""Masked teacher-distillation steps for retrieval architectures.

The governed collator stores teacher scores only for candidates belonging to each query
and uses NaN elsewhere. These losses therefore normalize teacher/student distributions
only over finite teacher subsets rather than requiring a dense all-query/all-document
teacher matrix. Nothing executes on import.
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

from training.torch_engine import StepResult
from training.torch_losses import SparsePenaltyWeights, in_batch_info_nce, sparse_retrieval_objective


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("distillation steps require the optional PyTorch dependency")


def masked_distillation_kl(
    student_logits: Any,
    teacher_logits: Any,
    *,
    mask: Any | None = None,
    temperature: float = 1.0,
    minimum_candidates: int = 2,
) -> Any:
    """Mean temperature-scaled KL(teacher||student) over row-specific finite subsets.

    Rows with fewer than ``minimum_candidates`` teacher scores are excluded. If no row
    is eligible, a differentiable zero scalar tied to the student graph is returned.
    """

    _require_torch()
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("student and teacher logits must be aligned [batch,candidates] tensors")
    selected_temperature = float(temperature)
    if not selected_temperature > 0.0:
        raise ValueError("temperature must be positive")
    if isinstance(minimum_candidates, bool) or not isinstance(minimum_candidates, int) or minimum_candidates < 2:
        raise ValueError("minimum_candidates must be an integer >= 2")
    valid = torch.isfinite(teacher_logits)
    if mask is not None:
        if mask.shape != teacher_logits.shape:
            raise ValueError("distillation mask must match logits shape")
        valid = valid & mask.to(device=valid.device, dtype=torch.bool)
    losses: list[Any] = []
    for row in range(student_logits.size(0)):
        selected = valid[row]
        if int(selected.sum().item()) < minimum_candidates:
            continue
        student = student_logits[row, selected] / selected_temperature
        teacher = teacher_logits[row, selected].detach() / selected_temperature
        loss = F.kl_div(
            F.log_softmax(student, dim=-1),
            F.softmax(teacher, dim=-1),
            reduction="sum",
        )
        losses.append(loss * selected_temperature * selected_temperature)
    if not losses:
        return student_logits.sum() * 0.0
    return torch.stack(losses).mean()


def _mask_false_negatives(scores: Any, batch: Mapping[str, Any]) -> Any:
    mask = batch.get("false_negative_mask")
    if mask is None:
        return scores
    return scores.masked_fill(mask.to(device=scores.device, dtype=torch.bool), torch.finfo(scores.dtype).min)


def _teacher(batch: Mapping[str, Any], scores: Any) -> tuple[Any, Any]:
    teacher = batch.get("teacher_scores")
    if teacher is None:
        return None, None
    teacher = teacher.to(device=scores.device, dtype=scores.dtype)
    if teacher.shape != scores.shape:
        raise ValueError("teacher score matrix does not match retrieval score matrix")
    return teacher, torch.isfinite(teacher)


@dataclass(frozen=True)
class DistillationConfig:
    retrieval_temperature: float = 0.05
    teacher_temperature: float = 1.0
    distillation_weight: float = 1.0
    minimum_teacher_candidates: int = 2

    def __post_init__(self) -> None:
        if float(self.retrieval_temperature) <= 0.0 or float(self.teacher_temperature) <= 0.0:
            raise ValueError("temperatures must be positive")
        if float(self.distillation_weight) < 0.0:
            raise ValueError("distillation_weight must be non-negative")
        if self.minimum_teacher_candidates < 2:
            raise ValueError("minimum_teacher_candidates must be >=2")


class DistilledDenseContrastiveStep:
    def __init__(self, config: DistillationConfig = DistillationConfig()) -> None:
        self.config = config

    def __call__(self, model: Any, batch: Mapping[str, Any]) -> StepResult:
        _require_torch()
        queries, documents = model(batch["query_inputs"], batch["document_inputs"])
        scores = _mask_false_negatives(queries @ documents.transpose(0, 1), batch)
        retrieval = in_batch_info_nce(
            scores,
            positive_indices=batch["positive_indices"],
            temperature=self.config.retrieval_temperature,
        )
        teacher, teacher_mask = _teacher(batch, scores)
        distillation = scores.sum() * 0.0 if teacher is None else masked_distillation_kl(
            scores,
            teacher,
            mask=teacher_mask,
            temperature=self.config.teacher_temperature,
            minimum_candidates=self.config.minimum_teacher_candidates,
        )
        total = retrieval + self.config.distillation_weight * distillation
        return StepResult(
            total,
            {
                "retrieval_loss": float(retrieval.detach().cpu()),
                "distillation_loss": float(distillation.detach().cpu()),
            },
        )


class DistilledSparseContrastiveStep:
    def __init__(
        self,
        config: DistillationConfig = DistillationConfig(retrieval_temperature=1.0),
        *,
        penalties: SparsePenaltyWeights = SparsePenaltyWeights(),
    ) -> None:
        self.config = config
        self.penalties = penalties

    def __call__(self, model: Any, batch: Mapping[str, Any]) -> StepResult:
        _require_torch()
        query_weights = model(**batch["query_inputs"])
        document_weights = model(**batch["document_inputs"])
        scores = _mask_false_negatives(query_weights @ document_weights.transpose(0, 1), batch)
        base = sparse_retrieval_objective(
            scores,
            batch["positive_indices"],
            query_weights,
            document_weights,
            temperature=self.config.retrieval_temperature,
            penalties=self.penalties,
        )
        teacher, teacher_mask = _teacher(batch, scores)
        distillation = scores.sum() * 0.0 if teacher is None else masked_distillation_kl(
            scores,
            teacher,
            mask=teacher_mask,
            temperature=self.config.teacher_temperature,
            minimum_candidates=self.config.minimum_teacher_candidates,
        )
        total = base.total + self.config.distillation_weight * distillation
        return StepResult(
            total,
            {
                "retrieval_loss": float(base.retrieval.detach().cpu()),
                "distillation_loss": float(distillation.detach().cpu()),
                "query_l1": float(base.query_l1.detach().cpu()),
                "document_l1": float(base.document_l1.detach().cpu()),
                "query_flops": float(base.query_flops.detach().cpu()),
                "document_flops": float(base.document_flops.detach().cpu()),
            },
        )


class DistilledColBERTContrastiveStep:
    def __init__(self, config: DistillationConfig = DistillationConfig()) -> None:
        self.config = config

    def __call__(self, model: Any, batch: Mapping[str, Any]) -> StepResult:
        _require_torch()
        query_embeddings, query_mask = model(**batch["query_inputs"])
        document_embeddings, document_mask = model(**batch["document_inputs"])
        module = model.module if hasattr(model, "module") else model
        scores = type(module).score_matrix(query_embeddings, query_mask, document_embeddings, document_mask)
        scores = _mask_false_negatives(scores, batch)
        retrieval = in_batch_info_nce(
            scores,
            positive_indices=batch["positive_indices"],
            temperature=self.config.retrieval_temperature,
        )
        teacher, teacher_mask = _teacher(batch, scores)
        distillation = scores.sum() * 0.0 if teacher is None else masked_distillation_kl(
            scores,
            teacher,
            mask=teacher_mask,
            temperature=self.config.teacher_temperature,
            minimum_candidates=self.config.minimum_teacher_candidates,
        )
        total = retrieval + self.config.distillation_weight * distillation
        return StepResult(
            total,
            {
                "retrieval_loss": float(retrieval.detach().cpu()),
                "distillation_loss": float(distillation.detach().cpu()),
            },
        )


__all__ = [
    "DistillationConfig",
    "DistilledColBERTContrastiveStep",
    "DistilledDenseContrastiveStep",
    "DistilledSparseContrastiveStep",
    "masked_distillation_kl",
]
