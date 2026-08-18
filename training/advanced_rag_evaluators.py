"""Trainer-compatible validation evaluators for grounded generation and dynamic RAG."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from training.advanced_rag_steps import DynamicRetrievalPolicyStep, GroundedGenerationStep
from training.torch_engine import move_to_device


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("advanced RAG evaluation requires optional PyTorch")


def _device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("validation model exposes no parameters") from exc


def _safe_mean(total: float, count: int, label: str) -> float:
    if count <= 0:
        raise ValueError(f"validation produced no observations for {label}")
    result = total / count
    if not math.isfinite(result):
        raise ValueError(f"validation metric {label} is non-finite")
    return result


def _masked_accuracy(logits: Any, targets: Any, mask: Any | None = None) -> tuple[float, int]:
    _require_torch()
    if mask is None:
        mask = torch.ones_like(targets, dtype=torch.bool)
    else:
        mask = mask.to(dtype=torch.bool)
    count = int(mask.sum().item())
    if count == 0:
        return 0.0, 0
    prediction = (logits >= 0).to(dtype=targets.dtype)
    correct = ((prediction == targets) & mask).sum().item()
    return float(correct), count


@dataclass(frozen=True)
class ValidationLimits:
    maximum_batches: int | None = None

    def __post_init__(self) -> None:
        if self.maximum_batches is not None and (isinstance(self.maximum_batches, bool) or not isinstance(self.maximum_batches, int) or self.maximum_batches <= 0):
            raise ValueError("maximum_batches must be positive or None")


class GroundedValidationEvaluator:
    """Evaluate each grounded curriculum stage with its exact training objective."""
    def __init__(self, dataloader: Iterable[Mapping[str, Any]], stage_steps: Sequence[GroundedGenerationStep], limits: ValidationLimits = ValidationLimits()) -> None:
        self.dataloader = dataloader
        self.stage_steps = tuple(stage_steps)
        self.limits = limits
        if not self.stage_steps:
            raise ValueError("at least one grounded validation step is required")

    def __call__(self, model: Any, *, stage_index: int, optimizer_step: int) -> Mapping[str, float]:
        _require_torch()
        if not 0 <= stage_index < len(self.stage_steps):
            raise ValueError("grounded validation stage_index is outside configured steps")
        device = _device(model)
        totals: dict[str, float] = {}
        batches = 0
        citation_correct = citation_count = 0
        support_correct = support_count = 0
        contradiction_correct = contradiction_count = 0
        abstention_correct = abstention_count = 0
        reflection_correct = reflection_count = 0
        step = self.stage_steps[stage_index]
        for raw_batch in self.dataloader:
            if self.limits.maximum_batches is not None and batches >= self.limits.maximum_batches:
                break
            batch = move_to_device(raw_batch, device)
            result = step(model, batch)
            batches += 1
            for key, value in result.metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            outputs = model(**batch["model_inputs"])
            if "citation_logits" in outputs:
                targets = batch["citation_targets"]
                mask = targets.ne(step.config.ignore_index)
                count = int(mask.sum().item())
                if count:
                    prediction = outputs["citation_logits"].argmax(dim=-1)
                    citation_correct += int(((prediction == targets) & mask).sum().item()); citation_count += count
            if "support_logits" in outputs:
                correct, count = _masked_accuracy(outputs["support_logits"], batch["support_targets"], batch.get("claim_mask")); support_correct += int(correct); support_count += count
            if "contradiction_logits" in outputs:
                correct, count = _masked_accuracy(outputs["contradiction_logits"], batch["contradiction_targets"], batch.get("claim_mask")); contradiction_correct += int(correct); contradiction_count += count
            if "abstention_logits" in outputs:
                correct, count = _masked_accuracy(outputs["abstention_logits"], batch["abstention_targets"]); abstention_correct += int(correct); abstention_count += count
            if "reflection_logits" in outputs:
                targets = batch["reflection_targets"]
                mask = targets.ne(step.config.ignore_index)
                count = int(mask.sum().item())
                if count:
                    reflection_correct += int(((outputs["reflection_logits"].argmax(dim=-1) == targets) & mask).sum().item()); reflection_count += count
        if batches == 0:
            raise ValueError("grounded validation dataloader produced no batches")
        metrics = {f"validation_{key}": value / batches for key, value in totals.items()}
        if "validation_grounded_total" in metrics:
            metrics["validation_primary"] = -metrics["validation_grounded_total"]
        if citation_count: metrics["validation_citation_accuracy"] = citation_correct / citation_count
        if support_count: metrics["validation_support_accuracy"] = support_correct / support_count
        if contradiction_count: metrics["validation_contradiction_accuracy"] = contradiction_correct / contradiction_count
        if abstention_count: metrics["validation_abstention_accuracy"] = abstention_correct / abstention_count
        if reflection_count: metrics["validation_reflection_accuracy"] = reflection_correct / reflection_count
        metrics["validation_batches"] = float(batches)
        metrics["validation_optimizer_step"] = float(optimizer_step)
        return metrics


class DynamicPolicyValidationEvaluator:
    """Evaluate action, value and information-need learning with exact stage objectives."""
    def __init__(self, dataloader: Iterable[Mapping[str, Any]], stage_steps: Sequence[DynamicRetrievalPolicyStep], limits: ValidationLimits = ValidationLimits()) -> None:
        self.dataloader = dataloader
        self.stage_steps = tuple(stage_steps)
        self.limits = limits
        if not self.stage_steps:
            raise ValueError("at least one dynamic validation step is required")

    def __call__(self, model: Any, *, stage_index: int, optimizer_step: int) -> Mapping[str, float]:
        _require_torch()
        if not 0 <= stage_index < len(self.stage_steps):
            raise ValueError("dynamic validation stage_index is outside configured steps")
        device = _device(model)
        step = self.stage_steps[stage_index]
        totals: dict[str, float] = {}
        batches = action_correct = action_count = need_correct = need_count = 0
        value_absolute_error = 0.0
        value_count = 0
        for raw_batch in self.dataloader:
            if self.limits.maximum_batches is not None and batches >= self.limits.maximum_batches:
                break
            batch = move_to_device(raw_batch, device)
            result = step(model, batch)
            batches += 1
            for key, value in result.metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            outputs = model(features=batch["features"], token_hidden=batch.get("token_hidden"), state_hidden=batch.get("state_hidden"), attention_mask=batch.get("attention_mask"))
            if "action_logits" in outputs:
                targets = batch["action_targets"]
                mask = targets.ne(step.config.ignore_index)
                count = int(mask.sum().item())
                action_correct += int(((outputs["action_logits"].argmax(dim=-1) == targets) & mask).sum().item()); action_count += count
            if "retrieval_value" in outputs:
                target = batch["realized_retrieval_gain"].to(dtype=outputs["retrieval_value"].dtype)
                value_absolute_error += float((outputs["retrieval_value"] - target).abs().sum().item()); value_count += int(target.numel())
            if "need_logits" in outputs and "need_target_mask" in batch:
                valid = batch.get("need_valid_mask", torch.ones_like(batch["need_target_mask"], dtype=torch.bool)).to(dtype=torch.bool)
                prediction = outputs["need_logits"] >= 0
                target = batch["need_target_mask"].to(dtype=torch.bool)
                need_correct += int(((prediction == target) & valid).sum().item()); need_count += int(valid.sum().item())
        if batches == 0:
            raise ValueError("dynamic validation dataloader produced no batches")
        metrics = {f"validation_{key}": value / batches for key, value in totals.items()}
        if "validation_dynamic_total" in metrics:
            metrics["validation_primary"] = -metrics["validation_dynamic_total"]
        if action_count: metrics["validation_action_accuracy"] = action_correct / action_count
        if value_count: metrics["validation_value_mae"] = value_absolute_error / value_count
        if need_count: metrics["validation_need_token_accuracy"] = need_correct / need_count
        metrics["validation_batches"] = float(batches)
        metrics["validation_optimizer_step"] = float(optimizer_step)
        return metrics


__all__ = ["DynamicPolicyValidationEvaluator", "GroundedValidationEvaluator", "ValidationLimits"]
