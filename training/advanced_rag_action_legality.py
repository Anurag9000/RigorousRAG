"""Legal-action masking and explicit value targets for authoritative dynamic-RAG training."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from training.advanced_rag_authoritative_data import LegalDynamicRagEpisodeStep
from training.advanced_rag_final_collation import FinalDynamicRagEpisodeCollator
from training.advanced_rag_steps import DynamicPolicyStepConfig
from training.advanced_rag_strict import StrictDynamicRetrievalPolicyStep


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("dynamic action-legality masking requires optional PyTorch")


class LegalActionDynamicRagEpisodeCollator(FinalDynamicRagEpisodeCollator):
    """Final dynamic collator plus exact legal action and state-value targets."""
    def __call__(self, examples: Sequence[LegalDynamicRagEpisodeStep]) -> dict[str, Any]:
        _require_torch()
        if not examples or any(not isinstance(item, LegalDynamicRagEpisodeStep) for item in examples):
            raise ValueError("authoritative dynamic batches require LegalDynamicRagEpisodeStep values")
        architecture_actions = tuple(self.architecture.actions)
        known = set(architecture_actions)
        rows = []
        for item in examples:
            if item.action not in known:
                raise ValueError(f"logged action {item.action.value} is absent from DynamicPolicyArchitecture.actions")
            legal = set(item.valid_actions) & known
            if item.action not in legal:
                raise ValueError("logged action is not legal under the architecture-specific action set")
            rows.append([action in legal for action in architecture_actions])
        batch = super().__call__(examples)
        batch["valid_action_mask"] = torch.tensor(rows, dtype=torch.bool)
        have_values = [item.value_target is not None for item in examples]
        if any(have_values) and not all(have_values):
            raise ValueError("a dynamic batch may not mix present and absent state-value targets")
        if all(have_values):
            batch["value_targets"] = torch.tensor([float(item.value_target) for item in examples], dtype=torch.float32)
        return batch


class _LegalityMaskedModel:
    def __init__(self, model: Any, valid_action_mask: Any) -> None:
        self.model = model
        self.valid_action_mask = valid_action_mask

    def __call__(self, **kwargs: Any) -> Mapping[str, Any]:
        _require_torch()
        output = self.model(**kwargs)
        if not isinstance(output, Mapping):
            raise ValueError("dynamic policy model must return a mapping")
        logits = output.get("action_logits")
        if logits is None or not torch.is_tensor(logits) or logits.ndim != 2:
            raise ValueError("dynamic policy model must expose [B,A] action_logits")
        mask = self.valid_action_mask.to(device=logits.device, dtype=torch.bool)
        if tuple(mask.shape) != tuple(logits.shape):
            raise ValueError("valid_action_mask must align exactly with action_logits")
        if torch.any(~mask.any(dim=-1)):
            raise ValueError("every dynamic state requires at least one valid action")
        floor = torch.finfo(logits.dtype).min
        selected = dict(output)
        selected["action_logits"] = logits.masked_fill(~mask, floor)
        return selected


class LegalActionDynamicRetrievalPolicyStep:
    """Strict dynamic objective after legality masking and value-target normalization."""
    def __init__(self, config: DynamicPolicyStepConfig, *, actions: tuple[Any, ...]) -> None:
        self.config = config
        self.inner = StrictDynamicRetrievalPolicyStep(config, actions=actions)

    def __call__(self, model: Any, batch: Mapping[str, Any]) -> Any:
        mask = batch.get("valid_action_mask")
        if mask is None:
            raise ValueError("authoritative dynamic policy step requires valid_action_mask")
        targets = batch.get("action_targets")
        if targets is None:
            raise ValueError("authoritative dynamic policy step requires action_targets")
        _require_torch()
        if mask.ndim != 2 or targets.ndim != 1 or mask.size(0) != targets.size(0):
            raise ValueError("valid_action_mask/action_targets shapes are incompatible")
        selected = mask.gather(1, targets.long().unsqueeze(1)).squeeze(1)
        if not bool(selected.all().item()):
            raise ValueError("one or more logged action targets are invalid in their state")
        normalized = dict(batch)
        if self.config.objective.value_weight > 0.0:
            value_targets = normalized.get("value_targets")
            if value_targets is None:
                raise ValueError("authoritative value learning requires explicit value_targets from trajectory materialization")
            normalized["realized_retrieval_gain"] = value_targets
        return self.inner(_LegalityMaskedModel(model, mask), normalized)


__all__ = ["LegalActionDynamicRagEpisodeCollator", "LegalActionDynamicRetrievalPolicyStep"]
