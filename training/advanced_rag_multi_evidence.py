"""Multi-positive citation supervision for the authoritative grounded training path."""
from __future__ import annotations

from typing import Any, Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from training.advanced_rag_data import GroundedGenerationExample
from training.advanced_rag_final_collation import FinalCausalGroundedCollator, FinalSeq2SeqGroundedCollator


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("multi-evidence grounded collation requires optional PyTorch")


def _attach_multi_evidence_targets(batch: dict[str, Any], examples: Sequence[GroundedGenerationExample], *, evidence_limit: int) -> dict[str, Any]:
    _require_torch()
    citation_targets = batch.get("citation_targets")
    claim_mask = batch.get("claim_mask")
    model_inputs = batch.get("model_inputs")
    if not torch.is_tensor(citation_targets) or citation_targets.ndim != 2 or not torch.is_tensor(claim_mask) or tuple(claim_mask.shape) != tuple(citation_targets.shape):
        raise ValueError("base grounded collator did not emit aligned citation_targets/claim_mask")
    if not isinstance(model_inputs, dict) or not torch.is_tensor(model_inputs.get("evidence_input_ids")):
        raise ValueError("base grounded collator did not emit evidence_input_ids")
    evidence_slots = int(model_inputs["evidence_input_ids"].size(1))
    target_mask = torch.zeros((len(examples), citation_targets.size(1), evidence_slots), dtype=torch.bool)
    for row, example in enumerate(examples):
        selected = example.evidence[:evidence_limit]
        index = {record.evidence_id: position for position, record in enumerate(selected)}
        for claim_index, claim in enumerate(example.claims[: citation_targets.size(1)]):
            for evidence_id in claim.evidence_ids:
                position = index.get(evidence_id)
                if position is not None:
                    target_mask[row, claim_index, position] = True
    batch["citation_target_mask"] = target_mask
    batch["citation_supervision_mask"] = target_mask.any(dim=-1) & claim_mask.to(dtype=torch.bool)
    return batch


class MultiEvidenceCausalGroundedCollator(FinalCausalGroundedCollator):
    def __call__(self, examples: Sequence[GroundedGenerationExample]) -> dict[str, Any]:
        batch = super().__call__(examples)
        return _attach_multi_evidence_targets(batch, examples, evidence_limit=self.config.evidence_limit)


class MultiEvidenceSeq2SeqGroundedCollator(FinalSeq2SeqGroundedCollator):
    def __call__(self, examples: Sequence[GroundedGenerationExample]) -> dict[str, Any]:
        batch = super().__call__(examples)
        return _attach_multi_evidence_targets(batch, examples, evidence_limit=self.config.evidence_limit)


__all__ = ["MultiEvidenceCausalGroundedCollator", "MultiEvidenceSeq2SeqGroundedCollator"]
