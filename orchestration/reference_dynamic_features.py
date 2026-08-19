"""Repository-owned dynamic-RAG feature and hidden-state reference providers.

Dynamic control remains server-owned: learned models score a closed action vocabulary while
this module deterministically constructs the policy feature vector from admitted analyzers
and hard runtime budgets. No analyzer/model is loaded on import.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from orchestration.dynamic_rag_runtime import DynamicRuntimeSnapshot
from training.dynamic_retrieval_policy import DynamicRetrievalBudget, DynamicRetrievalFeatures


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


def _unit(value: Any, label: str) -> float:
    selected = _finite(value, label)
    if not 0.0 <= selected <= 1.0:
        raise ValueError(f"{label} must lie in [0,1]")
    return selected


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


@dataclass(frozen=True)
class GenerationUncertaintySignals:
    token_entropy: float
    top1_margin: float

    def __post_init__(self) -> None:
        entropy = _finite(self.token_entropy, "token_entropy")
        if entropy < 0.0:
            raise ValueError("token_entropy must be non-negative")
        object.__setattr__(self, "token_entropy", entropy)
        object.__setattr__(self, "top1_margin", _unit(self.top1_margin, "top1_margin"))


@dataclass(frozen=True)
class EvidenceStateSignals:
    evidence_sufficiency: float
    semantic_support: float
    contradiction_risk: float
    citation_coverage: float

    def __post_init__(self) -> None:
        for name in ("evidence_sufficiency", "semantic_support", "contradiction_risk", "citation_coverage"):
            object.__setattr__(self, name, _unit(getattr(self, name), name))


@dataclass(frozen=True)
class ContextStateSignals:
    context_novelty: float
    unresolved_entity_ratio: float
    temporal_uncertainty: float

    def __post_init__(self) -> None:
        for name in ("context_novelty", "unresolved_entity_ratio", "temporal_uncertainty"):
            object.__setattr__(self, name, _unit(getattr(self, name), name))


class GenerationUncertaintyAnalyzer(Protocol):
    @property
    def contract_sha256(self) -> str: ...
    def analyze(self, snapshot: DynamicRuntimeSnapshot) -> GenerationUncertaintySignals: ...


class EvidenceStateAnalyzer(Protocol):
    @property
    def contract_sha256(self) -> str: ...
    def analyze(self, snapshot: DynamicRuntimeSnapshot) -> EvidenceStateSignals: ...


class ContextStateAnalyzer(Protocol):
    @property
    def contract_sha256(self) -> str: ...
    def analyze(self, snapshot: DynamicRuntimeSnapshot) -> ContextStateSignals: ...


class ElapsedBudgetAnalyzer(Protocol):
    @property
    def contract_sha256(self) -> str: ...
    def fraction(self, snapshot: DynamicRuntimeSnapshot) -> float: ...


class ReferenceDynamicFeatureProvider:
    """Concrete ``DynamicFeatureProvider`` composed from admitted bounded analyzers."""
    def __init__(self, *, budget: DynamicRetrievalBudget, generation: GenerationUncertaintyAnalyzer, evidence: EvidenceStateAnalyzer, context: ContextStateAnalyzer, elapsed: ElapsedBudgetAnalyzer) -> None:
        if not isinstance(budget, DynamicRetrievalBudget):
            raise ValueError("budget must be DynamicRetrievalBudget")
        self.budget = budget
        self.generation = generation
        self.evidence = evidence
        self.context = context
        self.elapsed = elapsed
        for name, provider in (("generation", generation), ("evidence", evidence), ("context", context), ("elapsed", elapsed)):
            digest = getattr(provider, "contract_sha256", None)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"{name} analyzer must expose a SHA-256 contract identity")

    @property
    def contract_sha256(self) -> str:
        return _digest({
            "schema": "rigorousrag-reference-dynamic-feature-provider/v1",
            "budget_sha256": self.budget.budget_sha256,
            "generation_contract_sha256": self.generation.contract_sha256,
            "evidence_contract_sha256": self.evidence.contract_sha256,
            "context_contract_sha256": self.context.contract_sha256,
            "elapsed_contract_sha256": self.elapsed.contract_sha256,
        })

    def features(self, snapshot: DynamicRuntimeSnapshot) -> DynamicRetrievalFeatures:
        if not isinstance(snapshot, DynamicRuntimeSnapshot):
            raise ValueError("snapshot must be DynamicRuntimeSnapshot")
        generation = self.generation.analyze(snapshot)
        evidence = self.evidence.analyze(snapshot)
        context = self.context.analyze(snapshot)
        retrieval_fraction = 0.0 if self.budget.max_retrievals == 0 else min(1.0, snapshot.state.retrievals / self.budget.max_retrievals)
        token_fraction = min(1.0, snapshot.state.generated_tokens / self.budget.max_generation_tokens)
        elapsed_fraction = _unit(self.elapsed.fraction(snapshot), "elapsed_budget_fraction")
        return DynamicRetrievalFeatures(
            token_entropy=generation.token_entropy,
            top1_margin=generation.top1_margin,
            evidence_sufficiency=evidence.evidence_sufficiency,
            semantic_support=evidence.semantic_support,
            contradiction_risk=evidence.contradiction_risk,
            citation_coverage=evidence.citation_coverage,
            context_novelty=context.context_novelty,
            unresolved_entity_ratio=context.unresolved_entity_ratio,
            temporal_uncertainty=context.temporal_uncertainty,
            retrieval_count_fraction=retrieval_fraction,
            token_budget_fraction=token_fraction,
            elapsed_budget_fraction=elapsed_fraction,
        )


class GeneratorHiddenStateAdapter:
    """Extract policy selector states from an exact admitted causal or seq2seq generator.

    Causal models use the final causal hidden states. Encoder-decoder models use encoder
    ``last_hidden_state`` over the visible context. The state summary is the actual final
    token whose attention-mask value is visible, so pooling is correct for either left- or
    right-padded reuse even though the authoritative training contract itself requires right
    padding.
    """
    def __init__(
        self,
        generator: Any,
        tokenizer: Any,
        *,
        generator_sha256: str,
        tokenizer_sha256: str,
        generator_family: str,
        max_length: int = 2048,
    ) -> None:
        self.generator_sha256 = _sha(generator_sha256, "generator_sha256")
        self.tokenizer_sha256 = _sha(tokenizer_sha256, "tokenizer_sha256")
        if generator_family not in {"causal_lm", "seq2seq_lm"}:
            raise ValueError("generator_family must be causal_lm or seq2seq_lm")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be positive")
        self.generator, self.tokenizer = generator, tokenizer
        self.generator_family, self.max_length = generator_family, max_length

    @property
    def contract_sha256(self) -> str:
        return _digest({
            "schema": "rigorousrag-generator-hidden-state-adapter/v3",
            "generator_sha256": self.generator_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "generator_family": self.generator_family,
            "max_length": self.max_length,
            "state_pooling": "actual_last_visible_token",
        })

    def encode(self, texts: list[str]) -> Mapping[str, Any]:
        if torch is None:
            raise RuntimeError("hidden-state extraction requires optional PyTorch")
        if not texts or any(not isinstance(text, str) or not text for text in texts):
            raise ValueError("texts must contain non-empty strings")
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        try:
            device = next(self.generator.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in encoded.items()}
        mask = inputs.get("attention_mask")
        if self.generator_family == "seq2seq_lm":
            if not hasattr(self.generator, "get_encoder"):
                raise ValueError("seq2seq generator does not expose get_encoder()")
            encoder = self.generator.get_encoder()
            with torch.no_grad():
                output = encoder(input_ids=inputs["input_ids"], attention_mask=mask, return_dict=True)
            token_hidden = getattr(output, "last_hidden_state", None)
            if token_hidden is None:
                raise ValueError("seq2seq encoder does not expose last_hidden_state")
        else:
            with torch.no_grad():
                output = self.generator(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = getattr(output, "hidden_states", None)
            if hidden_states is None or not hidden_states:
                raise ValueError("causal generator does not expose hidden_states")
            token_hidden = hidden_states[-1]
        if token_hidden.ndim != 3:
            raise ValueError("generator hidden states must have shape [B,T,H]")
        if mask is None:
            mask = torch.ones(token_hidden.shape[:2], device=token_hidden.device, dtype=torch.long)
        if tuple(mask.shape) != tuple(token_hidden.shape[:2]):
            raise ValueError("generator attention mask does not align with hidden states")
        visible = mask.to(dtype=torch.bool)
        if torch.any(~visible.any(dim=1)):
            raise ValueError("generator attention mask contains a row with no visible tokens")
        positions = torch.arange(token_hidden.size(1), device=token_hidden.device).unsqueeze(0).expand(token_hidden.size(0), -1)
        last_visible = positions.masked_fill(~visible, -1).max(dim=1).values
        state_hidden = token_hidden[torch.arange(token_hidden.size(0), device=token_hidden.device), last_visible]
        return {
            "token_hidden": token_hidden.detach().cpu(),
            "state_hidden": state_hidden.detach().cpu(),
            "attention_mask": mask.detach().cpu(),
        }


__all__ = [
    "ContextStateAnalyzer", "ContextStateSignals", "ElapsedBudgetAnalyzer", "EvidenceStateAnalyzer",
    "EvidenceStateSignals", "GenerationUncertaintyAnalyzer", "GenerationUncertaintySignals",
    "GeneratorHiddenStateAdapter", "ReferenceDynamicFeatureProvider",
]
