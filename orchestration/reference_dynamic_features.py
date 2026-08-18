"""Repository-owned dynamic-RAG feature and hidden-state reference providers.

Dynamic control remains server-owned: learned models score a closed action vocabulary while
this module deterministically constructs the policy feature vector from admitted analyzers
and hard runtime budgets. No analyzer/model is loaded on import.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
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
    """Explicit adapter that obtains policy need-selector states from an admitted generator.

    ``generator`` must follow the common Hugging-Face-like contract and expose hidden states.
    Execution happens only when ``encode`` is called. Returned tensors are detached because
    this adapter is intended for cached/logged policy training. Joint differentiable training
    can instead route the same hidden tensors directly inside a composed model.
    """
    def __init__(self, generator: Any, tokenizer: Any, *, generator_sha256: str, tokenizer_sha256: str, max_length: int = 2048) -> None:
        if len(generator_sha256) != 64 or len(tokenizer_sha256) != 64:
            raise ValueError("generator/tokenizer identities must be SHA-256")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be positive")
        self.generator, self.tokenizer = generator, tokenizer
        self.generator_sha256, self.tokenizer_sha256, self.max_length = generator_sha256, tokenizer_sha256, max_length

    @property
    def contract_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-generator-hidden-state-adapter/v1", "generator_sha256": self.generator_sha256, "tokenizer_sha256": self.tokenizer_sha256, "max_length": self.max_length})

    def encode(self, texts: list[str]) -> Mapping[str, Any]:
        if torch is None:
            raise RuntimeError("hidden-state extraction requires optional PyTorch")
        if not texts:
            raise ValueError("texts may not be empty")
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        device = next(self.generator.parameters()).device
        inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in encoded.items()}
        with torch.no_grad():
            output = self.generator(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None:
            hidden_states = getattr(output, "decoder_hidden_states", None)
        if hidden_states is None or not hidden_states:
            raise ValueError("generator does not expose hidden states")
        token_hidden = hidden_states[-1]
        mask = inputs.get("attention_mask")
        if mask is None:
            mask = torch.ones(token_hidden.shape[:2], device=token_hidden.device, dtype=torch.long)
        lengths = mask.long().sum(dim=1).clamp_min(1)
        state_hidden = token_hidden[torch.arange(token_hidden.size(0), device=token_hidden.device), lengths - 1]
        return {"token_hidden": token_hidden.detach().cpu(), "state_hidden": state_hidden.detach().cpu(), "attention_mask": mask.detach().cpu()}


__all__ = ["ContextStateAnalyzer", "ContextStateSignals", "ElapsedBudgetAnalyzer", "EvidenceStateAnalyzer", "EvidenceStateSignals", "GenerationUncertaintyAnalyzer", "GenerationUncertaintySignals", "GeneratorHiddenStateAdapter", "ReferenceDynamicFeatureProvider"]
