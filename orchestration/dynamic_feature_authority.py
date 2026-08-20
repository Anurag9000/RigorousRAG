"""Concrete, contract-bound feature computation for dynamic-RAG control.

``DynamicFeatureProvider`` in the runtime loop is intentionally a protocol.  This module supplies
the production reference composition without pretending that semantic confidence can be derived
from cheap text heuristics:

* next-token entropy and top-1 margin come from an explicit uncertainty provider;
* evidence sufficiency, citation coverage, novelty, unresolved-entity ratio and temporal
  uncertainty come from an explicit semantic-signal provider;
* semantic support / contradiction use the server-owned verification observation when present,
  otherwise the semantic provider's values;
* retrieval/token/elapsed budget fractions are computed from the exact runtime state/policy.

A local Hugging Face next-token provider is included for already-admitted local model/tokenizer
objects.  Because ``DynamicRuntimeSnapshot`` intentionally does not carry the original request or
evidence text, that provider also requires an explicit bound model-context provider.  Nothing is
loaded or executed on import.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence

try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency.
    torch = None  # type: ignore[assignment]

from orchestration.dynamic_rag_runtime import DynamicRuntimeSnapshot
from training.dynamic_retrieval_policy import DynamicRetrievalBudget, DynamicRetrievalFeatures

_HEX = frozenset("0123456789abcdef")
_MAX_TEXT = 10_000_000
_MAX_LENGTH = 10_000_000


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


def _unit(value: Any, label: str) -> float:
    selected = _finite(value, label)
    if not 0.0 <= selected <= 1.0:
        raise ValueError(f"{label} must lie in [0,1]")
    return selected


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("local dynamic uncertainty computation requires optional PyTorch")


def _device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("local uncertainty model exposes no parameters") from exc


@dataclass(frozen=True)
class DynamicSemanticSignals:
    evidence_sufficiency: float
    semantic_support: float
    contradiction_risk: float
    citation_coverage: float
    context_novelty: float
    unresolved_entity_ratio: float
    temporal_uncertainty: float

    def __post_init__(self) -> None:
        for name in (
            "evidence_sufficiency",
            "semantic_support",
            "contradiction_risk",
            "citation_coverage",
            "context_novelty",
            "unresolved_entity_ratio",
            "temporal_uncertainty",
        ):
            object.__setattr__(self, name, _unit(getattr(self, name), name))


@dataclass(frozen=True)
class NextTokenUncertainty:
    token_entropy: float
    top1_margin: float

    def __post_init__(self) -> None:
        entropy = _finite(self.token_entropy, "token_entropy")
        if entropy < 0.0:
            raise ValueError("token_entropy must be non-negative")
        object.__setattr__(self, "token_entropy", entropy)
        object.__setattr__(self, "top1_margin", _unit(self.top1_margin, "top1_margin"))


class DynamicSemanticSignalProvider(Protocol):
    @property
    def contract_sha256(self) -> str: ...
    def signals(self, snapshot: DynamicRuntimeSnapshot) -> DynamicSemanticSignals: ...


class NextTokenUncertaintyProvider(Protocol):
    @property
    def contract_sha256(self) -> str: ...
    def uncertainty(self, snapshot: DynamicRuntimeSnapshot) -> NextTokenUncertainty: ...


class BoundDynamicModelContextProvider(Protocol):
    """Server-owned reconstruction of exact model text/context for one runtime snapshot."""
    @property
    def contract_sha256(self) -> str: ...
    def model_text(self, snapshot: DynamicRuntimeSnapshot) -> str: ...


@dataclass(frozen=True)
class DynamicFeatureAuthorityPolicy:
    budget: DynamicRetrievalBudget
    maximum_iterations: int
    prefer_verification_observation: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.budget, DynamicRetrievalBudget):
            raise ValueError("budget must be DynamicRetrievalBudget")
        if isinstance(self.maximum_iterations, bool) or not isinstance(self.maximum_iterations, int) or self.maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be positive")
        if not isinstance(self.prefer_verification_observation, bool):
            raise ValueError("prefer_verification_observation must be boolean")

    @property
    def policy_sha256(self) -> str:
        return _digest({
            "schema": "rigorousrag-dynamic-feature-authority-policy/v1",
            "budget_sha256": self.budget.budget_sha256,
            "maximum_iterations": self.maximum_iterations,
            "prefer_verification_observation": self.prefer_verification_observation,
        })


class ReferenceDynamicFeatureProvider:
    """Production reference implementation of the runtime DynamicFeatureProvider protocol."""

    def __init__(
        self,
        *,
        uncertainty_provider: NextTokenUncertaintyProvider,
        semantic_provider: DynamicSemanticSignalProvider,
        policy: DynamicFeatureAuthorityPolicy,
    ) -> None:
        if not isinstance(policy, DynamicFeatureAuthorityPolicy):
            raise ValueError("policy must be DynamicFeatureAuthorityPolicy")
        self.uncertainty_provider = uncertainty_provider
        self.semantic_provider = semantic_provider
        self.policy = policy
        self._uncertainty_sha = _sha(
            getattr(uncertainty_provider, "contract_sha256", None),
            "uncertainty provider contract_sha256",
        )
        self._semantic_sha = _sha(
            getattr(semantic_provider, "contract_sha256", None),
            "semantic provider contract_sha256",
        )

    @property
    def contract_sha256(self) -> str:
        return _digest({
            "schema": "rigorousrag-reference-dynamic-feature-provider/v1",
            "uncertainty_provider_sha256": self._uncertainty_sha,
            "semantic_provider_sha256": self._semantic_sha,
            "policy_sha256": self.policy.policy_sha256,
            "semantics": {
                "uncertainty": "next_token_distribution",
                "verification_override": self.policy.prefer_verification_observation,
                "budget_fractions": "used_over_exact_configured_budget_clamped_to_unit_interval",
            },
        })

    @staticmethod
    def _fraction(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return min(1.0, max(0.0, float(numerator) / float(denominator)))

    def features(self, snapshot: DynamicRuntimeSnapshot) -> DynamicRetrievalFeatures:
        if not isinstance(snapshot, DynamicRuntimeSnapshot):
            raise ValueError("snapshot must be DynamicRuntimeSnapshot")
        uncertainty = self.uncertainty_provider.uncertainty(snapshot)
        semantic = self.semantic_provider.signals(snapshot)
        if not isinstance(uncertainty, NextTokenUncertainty):
            raise ValueError("uncertainty provider returned the wrong type")
        if not isinstance(semantic, DynamicSemanticSignals):
            raise ValueError("semantic provider returned the wrong type")

        support = semantic.semantic_support
        contradiction = semantic.contradiction_risk
        if self.policy.prefer_verification_observation and snapshot.verification is not None:
            support = snapshot.verification.support_score
            contradiction = snapshot.verification.contradiction_score

        return DynamicRetrievalFeatures(
            token_entropy=uncertainty.token_entropy,
            top1_margin=uncertainty.top1_margin,
            evidence_sufficiency=semantic.evidence_sufficiency,
            semantic_support=support,
            contradiction_risk=contradiction,
            citation_coverage=semantic.citation_coverage,
            context_novelty=semantic.context_novelty,
            unresolved_entity_ratio=semantic.unresolved_entity_ratio,
            temporal_uncertainty=semantic.temporal_uncertainty,
            retrieval_count_fraction=self._fraction(snapshot.state.retrievals, self.policy.budget.max_retrievals),
            token_budget_fraction=self._fraction(snapshot.state.generated_tokens, self.policy.budget.max_generation_tokens),
            elapsed_budget_fraction=self._fraction(snapshot.iteration, self.policy.maximum_iterations),
        )


@dataclass(frozen=True)
class LocalNextTokenUncertaintyConfig:
    max_length: int = 4096
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.max_length, bool) or not isinstance(self.max_length, int) or not 1 <= self.max_length <= _MAX_LENGTH:
            raise ValueError("max_length must be a positive bounded integer")
        temperature = _finite(self.temperature, "temperature")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        object.__setattr__(self, "temperature", temperature)

    @property
    def config_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-local-next-token-uncertainty-config/v1", **asdict(self)})


class LocalNextTokenUncertaintyProvider:
    """Compute entropy/margin from an admitted local causal language model's next-token logits."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        model_sha256: str,
        tokenizer_sha256: str,
        context_provider: BoundDynamicModelContextProvider,
        config: LocalNextTokenUncertaintyConfig = LocalNextTokenUncertaintyConfig(),
    ) -> None:
        if not isinstance(config, LocalNextTokenUncertaintyConfig):
            raise ValueError("config must be LocalNextTokenUncertaintyConfig")
        self.model = model
        self.tokenizer = tokenizer
        self.model_sha256 = _sha(model_sha256, "model_sha256")
        self.tokenizer_sha256 = _sha(tokenizer_sha256, "tokenizer_sha256")
        self.context_provider = context_provider
        self.config = config
        self._context_sha = _sha(
            getattr(context_provider, "contract_sha256", None),
            "model context provider contract_sha256",
        )

    @property
    def contract_sha256(self) -> str:
        return _digest({
            "schema": "rigorousrag-local-next-token-uncertainty-provider/v1",
            "model_sha256": self.model_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "context_provider_sha256": self._context_sha,
            "config_sha256": self.config.config_sha256,
        })

    def uncertainty(self, snapshot: DynamicRuntimeSnapshot) -> NextTokenUncertainty:
        _require_torch()
        text = self.context_provider.model_text(snapshot)
        if not isinstance(text, str) or not text or len(text) > _MAX_TEXT or "\x00" in text:
            raise ValueError("model context provider returned invalid text")
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        if "input_ids" not in encoded or "attention_mask" not in encoded:
            raise ValueError("local tokenizer must return input_ids and attention_mask")
        device = _device(self.model)
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in encoded.items()
        }
        self.model.eval()
        with torch.no_grad():
            output = self.model(**inputs, return_dict=True)
        logits = output.get("logits") if isinstance(output, Mapping) else getattr(output, "logits", None)
        if logits is None or not torch.is_tensor(logits) or logits.ndim != 3 or logits.size(0) != 1:
            raise ValueError("local causal generator must expose [1,T,V] logits")
        attention = inputs["attention_mask"].to(dtype=torch.bool)
        positions = torch.arange(logits.size(1), device=device).unsqueeze(0)
        last_visible = positions.masked_fill(~attention, -1).max(dim=1).values
        if int(last_visible.item()) < 0:
            raise ValueError("local uncertainty tokenizer produced no visible token")
        next_logits = logits[0, int(last_visible.item())].float() / self.config.temperature
        probabilities = torch.softmax(next_logits, dim=-1)
        if probabilities.numel() < 2:
            raise ValueError("next-token distribution requires at least two vocabulary entries")
        entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum()
        top = torch.topk(probabilities, k=2).values
        return NextTokenUncertainty(
            token_entropy=float(entropy.detach().cpu().item()),
            top1_margin=float((top[0] - top[1]).detach().cpu().item()),
        )


__all__ = [
    "BoundDynamicModelContextProvider",
    "DynamicFeatureAuthorityPolicy",
    "DynamicSemanticSignalProvider",
    "DynamicSemanticSignals",
    "LocalNextTokenUncertaintyConfig",
    "LocalNextTokenUncertaintyProvider",
    "NextTokenUncertainty",
    "NextTokenUncertaintyProvider",
    "ReferenceDynamicFeatureProvider",
]
