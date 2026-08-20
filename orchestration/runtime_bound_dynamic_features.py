"""Runtime-policy binding for dynamic-RAG feature providers.

Dynamic retrieval features include three structural fractions derived from the active runtime
budget. A semantically valid feature provider must not be able to compute those fractions against
a different budget while the runtime enforces another. This transparent wrapper binds the inner
provider contract to the exact ``DynamicRagRuntimePolicy`` and checks the structural fractions on
every snapshot before returning features to the controller.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from orchestration.dynamic_rag_runtime import DynamicFeatureProvider, DynamicRagRuntimePolicy, DynamicRuntimeSnapshot
from training.dynamic_retrieval_policy import DynamicRetrievalFeatures

_HEX = frozenset("0123456789abcdef")
_TOLERANCE = 1e-12


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return min(1.0, max(0.0, float(numerator) / float(denominator)))


def _same(actual: float, expected: float, label: str) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > _TOLERANCE:
        raise ValueError(f"dynamic feature {label} is not bound to the active runtime policy")


class RuntimeBoundDynamicFeatureProvider:
    """Transparent DynamicFeatureProvider that proves exact structural runtime binding."""

    def __init__(self, inner: DynamicFeatureProvider, runtime_policy: DynamicRagRuntimePolicy) -> None:
        if not isinstance(runtime_policy, DynamicRagRuntimePolicy):
            raise ValueError("runtime_policy must be DynamicRagRuntimePolicy")
        self.inner = inner
        self.runtime_policy = runtime_policy
        self.inner_contract_sha256 = _sha(
            getattr(inner, "contract_sha256", None),
            "inner feature provider contract_sha256",
        )

    @property
    def contract_sha256(self) -> str:
        return _digest({
            "schema": "rigorousrag-runtime-bound-dynamic-feature-provider/v1",
            "inner_feature_provider_sha256": self.inner_contract_sha256,
            "runtime_policy_sha256": self.runtime_policy.policy_sha256,
            "structural_features": {
                "retrieval_count_fraction": "state.retrievals/runtime_budget.max_retrievals",
                "token_budget_fraction": "state.generated_tokens/runtime_budget.max_generation_tokens",
                "elapsed_budget_fraction": "snapshot.iteration/runtime_policy.maximum_iterations",
            },
        })

    def features(self, snapshot: DynamicRuntimeSnapshot) -> DynamicRetrievalFeatures:
        if not isinstance(snapshot, DynamicRuntimeSnapshot):
            raise ValueError("snapshot must be DynamicRuntimeSnapshot")
        value = self.inner.features(snapshot)
        if not isinstance(value, DynamicRetrievalFeatures):
            raise ValueError("inner dynamic feature provider returned invalid feature type")
        budget = self.runtime_policy.budget
        _same(
            value.retrieval_count_fraction,
            _fraction(snapshot.state.retrievals, budget.max_retrievals),
            "retrieval_count_fraction",
        )
        _same(
            value.token_budget_fraction,
            _fraction(snapshot.state.generated_tokens, budget.max_generation_tokens),
            "token_budget_fraction",
        )
        _same(
            value.elapsed_budget_fraction,
            _fraction(snapshot.iteration, self.runtime_policy.maximum_iterations),
            "elapsed_budget_fraction",
        )
        return value


__all__ = ["RuntimeBoundDynamicFeatureProvider"]
