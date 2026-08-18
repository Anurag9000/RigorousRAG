"""Deterministic supervision and cache materialization for advanced RAG training.

The training losses intentionally consume explicit tensors. This module defines how those
supervision tensors are produced and persisted once operators elect to execute admitted
teacher/reference/generator/retriever models. Merely importing this module performs no model
execution, network access, dataset download, or training.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from training.advanced_rag_data import DynamicRagEpisodeStep, GroundedGenerationExample
from training.dynamic_retrieval_policy import DynamicRetrievalAction

_HEX = frozenset("0123456789abcdef")


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("supervision materialization requires optional PyTorch")


def _identifier(value: Any, label: str, maximum: int = 2000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _sha256(value: Any, label: str) -> str:
    selected = _identifier(value, label, 64).lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SupervisionCacheIdentity:
    cache_kind: str
    producer_sha256: str
    tokenizer_sha256: str
    dataset_manifest_sha256: str
    source_commit: str
    config_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_kind", _identifier(self.cache_kind, "cache_kind", 160))
        for name in ("producer_sha256", "tokenizer_sha256", "dataset_manifest_sha256", "config_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        commit = _identifier(self.source_commit, "source_commit", 64).lower()
        if len(commit) not in {40, 64} or any(ch not in _HEX for ch in commit):
            raise ValueError("source_commit must be a full Git object id")
        object.__setattr__(self, "source_commit", commit)

    @property
    def digest(self) -> str:
        return canonical_digest({"schema": "rigorousrag-supervision-cache-identity/v1", **asdict(self)})


class SequenceReferenceProvider(Protocol):
    """Admitted reference policy used later to materialize DPO sequence log probabilities."""
    def sequence_log_probabilities(self, examples: Sequence[GroundedGenerationExample]) -> Sequence[tuple[float, float]]: ...


class TeacherLogitProvider(Protocol):
    """Admitted teacher used later to materialize token-logit tensors."""
    def token_logits(self, examples: Sequence[GroundedGenerationExample]) -> Sequence[Any]: ...


class DocumentUtilityProvider(Protocol):
    """Admitted generator/retriever stack used later to score candidate document utility."""
    def document_log_likelihoods(self, examples: Sequence[GroundedGenerationExample]) -> Sequence[Any]: ...


class CounterfactualActionProvider(Protocol):
    """Admitted runtime simulator/logger used later to estimate per-action utility."""
    def action_utilities(self, step: DynamicRagEpisodeStep) -> Mapping[DynamicRetrievalAction, float]: ...


@dataclass(frozen=True)
class DynamicRewardConfig:
    discount: float = 0.99
    gae_lambda: float = 0.95
    retrieval_cost: float = 0.02
    verification_cost: float = 0.01
    abstention_cost: float = 0.05

    def __post_init__(self) -> None:
        for name in ("discount", "gae_lambda"):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
            object.__setattr__(self, name, value)
        for name in ("retrieval_cost", "verification_cost", "abstention_cost"):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)

    def action_cost(self, action: DynamicRetrievalAction) -> float:
        if action == DynamicRetrievalAction.RETRIEVE:
            return self.retrieval_cost
        if action == DynamicRetrievalAction.VERIFY:
            return self.verification_cost
        if action == DynamicRetrievalAction.ABSTAIN:
            return self.abstention_cost
        return 0.0


@dataclass(frozen=True)
class DynamicTrajectoryTargets:
    returns: tuple[float, ...]
    advantages: tuple[float, ...]


def discounted_returns(rewards: Sequence[float], *, discount: float = 0.99, bootstrap: float = 0.0) -> tuple[float, ...]:
    gamma = _finite(discount, "discount")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("discount must lie in [0,1]")
    running = _finite(bootstrap, "bootstrap")
    result = [0.0] * len(rewards)
    for index in range(len(rewards) - 1, -1, -1):
        running = _finite(rewards[index], f"reward[{index}]") + gamma * running
        result[index] = running
    return tuple(result)


def generalized_advantage_estimate(rewards: Sequence[float], values: Sequence[float], *, discount: float = 0.99, gae_lambda: float = 0.95, bootstrap_value: float = 0.0) -> DynamicTrajectoryTargets:
    if len(rewards) != len(values):
        raise ValueError("rewards and values must align")
    gamma, lam = _finite(discount, "discount"), _finite(gae_lambda, "gae_lambda")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= lam <= 1.0:
        raise ValueError("discount and gae_lambda must lie in [0,1]")
    next_value = _finite(bootstrap_value, "bootstrap_value")
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        reward = _finite(rewards[index], f"reward[{index}]")
        value = _finite(values[index], f"value[{index}]")
        delta = reward + gamma * next_value - value
        gae = delta + gamma * lam * gae
        advantages[index] = gae
        next_value = value
    returns = tuple(_finite(values[i], f"value[{i}]") + advantages[i] for i in range(len(values)))
    return DynamicTrajectoryTargets(returns=returns, advantages=tuple(advantages))


def importance_ratio(target_action_probability: float, behavior_action_probability: float, *, maximum: float = 100.0) -> float:
    target = _finite(target_action_probability, "target_action_probability")
    behavior = _finite(behavior_action_probability, "behavior_action_probability")
    if not 0.0 <= target <= 1.0 or not 0.0 < behavior <= 1.0:
        raise ValueError("target probability must be [0,1] and behavior probability must be (0,1]")
    cap = _finite(maximum, "maximum")
    if cap <= 0.0:
        raise ValueError("maximum must be positive")
    return min(cap, target / behavior)


def counterfactual_action_target(utilities: Mapping[DynamicRetrievalAction, float], config: DynamicRewardConfig = DynamicRewardConfig()) -> tuple[DynamicRetrievalAction, float]:
    if not utilities:
        raise ValueError("counterfactual utilities are required")
    adjusted: list[tuple[float, str, DynamicRetrievalAction]] = []
    for raw_action, raw_utility in utilities.items():
        action = raw_action if isinstance(raw_action, DynamicRetrievalAction) else DynamicRetrievalAction(raw_action)
        utility = _finite(raw_utility, f"utility[{action.value}]") - config.action_cost(action)
        adjusted.append((utility, action.value, action))
    adjusted.sort(key=lambda item: (-item[0], item[1]))
    best = adjusted[0]
    baseline = next((item[0] for item in adjusted if item[2] == DynamicRetrievalAction.CONTINUE), 0.0)
    return best[2], best[0] - baseline


def trajectory_rewards(steps: Sequence[DynamicRagEpisodeStep], config: DynamicRewardConfig = DynamicRewardConfig()) -> tuple[float, ...]:
    rewards = []
    for step in steps:
        reward = step.realized_retrieval_gain - config.action_cost(step.action)
        if step.terminal_utility is not None:
            reward += step.terminal_utility
        rewards.append(_finite(reward, "trajectory reward"))
    return tuple(rewards)


class SafetensorSupervisionCache:
    """Content-addressed tensor sidecars with immutable JSON manifests.

    Safetensors is used deliberately; no pickle-bearing cache is accepted. The cache identity
    binds producer/tokenizer/dataset/source/config so stale teacher or hidden-state tensors
    cannot silently enter a different training run.
    """
    def __init__(self, root: str | Path, identity: SupervisionCacheIdentity) -> None:
        if not isinstance(identity, SupervisionCacheIdentity):
            raise ValueError("identity must be SupervisionCacheIdentity")
        self.root = Path(root).expanduser().resolve()
        self.identity = identity
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, key: str) -> tuple[Path, Path]:
        selected = _identifier(key, "cache key", 1000)
        digest = hashlib.sha256(selected.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.safetensors", self.root / f"{digest}.json"

    def put(self, key: str, tensors: Mapping[str, Any]) -> str:
        _require_torch()
        try:
            from safetensors.torch import save_file
        except Exception as exc:
            raise RuntimeError("safetensors is required for supervision cache writes") from exc
        if not tensors:
            raise ValueError("cache tensor mapping may not be empty")
        normalized = {}
        for name, tensor in tensors.items():
            tensor_name = _identifier(str(name), "tensor name", 300)
            if not torch.is_tensor(tensor):
                raise ValueError("supervision cache values must be tensors")
            normalized[tensor_name] = tensor.detach().cpu().contiguous()
        tensor_path, manifest_path = self._paths(key)
        with tempfile.NamedTemporaryFile(prefix=".rag-cache-", suffix=".safetensors", dir=self.root, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            save_file(normalized, str(temporary))
            tensor_sha = hashlib.sha256(temporary.read_bytes()).hexdigest()
            os.replace(temporary, tensor_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        manifest = {"schema": "rigorousrag-supervision-cache-entry/v1", "key": key, "identity_sha256": self.identity.digest, "tensor_sha256": tensor_sha, "tensor_names": sorted(normalized)}
        payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
        descriptor, temp_name = tempfile.mkstemp(prefix=".rag-cache-manifest-", suffix=".json", dir=self.root)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload); handle.flush(); os.fsync(handle.fileno())
            os.replace(temp_name, manifest_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return tensor_sha

    def get(self, key: str) -> Mapping[str, Any]:
        _require_torch()
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError("safetensors is required for supervision cache reads") from exc
        tensor_path, manifest_path = self._paths(key)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "rigorousrag-supervision-cache-entry/v1" or manifest.get("identity_sha256") != self.identity.digest or manifest.get("key") != key:
            raise ValueError("supervision cache manifest identity mismatch")
        actual = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
        if actual != manifest.get("tensor_sha256"):
            raise ValueError("supervision cache tensor digest mismatch")
        tensors = load_file(str(tensor_path), device="cpu")
        if sorted(tensors) != list(manifest.get("tensor_names") or []):
            raise ValueError("supervision cache tensor names differ from manifest")
        return tensors


class GroundedSupervisionMaterializer:
    """Explicit execution object for teacher/reference/retriever supervision caches."""
    def __init__(self, *, reference_provider: SequenceReferenceProvider | None = None, teacher_provider: TeacherLogitProvider | None = None, document_utility_provider: DocumentUtilityProvider | None = None) -> None:
        self.reference_provider = reference_provider
        self.teacher_provider = teacher_provider
        self.document_utility_provider = document_utility_provider

    def preference_log_probs(self, examples: Sequence[GroundedGenerationExample]) -> tuple[tuple[float, float], ...]:
        if self.reference_provider is None:
            raise RuntimeError("reference provider is not configured")
        values = tuple(self.reference_provider.sequence_log_probabilities(examples))
        if len(values) != len(examples):
            raise ValueError("reference provider returned wrong number of preference scores")
        return tuple((_finite(a, "chosen log probability"), _finite(b, "rejected log probability")) for a, b in values)

    def teacher_logits(self, examples: Sequence[GroundedGenerationExample]) -> tuple[Any, ...]:
        if self.teacher_provider is None:
            raise RuntimeError("teacher provider is not configured")
        values = tuple(self.teacher_provider.token_logits(examples))
        if len(values) != len(examples):
            raise ValueError("teacher provider returned wrong number of logit tensors")
        return values

    def document_utilities(self, examples: Sequence[GroundedGenerationExample]) -> tuple[Any, ...]:
        if self.document_utility_provider is None:
            raise RuntimeError("document utility provider is not configured")
        values = tuple(self.document_utility_provider.document_log_likelihoods(examples))
        if len(values) != len(examples):
            raise ValueError("document utility provider returned wrong number of tensors")
        return values


__all__ = ["CounterfactualActionProvider", "DocumentUtilityProvider", "DynamicRewardConfig", "DynamicTrajectoryTargets", "GroundedSupervisionMaterializer", "SafetensorSupervisionCache", "SequenceReferenceProvider", "SupervisionCacheIdentity", "TeacherLogitProvider", "canonical_digest", "counterfactual_action_target", "discounted_returns", "generalized_advantage_estimate", "importance_ratio", "trajectory_rewards"]
