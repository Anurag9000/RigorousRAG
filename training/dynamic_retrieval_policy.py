"""Generation-time retrieval-control learning and bounded runtime state for RigorousRAG.

Query-level plan ranking answers *which retrieval strategy should a request use*.  Dynamic
RAG has a different control problem: while an answer is being generated, decide whether the
next step should continue generation, retrieve new evidence, verify existing evidence,
abstain, or stop, and identify which part of the generation context expresses the current
information need.

This module provides source-only, executable PyTorch/reference machinery for that second
problem.  It does not call a generator or retriever, download a model, or start a training
loop.  The public contracts deliberately separate:

* action-policy learning;
* information-need span selection;
* counterfactual retrieval-value prediction;
* optional off-policy policy-gradient refinement;
* hard runtime budgets and action masking; and
* immutable plan/checkpoint identity.

The implementation is inspired by the family of on-demand/self-reflective and dynamic RAG
methods, but it exposes repository-owned closed action semantics instead of granting model
text authority to trigger tools directly.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

try:  # Optional training dependency.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency boundary.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

_HEX = frozenset("0123456789abcdef")
_MAX_FEATURES = 256
_MAX_HIDDEN = 1_000_000
_MAX_STAGES = 64
_MAX_TOKENS = 10_000_000
_MAX_RETRIEVALS = 100_000


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum:
        raise ValueError(f"{label} is empty or too long")
    if any(ord(character) < 32 or ord(character) == 127 for character in selected):
        raise ValueError(f"{label} contains control characters")
    return selected


def _sha256(value: Any, label: str) -> str:
    selected = _identifier(value, label, 64).lower()
    if len(selected) != 64 or any(character not in _HEX for character in selected):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return selected


def _git_commit(value: Any) -> str:
    selected = _identifier(value, "source_commit", 64).lower()
    if len(selected) not in {40, 64} or any(character not in _HEX for character in selected):
        raise ValueError("source_commit must be a full 40- or 64-character Git object id")
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


def _bounded(value: Any, label: str, minimum: float, maximum: float) -> float:
    selected = _finite(value, label)
    if not minimum <= selected <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
    return selected


def _nonnegative(value: Any, label: str, maximum: float = 1e9) -> float:
    return _bounded(value, label, 0.0, maximum)


def _positive(value: Any, label: str, maximum: float = 1e9) -> float:
    selected = _finite(value, label)
    if not 0.0 < selected <= maximum:
        raise ValueError(f"{label} must be positive and bounded")
    return selected


def _bounded_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer between {minimum} and {maximum}")
    return value


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError("dynamic-retrieval policy learning requires the optional PyTorch dependency")


class DynamicRetrievalAction(str, Enum):
    CONTINUE = "continue"
    RETRIEVE = "retrieve"
    VERIFY = "verify"
    ABSTAIN = "abstain"
    STOP = "stop"


class DynamicRetrievalStageKind(str, Enum):
    IMITATION = "imitation"
    NEED_SELECTION = "need_selection"
    VALUE = "value"
    OFF_POLICY = "off_policy"
    JOINT = "joint"


DEFAULT_FEATURE_NAMES: tuple[str, ...] = (
    "token_entropy",
    "top1_margin",
    "evidence_sufficiency",
    "semantic_support",
    "contradiction_risk",
    "citation_coverage",
    "context_novelty",
    "unresolved_entity_ratio",
    "temporal_uncertainty",
    "retrieval_count_fraction",
    "token_budget_fraction",
    "elapsed_budget_fraction",
)


@dataclass(frozen=True)
class DynamicRetrievalFeatures:
    token_entropy: float
    top1_margin: float
    evidence_sufficiency: float
    semantic_support: float
    contradiction_risk: float
    citation_coverage: float
    context_novelty: float
    unresolved_entity_ratio: float
    temporal_uncertainty: float
    retrieval_count_fraction: float
    token_budget_fraction: float
    elapsed_budget_fraction: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_entropy", _nonnegative(self.token_entropy, "token_entropy", 1e6))
        for name in (
            "top1_margin",
            "evidence_sufficiency",
            "semantic_support",
            "contradiction_risk",
            "citation_coverage",
            "context_novelty",
            "unresolved_entity_ratio",
            "temporal_uncertainty",
            "retrieval_count_fraction",
            "token_budget_fraction",
            "elapsed_budget_fraction",
        ):
            object.__setattr__(self, name, _bounded(getattr(self, name), name, 0.0, 1.0))

    def vector(self, names: Sequence[str] = DEFAULT_FEATURE_NAMES) -> tuple[float, ...]:
        selected = tuple(names)
        if not selected or len(selected) > _MAX_FEATURES:
            raise ValueError("feature names must be non-empty and bounded")
        if len(set(selected)) != len(selected):
            raise ValueError("feature names must be unique")
        values: list[float] = []
        for name in selected:
            if name not in DEFAULT_FEATURE_NAMES:
                raise ValueError(f"unsupported dynamic-retrieval feature: {name}")
            values.append(float(getattr(self, name)))
        return tuple(values)


@dataclass(frozen=True)
class DynamicRetrievalBudget:
    max_generation_tokens: int
    max_retrievals: int
    max_verifications: int
    min_tokens_before_retrieval: int = 0
    min_tokens_before_stop: int = 1
    max_consecutive_retrievals: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_generation_tokens",
            _bounded_int(self.max_generation_tokens, "max_generation_tokens", 1, _MAX_TOKENS),
        )
        object.__setattr__(self, "max_retrievals", _bounded_int(self.max_retrievals, "max_retrievals", 0, _MAX_RETRIEVALS))
        object.__setattr__(
            self,
            "max_verifications",
            _bounded_int(self.max_verifications, "max_verifications", 0, _MAX_RETRIEVALS),
        )
        object.__setattr__(
            self,
            "min_tokens_before_retrieval",
            _bounded_int(self.min_tokens_before_retrieval, "min_tokens_before_retrieval", 0, self.max_generation_tokens),
        )
        object.__setattr__(
            self,
            "min_tokens_before_stop",
            _bounded_int(self.min_tokens_before_stop, "min_tokens_before_stop", 0, self.max_generation_tokens),
        )
        object.__setattr__(
            self,
            "max_consecutive_retrievals",
            _bounded_int(self.max_consecutive_retrievals, "max_consecutive_retrievals", 1, max(1, self.max_retrievals or 1)),
        )

    @property
    def budget_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-dynamic-retrieval-budget/v1", **asdict(self)})


@dataclass(frozen=True)
class DynamicPolicyArchitecture:
    feature_names: tuple[str, ...] = DEFAULT_FEATURE_NAMES
    hidden_size: int = 128
    context_hidden_size: int = 768
    need_projection_size: int = 128
    actions: tuple[DynamicRetrievalAction, ...] = tuple(DynamicRetrievalAction)

    def __post_init__(self) -> None:
        names = tuple(_identifier(name, "feature name", 80) for name in self.feature_names)
        if not names or len(names) > _MAX_FEATURES or len(set(names)) != len(names):
            raise ValueError("feature_names must be a unique non-empty bounded sequence")
        if any(name not in DEFAULT_FEATURE_NAMES for name in names):
            raise ValueError("feature_names contains an unsupported feature")
        object.__setattr__(self, "feature_names", names)
        for name in ("hidden_size", "context_hidden_size", "need_projection_size"):
            object.__setattr__(self, name, _bounded_int(getattr(self, name), name, 1, _MAX_HIDDEN))
        actions = tuple(action if isinstance(action, DynamicRetrievalAction) else DynamicRetrievalAction(action) for action in self.actions)
        if not actions or len(set(actions)) != len(actions):
            raise ValueError("actions must be unique and non-empty")
        required = {DynamicRetrievalAction.CONTINUE, DynamicRetrievalAction.RETRIEVE, DynamicRetrievalAction.ABSTAIN, DynamicRetrievalAction.STOP}
        if not required.issubset(set(actions)):
            raise ValueError("dynamic policy must contain continue/retrieve/abstain/stop actions")
        object.__setattr__(self, "actions", actions)

    @property
    def architecture_sha256(self) -> str:
        return _digest(
            {
                "schema": "rigorousrag-dynamic-retrieval-architecture/v1",
                "feature_names": self.feature_names,
                "hidden_size": self.hidden_size,
                "context_hidden_size": self.context_hidden_size,
                "need_projection_size": self.need_projection_size,
                "actions": [action.value for action in self.actions],
            }
        )


@dataclass(frozen=True)
class DynamicPolicyObjective:
    action_weight: float = 1.0
    need_selection_weight: float = 0.0
    value_weight: float = 0.0
    policy_gradient_weight: float = 0.0
    retrieval_cost_weight: float = 0.0
    verification_cost_weight: float = 0.0
    abstention_cost_weight: float = 0.0
    entropy_bonus_weight: float = 0.0

    def __post_init__(self) -> None:
        total = 0.0
        for name in (
            "action_weight",
            "need_selection_weight",
            "value_weight",
            "policy_gradient_weight",
            "retrieval_cost_weight",
            "verification_cost_weight",
            "abstention_cost_weight",
            "entropy_bonus_weight",
        ):
            value = _nonnegative(getattr(self, name), name, 1e6)
            object.__setattr__(self, name, value)
            if name not in {"retrieval_cost_weight", "verification_cost_weight", "abstention_cost_weight", "entropy_bonus_weight"}:
                total += value
        if total <= 0.0:
            raise ValueError("at least one learnable dynamic-policy objective must be positive")

    @property
    def objective_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-dynamic-retrieval-objective/v1", **asdict(self)})


@dataclass(frozen=True)
class DynamicPolicyStage:
    name: str
    kind: DynamicRetrievalStageKind
    max_steps: int
    checkpoint_every_steps: int
    learning_rate: float
    objective: DynamicPolicyObjective

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "stage name", 160))
        if not isinstance(self.kind, DynamicRetrievalStageKind):
            object.__setattr__(self, "kind", DynamicRetrievalStageKind(self.kind))
        object.__setattr__(self, "max_steps", _bounded_int(self.max_steps, "max_steps", 1, 10**12))
        object.__setattr__(
            self,
            "checkpoint_every_steps",
            _bounded_int(self.checkpoint_every_steps, "checkpoint_every_steps", 1, self.max_steps),
        )
        object.__setattr__(self, "learning_rate", _positive(self.learning_rate, "learning_rate", 10.0))
        if not isinstance(self.objective, DynamicPolicyObjective):
            raise ValueError("objective must be DynamicPolicyObjective")

    @property
    def stage_sha256(self) -> str:
        return _digest(
            {
                "schema": "rigorousrag-dynamic-retrieval-stage/v1",
                "name": self.name,
                "kind": self.kind.value,
                "max_steps": self.max_steps,
                "checkpoint_every_steps": self.checkpoint_every_steps,
                "learning_rate": self.learning_rate,
                "objective_sha256": self.objective.objective_sha256,
            }
        )


@dataclass(frozen=True)
class DynamicPolicyTrainingPlan:
    run_id: str
    architecture: DynamicPolicyArchitecture
    budget: DynamicRetrievalBudget
    dataset_manifest_sha256: str
    base_generator_sha256: str
    retrieval_stack_sha256: str
    source_commit: str
    stages: tuple[DynamicPolicyStage, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id", 240))
        if not isinstance(self.architecture, DynamicPolicyArchitecture):
            raise ValueError("architecture must be DynamicPolicyArchitecture")
        if not isinstance(self.budget, DynamicRetrievalBudget):
            raise ValueError("budget must be DynamicRetrievalBudget")
        for name in ("dataset_manifest_sha256", "base_generator_sha256", "retrieval_stack_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(self, "source_commit", _git_commit(self.source_commit))
        stages = tuple(self.stages)
        if not stages or len(stages) > _MAX_STAGES or any(not isinstance(stage, DynamicPolicyStage) for stage in stages):
            raise ValueError("stages must be a non-empty bounded DynamicPolicyStage sequence")
        if len({stage.name for stage in stages}) != len(stages):
            raise ValueError("dynamic-policy stage names must be unique")
        object.__setattr__(self, "stages", stages)

    @property
    def plan_sha256(self) -> str:
        return _digest(
            {
                "schema": "rigorousrag-dynamic-policy-training-plan/v1",
                "run_id": self.run_id,
                "architecture_sha256": self.architecture.architecture_sha256,
                "budget_sha256": self.budget.budget_sha256,
                "dataset_manifest_sha256": self.dataset_manifest_sha256,
                "base_generator_sha256": self.base_generator_sha256,
                "retrieval_stack_sha256": self.retrieval_stack_sha256,
                "source_commit": self.source_commit,
                "stage_sha256s": [stage.stage_sha256 for stage in self.stages],
            }
        )

    def checkpoint_bindings(self, *, stage_index: int, data_cursor_sha256: str, replay_buffer_sha256: str | None = None) -> Mapping[str, Any]:
        if isinstance(stage_index, bool) or not isinstance(stage_index, int) or not 0 <= stage_index < len(self.stages):
            raise ValueError("stage_index is outside the dynamic-policy training plan")
        stage = self.stages[stage_index]
        return {
            "schema": "rigorousrag-dynamic-policy-checkpoint-binding/v1",
            "run_id": self.run_id,
            "training_config_digest": self.plan_sha256,
            "data_manifest_digest": self.dataset_manifest_sha256,
            "source_commit": self.source_commit,
            "stage_index": stage_index,
            "stage_name": stage.name,
            "stage_sha256": stage.stage_sha256,
            "data_cursor_sha256": _sha256(data_cursor_sha256, "data_cursor_sha256"),
            "replay_buffer_sha256": None if replay_buffer_sha256 is None else _sha256(replay_buffer_sha256, "replay_buffer_sha256"),
        }


if nn is not None:

    class DynamicRetrievalController(nn.Module):
        """Small policy/value network over bounded server-owned retrieval signals."""

        def __init__(self, config: DynamicPolicyArchitecture) -> None:
            super().__init__()
            if not isinstance(config, DynamicPolicyArchitecture):
                raise ValueError("config must be DynamicPolicyArchitecture")
            self.config = config
            width = len(config.feature_names)
            self.encoder = nn.Sequential(
                nn.Linear(width, config.hidden_size),
                nn.GELU(),
                nn.LayerNorm(config.hidden_size),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
            )
            self.action_head = nn.Linear(config.hidden_size, len(config.actions))
            self.value_head = nn.Linear(config.hidden_size, 1)

        def forward(self, features: Any) -> Mapping[str, Any]:
            if features.ndim != 2 or features.size(-1) != len(self.config.feature_names):
                raise ValueError("dynamic policy features must have shape [B,F] matching the architecture")
            hidden = self.encoder(features)
            return {
                "action_logits": self.action_head(hidden),
                "retrieval_value": self.value_head(hidden).squeeze(-1),
            }


    class InformationNeedSelector(nn.Module):
        """Scores generated-context tokens for their relevance to the current information need."""

        def __init__(self, config: DynamicPolicyArchitecture) -> None:
            super().__init__()
            if not isinstance(config, DynamicPolicyArchitecture):
                raise ValueError("config must be DynamicPolicyArchitecture")
            self.config = config
            self.token_projection = nn.Linear(config.context_hidden_size, config.need_projection_size, bias=False)
            self.state_projection = nn.Linear(config.context_hidden_size, config.need_projection_size, bias=False)

        def forward(self, token_hidden: Any, state_hidden: Any, attention_mask: Any | None = None) -> Any:
            if token_hidden.ndim != 3 or state_hidden.ndim != 2:
                raise ValueError("information-need hidden states must have shapes [B,T,H] and [B,H]")
            if token_hidden.size(0) != state_hidden.size(0) or token_hidden.size(-1) != self.config.context_hidden_size or state_hidden.size(-1) != self.config.context_hidden_size:
                raise ValueError("information-need hidden-state shapes do not match architecture")
            token_key = self.token_projection(token_hidden)
            state_key = self.state_projection(state_hidden).unsqueeze(1)
            logits = (token_key * state_key).sum(dim=-1) / math.sqrt(float(self.config.need_projection_size))
            if attention_mask is not None:
                if attention_mask.shape != logits.shape:
                    raise ValueError("attention_mask must align with information-need logits")
                logits = logits.masked_fill(~attention_mask.to(dtype=torch.bool), torch.finfo(logits.dtype).min)
            return logits

else:

    class DynamicRetrievalController:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()

    class InformationNeedSelector:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


@dataclass
class TensorDynamicPolicyLoss:
    total: Any
    action: Any | None = None
    need_selection: Any | None = None
    value: Any | None = None
    policy_gradient: Any | None = None
    retrieval_cost: Any | None = None
    verification_cost: Any | None = None
    abstention_cost: Any | None = None
    entropy: Any | None = None


def action_imitation_loss(logits: Any, targets: Any, *, class_weights: Any | None = None, ignore_index: int = -100) -> Any:
    _require_torch()
    if logits.ndim != 2 or targets.ndim != 1 or logits.size(0) != targets.size(0):
        raise ValueError("action logits/targets must have shapes [B,A] and [B]")
    if class_weights is not None and (class_weights.ndim != 1 or class_weights.numel() != logits.size(1)):
        raise ValueError("class_weights must contain one weight per action")
    return F.cross_entropy(logits, targets.long(), weight=class_weights, ignore_index=ignore_index)


def information_need_bce_loss(logits: Any, target_mask: Any, *, valid_mask: Any | None = None) -> Any:
    _require_torch()
    if logits.ndim != 2 or target_mask.shape != logits.shape:
        raise ValueError("information-need logits/targets must have aligned [B,T] shapes")
    targets = target_mask.to(dtype=logits.dtype)
    if torch.any(targets < 0.0) or torch.any(targets > 1.0):
        raise ValueError("information-need targets must lie in [0,1]")
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if valid_mask is None:
        return loss.mean()
    if valid_mask.shape != logits.shape:
        raise ValueError("valid_mask must align with information-need logits")
    selected = valid_mask.to(dtype=loss.dtype)
    denominator = selected.sum()
    if float(denominator.detach().item()) <= 0.0:
        raise ValueError("valid_mask selects no information-need tokens")
    return (loss * selected).sum() / denominator


def retrieval_value_loss(predicted_value: Any, realized_gain: Any, *, huber_delta: float = 1.0) -> Any:
    _require_torch()
    if predicted_value.shape != realized_gain.shape or predicted_value.numel() == 0:
        raise ValueError("predicted and realized retrieval values must be non-empty and aligned")
    selected_delta = _positive(huber_delta, "huber_delta", 1e6)
    return F.huber_loss(predicted_value, realized_gain.detach().to(dtype=predicted_value.dtype), delta=selected_delta)


def offpolicy_policy_gradient_loss(
    selected_action_log_prob: Any,
    advantage: Any,
    *,
    importance_ratio: Any | None = None,
    max_importance_ratio: float = 10.0,
) -> Any:
    """Bounded off-policy policy-gradient objective over logged dynamic-RAG episodes."""

    _require_torch()
    if selected_action_log_prob.shape != advantage.shape or selected_action_log_prob.numel() == 0:
        raise ValueError("action log-probabilities and advantages must be non-empty and aligned")
    weight = torch.ones_like(selected_action_log_prob)
    if importance_ratio is not None:
        if importance_ratio.shape != selected_action_log_prob.shape:
            raise ValueError("importance_ratio must align with selected action log-probability")
        cap = _positive(max_importance_ratio, "max_importance_ratio", 1e6)
        weight = torch.clamp(importance_ratio.detach().to(dtype=selected_action_log_prob.dtype), min=0.0, max=cap)
    return -(weight * advantage.detach().to(dtype=selected_action_log_prob.dtype) * selected_action_log_prob).mean()


def action_cost_expectations(
    logits: Any,
    *,
    actions: Sequence[DynamicRetrievalAction],
) -> Mapping[str, Any]:
    _require_torch()
    selected_actions = tuple(action if isinstance(action, DynamicRetrievalAction) else DynamicRetrievalAction(action) for action in actions)
    if logits.ndim != 2 or logits.size(1) != len(selected_actions):
        raise ValueError("action logits must align with the supplied action vocabulary")
    probabilities = F.softmax(logits, dim=-1)

    def probability(action: DynamicRetrievalAction) -> Any:
        if action not in selected_actions:
            return logits.new_zeros(())
        return probabilities[:, selected_actions.index(action)].mean()

    entropy = -(probabilities * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    return {
        "retrieval_cost": probability(DynamicRetrievalAction.RETRIEVE),
        "verification_cost": probability(DynamicRetrievalAction.VERIFY),
        "abstention_cost": probability(DynamicRetrievalAction.ABSTAIN),
        "entropy": entropy,
    }


def dynamic_policy_objective(
    *,
    objective: DynamicPolicyObjective,
    action: Any | None = None,
    need_selection: Any | None = None,
    value: Any | None = None,
    policy_gradient: Any | None = None,
    retrieval_cost: Any | None = None,
    verification_cost: Any | None = None,
    abstention_cost: Any | None = None,
    entropy: Any | None = None,
) -> TensorDynamicPolicyLoss:
    _require_torch()
    if not isinstance(objective, DynamicPolicyObjective):
        raise ValueError("objective must be DynamicPolicyObjective")
    weighted = (
        ("action", objective.action_weight, action, 1.0),
        ("need_selection", objective.need_selection_weight, need_selection, 1.0),
        ("value", objective.value_weight, value, 1.0),
        ("policy_gradient", objective.policy_gradient_weight, policy_gradient, 1.0),
        ("retrieval_cost", objective.retrieval_cost_weight, retrieval_cost, 1.0),
        ("verification_cost", objective.verification_cost_weight, verification_cost, 1.0),
        ("abstention_cost", objective.abstention_cost_weight, abstention_cost, 1.0),
        ("entropy", objective.entropy_bonus_weight, entropy, -1.0),
    )
    total = None
    for name, coefficient, component, sign in weighted:
        if coefficient <= 0.0:
            continue
        if component is None:
            raise ValueError(f"dynamic-policy component {name} is weighted but missing")
        if getattr(component, "numel", lambda: 0)() != 1:
            raise ValueError(f"dynamic-policy component {name} must be scalar")
        value_to_add = sign * coefficient * component
        total = value_to_add if total is None else total + value_to_add
    if total is None:
        raise ValueError("dynamic-policy objective selected no active component")
    return TensorDynamicPolicyLoss(
        total=total,
        action=action,
        need_selection=need_selection,
        value=value,
        policy_gradient=policy_gradient,
        retrieval_cost=retrieval_cost,
        verification_cost=verification_cost,
        abstention_cost=abstention_cost,
        entropy=entropy,
    )


@dataclass(frozen=True)
class DynamicRetrievalRuntimeState:
    generated_tokens: int = 0
    retrievals: int = 0
    verifications: int = 0
    consecutive_retrievals: int = 0
    terminal: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "generated_tokens", _bounded_int(self.generated_tokens, "generated_tokens", 0, _MAX_TOKENS))
        object.__setattr__(self, "retrievals", _bounded_int(self.retrievals, "retrievals", 0, _MAX_RETRIEVALS))
        object.__setattr__(self, "verifications", _bounded_int(self.verifications, "verifications", 0, _MAX_RETRIEVALS))
        object.__setattr__(
            self,
            "consecutive_retrievals",
            _bounded_int(self.consecutive_retrievals, "consecutive_retrievals", 0, _MAX_RETRIEVALS),
        )
        if not isinstance(self.terminal, bool):
            raise ValueError("terminal must be boolean")


def allowed_actions(state: DynamicRetrievalRuntimeState, budget: DynamicRetrievalBudget, *, verification_enabled: bool = True) -> tuple[DynamicRetrievalAction, ...]:
    if not isinstance(state, DynamicRetrievalRuntimeState) or not isinstance(budget, DynamicRetrievalBudget):
        raise ValueError("state and budget have invalid types")
    if state.terminal:
        return ()
    if state.generated_tokens >= budget.max_generation_tokens:
        return (DynamicRetrievalAction.ABSTAIN, DynamicRetrievalAction.STOP)
    actions: list[DynamicRetrievalAction] = [DynamicRetrievalAction.CONTINUE, DynamicRetrievalAction.ABSTAIN]
    if state.generated_tokens >= budget.min_tokens_before_stop:
        actions.append(DynamicRetrievalAction.STOP)
    if (
        state.generated_tokens >= budget.min_tokens_before_retrieval
        and state.retrievals < budget.max_retrievals
        and state.consecutive_retrievals < budget.max_consecutive_retrievals
    ):
        actions.append(DynamicRetrievalAction.RETRIEVE)
    if verification_enabled and state.verifications < budget.max_verifications:
        actions.append(DynamicRetrievalAction.VERIFY)
    return tuple(actions)


def mask_action_logits(
    logits: Any,
    *,
    action_vocabulary: Sequence[DynamicRetrievalAction],
    allowed: Sequence[DynamicRetrievalAction],
) -> Any:
    _require_torch()
    vocabulary = tuple(action if isinstance(action, DynamicRetrievalAction) else DynamicRetrievalAction(action) for action in action_vocabulary)
    selected = set(action if isinstance(action, DynamicRetrievalAction) else DynamicRetrievalAction(action) for action in allowed)
    if logits.ndim < 1 or logits.size(-1) != len(vocabulary):
        raise ValueError("action logits do not align with action vocabulary")
    if not selected or not selected.issubset(set(vocabulary)):
        raise ValueError("allowed actions must be a non-empty subset of the action vocabulary")
    mask = torch.tensor([action in selected for action in vocabulary], device=logits.device, dtype=torch.bool)
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def transition_runtime_state(state: DynamicRetrievalRuntimeState, action: DynamicRetrievalAction, budget: DynamicRetrievalBudget, *, generated_tokens_delta: int = 0) -> DynamicRetrievalRuntimeState:
    if not isinstance(action, DynamicRetrievalAction):
        action = DynamicRetrievalAction(action)
    permitted = allowed_actions(state, budget)
    if action not in permitted:
        raise ValueError("dynamic retrieval action is not permitted by the current hard budget")
    token_delta = _bounded_int(generated_tokens_delta, "generated_tokens_delta", 0, _MAX_TOKENS)
    if action != DynamicRetrievalAction.CONTINUE and token_delta != 0:
        raise ValueError("only continue may advance generated token count")
    generated = state.generated_tokens + token_delta
    if generated > budget.max_generation_tokens:
        raise ValueError("generation token budget exceeded")
    retrievals = state.retrievals + (1 if action == DynamicRetrievalAction.RETRIEVE else 0)
    verifications = state.verifications + (1 if action == DynamicRetrievalAction.VERIFY else 0)
    consecutive = state.consecutive_retrievals + 1 if action == DynamicRetrievalAction.RETRIEVE else 0
    return DynamicRetrievalRuntimeState(
        generated_tokens=generated,
        retrievals=retrievals,
        verifications=verifications,
        consecutive_retrievals=consecutive,
        terminal=action in {DynamicRetrievalAction.ABSTAIN, DynamicRetrievalAction.STOP},
    )


@dataclass(frozen=True)
class InformationNeedSpan:
    start_token: int
    end_token: int
    score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_token", _bounded_int(self.start_token, "start_token", 0, _MAX_TOKENS))
        object.__setattr__(self, "end_token", _bounded_int(self.end_token, "end_token", self.start_token + 1, _MAX_TOKENS))
        object.__setattr__(self, "score", _finite(self.score, "score"))


def select_information_need_spans(
    token_scores: Sequence[Any],
    *,
    threshold: float = 0.5,
    max_spans: int = 4,
    max_span_tokens: int = 64,
) -> tuple[InformationNeedSpan, ...]:
    """Convert per-token need probabilities into deterministic bounded contiguous spans."""

    if not token_scores or len(token_scores) > _MAX_TOKENS:
        raise ValueError("token_scores must be a non-empty bounded sequence")
    selected_threshold = _bounded(threshold, "threshold", 0.0, 1.0)
    selected_max_spans = _bounded_int(max_spans, "max_spans", 1, 10_000)
    selected_max_tokens = _bounded_int(max_span_tokens, "max_span_tokens", 1, _MAX_TOKENS)
    probabilities = tuple(_bounded(value, "token need score", 0.0, 1.0) for value in token_scores)
    raw: list[InformationNeedSpan] = []
    start: int | None = None
    values: list[float] = []
    for index, score in enumerate((*probabilities, -1.0)):
        active = index < len(probabilities) and score >= selected_threshold
        if active and start is None:
            start = index
            values = [score]
        elif active:
            values.append(score)
        elif start is not None:
            stop = index
            cursor = start
            while cursor < stop:
                end = min(stop, cursor + selected_max_tokens)
                local = probabilities[cursor:end]
                raw.append(InformationNeedSpan(cursor, end, sum(local) / len(local)))
                cursor = end
            start = None
            values = []
    raw.sort(key=lambda span: (-span.score, span.start_token, span.end_token))
    return tuple(raw[:selected_max_spans])


__all__ = [
    "DEFAULT_FEATURE_NAMES",
    "DynamicPolicyArchitecture",
    "DynamicPolicyObjective",
    "DynamicPolicyStage",
    "DynamicPolicyTrainingPlan",
    "DynamicRetrievalAction",
    "DynamicRetrievalBudget",
    "DynamicRetrievalController",
    "DynamicRetrievalFeatures",
    "DynamicRetrievalRuntimeState",
    "DynamicRetrievalStageKind",
    "InformationNeedSelector",
    "InformationNeedSpan",
    "TensorDynamicPolicyLoss",
    "action_cost_expectations",
    "action_imitation_loss",
    "allowed_actions",
    "dynamic_policy_objective",
    "information_need_bce_loss",
    "mask_action_logits",
    "offpolicy_policy_gradient_loss",
    "retrieval_value_loss",
    "select_information_need_spans",
    "transition_runtime_state",
]
