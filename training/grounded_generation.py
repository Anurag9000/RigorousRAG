"""Grounded-generator training architectures, objectives and resume bindings for RigorousRAG.

The repository already contains executable training support for retrievers, learned sparse
models, late interaction, rerankers, query planning and cross-profile fusion.  This module
closes the complementary generator-side learning surface without loading a model or dataset
at import time.

The design is intentionally backend-neutral at the language-model boundary while providing
executable PyTorch auxiliary heads and differentiable reference losses for:

* supervised token generation;
* claim-to-evidence citation attribution;
* claim support and contradiction discrimination;
* abstention and reflection/action prediction;
* unsupported-token probability-mass unlikelihood;
* grounded preference optimization (DPO-style chosen/rejected pairs);
* teacher distillation; and
* LM-supervised retriever coupling (REPLUG-style document-utility distillation).

Checkpoint persistence is delegated to :mod:`training.checkpointing`.  A
:class:`GroundedTrainingPlan` is content-addressed and is intended to be supplied as the
training-configuration identity in that checkpoint manifest, while the exact governed
training split remains the data-manifest identity.  No checkpoint, download, optimizer step
or model execution occurs merely by importing this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

try:  # Optional training dependency.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency boundary.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

_HEX = frozenset("0123456789abcdef")
_EPS = 1e-12
_MAX_STAGES = 64
_MAX_REFLECTION_ACTIONS = 32
_MAX_HIDDEN = 1_000_000


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


def _nonnegative(value: Any, label: str, maximum: float = 1e9) -> float:
    selected = _finite(value, label)
    if not 0.0 <= selected <= maximum:
        raise ValueError(f"{label} must be non-negative and bounded")
    return selected


def _positive(value: Any, label: str, maximum: float = 1e9) -> float:
    selected = _finite(value, label)
    if not 0.0 < selected <= maximum:
        raise ValueError(f"{label} must be positive and bounded")
    return selected


def _positive_int(value: Any, label: str, maximum: int = 10**12) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{label} must be a positive bounded integer")
    return value


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError("grounded-generator training requires the optional PyTorch dependency")


class GroundedStageKind(str, Enum):
    SUPERVISED = "supervised"
    ATTRIBUTION = "attribution"
    GROUNDING = "grounding"
    RETRIEVER_COUPLING = "retriever_coupling"
    PREFERENCE = "preference"
    REFLECTION = "reflection"
    JOINT = "joint"


class ReflectionAction(str, Enum):
    CONTINUE = "continue"
    RETRIEVE = "retrieve"
    VERIFY = "verify"
    CITE = "cite"
    ABSTAIN = "abstain"
    STOP = "stop"


@dataclass(frozen=True)
class GroundedObjectiveWeights:
    token_nll: float = 1.0
    citation: float = 0.0
    support: float = 0.0
    contradiction: float = 0.0
    abstention: float = 0.0
    reflection: float = 0.0
    unsupported_unlikelihood: float = 0.0
    preference: float = 0.0
    teacher_distillation: float = 0.0
    retriever_coupling: float = 0.0

    def __post_init__(self) -> None:
        total = 0.0
        for name in (
            "token_nll",
            "citation",
            "support",
            "contradiction",
            "abstention",
            "reflection",
            "unsupported_unlikelihood",
            "preference",
            "teacher_distillation",
            "retriever_coupling",
        ):
            value = _nonnegative(getattr(self, name), name, 1e6)
            object.__setattr__(self, name, value)
            total += value
        if total <= 0.0:
            raise ValueError("at least one grounded objective weight must be positive")

    @property
    def objective_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-grounded-objective/v1", **asdict(self)})


@dataclass(frozen=True)
class GroundedGenerationArchitectureConfig:
    hidden_size: int
    attribution_size: int = 256
    reflection_actions: tuple[ReflectionAction, ...] = tuple(ReflectionAction)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_size", _positive_int(self.hidden_size, "hidden_size", _MAX_HIDDEN))
        object.__setattr__(
            self,
            "attribution_size",
            _positive_int(self.attribution_size, "attribution_size", _MAX_HIDDEN),
        )
        raw_actions = tuple(self.reflection_actions)
        if not raw_actions or len(raw_actions) > _MAX_REFLECTION_ACTIONS:
            raise ValueError("reflection_actions must be non-empty and bounded")
        actions = tuple(action if isinstance(action, ReflectionAction) else ReflectionAction(action) for action in raw_actions)
        if len(set(actions)) != len(actions):
            raise ValueError("reflection_actions must be unique")
        object.__setattr__(self, "reflection_actions", actions)

    @property
    def architecture_sha256(self) -> str:
        return _digest(
            {
                "schema": "rigorousrag-grounded-generation-architecture/v1",
                "hidden_size": self.hidden_size,
                "attribution_size": self.attribution_size,
                "reflection_actions": [action.value for action in self.reflection_actions],
            }
        )


@dataclass(frozen=True)
class GroundedTrainingStage:
    name: str
    kind: GroundedStageKind
    max_steps: int
    checkpoint_every_steps: int
    learning_rate: float
    objective: GroundedObjectiveWeights

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "stage name", 160))
        if not isinstance(self.kind, GroundedStageKind):
            object.__setattr__(self, "kind", GroundedStageKind(self.kind))
        object.__setattr__(self, "max_steps", _positive_int(self.max_steps, "max_steps"))
        object.__setattr__(
            self,
            "checkpoint_every_steps",
            _positive_int(self.checkpoint_every_steps, "checkpoint_every_steps"),
        )
        if self.checkpoint_every_steps > self.max_steps:
            raise ValueError("checkpoint_every_steps may not exceed max_steps")
        object.__setattr__(self, "learning_rate", _positive(self.learning_rate, "learning_rate", 10.0))
        if not isinstance(self.objective, GroundedObjectiveWeights):
            raise ValueError("objective must be GroundedObjectiveWeights")

    @property
    def stage_sha256(self) -> str:
        return _digest(
            {
                "schema": "rigorousrag-grounded-training-stage/v1",
                "name": self.name,
                "kind": self.kind.value,
                "max_steps": self.max_steps,
                "checkpoint_every_steps": self.checkpoint_every_steps,
                "learning_rate": self.learning_rate,
                "objective_sha256": self.objective.objective_sha256,
            }
        )


@dataclass(frozen=True)
class GroundedTrainingPlan:
    run_id: str
    architecture: GroundedGenerationArchitectureConfig
    base_model_sha256: str
    tokenizer_sha256: str
    dataset_manifest_sha256: str
    source_commit: str
    stages: tuple[GroundedTrainingStage, ...]
    retriever_stack_sha256: str | None = None
    teacher_model_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id", 240))
        if not isinstance(self.architecture, GroundedGenerationArchitectureConfig):
            raise ValueError("architecture must be GroundedGenerationArchitectureConfig")
        for name in ("base_model_sha256", "tokenizer_sha256", "dataset_manifest_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(self, "source_commit", _git_commit(self.source_commit))
        selected_stages = tuple(self.stages)
        if not selected_stages or len(selected_stages) > _MAX_STAGES:
            raise ValueError("stages must be a non-empty bounded sequence")
        if any(not isinstance(stage, GroundedTrainingStage) for stage in selected_stages):
            raise ValueError("every stage must be GroundedTrainingStage")
        if len({stage.name for stage in selected_stages}) != len(selected_stages):
            raise ValueError("stage names must be unique")
        object.__setattr__(self, "stages", selected_stages)
        for name in ("retriever_stack_sha256", "teacher_model_sha256"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _sha256(value, name))

    @property
    def plan_sha256(self) -> str:
        return _digest(
            {
                "schema": "rigorousrag-grounded-training-plan/v1",
                "run_id": self.run_id,
                "architecture_sha256": self.architecture.architecture_sha256,
                "base_model_sha256": self.base_model_sha256,
                "tokenizer_sha256": self.tokenizer_sha256,
                "dataset_manifest_sha256": self.dataset_manifest_sha256,
                "source_commit": self.source_commit,
                "stage_sha256s": [stage.stage_sha256 for stage in self.stages],
                "retriever_stack_sha256": self.retriever_stack_sha256,
                "teacher_model_sha256": self.teacher_model_sha256,
            }
        )

    def checkpoint_bindings(self, *, stage_index: int, data_cursor_sha256: str) -> Mapping[str, Any]:
        if isinstance(stage_index, bool) or not isinstance(stage_index, int) or not 0 <= stage_index < len(self.stages):
            raise ValueError("stage_index is outside the training plan")
        cursor = _sha256(data_cursor_sha256, "data_cursor_sha256")
        stage = self.stages[stage_index]
        return {
            "schema": "rigorousrag-grounded-checkpoint-binding/v1",
            "run_id": self.run_id,
            "training_config_digest": self.plan_sha256,
            "data_manifest_digest": self.dataset_manifest_sha256,
            "source_commit": self.source_commit,
            "stage_index": stage_index,
            "stage_name": stage.name,
            "stage_sha256": stage.stage_sha256,
            "objective_sha256": stage.objective.objective_sha256,
            "data_cursor_sha256": cursor,
        }


if nn is not None:

    class GroundedAuxiliaryHeads(nn.Module):
        """Auxiliary attribution/support/abstention/reflection heads over LM hidden states.

        The base causal/seq2seq language model remains injected by deployment/training code.
        Claim and evidence representations must be produced from the exact released training
        context; this head never performs retrieval itself.
        """

        def __init__(self, config: GroundedGenerationArchitectureConfig) -> None:
            super().__init__()
            if not isinstance(config, GroundedGenerationArchitectureConfig):
                raise ValueError("config must be GroundedGenerationArchitectureConfig")
            self.config = config
            self.claim_projection = nn.Linear(config.hidden_size, config.attribution_size, bias=False)
            self.evidence_projection = nn.Linear(config.hidden_size, config.attribution_size, bias=False)
            self.support_head = nn.Linear(config.hidden_size, 1)
            self.contradiction_head = nn.Linear(config.hidden_size, 1)
            self.abstention_head = nn.Linear(config.hidden_size, 1)
            self.reflection_head = nn.Linear(config.hidden_size, len(config.reflection_actions))

        def forward(self, claim_hidden: Any, evidence_hidden: Any, generation_hidden: Any) -> Mapping[str, Any]:
            if claim_hidden.ndim != 3 or evidence_hidden.ndim != 3 or generation_hidden.ndim != 2:
                raise ValueError("claim/evidence/generation hidden states must be [B,C,H], [B,E,H], [B,H]")
            batch = claim_hidden.size(0)
            if evidence_hidden.size(0) != batch or generation_hidden.size(0) != batch:
                raise ValueError("grounded hidden-state batches must align")
            hidden = self.config.hidden_size
            if claim_hidden.size(-1) != hidden or evidence_hidden.size(-1) != hidden or generation_hidden.size(-1) != hidden:
                raise ValueError("hidden-state width does not match architecture config")
            claim_keys = self.claim_projection(claim_hidden)
            evidence_keys = self.evidence_projection(evidence_hidden)
            citation_logits = torch.einsum("bch,beh->bce", claim_keys, evidence_keys) / math.sqrt(
                float(self.config.attribution_size)
            )
            return {
                "citation_logits": citation_logits,
                "support_logits": self.support_head(claim_hidden).squeeze(-1),
                "contradiction_logits": self.contradiction_head(claim_hidden).squeeze(-1),
                "abstention_logits": self.abstention_head(generation_hidden).squeeze(-1),
                "reflection_logits": self.reflection_head(generation_hidden),
            }

else:

    class GroundedAuxiliaryHeads:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


@dataclass
class TensorGroundedLossBreakdown:
    total: Any
    token_nll: Any | None = None
    citation: Any | None = None
    support: Any | None = None
    contradiction: Any | None = None
    abstention: Any | None = None
    reflection: Any | None = None
    unsupported_unlikelihood: Any | None = None
    preference: Any | None = None
    teacher_distillation: Any | None = None
    retriever_coupling: Any | None = None


def masked_token_nll(logits: Any, labels: Any, *, ignore_index: int = -100) -> Any:
    _require_torch()
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("token logits/labels must have shapes [B,T,V] and [B,T]")
    if logits.size(-1) < 2:
        raise ValueError("token vocabulary must contain at least two entries")
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1).long(), ignore_index=ignore_index)


def sequence_log_prob(logits: Any, labels: Any, *, ignore_index: int = -100) -> Any:
    """Return one summed log-probability per sequence for preference objectives."""

    _require_torch()
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("token logits/labels must have aligned [B,T,V]/[B,T] shapes")
    log_probability = F.log_softmax(logits, dim=-1)
    safe_labels = labels.long().clone()
    mask = safe_labels.ne(ignore_index)
    safe_labels = safe_labels.masked_fill(~mask, 0)
    selected = log_probability.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (selected * mask.to(dtype=selected.dtype)).sum(dim=-1)


def citation_pointer_loss(citation_logits: Any, targets: Any, *, ignore_index: int = -100) -> Any:
    _require_torch()
    if citation_logits.ndim != 3 or targets.ndim != 2 or citation_logits.shape[:2] != targets.shape:
        raise ValueError("citation logits/targets must have shapes [B,C,E] and [B,C]")
    if citation_logits.size(-1) < 1:
        raise ValueError("citation loss requires at least one evidence slot")
    return F.cross_entropy(
        citation_logits.reshape(-1, citation_logits.size(-1)),
        targets.reshape(-1).long(),
        ignore_index=ignore_index,
    )


def binary_supervision_loss(logits: Any, targets: Any, *, mask: Any | None = None) -> Any:
    _require_torch()
    if logits.shape != targets.shape or logits.numel() == 0:
        raise ValueError("binary logits and targets must be non-empty and aligned")
    selected_targets = targets.to(dtype=logits.dtype)
    if torch.any(selected_targets < 0.0) or torch.any(selected_targets > 1.0):
        raise ValueError("binary targets must lie in [0,1]")
    loss = F.binary_cross_entropy_with_logits(logits, selected_targets, reduction="none")
    if mask is None:
        return loss.mean()
    if mask.shape != logits.shape:
        raise ValueError("binary supervision mask must align with logits")
    selected_mask = mask.to(dtype=loss.dtype)
    denominator = selected_mask.sum()
    if float(denominator.detach().item()) <= 0.0:
        raise ValueError("binary supervision mask selects no examples")
    return (loss * selected_mask).sum() / denominator


def reflection_action_loss(logits: Any, targets: Any, *, ignore_index: int = -100) -> Any:
    _require_torch()
    if logits.ndim != 2 or targets.ndim != 1 or logits.size(0) != targets.size(0):
        raise ValueError("reflection logits/targets must have shapes [B,A] and [B]")
    return F.cross_entropy(logits, targets.long(), ignore_index=ignore_index)


def unsupported_mass_unlikelihood(token_logits: Any, unsupported_token_mask: Any, *, eps: float = 1e-6) -> Any:
    """Penalize probability mass assigned to annotated unsupported alternatives.

    ``unsupported_token_mask`` is a boolean tensor with the exact shape of ``token_logits``.
    The loss is ``-log(1 - p(unsupported))`` averaged across positions that contain at least
    one annotated unsupported token.  It is intentionally annotation-driven; this module
    never invents unsupported-token labels from model text.
    """

    _require_torch()
    if token_logits.shape != unsupported_token_mask.shape or token_logits.ndim != 3:
        raise ValueError("unsupported_token_mask must exactly match [B,T,V] token logits")
    mask = unsupported_token_mask.to(dtype=torch.bool)
    active = mask.any(dim=-1)
    if not bool(active.any().detach().item()):
        raise ValueError("unsupported-token supervision selects no positions")
    probability = F.softmax(token_logits, dim=-1)
    unsupported_mass = (probability * mask.to(dtype=probability.dtype)).sum(dim=-1)
    selected_eps = _positive(eps, "eps", 0.1)
    safe = torch.clamp(1.0 - unsupported_mass, min=selected_eps, max=1.0)
    return (-torch.log(safe))[active].mean()


def dpo_grounded_preference_loss(
    chosen_log_prob: Any,
    rejected_log_prob: Any,
    reference_chosen_log_prob: Any,
    reference_rejected_log_prob: Any,
    *,
    beta: float = 0.1,
) -> Any:
    """DPO-style preference loss for grounded-vs-unsupported response pairs."""

    _require_torch()
    if not (
        chosen_log_prob.shape
        == rejected_log_prob.shape
        == reference_chosen_log_prob.shape
        == reference_rejected_log_prob.shape
    ) or chosen_log_prob.numel() == 0:
        raise ValueError("preference log-probability tensors must be non-empty and aligned")
    selected_beta = _positive(beta, "beta", 1_000.0)
    policy_margin = chosen_log_prob - rejected_log_prob
    reference_margin = reference_chosen_log_prob.detach() - reference_rejected_log_prob.detach()
    return -F.logsigmoid(selected_beta * (policy_margin - reference_margin)).mean()


def teacher_token_distillation_kl(student_logits: Any, teacher_logits: Any, *, temperature: float = 1.0) -> Any:
    _require_torch()
    if student_logits.shape != teacher_logits.shape or student_logits.numel() == 0:
        raise ValueError("student and teacher token logits must be non-empty and aligned")
    selected_temperature = _positive(temperature, "temperature", 1_000.0)
    student_log = F.log_softmax(student_logits / selected_temperature, dim=-1)
    teacher_probability = F.softmax(teacher_logits.detach() / selected_temperature, dim=-1)
    return (
        F.kl_div(student_log, teacher_probability, reduction="batchmean")
        * selected_temperature
        * selected_temperature
    )


def lm_supervised_retriever_kl(
    retriever_logits: Any,
    document_lm_log_likelihood: Any,
    *,
    temperature: float = 1.0,
    candidate_mask: Any | None = None,
) -> Any:
    """Distill LM document utility into a retriever distribution.

    ``document_lm_log_likelihood[b, d]`` is the frozen/reference generator log-likelihood
    of the target when conditioned on candidate document ``d``.  The target document
    distribution is derived only from those measured training values; no relevance label is
    fabricated here.
    """

    _require_torch()
    if retriever_logits.ndim != 2 or retriever_logits.shape != document_lm_log_likelihood.shape:
        raise ValueError("retriever logits and LM utility must have aligned [B,D] shapes")
    if retriever_logits.size(1) < 1:
        raise ValueError("retriever coupling requires at least one document candidate")
    selected_temperature = _positive(temperature, "temperature", 1_000.0)
    selected_retriever = retriever_logits
    selected_utility = document_lm_log_likelihood.detach()
    if candidate_mask is not None:
        if candidate_mask.shape != retriever_logits.shape:
            raise ValueError("candidate_mask must align with retriever logits")
        mask = candidate_mask.to(dtype=torch.bool)
        if torch.any(~mask.any(dim=-1)):
            raise ValueError("every retriever-coupling row requires at least one candidate")
        floor = torch.finfo(retriever_logits.dtype).min
        selected_retriever = retriever_logits.masked_fill(~mask, floor)
        selected_utility = selected_utility.masked_fill(~mask, floor)
    student_log = F.log_softmax(selected_retriever / selected_temperature, dim=-1)
    target = F.softmax(selected_utility / selected_temperature, dim=-1)
    return (
        F.kl_div(student_log, target, reduction="batchmean")
        * selected_temperature
        * selected_temperature
    )


def grounded_generation_objective(
    *,
    weights: GroundedObjectiveWeights,
    token_nll: Any | None = None,
    citation: Any | None = None,
    support: Any | None = None,
    contradiction: Any | None = None,
    abstention: Any | None = None,
    reflection: Any | None = None,
    unsupported_unlikelihood: Any | None = None,
    preference: Any | None = None,
    teacher_distillation: Any | None = None,
    retriever_coupling: Any | None = None,
) -> TensorGroundedLossBreakdown:
    """Combine already-computed differentiable components under an immutable weight set."""

    _require_torch()
    if not isinstance(weights, GroundedObjectiveWeights):
        raise ValueError("weights must be GroundedObjectiveWeights")
    components = {
        "token_nll": token_nll,
        "citation": citation,
        "support": support,
        "contradiction": contradiction,
        "abstention": abstention,
        "reflection": reflection,
        "unsupported_unlikelihood": unsupported_unlikelihood,
        "preference": preference,
        "teacher_distillation": teacher_distillation,
        "retriever_coupling": retriever_coupling,
    }
    total = None
    for name, value in components.items():
        weight = float(getattr(weights, name))
        if weight <= 0.0:
            continue
        if value is None:
            raise ValueError(f"objective {name} is weighted but no loss tensor was supplied")
        if getattr(value, "numel", lambda: 0)() != 1:
            raise ValueError(f"objective {name} must be a scalar tensor")
        weighted = value * weight
        total = weighted if total is None else total + weighted
    if total is None:
        raise ValueError("grounded objective selected no active loss")
    return TensorGroundedLossBreakdown(total=total, **components)


@runtime_checkable
class GroundedGeneratorTrainingBackend(Protocol):
    """Backend contract for a base LM plus the exact released grounded context.

    Implementations may wrap a local Hugging Face model, another admitted local model, or a
    governed training service.  The backend must not change evidence identities, citation
    slots, labels or data-order identity supplied by the caller.
    """

    @property
    def model_sha256(self) -> str:
        ...

    @property
    def tokenizer_sha256(self) -> str:
        ...

    def forward_training(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


def assert_backend_matches_plan(backend: GroundedGeneratorTrainingBackend, plan: GroundedTrainingPlan) -> None:
    if not isinstance(plan, GroundedTrainingPlan):
        raise ValueError("plan must be GroundedTrainingPlan")
    model_sha = _sha256(backend.model_sha256, "backend model_sha256")
    tokenizer_sha = _sha256(backend.tokenizer_sha256, "backend tokenizer_sha256")
    if model_sha != plan.base_model_sha256:
        raise ValueError("generator backend model does not match immutable training plan")
    if tokenizer_sha != plan.tokenizer_sha256:
        raise ValueError("generator backend tokenizer does not match immutable training plan")


__all__ = [
    "GroundedAuxiliaryHeads",
    "GroundedGenerationArchitectureConfig",
    "GroundedGeneratorTrainingBackend",
    "GroundedObjectiveWeights",
    "GroundedStageKind",
    "GroundedTrainingPlan",
    "GroundedTrainingStage",
    "ReflectionAction",
    "TensorGroundedLossBreakdown",
    "assert_backend_matches_plan",
    "binary_supervision_loss",
    "citation_pointer_loss",
    "dpo_grounded_preference_loss",
    "grounded_generation_objective",
    "lm_supervised_retriever_kl",
    "masked_token_nll",
    "reflection_action_loss",
    "sequence_log_prob",
    "teacher_token_distillation_kl",
    "unsupported_mass_unlikelihood",
]
