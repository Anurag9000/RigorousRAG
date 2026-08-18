"""Evaluation metrics for generation-time dynamic retrieval policies.

The metrics here are intentionally episode based.  They distinguish whether a policy
retrieved at the right *time* from whether the final answer happened to be correct, and make
retrieval cost/latency and abstention explicit.  This is a source-only evaluator: it consumes
already-produced observations and performs no generation or retrieval itself.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from training.dynamic_retrieval_policy import DynamicRetrievalAction

_MAX_STEPS = 100_000_000
_MAX_EPISODES = 10_000_000


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
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


def _nonnegative(value: Any, label: str, maximum: float = 1e18) -> float:
    selected = _finite(value, label)
    if not 0.0 <= selected <= maximum:
        raise ValueError(f"{label} must be non-negative and bounded")
    return selected


def _bounded_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label} is outside its allowed integer range")
    return value


@dataclass(frozen=True)
class DynamicStepObservation:
    episode_id: str
    step_index: int
    action: DynamicRetrievalAction
    oracle_action: DynamicRetrievalAction | None
    selected_action_probability: float
    retrieval_gain: float
    retrieval_cost: float
    latency_ms: float
    generated_tokens: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _identifier(self.episode_id, "episode_id", 240))
        object.__setattr__(self, "step_index", _bounded_int(self.step_index, "step_index", 0, _MAX_STEPS))
        if not isinstance(self.action, DynamicRetrievalAction):
            object.__setattr__(self, "action", DynamicRetrievalAction(self.action))
        if self.oracle_action is not None and not isinstance(self.oracle_action, DynamicRetrievalAction):
            object.__setattr__(self, "oracle_action", DynamicRetrievalAction(self.oracle_action))
        object.__setattr__(self, "selected_action_probability", _unit(self.selected_action_probability, "selected_action_probability"))
        object.__setattr__(self, "retrieval_gain", _finite(self.retrieval_gain, "retrieval_gain"))
        object.__setattr__(self, "retrieval_cost", _nonnegative(self.retrieval_cost, "retrieval_cost"))
        object.__setattr__(self, "latency_ms", _nonnegative(self.latency_ms, "latency_ms"))
        object.__setattr__(self, "generated_tokens", _bounded_int(self.generated_tokens, "generated_tokens", 0, _MAX_STEPS))
        if self.action != DynamicRetrievalAction.RETRIEVE and (self.retrieval_cost != 0.0 or self.retrieval_gain != 0.0):
            raise ValueError("non-retrieval steps may not claim retrieval cost or gain")


@dataclass(frozen=True)
class DynamicEpisodeOutcome:
    episode_id: str
    final_supported: bool
    final_contradicted: bool
    abstained: bool
    answer_utility: float
    oracle_utility: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _identifier(self.episode_id, "episode_id", 240))
        for name in ("final_supported", "final_contradicted", "abstained"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        object.__setattr__(self, "answer_utility", _finite(self.answer_utility, "answer_utility"))
        if self.oracle_utility is not None:
            object.__setattr__(self, "oracle_utility", _finite(self.oracle_utility, "oracle_utility"))


@dataclass(frozen=True)
class DynamicRagReport:
    episode_count: int
    step_count: int
    action_accuracy: float | None
    retrieve_precision: float | None
    retrieve_recall: float | None
    mean_retrievals_per_episode: float
    useful_retrieval_rate: float | None
    unnecessary_retrieval_rate: float | None
    mean_retrieval_gain: float
    mean_retrieval_cost: float
    mean_step_latency_ms: float
    mean_generated_tokens_per_step: float
    final_supported_rate: float
    contradiction_rate: float
    abstention_rate: float
    mean_answer_utility: float
    mean_oracle_regret: float | None
    selected_action_brier: float | None
    report_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_count", _bounded_int(self.episode_count, "episode_count", 1, _MAX_EPISODES))
        object.__setattr__(self, "step_count", _bounded_int(self.step_count, "step_count", 1, _MAX_STEPS))
        for name in (
            "action_accuracy",
            "retrieve_precision",
            "retrieve_recall",
            "useful_retrieval_rate",
            "unnecessary_retrieval_rate",
            "selected_action_brier",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _unit(value, name))
        for name in ("final_supported_rate", "contradiction_rate", "abstention_rate"):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        for name in (
            "mean_retrievals_per_episode",
            "mean_retrieval_gain",
            "mean_retrieval_cost",
            "mean_step_latency_ms",
            "mean_generated_tokens_per_step",
            "mean_answer_utility",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.mean_retrievals_per_episode < 0.0 or self.mean_retrieval_cost < 0.0 or self.mean_step_latency_ms < 0.0 or self.mean_generated_tokens_per_step < 0.0:
            raise ValueError("count/cost/latency/token means must be non-negative")
        if self.mean_oracle_regret is not None:
            object.__setattr__(self, "mean_oracle_regret", _finite(self.mean_oracle_regret, "mean_oracle_regret"))
        expected = _digest(self._payload())
        if self.report_sha256 != expected:
            raise ValueError("dynamic RAG report digest mismatch")

    def _payload(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-dynamic-rag-report/v1",
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "action_accuracy": self.action_accuracy,
            "retrieve_precision": self.retrieve_precision,
            "retrieve_recall": self.retrieve_recall,
            "mean_retrievals_per_episode": self.mean_retrievals_per_episode,
            "useful_retrieval_rate": self.useful_retrieval_rate,
            "unnecessary_retrieval_rate": self.unnecessary_retrieval_rate,
            "mean_retrieval_gain": self.mean_retrieval_gain,
            "mean_retrieval_cost": self.mean_retrieval_cost,
            "mean_step_latency_ms": self.mean_step_latency_ms,
            "mean_generated_tokens_per_step": self.mean_generated_tokens_per_step,
            "final_supported_rate": self.final_supported_rate,
            "contradiction_rate": self.contradiction_rate,
            "abstention_rate": self.abstention_rate,
            "mean_answer_utility": self.mean_answer_utility,
            "mean_oracle_regret": self.mean_oracle_regret,
            "selected_action_brier": self.selected_action_brier,
        }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires values")
    return sum(values) / len(values)


def _optional_rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def build_dynamic_rag_report(
    steps: Sequence[DynamicStepObservation],
    outcomes: Sequence[DynamicEpisodeOutcome],
    *,
    useful_retrieval_gain_threshold: float = 0.0,
) -> DynamicRagReport:
    selected_steps = tuple(steps)
    selected_outcomes = tuple(outcomes)
    if not selected_steps or len(selected_steps) > _MAX_STEPS:
        raise ValueError("steps must be a non-empty bounded sequence")
    if not selected_outcomes or len(selected_outcomes) > _MAX_EPISODES:
        raise ValueError("outcomes must be a non-empty bounded sequence")
    if any(not isinstance(step, DynamicStepObservation) for step in selected_steps):
        raise ValueError("steps contain an invalid observation")
    if any(not isinstance(outcome, DynamicEpisodeOutcome) for outcome in selected_outcomes):
        raise ValueError("outcomes contain an invalid episode")
    threshold = _finite(useful_retrieval_gain_threshold, "useful_retrieval_gain_threshold")

    outcome_ids = [outcome.episode_id for outcome in selected_outcomes]
    if len(set(outcome_ids)) != len(outcome_ids):
        raise ValueError("episode outcomes must be unique")
    outcome_set = set(outcome_ids)
    if any(step.episode_id not in outcome_set for step in selected_steps):
        raise ValueError("every dynamic step must have a matching episode outcome")

    seen_step_keys: set[tuple[str, int]] = set()
    for step in selected_steps:
        key = (step.episode_id, step.step_index)
        if key in seen_step_keys:
            raise ValueError("dynamic step identities must be unique")
        seen_step_keys.add(key)

    labelled = [step for step in selected_steps if step.oracle_action is not None]
    action_accuracy = None if not labelled else sum(step.action == step.oracle_action for step in labelled) / len(labelled)
    oracle_retrieves = [step for step in labelled if step.oracle_action == DynamicRetrievalAction.RETRIEVE]
    policy_retrieves_labelled = [step for step in labelled if step.action == DynamicRetrievalAction.RETRIEVE]
    true_positive_retrieves = sum(
        step.action == DynamicRetrievalAction.RETRIEVE and step.oracle_action == DynamicRetrievalAction.RETRIEVE
        for step in labelled
    )
    retrieve_precision = _optional_rate(true_positive_retrieves, len(policy_retrieves_labelled))
    retrieve_recall = _optional_rate(true_positive_retrieves, len(oracle_retrieves))

    retrieval_steps = [step for step in selected_steps if step.action == DynamicRetrievalAction.RETRIEVE]
    useful = [step for step in retrieval_steps if step.retrieval_gain > threshold]
    unnecessary = [step for step in retrieval_steps if step.retrieval_gain <= threshold]
    useful_rate = _optional_rate(len(useful), len(retrieval_steps))
    unnecessary_rate = _optional_rate(len(unnecessary), len(retrieval_steps))

    retrievals_by_episode = {episode_id: 0 for episode_id in outcome_set}
    for step in retrieval_steps:
        retrievals_by_episode[step.episode_id] += 1

    oracle_pairs = [outcome for outcome in selected_outcomes if outcome.oracle_utility is not None]
    mean_regret = None
    if oracle_pairs:
        mean_regret = _mean([max(0.0, float(outcome.oracle_utility) - outcome.answer_utility) for outcome in oracle_pairs])

    selected_action_brier = None
    if labelled:
        # With only the probability assigned to the selected action logged, this is a
        # conservative one-vs-rest calibration diagnostic, not a full multiclass Brier score.
        selected_action_brier = _mean(
            [
                (step.selected_action_probability - (1.0 if step.action == step.oracle_action else 0.0)) ** 2
                for step in labelled
            ]
        )

    payload: dict[str, Any] = {
        "episode_count": len(selected_outcomes),
        "step_count": len(selected_steps),
        "action_accuracy": action_accuracy,
        "retrieve_precision": retrieve_precision,
        "retrieve_recall": retrieve_recall,
        "mean_retrievals_per_episode": _mean([float(value) for value in retrievals_by_episode.values()]),
        "useful_retrieval_rate": useful_rate,
        "unnecessary_retrieval_rate": unnecessary_rate,
        "mean_retrieval_gain": 0.0 if not retrieval_steps else _mean([step.retrieval_gain for step in retrieval_steps]),
        "mean_retrieval_cost": 0.0 if not retrieval_steps else _mean([step.retrieval_cost for step in retrieval_steps]),
        "mean_step_latency_ms": _mean([step.latency_ms for step in selected_steps]),
        "mean_generated_tokens_per_step": _mean([float(step.generated_tokens) for step in selected_steps]),
        "final_supported_rate": sum(outcome.final_supported for outcome in selected_outcomes) / len(selected_outcomes),
        "contradiction_rate": sum(outcome.final_contradicted for outcome in selected_outcomes) / len(selected_outcomes),
        "abstention_rate": sum(outcome.abstained for outcome in selected_outcomes) / len(selected_outcomes),
        "mean_answer_utility": _mean([outcome.answer_utility for outcome in selected_outcomes]),
        "mean_oracle_regret": mean_regret,
        "selected_action_brier": selected_action_brier,
    }
    digest = _digest({"schema": "rigorousrag-dynamic-rag-report/v1", **payload})
    return DynamicRagReport(report_sha256=digest, **payload)


__all__ = [
    "DynamicEpisodeOutcome",
    "DynamicRagReport",
    "DynamicStepObservation",
    "build_dynamic_rag_report",
]
