"""Durable dynamic-RAG trajectory target materialization.

This module turns logged generation-time episodes into immutable local training JSONL after
operators explicitly execute admitted value/counterfactual providers. It preserves legal
action sets, measures counterfactual improvement against the actual logged action rather than
an assumed CONTINUE baseline, validates every numeric target as finite, and stores GAE returns
as explicit state-value targets.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_authoritative_data import LegalDynamicRagEpisodeStep
from training.advanced_rag_data import DynamicRagEpisodeStep
from training.advanced_rag_supervision import CounterfactualActionProvider, DynamicRewardConfig, generalized_advantage_estimate, trajectory_rewards
from training.dynamic_retrieval_policy import DynamicRetrievalAction


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: str, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
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


class LoggedValueProvider(Protocol):
    @property
    def contract_sha256(self) -> str: ...
    def values(self, steps: Sequence[DynamicRagEpisodeStep]) -> Sequence[float]: ...


@dataclass(frozen=True)
class TrajectoryMaterializationIdentity:
    source_dataset_sha256: str
    dataset_manifest_sha256: str
    runtime_stack_sha256: str
    feature_provider_sha256: str
    behavior_policy_sha256: str
    value_provider_sha256: str
    counterfactual_provider_sha256: str | None
    source_commit: str
    reward_config: DynamicRewardConfig = DynamicRewardConfig()

    def __post_init__(self) -> None:
        for name in ("source_dataset_sha256", "dataset_manifest_sha256", "runtime_stack_sha256", "feature_provider_sha256", "behavior_policy_sha256", "value_provider_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if self.counterfactual_provider_sha256 is not None:
            object.__setattr__(self, "counterfactual_provider_sha256", _sha(self.counterfactual_provider_sha256, "counterfactual_provider_sha256"))
        commit = str(self.source_commit).strip().lower()
        if len(commit) not in {40, 64} or any(ch not in "0123456789abcdef" for ch in commit):
            raise ValueError("source_commit must be a full Git object id")
        object.__setattr__(self, "source_commit", commit)
        if not isinstance(self.reward_config, DynamicRewardConfig):
            raise ValueError("reward_config must be DynamicRewardConfig")

    @property
    def identity_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-dynamic-trajectory-materialization-identity/v1", **{k: v for k, v in asdict(self).items() if k != "reward_config"}, "reward_config": asdict(self.reward_config)})


@dataclass(frozen=True)
class MaterializedTrajectoryReceipt:
    output_path: str
    output_sha256: str
    record_count: int
    episode_count: int
    identity_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("output_sha256", "identity_sha256", "receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if isinstance(self.record_count, bool) or not isinstance(self.record_count, int) or self.record_count <= 0:
            raise ValueError("record_count must be positive")
        if isinstance(self.episode_count, bool) or not isinstance(self.episode_count, int) or self.episode_count <= 0:
            raise ValueError("episode_count must be positive")
        expected = _digest({
            "schema": "rigorousrag-dynamic-trajectory-materialization-receipt/v1",
            "output_sha256": self.output_sha256,
            "record_count": self.record_count,
            "episode_count": self.episode_count,
            "identity_sha256": self.identity_sha256,
        })
        if expected != self.receipt_sha256:
            raise ValueError("trajectory materialization receipt digest mismatch")


def _record(step: DynamicRagEpisodeStep) -> Mapping[str, Any]:
    result: dict[str, Any] = {
        "episode_id": step.episode_id,
        "step_id": step.step_id,
        "context": step.context,
        "features": dict(step.features),
        "action": step.action.value,
        "realized_retrieval_gain": step.realized_retrieval_gain,
        "behavior_action_probability": step.behavior_action_probability,
        "advantage": step.advantage,
        "need_spans": [asdict(span) for span in step.need_spans],
        "hidden_state_cache_key": step.hidden_state_cache_key,
        "terminal_utility": step.terminal_utility,
        "metadata": dict(step.metadata),
    }
    if isinstance(step, LegalDynamicRagEpisodeStep):
        result["valid_actions"] = [action.value for action in step.valid_actions]
        result["value_target"] = step.value_target
    return result


def _legal_counterfactual_utilities(step: DynamicRagEpisodeStep, utilities: Mapping[Any, float]) -> Mapping[DynamicRetrievalAction, float]:
    normalized: dict[DynamicRetrievalAction, float] = {}
    for raw_action, raw_value in utilities.items():
        action = raw_action if isinstance(raw_action, DynamicRetrievalAction) else DynamicRetrievalAction(raw_action)
        if action in normalized:
            raise ValueError(f"counterfactual provider duplicated action {action.value}")
        normalized[action] = _finite(raw_value, f"counterfactual utility {action.value}")
    if not normalized:
        raise ValueError("counterfactual provider returned no action utilities")
    if isinstance(step, LegalDynamicRagEpisodeStep):
        legal = set(step.valid_actions)
        normalized = {action: value for action, value in normalized.items() if action in legal}
        if not normalized:
            raise ValueError("counterfactual provider returned no utility for any legal action")
    if step.action not in normalized:
        raise ValueError("counterfactual provider must score the actual logged action baseline")
    return normalized


def _counterfactual_target_against_logged_action(
    step: DynamicRagEpisodeStep,
    utilities: Mapping[DynamicRetrievalAction, float],
    reward_config: DynamicRewardConfig,
) -> tuple[DynamicRetrievalAction, float]:
    adjusted = {action: _finite(value, f"counterfactual utility {action.value}") - reward_config.action_cost(action) for action, value in utilities.items()}
    baseline = adjusted[step.action]
    best = min(adjusted, key=lambda action: (-adjusted[action], action.value))
    return best, _finite(adjusted[best] - baseline, "counterfactual gain over logged action")


def materialize_dynamic_trajectories(
    steps: Sequence[DynamicRagEpisodeStep],
    *,
    identity: TrajectoryMaterializationIdentity,
    value_provider: LoggedValueProvider,
    output_path: str | Path,
    counterfactual_provider: CounterfactualActionProvider | None = None,
) -> MaterializedTrajectoryReceipt:
    """Compute value/advantage/counterfactual targets and atomically write governed JSONL."""
    if not steps:
        raise ValueError("trajectory materialization requires at least one logged step")
    if any(not isinstance(step, DynamicRagEpisodeStep) for step in steps):
        raise ValueError("steps must contain DynamicRagEpisodeStep values")
    if getattr(value_provider, "contract_sha256", None) != identity.value_provider_sha256:
        raise ValueError("value provider contract does not match materialization identity")
    if counterfactual_provider is not None:
        if getattr(counterfactual_provider, "contract_sha256", None) != identity.counterfactual_provider_sha256:
            raise ValueError("counterfactual provider contract does not match materialization identity")
    elif identity.counterfactual_provider_sha256 is not None:
        raise ValueError("materialization identity requires a counterfactual provider")

    by_episode: dict[str, list[DynamicRagEpisodeStep]] = {}
    seen: set[tuple[str, str]] = set()
    for step in steps:
        key = (step.episode_id, step.step_id)
        if key in seen:
            raise ValueError(f"duplicate trajectory step identity: {key}")
        seen.add(key)
        by_episode.setdefault(step.episode_id, []).append(step)

    materialized: list[DynamicRagEpisodeStep] = []
    for episode_id in sorted(by_episode):
        episode = by_episode[episode_id]
        values = tuple(_finite(value, "logged state value") for value in value_provider.values(episode))
        if len(values) != len(episode):
            raise ValueError("value provider returned the wrong number of values")
        rewards = trajectory_rewards(episode, identity.reward_config)
        targets = generalized_advantage_estimate(rewards, values, discount=identity.reward_config.discount, gae_lambda=identity.reward_config.gae_lambda, bootstrap_value=0.0)
        for index, step in enumerate(episode):
            metadata = dict(step.metadata)
            metadata["trajectory_identity_sha256"] = identity.identity_sha256
            if counterfactual_provider is not None:
                utilities = _legal_counterfactual_utilities(step, counterfactual_provider.action_utilities(step))
                action, gain = _counterfactual_target_against_logged_action(step, utilities, identity.reward_config)
                metadata["counterfactual_best_action"] = action.value
                metadata["counterfactual_logged_action"] = step.action.value
                metadata["counterfactual_gain_over_logged_action"] = format(gain, ".17g")
            if isinstance(step, LegalDynamicRagEpisodeStep):
                materialized.append(replace(step, advantage=_finite(targets.advantages[index], "advantage"), value_target=_finite(targets.returns[index], "value target"), metadata=metadata))
            else:
                metadata["return_target"] = format(_finite(targets.returns[index], "value target"), ".17g")
                materialized.append(replace(step, advantage=_finite(targets.advantages[index], "advantage"), metadata=metadata))

    destination = safe_advanced_path(output_path, label="dynamic trajectory output", must_exist=False)
    if destination.exists() and destination.is_dir():
        raise ValueError("dynamic trajectory output must be a file path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for step in materialized:
                line = _canonical(_record(step)) + b"\n"
                handle.write(line); digest.update(line)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)

    output_sha = digest.hexdigest()
    unsigned = {
        "schema": "rigorousrag-dynamic-trajectory-materialization-receipt/v1",
        "output_sha256": output_sha,
        "record_count": len(materialized),
        "episode_count": len(by_episode),
        "identity_sha256": identity.identity_sha256,
    }
    return MaterializedTrajectoryReceipt(
        output_path=str(destination), output_sha256=output_sha, record_count=len(materialized),
        episode_count=len(by_episode), identity_sha256=identity.identity_sha256, receipt_sha256=_digest(unsigned),
    )


__all__ = ["LoggedValueProvider", "MaterializedTrajectoryReceipt", "TrajectoryMaterializationIdentity", "materialize_dynamic_trajectories"]
