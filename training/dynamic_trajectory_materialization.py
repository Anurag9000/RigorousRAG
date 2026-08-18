"""Durable dynamic-RAG trajectory target materialization.

This module turns logged generation-time episodes into immutable local training JSONL after
operators explicitly execute admitted value/counterfactual providers. It does not run any
model, retrieval stack, or dataset acquisition on import.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from training.advanced_rag_data import DynamicRagEpisodeStep
from training.advanced_rag_supervision import (
    CounterfactualActionProvider,
    DynamicRewardConfig,
    counterfactual_action_target,
    generalized_advantage_estimate,
    trajectory_rewards,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: str, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
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
        for name in (
            "source_dataset_sha256",
            "dataset_manifest_sha256",
            "runtime_stack_sha256",
            "feature_provider_sha256",
            "behavior_policy_sha256",
            "value_provider_sha256",
        ):
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
        return _digest({
            "schema": "rigorousrag-dynamic-trajectory-materialization-identity/v1",
            **{k: v for k, v in asdict(self).items() if k != "reward_config"},
            "reward_config": asdict(self.reward_config),
        })


@dataclass(frozen=True)
class MaterializedTrajectoryReceipt:
    output_path: str
    output_sha256: str
    record_count: int
    episode_count: int
    identity_sha256: str
    receipt_sha256: str


def _record(step: DynamicRagEpisodeStep) -> Mapping[str, Any]:
    return {
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
        provider_sha = getattr(counterfactual_provider, "contract_sha256", None)
        if provider_sha != identity.counterfactual_provider_sha256:
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
        values = tuple(float(value) for value in value_provider.values(episode))
        if len(values) != len(episode):
            raise ValueError("value provider returned the wrong number of values")
        rewards = trajectory_rewards(episode, identity.reward_config)
        targets = generalized_advantage_estimate(
            rewards,
            values,
            discount=identity.reward_config.discount,
            gae_lambda=identity.reward_config.gae_lambda,
            bootstrap_value=0.0,
        )
        for index, step in enumerate(episode):
            metadata = dict(step.metadata)
            metadata["trajectory_identity_sha256"] = identity.identity_sha256
            metadata["return_target"] = format(targets.returns[index], ".17g")
            if counterfactual_provider is not None:
                action, gain = counterfactual_action_target(counterfactual_provider.action_utilities(step), identity.reward_config)
                metadata["counterfactual_best_action"] = action.value
                metadata["counterfactual_gain_over_continue"] = format(gain, ".17g")
            materialized.append(replace(step, advantage=targets.advantages[index], metadata=metadata))

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for step in materialized:
                line = _canonical(_record(step)) + b"\n"
                handle.write(line)
                digest.update(line)
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
        output_path=str(destination),
        output_sha256=output_sha,
        record_count=len(materialized),
        episode_count=len(by_episode),
        identity_sha256=identity.identity_sha256,
        receipt_sha256=_digest(unsigned),
    )


__all__ = ["LoggedValueProvider", "MaterializedTrajectoryReceipt", "TrajectoryMaterializationIdentity", "materialize_dynamic_trajectories"]
