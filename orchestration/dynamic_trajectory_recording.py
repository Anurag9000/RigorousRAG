"""Privacy-preserving trajectory recording wrappers for the bounded dynamic-RAG runtime.

The runtime itself continues to retain only trace hashes. These request-scoped wrappers
observe the same snapshot/features/scores already passed to feature and policy providers and
materialize training-ready legal-action decision records outside the runtime result.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from orchestration.dynamic_rag_runtime import DynamicFeatureProvider, DynamicPolicyProvider, DynamicRuntimeSnapshot
from training.advanced_rag_authoritative_data import LegalDynamicRagEpisodeStep
from training.dynamic_retrieval_policy import DynamicRetrievalAction, DynamicRetrievalBudget, DynamicRetrievalFeatures, allowed_actions

_MAX_CONTEXT = 5_000_000


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _choose(scores: Mapping[DynamicRetrievalAction, float], permitted: Sequence[DynamicRetrievalAction]) -> DynamicRetrievalAction:
    normalized = {action if isinstance(action, DynamicRetrievalAction) else DynamicRetrievalAction(action): float(score) for action, score in scores.items()}
    if set(normalized) != set(DynamicRetrievalAction):
        raise ValueError("trajectory recorder requires a complete closed action-score map")
    selected = tuple(permitted)
    if not selected:
        raise ValueError("trajectory recorder observed a state with no legal actions")
    return min(selected, key=lambda action: (-normalized[action], action.value))


@dataclass
class DynamicDecisionTrajectoryRecorder:
    request_text: str
    episode_id: str
    budget: DynamicRetrievalBudget

    def __post_init__(self) -> None:
        if not isinstance(self.request_text, str) or not self.request_text or "\x00" in self.request_text:
            raise ValueError("request_text must be released non-empty text")
        if len(self.request_text) > _MAX_CONTEXT:
            raise ValueError("request_text exceeds recorder character bound")
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("episode_id is required")
        if not isinstance(self.budget, DynamicRetrievalBudget):
            raise ValueError("budget must be DynamicRetrievalBudget")
        self.request_sha256 = _sha_text(self.request_text)
        self._snapshots: dict[str, tuple[DynamicRuntimeSnapshot, DynamicRetrievalFeatures]] = {}
        self._steps: list[LegalDynamicRagEpisodeStep] = []

    @property
    def recorder_sha256(self) -> str:
        return _digest({
            "schema": "rigorousrag-dynamic-decision-trajectory-recorder/v1",
            "request_sha256": self.request_sha256,
            "episode_id": self.episode_id,
            "budget_sha256": self.budget.budget_sha256,
            "behavior_policy": "deterministic_server_argmax",
        })

    def observe_features(self, snapshot: DynamicRuntimeSnapshot, features: DynamicRetrievalFeatures) -> None:
        if snapshot.request_sha256 != self.request_sha256:
            raise ValueError("snapshot request identity differs from trajectory recorder")
        if not isinstance(features, DynamicRetrievalFeatures):
            raise ValueError("features must be DynamicRetrievalFeatures")
        digest = snapshot.snapshot_sha256
        if digest in self._snapshots:
            raise ValueError("trajectory recorder observed duplicate snapshot identity")
        self._snapshots[digest] = (snapshot, features)

    def observe_scores(self, snapshot_sha256: str, scores: Mapping[DynamicRetrievalAction, float]) -> None:
        if snapshot_sha256 not in self._snapshots:
            raise ValueError("policy score arrived before its feature snapshot")
        snapshot, features = self._snapshots.pop(snapshot_sha256)
        permitted = tuple(allowed_actions(snapshot.state, self.budget, verification_enabled=True))
        action = _choose(scores, permitted)
        context = self.request_text if not snapshot.generated_text else self.request_text + "\n\n" + snapshot.generated_text
        if len(context) > _MAX_CONTEXT:
            raise ValueError("recorded dynamic context exceeds character bound")
        step = LegalDynamicRagEpisodeStep(
            episode_id=self.episode_id,
            step_id=f"{snapshot.iteration:08d}",
            context=context,
            features={name: value for name, value in zip(features.vector(), ())},
            action=action,
            realized_retrieval_gain=0.0,
            behavior_action_probability=1.0,
            advantage=None,
            need_spans=(),
            hidden_state_cache_key=None,
            terminal_utility=None,
            metadata={
                "snapshot_sha256": snapshot.snapshot_sha256,
                "recorder_sha256": self.recorder_sha256,
            },
            valid_actions=permitted,
        )
        # Build feature mapping explicitly after validation to preserve canonical names/order.
        object.__setattr__(step, "features", {name: float(getattr(features, name)) for name in features.__dataclass_fields__})
        self._steps.append(step)

    def steps(self) -> tuple[LegalDynamicRagEpisodeStep, ...]:
        if self._snapshots:
            raise ValueError("trajectory recorder has snapshots without corresponding policy scores")
        return tuple(self._steps)

    def finalize(self, *, terminal_utility: float | None = None) -> tuple[LegalDynamicRagEpisodeStep, ...]:
        selected = self.steps()
        if not selected:
            raise ValueError("trajectory recorder contains no decisions")
        if terminal_utility is None:
            return selected
        utility = float(terminal_utility)
        return tuple(replace(step, terminal_utility=utility if index == len(selected) - 1 else None) for index, step in enumerate(selected))


class RecordingDynamicFeatureProvider:
    def __init__(self, inner: DynamicFeatureProvider, recorder: DynamicDecisionTrajectoryRecorder) -> None:
        self.inner, self.recorder = inner, recorder

    @property
    def contract_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-recording-feature-provider/v1", "inner_sha256": self.inner.contract_sha256, "recorder_sha256": self.recorder.recorder_sha256})

    def features(self, snapshot: DynamicRuntimeSnapshot) -> DynamicRetrievalFeatures:
        features = self.inner.features(snapshot)
        self.recorder.observe_features(snapshot, features)
        return features


class RecordingDynamicPolicyProvider:
    def __init__(self, inner: DynamicPolicyProvider, recorder: DynamicDecisionTrajectoryRecorder) -> None:
        self.inner, self.recorder = inner, recorder

    @property
    def artifact_sha256(self) -> str:
        return self.inner.artifact_sha256

    @property
    def contract_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-recording-policy-provider/v1", "inner_sha256": self.inner.contract_sha256, "recorder_sha256": self.recorder.recorder_sha256})

    def action_scores(self, features: DynamicRetrievalFeatures, *, snapshot_sha256: str) -> Mapping[DynamicRetrievalAction, float]:
        scores = self.inner.action_scores(features, snapshot_sha256=snapshot_sha256)
        self.recorder.observe_scores(snapshot_sha256, scores)
        return scores


def wrap_dynamic_providers_for_trajectory(
    feature_provider: DynamicFeatureProvider,
    policy_provider: DynamicPolicyProvider,
    recorder: DynamicDecisionTrajectoryRecorder,
) -> tuple[RecordingDynamicFeatureProvider, RecordingDynamicPolicyProvider]:
    return RecordingDynamicFeatureProvider(feature_provider, recorder), RecordingDynamicPolicyProvider(policy_provider, recorder)


__all__ = [
    "DynamicDecisionTrajectoryRecorder", "RecordingDynamicFeatureProvider",
    "RecordingDynamicPolicyProvider", "wrap_dynamic_providers_for_trajectory",
]
