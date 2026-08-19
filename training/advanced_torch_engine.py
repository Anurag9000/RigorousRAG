"""Authoritative Torch engine refinements for advanced-RAG checkpoint/resume.

A content-addressed checkpoint cannot serialize its own not-yet-known digest inside
``TrainerState.best_checkpoint_digest`` without a self-reference. When an evaluation produces a
new best, the immutable candidate therefore stores the *previous* best digest and the external
verified ``best.json`` pointer is updated after the candidate digest exists. This subclass
repairs that one unavoidable representation edge on resume while remaining lineage-aware and
fail-closed.
"""
from __future__ import annotations

import math
from typing import Any

from training.advanced_checkpoint_authority import AdvancedCheckpointManager
from training.checkpoint_control import read_checkpoint_pointer
from training.checkpointing import TensorCheckpointManifest, TrainerState
from training.torch_engine import StageRuntime, TorchTrainingEngine

_MAX_LINEAGE_DEPTH = 1_000_000


def _finite_equal(left: Any, right: Any) -> bool:
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(a) and math.isfinite(b) and a == b


class AuthoritativeTorchTrainingEngine(TorchTrainingEngine):
    """Torch engine with exact best-pointer repair for advanced content-addressed resumes."""

    def __init__(self, model: Any, config: Any, checkpoints: AdvancedCheckpointManager) -> None:
        if not isinstance(checkpoints, AdvancedCheckpointManager):
            raise ValueError("authoritative advanced engine requires AdvancedCheckpointManager")
        super().__init__(model, config, checkpoints)

    def _assert_manifest_authority(self, manifest: TensorCheckpointManifest) -> None:
        failures = []
        if manifest.run_id != self.config.run_id:
            failures.append("run_id")
        if manifest.source_commit != self.config.source_commit:
            failures.append("source_commit")
        if manifest.training_config_digest != self.config.digest:
            failures.append("training_config_digest")
        if manifest.dataset_manifest_digest != self.config.dataset_manifest_digest:
            failures.append("dataset_manifest_digest")
        if manifest.model_architecture != self.config.model_architecture:
            failures.append("model_architecture")
        if failures:
            raise ValueError(f"checkpoint lineage differs from authoritative run: {','.join(failures)}")

    def _lineage(self, digest: str) -> dict[str, TensorCheckpointManifest]:
        lineage: dict[str, TensorCheckpointManifest] = {}
        current: str | None = digest
        while current is not None:
            if len(lineage) >= _MAX_LINEAGE_DEPTH:
                raise ValueError("checkpoint lineage exceeds safety bound")
            if current in lineage:
                raise ValueError("checkpoint lineage contains a parent cycle")
            _, manifest = self.checkpoints.verify(current)
            self._assert_manifest_authority(manifest)
            if manifest.digest != current:
                raise ValueError("checkpoint lineage digest differs from requested parent")
            lineage[current] = manifest
            current = manifest.parent_checkpoint_digest
        return lineage

    def _metric_matches(self, manifest: TensorCheckpointManifest, state: TrainerState) -> bool:
        metric_name = self.config.early_stopping_metric
        if metric_name is None or state.best_metric is None:
            return False
        return metric_name in manifest.metric_snapshot and _finite_equal(manifest.metric_snapshot[metric_name], state.best_metric)

    def _repair_best_checkpoint(self, resume_digest: str, state: TrainerState) -> TrainerState:
        if self.config.early_stopping_metric is None or state.best_metric is None:
            if state.best_metric is None and state.best_checkpoint_digest is not None:
                raise ValueError("resume state has best checkpoint without best metric")
            return state

        lineage = self._lineage(resume_digest)
        stored = state.best_checkpoint_digest
        if stored is not None:
            if stored not in lineage:
                raise ValueError("serialized best checkpoint is outside resumed checkpoint ancestry")
            if self._metric_matches(lineage[stored], state):
                return state

        # Newly-best self-reference case: the candidate contains the updated metric but could
        # not contain its own future content digest. This is the primary repair path.
        current_manifest = lineage[resume_digest]
        if self._metric_matches(current_manifest, state):
            return TrainerState(
                run_id=state.run_id,
                cursor=state.cursor,
                best_metric=state.best_metric,
                best_checkpoint_digest=resume_digest,
                early_stopping_bad_steps=state.early_stopping_bad_steps,
                stage_name=state.stage_name,
            )

        # The external pointer is convenience authority only when it is in this exact ancestry
        # and its metric proves it represents the serialized best value. A pointer to a future
        # continuation of the same run is deliberately ignored when resuming an older branch.
        try:
            pointed = read_checkpoint_pointer(self.checkpoints, "best")
        except FileNotFoundError:
            pointed = None
        if pointed is not None and pointed in lineage and self._metric_matches(lineage[pointed], state):
            return TrainerState(
                run_id=state.run_id,
                cursor=state.cursor,
                best_metric=state.best_metric,
                best_checkpoint_digest=pointed,
                early_stopping_bad_steps=state.early_stopping_bad_steps,
                stage_name=state.stage_name,
            )

        raise ValueError("unable to reconstruct authoritative best checkpoint from resumed lineage")

    def _resume(self, digest: str, runtime: StageRuntime) -> TrainerState:
        state = super()._resume(digest, runtime)
        return self._repair_best_checkpoint(digest, state)


__all__ = ["AuthoritativeTorchTrainingEngine"]
