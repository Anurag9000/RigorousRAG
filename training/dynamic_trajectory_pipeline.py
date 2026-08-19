"""Explicitly non-canonical one-pass dynamic-RAG trajectory composition.

The final training workflow MUST use ``dynamic_canonical_training_data_pipeline`` because the
hidden-state cache identity must bind the final published dataset manifest. This legacy helper
pre-dates that two-phase authority order and cannot prove that property from its signature.
It therefore refuses execution by default and requires an explicit ``allow_noncanonical=True``
opt-in for research/compatibility use. No operator/recipe path uses this helper.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.advanced_rag_authoritative_data import LegalDynamicRagEpisodeStep
from training.advanced_rag_strict_cache import AuthoritativeSafetensorSupervisionCache
from training.advanced_rag_supervision import CounterfactualActionProvider
from training.dynamic_reward_supervision import RealizedRetrievalGainProvider, RealizedRetrievalGainReceipt, apply_realized_retrieval_gains
from training.dynamic_trajectory_materialization import LoggedValueProvider, MaterializedTrajectoryReceipt, TrajectoryMaterializationIdentity, materialize_dynamic_trajectories
from training.dynamic_trajectory_preparation import BoundGeneratorHiddenStateProvider, DynamicTrajectoryPreparationReceipt, InformationNeedAnnotationProvider, prepare_dynamic_trajectory_supervision


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


@dataclass(frozen=True)
class DynamicTrajectoryPipelineReceipt:
    preparation_receipt_sha256: str
    realized_gain_receipt_sha256: str | None
    materialization_receipt_sha256: str
    final_output_sha256: str
    record_count: int
    episode_count: int
    promotable: bool
    pipeline_receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("preparation_receipt_sha256", "materialization_receipt_sha256", "final_output_sha256", "pipeline_receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if self.realized_gain_receipt_sha256 is not None:
            object.__setattr__(self, "realized_gain_receipt_sha256", _sha(self.realized_gain_receipt_sha256, "realized_gain_receipt_sha256"))
        for name in ("record_count", "episode_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.promotable:
            raise ValueError("legacy one-pass dynamic trajectory receipt may never be promotable")
        if _digest(self._payload()) != self.pipeline_receipt_sha256:
            raise ValueError("dynamic trajectory pipeline receipt digest mismatch")

    def _payload(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-dynamic-trajectory-pipeline-receipt/v2",
            "preparation_receipt_sha256": self.preparation_receipt_sha256,
            "realized_gain_receipt_sha256": self.realized_gain_receipt_sha256,
            "materialization_receipt_sha256": self.materialization_receipt_sha256,
            "final_output_sha256": self.final_output_sha256,
            "record_count": self.record_count,
            "episode_count": self.episode_count,
            "promotable": False,
            "authority_note": "noncanonical_one_pass_hidden_cache_does_not_bind_final_published_dataset_manifest",
        }


def prepare_and_materialize_dynamic_trajectories(
    steps: Sequence[LegalDynamicRagEpisodeStep],
    *,
    hidden_provider: BoundGeneratorHiddenStateProvider,
    hidden_cache: AuthoritativeSafetensorSupervisionCache,
    annotation_provider: InformationNeedAnnotationProvider | None,
    value_provider: LoggedValueProvider,
    materialization_identity: TrajectoryMaterializationIdentity,
    output_path: str | Path,
    realized_gain_provider: RealizedRetrievalGainProvider | None = None,
    counterfactual_provider: CounterfactualActionProvider | None = None,
    require_need_annotations: bool = True,
    allow_noncanonical: bool = False,
) -> tuple[MaterializedTrajectoryReceipt, DynamicTrajectoryPipelineReceipt]:
    """Run legacy one-pass preparation only after explicit non-canonical opt-in.

    Output from this helper must not be used as the final governed training dataset. Use
    ``build_canonical_dynamic_training_data`` for final training/promotion workflows.
    """
    if not allow_noncanonical:
        raise ValueError(
            "legacy one-pass dynamic trajectory preparation is non-promotable; "
            "use training.dynamic_canonical_training_data_pipeline or explicitly set allow_noncanonical=True for research"
        )
    prepared, preparation = prepare_dynamic_trajectory_supervision(
        steps,
        hidden_provider=hidden_provider,
        cache=hidden_cache,
        annotation_provider=annotation_provider,
        require_need_annotations=require_need_annotations,
    )
    gain_receipt: RealizedRetrievalGainReceipt | None = None
    selected = prepared
    if realized_gain_provider is not None:
        selected, gain_receipt = apply_realized_retrieval_gains(prepared, realized_gain_provider)
    materialized = materialize_dynamic_trajectories(
        selected,
        identity=materialization_identity,
        value_provider=value_provider,
        output_path=output_path,
        counterfactual_provider=counterfactual_provider,
    )
    unsigned = {
        "schema": "rigorousrag-dynamic-trajectory-pipeline-receipt/v2",
        "preparation_receipt_sha256": preparation.receipt_sha256,
        "realized_gain_receipt_sha256": None if gain_receipt is None else gain_receipt.receipt_sha256,
        "materialization_receipt_sha256": materialized.receipt_sha256,
        "final_output_sha256": materialized.output_sha256,
        "record_count": materialized.record_count,
        "episode_count": materialized.episode_count,
        "promotable": False,
        "authority_note": "noncanonical_one_pass_hidden_cache_does_not_bind_final_published_dataset_manifest",
    }
    receipt = DynamicTrajectoryPipelineReceipt(
        preparation_receipt_sha256=preparation.receipt_sha256,
        realized_gain_receipt_sha256=unsigned["realized_gain_receipt_sha256"],
        materialization_receipt_sha256=materialized.receipt_sha256,
        final_output_sha256=materialized.output_sha256,
        record_count=materialized.record_count,
        episode_count=materialized.episode_count,
        promotable=False,
        pipeline_receipt_sha256=_digest(unsigned),
    )
    return materialized, receipt


__all__ = ["DynamicTrajectoryPipelineReceipt", "prepare_and_materialize_dynamic_trajectories"]
