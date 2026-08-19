"""Canonical two-phase source pipeline from recorded dynamic decisions to trainable data.

This is the authoritative composition for *new* dynamic-RAG training data. It deliberately
supersedes the older one-pass hidden-cache-first helper for final-manifest-bound runs:

recorded legal decisions
 -> deterministic hidden-key / reviewed need-span planning (no hidden tensor writes)
 -> governed realized retrieval-gain binding
 -> value / GAE / legal counterfactual materialization
 -> episode-isolated final DatasetManifest publication
 -> final-manifest-bound hidden-state cache materialization and sealing.

The providers supplied to this function may execute later when an operator explicitly invokes
it; importing this module performs no model execution, network access, dataset download or
training.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_authoritative_data import LegalDynamicRagEpisodeStep
from training.advanced_rag_strict_cache import AuthoritativeSafetensorSupervisionCache
from training.advanced_rag_supervision import CounterfactualActionProvider, DynamicRewardConfig, SupervisionCacheIdentity
from training.dynamic_dataset_io import VerifiedDynamicDatasetPublication, verify_dynamic_dataset_publication
from training.dynamic_dataset_publication import DynamicDatasetGovernance, DynamicDatasetPublicationReceipt, DynamicTrajectorySource, EpisodeSplitPolicy, publish_dynamic_training_dataset
from training.dynamic_manifest_bound_hidden_cache import DynamicHiddenSupervisionPlanReceipt, ManifestBoundHiddenCacheReceipt, materialize_manifest_bound_hidden_cache, plan_dynamic_hidden_supervision
from training.dynamic_reward_supervision import RealizedRetrievalGainProvider, RealizedRetrievalGainReceipt, apply_realized_retrieval_gains
from training.dynamic_trajectory_materialization import LoggedValueProvider, MaterializedTrajectoryReceipt, TrajectoryMaterializationIdentity, materialize_dynamic_trajectories
from training.dynamic_trajectory_preparation import BoundGeneratorHiddenStateProvider, InformationNeedAnnotationProvider

_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _commit(value: Any) -> str:
    selected = str(value).strip().lower()
    if len(selected) not in {40, 64} or any(ch not in _HEX for ch in selected):
        raise ValueError("source_commit must be a full Git object id")
    return selected


def _proven_gain(step: LegalDynamicRagEpisodeStep) -> bool:
    marker = step.metadata.get("realized_retrieval_gain_provider_sha256")
    if not isinstance(marker, str):
        return False
    selected = marker.strip().lower()
    return len(selected) == 64 and all(ch in _HEX for ch in selected)


@dataclass(frozen=True)
class DynamicRuntimeTrainingLineage:
    source_dataset_sha256: str
    source_dataset_manifest_sha256: str
    runtime_stack_sha256: str
    feature_provider_sha256: str
    behavior_policy_sha256: str
    source_commit: str
    reward_config: DynamicRewardConfig = DynamicRewardConfig()

    def __post_init__(self) -> None:
        for name in ("source_dataset_sha256", "source_dataset_manifest_sha256", "runtime_stack_sha256", "feature_provider_sha256", "behavior_policy_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        object.__setattr__(self, "source_commit", _commit(self.source_commit))
        if not isinstance(self.reward_config, DynamicRewardConfig):
            raise ValueError("reward_config must be DynamicRewardConfig")

    @property
    def lineage_sha256(self) -> str:
        return _digest({
            "schema": "rigorousrag-dynamic-runtime-training-lineage/v1",
            "source_dataset_sha256": self.source_dataset_sha256,
            "source_dataset_manifest_sha256": self.source_dataset_manifest_sha256,
            "runtime_stack_sha256": self.runtime_stack_sha256,
            "feature_provider_sha256": self.feature_provider_sha256,
            "behavior_policy_sha256": self.behavior_policy_sha256,
            "source_commit": self.source_commit,
            "reward_config": {
                "discount": self.reward_config.discount,
                "gae_lambda": self.reward_config.gae_lambda,
                "retrieval_cost": self.reward_config.retrieval_cost,
                "verification_cost": self.reward_config.verification_cost,
                "abstention_cost": self.reward_config.abstention_cost,
            },
        })


@dataclass(frozen=True)
class DynamicPrepublicationReceipt:
    runtime_lineage_sha256: str
    hidden_plan_receipt_sha256: str
    realized_gain_receipt_sha256: str | None
    preexisting_gain_provenance_sha256: str | None
    trajectory_materialization_receipt_sha256: str
    materialized_output_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("runtime_lineage_sha256", "hidden_plan_receipt_sha256", "trajectory_materialization_receipt_sha256", "materialized_output_sha256", "receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        for name in ("realized_gain_receipt_sha256", "preexisting_gain_provenance_sha256"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _sha(value, name))
        if (self.realized_gain_receipt_sha256 is None) == (self.preexisting_gain_provenance_sha256 is None):
            raise ValueError("prepublication receipt requires exactly one realized-gain provenance mode")
        if _digest(self.unsigned()) != self.receipt_sha256:
            raise ValueError("dynamic prepublication receipt digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-dynamic-prepublication-receipt/v1",
            "runtime_lineage_sha256": self.runtime_lineage_sha256,
            "hidden_plan_receipt_sha256": self.hidden_plan_receipt_sha256,
            "realized_gain_receipt_sha256": self.realized_gain_receipt_sha256,
            "preexisting_gain_provenance_sha256": self.preexisting_gain_provenance_sha256,
            "trajectory_materialization_receipt_sha256": self.trajectory_materialization_receipt_sha256,
            "materialized_output_sha256": self.materialized_output_sha256,
        }


@dataclass(frozen=True)
class CanonicalDynamicTrainingDataReceipt:
    prepublication_receipt_sha256: str
    dataset_publication_receipt_sha256: str
    dataset_manifest_sha256: str
    hidden_cache_identity_sha256: str
    hidden_cache_contract_sha256: str
    hidden_cache_receipt_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("prepublication_receipt_sha256", "dataset_publication_receipt_sha256", "dataset_manifest_sha256", "hidden_cache_identity_sha256", "hidden_cache_contract_sha256", "hidden_cache_receipt_sha256", "receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if _digest(self.unsigned()) != self.receipt_sha256:
            raise ValueError("canonical dynamic training-data receipt digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-canonical-dynamic-training-data-receipt/v1",
            "prepublication_receipt_sha256": self.prepublication_receipt_sha256,
            "dataset_publication_receipt_sha256": self.dataset_publication_receipt_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "hidden_cache_identity_sha256": self.hidden_cache_identity_sha256,
            "hidden_cache_contract_sha256": self.hidden_cache_contract_sha256,
            "hidden_cache_receipt_sha256": self.hidden_cache_receipt_sha256,
        }


@dataclass(frozen=True)
class CanonicalDynamicTrainingDataResult:
    dataset: VerifiedDynamicDatasetPublication
    hidden_cache: AuthoritativeSafetensorSupervisionCache
    hidden_plan_receipt: DynamicHiddenSupervisionPlanReceipt
    gain_receipt: RealizedRetrievalGainReceipt | None
    trajectory_receipt: MaterializedTrajectoryReceipt
    publication_receipt: DynamicDatasetPublicationReceipt
    hidden_cache_receipt: ManifestBoundHiddenCacheReceipt
    receipt: CanonicalDynamicTrainingDataReceipt


def _materialization_identity(
    lineage: DynamicRuntimeTrainingLineage,
    *,
    value_provider: LoggedValueProvider,
    counterfactual_provider: CounterfactualActionProvider | None,
) -> TrajectoryMaterializationIdentity:
    value_sha = _sha(getattr(value_provider, "contract_sha256", None), "value provider contract_sha256")
    counterfactual_sha = None if counterfactual_provider is None else _sha(getattr(counterfactual_provider, "contract_sha256", None), "counterfactual provider contract_sha256")
    return TrajectoryMaterializationIdentity(
        source_dataset_sha256=lineage.source_dataset_sha256,
        dataset_manifest_sha256=lineage.source_dataset_manifest_sha256,
        runtime_stack_sha256=lineage.runtime_stack_sha256,
        feature_provider_sha256=lineage.feature_provider_sha256,
        behavior_policy_sha256=lineage.behavior_policy_sha256,
        value_provider_sha256=value_sha,
        counterfactual_provider_sha256=counterfactual_sha,
        source_commit=lineage.source_commit,
        reward_config=lineage.reward_config,
    )


def _preexisting_gain_provenance(steps: Sequence[LegalDynamicRagEpisodeStep]) -> str:
    markers = []
    for step in steps:
        if not _proven_gain(step):
            raise ValueError(
                f"dynamic step {step.episode_id}:{step.step_id} has no governed realized-retrieval-gain provenance; "
                "supply realized_gain_provider instead of using recorder placeholder gains"
            )
        markers.append({
            "episode_id": step.episode_id,
            "step_id": step.step_id,
            "provider_sha256": str(step.metadata["realized_retrieval_gain_provider_sha256"]).strip().lower(),
            "gain": step.realized_retrieval_gain,
        })
    return _digest({"schema": "rigorousrag-preexisting-realized-gain-provenance/v1", "records": markers})


def build_canonical_dynamic_training_data(
    steps: Sequence[LegalDynamicRagEpisodeStep],
    *,
    hidden_provider: BoundGeneratorHiddenStateProvider,
    annotation_provider: InformationNeedAnnotationProvider | None,
    realized_gain_provider: RealizedRetrievalGainProvider | None,
    value_provider: LoggedValueProvider,
    counterfactual_provider: CounterfactualActionProvider | None,
    runtime_lineage: DynamicRuntimeTrainingLineage,
    governance: DynamicDatasetGovernance,
    split_policy: EpisodeSplitPolicy,
    workspace: str | Path,
    hidden_cache_root: str | Path,
    require_need_annotations: bool = True,
) -> CanonicalDynamicTrainingDataResult:
    """Build final manifest-bound dynamic training inputs in the only non-circular order."""
    selected = tuple(steps)
    if not selected or any(not isinstance(step, LegalDynamicRagEpisodeStep) for step in selected):
        raise ValueError("steps must contain a non-empty LegalDynamicRagEpisodeStep sequence")
    if not isinstance(runtime_lineage, DynamicRuntimeTrainingLineage):
        raise ValueError("runtime_lineage must be DynamicRuntimeTrainingLineage")
    root = safe_advanced_path(workspace, label="canonical dynamic training-data workspace", must_exist=False)
    if root.exists() and not root.is_dir():
        raise ValueError("canonical dynamic training-data workspace must be a directory")
    root.mkdir(parents=True, exist_ok=True)

    planned, hidden_plan = plan_dynamic_hidden_supervision(
        selected,
        hidden_provider=hidden_provider,
        annotation_provider=annotation_provider,
        require_need_annotations=require_need_annotations,
    )

    gain_receipt: RealizedRetrievalGainReceipt | None = None
    if realized_gain_provider is not None:
        gain_bound, gain_receipt = apply_realized_retrieval_gains(planned, realized_gain_provider)
        preexisting_gain_sha = None
    else:
        gain_bound = planned
        preexisting_gain_sha = _preexisting_gain_provenance(gain_bound)

    materialization_identity = _materialization_identity(runtime_lineage, value_provider=value_provider, counterfactual_provider=counterfactual_provider)
    materialized = materialize_dynamic_trajectories(
        gain_bound,
        identity=materialization_identity,
        value_provider=value_provider,
        output_path=root / "materialized.dynamic.jsonl",
        counterfactual_provider=counterfactual_provider,
    )
    pre_unsigned = {
        "schema": "rigorousrag-dynamic-prepublication-receipt/v1",
        "runtime_lineage_sha256": runtime_lineage.lineage_sha256,
        "hidden_plan_receipt_sha256": hidden_plan.receipt_sha256,
        "realized_gain_receipt_sha256": None if gain_receipt is None else gain_receipt.receipt_sha256,
        "preexisting_gain_provenance_sha256": preexisting_gain_sha,
        "trajectory_materialization_receipt_sha256": materialized.receipt_sha256,
        "materialized_output_sha256": materialized.output_sha256,
    }
    prepublication = DynamicPrepublicationReceipt(
        runtime_lineage_sha256=runtime_lineage.lineage_sha256,
        hidden_plan_receipt_sha256=hidden_plan.receipt_sha256,
        realized_gain_receipt_sha256=pre_unsigned["realized_gain_receipt_sha256"],
        preexisting_gain_provenance_sha256=preexisting_gain_sha,
        trajectory_materialization_receipt_sha256=materialized.receipt_sha256,
        materialized_output_sha256=materialized.output_sha256,
        receipt_sha256=_digest(pre_unsigned),
    )

    trajectory_source = DynamicTrajectorySource(
        path=materialized.output_path,
        sha256=materialized.output_sha256,
        lineage_receipt_sha256=prepublication.receipt_sha256,
    )
    manifest, publication = publish_dynamic_training_dataset(
        (trajectory_source,),
        governance=governance,
        split_policy=split_policy,
        output_dir=root / "published",
    )
    verified = verify_dynamic_dataset_publication(
        root / "published" / "publication_receipt.json",
        sources=(trajectory_source,),
        require_promotable=governance.require_promotable,
    )
    if verified.manifest.manifest_digest != manifest.manifest_digest or verified.receipt.receipt_sha256 != publication.receipt_sha256:
        raise RuntimeError("dynamic publication verification returned a different final identity")

    cache_root = safe_advanced_path(hidden_cache_root, label="final dynamic hidden-cache root", must_exist=False)
    if cache_root.exists():
        if not cache_root.is_dir():
            raise ValueError("final dynamic hidden-cache root must be a directory")
        if any(cache_root.iterdir()):
            raise ValueError("final dynamic hidden-cache root must be empty before canonical materialization")
    hidden_provider_sha = _sha(getattr(hidden_provider, "contract_sha256", None), "hidden provider contract_sha256")
    generator_sha = _sha(getattr(hidden_provider, "generator_sha256", None), "hidden provider generator_sha256")
    tokenizer_sha = _sha(getattr(hidden_provider, "tokenizer_sha256", None), "hidden provider tokenizer_sha256")
    hidden_config_sha = _digest({
        "schema": "rigorousrag-final-dynamic-hidden-cache-config/v1",
        "hidden_provider_sha256": hidden_provider_sha,
        "hidden_plan_receipt_sha256": hidden_plan.receipt_sha256,
        "prepublication_receipt_sha256": prepublication.receipt_sha256,
        "dataset_publication_receipt_sha256": publication.receipt_sha256,
        "dataset_manifest_sha256": verified.manifest.manifest_digest,
    })
    cache_identity = SupervisionCacheIdentity(
        cache_kind="generator_hidden_states",
        producer_sha256=generator_sha,
        tokenizer_sha256=tokenizer_sha,
        dataset_manifest_sha256=verified.manifest.manifest_digest,
        source_commit=runtime_lineage.source_commit,
        config_sha256=hidden_config_sha,
    )
    hidden_cache = AuthoritativeSafetensorSupervisionCache(cache_root, cache_identity)
    hidden_cache_receipt = materialize_manifest_bound_hidden_cache(verified, hidden_provider=hidden_provider, cache=hidden_cache)

    unsigned = {
        "schema": "rigorousrag-canonical-dynamic-training-data-receipt/v1",
        "prepublication_receipt_sha256": prepublication.receipt_sha256,
        "dataset_publication_receipt_sha256": publication.receipt_sha256,
        "dataset_manifest_sha256": verified.manifest.manifest_digest,
        "hidden_cache_identity_sha256": cache_identity.digest,
        "hidden_cache_contract_sha256": hidden_cache_receipt.cache_contract_sha256,
        "hidden_cache_receipt_sha256": hidden_cache_receipt.receipt_sha256,
    }
    receipt = CanonicalDynamicTrainingDataReceipt(
        prepublication_receipt_sha256=prepublication.receipt_sha256,
        dataset_publication_receipt_sha256=publication.receipt_sha256,
        dataset_manifest_sha256=verified.manifest.manifest_digest,
        hidden_cache_identity_sha256=cache_identity.digest,
        hidden_cache_contract_sha256=hidden_cache_receipt.cache_contract_sha256,
        hidden_cache_receipt_sha256=hidden_cache_receipt.receipt_sha256,
        receipt_sha256=_digest(unsigned),
    )
    return CanonicalDynamicTrainingDataResult(
        dataset=verified,
        hidden_cache=hidden_cache,
        hidden_plan_receipt=hidden_plan,
        gain_receipt=gain_receipt,
        trajectory_receipt=materialized,
        publication_receipt=publication,
        hidden_cache_receipt=hidden_cache_receipt,
        receipt=receipt,
    )


__all__ = [
    "CanonicalDynamicTrainingDataReceipt",
    "CanonicalDynamicTrainingDataResult",
    "DynamicPrepublicationReceipt",
    "DynamicRuntimeTrainingLineage",
    "build_canonical_dynamic_training_data",
]
