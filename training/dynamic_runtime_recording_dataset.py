"""Governed dataset adapter for authoritative dynamic-runtime episode recordings.

This module composes existing authorities instead of inventing another dataset format:

``RecordedDynamicEpisodeReceipt``
 -> ``DynamicTrajectorySource``
 -> ``publish_dynamic_training_dataset``
 -> verified source DatasetManifest
 -> exact ``DynamicRuntimeTrainingLineage`` for canonical target materialization.

The adapter requires a coherent runtime cohort: runtime policy, feature provider, policy artifact /
contract, deterministic behavior-policy contract, training-context provider, terminal-utility
semantics and complete runtime provider contract must match across every episode. Every episode
must also pass strict raw-runtime verification before it can enter the dataset.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from orchestration.dynamic_training_episode_recording import RecordedDynamicEpisodeReceipt
from orchestration.strict_dynamic_training_episode_io import verify_recorded_dynamic_episode_strict
from training.advanced_rag_supervision import DynamicRewardConfig
from training.dynamic_canonical_training_data_pipeline import DynamicRuntimeTrainingLineage
from training.dynamic_dataset_io import VerifiedDynamicDatasetPublication, verify_dynamic_dataset_publication
from training.dynamic_dataset_publication import (
    DynamicDatasetGovernance,
    DynamicTrajectorySource,
    EpisodeSplitPolicy,
    publish_dynamic_training_dataset,
)
from training.production_canonical_limits import assert_production_split_sequence

_MAX_EPISODE_SOURCES = 100_000


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _cohort_value(receipts: Sequence[RecordedDynamicEpisodeReceipt], field: str) -> str:
    values = {getattr(item, field) for item in receipts}
    if len(values) != 1:
        raise ValueError(f"recorded dynamic episode cohort has inconsistent {field}")
    selected = next(iter(values))
    if not isinstance(selected, str) or not selected:
        raise ValueError(f"recorded dynamic episode cohort has invalid {field}")
    return selected


def _cohort_optional_value(receipts: Sequence[RecordedDynamicEpisodeReceipt], field: str) -> str | None:
    values = {getattr(item, field) for item in receipts}
    if len(values) != 1:
        raise ValueError(f"recorded dynamic episode cohort has inconsistent {field}")
    selected = next(iter(values))
    if selected is None:
        return None
    if not isinstance(selected, str) or not selected:
        raise ValueError(f"recorded dynamic episode cohort has invalid {field}")
    return selected


def _episode_receipts(paths: Sequence[str | Path]) -> tuple[RecordedDynamicEpisodeReceipt, ...]:
    selected = tuple(paths)
    if not selected or len(selected) > _MAX_EPISODE_SOURCES:
        raise ValueError(f"recorded dynamic dataset requires 1..{_MAX_EPISODE_SOURCES} episode receipts")
    receipts = tuple(verify_recorded_dynamic_episode_strict(path) for path in selected)
    episode_ids = [item.episode_id for item in receipts]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("recorded dynamic dataset episode ids must be globally unique")
    return tuple(sorted(receipts, key=lambda item: item.episode_id))


def _sources(receipts: Sequence[RecordedDynamicEpisodeReceipt]) -> tuple[DynamicTrajectorySource, ...]:
    return tuple(
        DynamicTrajectorySource(
            path=item.output_path,
            sha256=item.output_sha256,
            lineage_receipt_sha256=item.receipt_sha256,
        )
        for item in receipts
    )


def _source_set_sha256(receipts: Sequence[RecordedDynamicEpisodeReceipt]) -> str:
    return _digest({
        "schema": "rigorousrag-dynamic-trajectory-source-set/v1",
        "sources": [
            {"sha256": item.output_sha256, "lineage_receipt_sha256": item.receipt_sha256}
            for item in receipts
        ],
    })


def _lineage(
    verified: VerifiedDynamicDatasetPublication,
    receipts: Sequence[RecordedDynamicEpisodeReceipt],
    *,
    source_commit: str,
    reward_config: DynamicRewardConfig,
) -> DynamicRuntimeTrainingLineage:
    if not isinstance(reward_config, DynamicRewardConfig):
        raise ValueError("reward_config must be DynamicRewardConfig")
    feature_sha = _cohort_value(receipts, "feature_provider_sha256")
    behavior_sha = _cohort_value(receipts, "behavior_policy_sha256")
    runtime_contract = _cohort_value(receipts, "runtime_provider_contract_sha256")
    _cohort_value(receipts, "runtime_policy_sha256")
    _cohort_value(receipts, "policy_artifact_sha256")
    _cohort_value(receipts, "policy_contract_sha256")
    _cohort_value(receipts, "context_provider_sha256")
    _cohort_optional_value(receipts, "terminal_utility_provider_sha256")
    return DynamicRuntimeTrainingLineage(
        source_dataset_sha256=verified.receipt.source_set_sha256,
        source_dataset_manifest_sha256=verified.manifest.manifest_digest,
        runtime_stack_sha256=runtime_contract,
        feature_provider_sha256=feature_sha,
        behavior_policy_sha256=behavior_sha,
        source_commit=source_commit,
        reward_config=reward_config,
    )


@dataclass(frozen=True)
class VerifiedRecordedDynamicRuntimeDataset:
    dataset: VerifiedDynamicDatasetPublication
    episode_receipts: tuple[RecordedDynamicEpisodeReceipt, ...]
    runtime_lineage: DynamicRuntimeTrainingLineage

    @property
    def source_shards(self) -> tuple[Mapping[str, Any], ...]:
        return tuple({
            "path": item.path,
            "sha256": item.sha256,
            "dataset_manifest_sha256": self.dataset.manifest.manifest_digest,
            "split_name": item.name,
            "expected_record_count": item.record_count,
        } for item in self.dataset.receipt.splits)

    @property
    def runtime_lineage_payload(self) -> Mapping[str, Any]:
        reward = self.runtime_lineage.reward_config
        return {
            "source_dataset_sha256": self.runtime_lineage.source_dataset_sha256,
            "source_dataset_manifest_sha256": self.runtime_lineage.source_dataset_manifest_sha256,
            "runtime_stack_sha256": self.runtime_lineage.runtime_stack_sha256,
            "feature_provider_sha256": self.runtime_lineage.feature_provider_sha256,
            "behavior_policy_sha256": self.runtime_lineage.behavior_policy_sha256,
            "source_commit": self.runtime_lineage.source_commit,
            "reward_config": {
                "discount": reward.discount,
                "gae_lambda": reward.gae_lambda,
                "retrieval_cost": reward.retrieval_cost,
                "verification_cost": reward.verification_cost,
                "abstention_cost": reward.abstention_cost,
            },
        }


def publish_recorded_dynamic_runtime_dataset(
    episode_receipt_paths: Sequence[str | Path],
    *,
    governance: DynamicDatasetGovernance,
    split_policy: EpisodeSplitPolicy,
    source_commit: str,
    reward_config: DynamicRewardConfig = DynamicRewardConfig(),
    output_dir: str | Path,
) -> VerifiedRecordedDynamicRuntimeDataset:
    receipts = _episode_receipts(episode_receipt_paths)
    for field in (
        "runtime_policy_sha256", "feature_provider_sha256", "policy_artifact_sha256",
        "policy_contract_sha256", "behavior_policy_sha256", "context_provider_sha256",
        "runtime_provider_contract_sha256",
    ):
        _cohort_value(receipts, field)
    _cohort_optional_value(receipts, "terminal_utility_provider_sha256")
    manifest, publication = publish_dynamic_training_dataset(
        _sources(receipts),
        governance=governance,
        split_policy=split_policy,
        output_dir=output_dir,
    )
    verified = verify_dynamic_dataset_publication(Path(publication.manifest_path).parent / "publication_receipt.json")
    if verified.manifest.manifest_digest != manifest.manifest_digest or verified.receipt.receipt_sha256 != publication.receipt_sha256:
        raise RuntimeError("recorded dynamic dataset identity changed on read-back")
    if verified.receipt.source_set_sha256 != _source_set_sha256(receipts):
        raise RuntimeError("recorded dynamic dataset source-set identity differs from episode authorities")
    assert_production_split_sequence(verified.receipt.splits, label="recorded dynamic dataset splits")
    return VerifiedRecordedDynamicRuntimeDataset(
        dataset=verified,
        episode_receipts=receipts,
        runtime_lineage=_lineage(verified, receipts, source_commit=source_commit, reward_config=reward_config),
    )


def verify_recorded_dynamic_runtime_dataset(
    publication_receipt_path: str | Path,
    episode_receipt_paths: Sequence[str | Path],
    *,
    source_commit: str,
    reward_config: DynamicRewardConfig = DynamicRewardConfig(),
) -> VerifiedRecordedDynamicRuntimeDataset:
    receipts = _episode_receipts(episode_receipt_paths)
    verified = verify_dynamic_dataset_publication(publication_receipt_path)
    if verified.receipt.source_set_sha256 != _source_set_sha256(receipts):
        raise ValueError("recorded dynamic dataset source-set differs from supplied episode authorities")
    assert_production_split_sequence(verified.receipt.splits, label="recorded dynamic dataset splits")
    return VerifiedRecordedDynamicRuntimeDataset(
        dataset=verified,
        episode_receipts=receipts,
        runtime_lineage=_lineage(verified, receipts, source_commit=source_commit, reward_config=reward_config),
    )


__all__ = [
    "VerifiedRecordedDynamicRuntimeDataset",
    "publish_recorded_dynamic_runtime_dataset",
    "verify_recorded_dynamic_runtime_dataset",
]
