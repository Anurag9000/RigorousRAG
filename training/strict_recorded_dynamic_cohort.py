"""Strict restart verification for recorded dynamic-RAG cohort authorities.

This is the production counterpart to the compatibility cohort verifier.  It performs one source
list pass and admits each episode only through ``verify_recorded_dynamic_episode_strict`` while
reusing the cohort receipt/source-set/runtime-lineage contracts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from orchestration.strict_dynamic_training_episode_io import verify_recorded_dynamic_episode_strict
from training.advanced_path_authority import safe_advanced_path
from training.dynamic_dataset_io import verify_dynamic_dataset_publication
from training.production_canonical_limits import assert_production_split_sequence
from training.recorded_dynamic_cohort_authority import (
    RecordedDynamicCohortReceipt,
    VerifiedRecordedDynamicCohort,
    _MAX_EPISODES,
    _MAX_SOURCE_LINE_BYTES,
    _RECEIPT_FILENAME,
    _SOURCE_FILENAME,
    _runtime_lineage,
    _sha,
    _source_set_stream_begin,
    _source_set_stream_item,
    _stream_sha,
    _strict_json,
)


def verify_recorded_dynamic_cohort_strict(path: str | Path) -> VerifiedRecordedDynamicCohort:
    raw_path = Path(path).expanduser()
    if raw_path.is_symlink():
        raise ValueError("recorded dynamic cohort receipt may not be a symlink")
    receipt_path = safe_advanced_path(raw_path, label="strict recorded dynamic cohort receipt", must_exist=True, require_file=True)
    root = receipt_path.parent
    if receipt_path != root / _RECEIPT_FILENAME:
        raise ValueError("recorded dynamic cohort receipt must use canonical filename")
    expected_children = {_SOURCE_FILENAME, _RECEIPT_FILENAME}
    if {item.name for item in root.iterdir()} != expected_children:
        raise ValueError("recorded dynamic cohort directory is not closed")
    if any(item.is_symlink() or not item.is_file() for item in root.iterdir()):
        raise ValueError("recorded dynamic cohort contains a non-regular child")
    raw = _strict_json(receipt_path, "strict recorded dynamic cohort receipt")
    expected = {
        "schema", "publication_receipt_path", "publication_receipt_sha256", "dataset_manifest_sha256",
        "dataset_source_set_sha256", "source_list_sha256", "episode_count", "record_count",
        "runtime_policy_sha256", "feature_provider_sha256", "policy_artifact_sha256",
        "policy_contract_sha256", "behavior_policy_sha256", "context_provider_sha256",
        "terminal_utility_provider_sha256", "runtime_provider_contract_sha256", "runtime_lineage",
        "runtime_lineage_sha256", "receipt_sha256",
    }
    if set(raw) != expected or raw.get("schema") != "rigorousrag-recorded-dynamic-cohort-receipt/v1":
        raise ValueError("unsupported recorded dynamic cohort receipt schema")
    receipt = RecordedDynamicCohortReceipt(**{key: value for key, value in raw.items() if key != "schema"})
    publication_path = Path(receipt.publication_receipt_path)
    if _stream_sha(publication_path) != receipt.publication_receipt_sha256:
        raise ValueError("recorded dynamic cohort publication receipt bytes changed")
    dataset = verify_dynamic_dataset_publication(publication_path)
    if dataset.manifest.manifest_digest != receipt.dataset_manifest_sha256 or dataset.receipt.source_set_sha256 != receipt.dataset_source_set_sha256:
        raise ValueError("recorded dynamic cohort dataset identity differs from receipt")
    assert_production_split_sequence(dataset.receipt.splits, label="strict recorded dynamic cohort dataset splits")

    source_path = root / _SOURCE_FILENAME
    if _stream_sha(source_path) != receipt.source_list_sha256:
        raise ValueError("recorded dynamic cohort source-list bytes changed")
    count = 0
    records = 0
    previous_episode: str | None = None
    source_set = hashlib.sha256()
    _source_set_stream_begin(source_set)
    common: dict[str, Any] = {}
    with source_path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if len(line) > _MAX_SOURCE_LINE_BYTES or count >= _MAX_EPISODES:
                raise ValueError("recorded dynamic cohort source list exceeds safety bound")
            try:
                item = json.loads(
                    line.decode("utf-8", errors="strict"),
                    parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
                )
            except Exception as exc:
                raise ValueError(f"recorded dynamic cohort source line {line_number} is invalid") from exc
            fields = {"episode_id", "episode_receipt_path", "episode_receipt_sha256", "output_sha256", "record_count"}
            if not isinstance(item, Mapping) or set(item) != fields:
                raise ValueError("recorded dynamic cohort source record fields are invalid")
            episode = verify_recorded_dynamic_episode_strict(item["episode_receipt_path"])
            if (
                episode.episode_id != item["episode_id"]
                or episode.receipt_sha256 != _sha(item["episode_receipt_sha256"], "episode_receipt_sha256")
                or episode.output_sha256 != _sha(item["output_sha256"], "output_sha256")
                or episode.record_count != item["record_count"]
            ):
                raise ValueError("recorded dynamic cohort source record differs from strict episode authority")
            if previous_episode is not None and episode.episode_id <= previous_episode:
                raise ValueError("recorded dynamic cohort source list must be strictly episode-id sorted")
            previous_episode = episode.episode_id
            _source_set_stream_item(
                source_set,
                output_sha256=episode.output_sha256,
                receipt_sha256=episode.receipt_sha256,
                first=(count == 0),
            )
            for field in (
                "runtime_policy_sha256", "feature_provider_sha256", "policy_artifact_sha256",
                "policy_contract_sha256", "behavior_policy_sha256", "context_provider_sha256",
                "terminal_utility_provider_sha256", "runtime_provider_contract_sha256",
            ):
                current = getattr(episode, field)
                if field not in common:
                    common[field] = current
                elif common[field] != current:
                    raise ValueError(f"recorded dynamic cohort has inconsistent {field}")
            records += episode.record_count
            count += 1
    source_set.update(b"]}")
    if count != receipt.episode_count or records != receipt.record_count:
        raise ValueError("recorded dynamic cohort episode/record counts differ")
    if source_set.hexdigest() != receipt.dataset_source_set_sha256:
        raise ValueError("recorded dynamic cohort source set differs from dataset publisher")
    for field in (
        "runtime_policy_sha256", "feature_provider_sha256", "policy_artifact_sha256",
        "policy_contract_sha256", "behavior_policy_sha256", "context_provider_sha256",
        "terminal_utility_provider_sha256", "runtime_provider_contract_sha256",
    ):
        if common.get(field) != getattr(receipt, field):
            raise ValueError(f"recorded dynamic cohort receipt differs from episode {field}")
    lineage = _runtime_lineage(receipt.runtime_lineage)
    if (
        lineage.source_dataset_sha256 != dataset.receipt.source_set_sha256
        or lineage.source_dataset_manifest_sha256 != dataset.manifest.manifest_digest
        or lineage.runtime_stack_sha256 != receipt.runtime_provider_contract_sha256
        or lineage.feature_provider_sha256 != receipt.feature_provider_sha256
        or lineage.behavior_policy_sha256 != receipt.behavior_policy_sha256
    ):
        raise ValueError("recorded dynamic cohort runtime lineage differs from strict dataset/episode authority")
    return VerifiedRecordedDynamicCohort(str(root), dataset, receipt)


__all__ = ["verify_recorded_dynamic_cohort_strict"]
