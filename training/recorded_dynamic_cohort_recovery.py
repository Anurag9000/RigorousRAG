"""Recovery sealing for an already-published recorded dynamic runtime dataset.

Dataset publication and cohort sealing are intentionally separate immutable authorities. If the
first succeeds and the second is interrupted, rerunning the runtime or republishing the dataset
would be unnecessary and undesirable. This module verifies the existing dataset plus strict
episode receipts, reconstructs the same runtime lineage, and atomically emits only the cohort
envelope.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_supervision import DynamicRewardConfig
from training.dynamic_runtime_recording_dataset import verify_recorded_dynamic_runtime_dataset
from training.recorded_dynamic_cohort_authority import (
    _RECEIPT_FILENAME,
    _SOURCE_FILENAME,
    _canonical,
    _common,
    _digest,
    _runtime_payload,
    _source_record,
    _stream_sha,
)
from training.strict_recorded_dynamic_cohort import verify_recorded_dynamic_cohort_strict


def seal_existing_recorded_dynamic_cohort(
    publication_receipt_path: str | Path,
    episode_receipt_paths: Sequence[str | Path],
    *,
    source_commit: str,
    reward_config: DynamicRewardConfig = DynamicRewardConfig(),
    cohort_output_dir: str | Path,
):
    recorded = verify_recorded_dynamic_runtime_dataset(
        publication_receipt_path,
        episode_receipt_paths,
        source_commit=source_commit,
        reward_config=reward_config,
    )
    receipts = recorded.episode_receipts
    root = safe_advanced_path(cohort_output_dir, label="recovered recorded dynamic cohort output", must_exist=False)
    if root.exists():
        raise ValueError("recovered recorded dynamic cohort output must not already exist")
    parent = safe_advanced_path(root.parent, label="recovered recorded dynamic cohort parent", must_exist=True, require_directory=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name or 'cohort'}-stage-", dir=parent))
    published = False
    try:
        source_path = stage / _SOURCE_FILENAME
        source_digest = hashlib.sha256()
        total_records = 0
        with source_path.open("xb") as handle:
            for receipt in receipts:
                encoded = _canonical(_source_record(receipt)) + b"\n"
                handle.write(encoded)
                source_digest.update(encoded)
                total_records += receipt.record_count
            handle.flush()
            os.fsync(handle.fileno())
        publication_path = safe_advanced_path(
            publication_receipt_path,
            label="recovered cohort dataset publication receipt",
            must_exist=True,
            require_file=True,
        )
        lineage_payload = _runtime_payload(recorded.runtime_lineage)
        unsigned = {
            "schema": "rigorousrag-recorded-dynamic-cohort-receipt/v1",
            "publication_receipt_path": str(publication_path),
            "publication_receipt_sha256": _stream_sha(publication_path),
            "dataset_manifest_sha256": recorded.dataset.manifest.manifest_digest,
            "dataset_source_set_sha256": recorded.dataset.receipt.source_set_sha256,
            "source_list_sha256": source_digest.hexdigest(),
            "episode_count": len(receipts),
            "record_count": total_records,
            "runtime_policy_sha256": _common(receipts, "runtime_policy_sha256"),
            "feature_provider_sha256": _common(receipts, "feature_provider_sha256"),
            "policy_artifact_sha256": _common(receipts, "policy_artifact_sha256"),
            "policy_contract_sha256": _common(receipts, "policy_contract_sha256"),
            "behavior_policy_sha256": _common(receipts, "behavior_policy_sha256"),
            "context_provider_sha256": _common(receipts, "context_provider_sha256"),
            "terminal_utility_provider_sha256": _common(receipts, "terminal_utility_provider_sha256", optional=True),
            "runtime_provider_contract_sha256": _common(receipts, "runtime_provider_contract_sha256"),
            "runtime_lineage": lineage_payload,
            "runtime_lineage_sha256": recorded.runtime_lineage.lineage_sha256,
        }
        payload = {**unsigned, "receipt_sha256": _digest(unsigned)}
        receipt_path = stage / _RECEIPT_FILENAME
        with receipt_path.open("xb") as handle:
            handle.write(_canonical(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        if {item.name for item in stage.iterdir()} != {_SOURCE_FILENAME, _RECEIPT_FILENAME}:
            raise RuntimeError("recovered recorded dynamic cohort directory is not closed")
        os.replace(stage, root)
        published = True
        return verify_recorded_dynamic_cohort_strict(root / _RECEIPT_FILENAME)
    except Exception:
        if published:
            shutil.rmtree(root, ignore_errors=True)
        else:
            shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = ["seal_existing_recorded_dynamic_cohort"]
