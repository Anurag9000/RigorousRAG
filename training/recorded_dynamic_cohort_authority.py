"""Sealed cohort authority for recorded dynamic-RAG runtime datasets.

The existing dynamic dataset publisher remains the byte/manifest authority.  This envelope seals
which authoritative runtime episode receipts formed that dataset and the one coherent runtime
lineage used for later canonical target materialization.  Verification streams the source list,
reopens every episode receipt, recomputes the publisher's exact source-set SHA incrementally and
never loads episode JSONL contents into memory.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from orchestration.dynamic_training_episode_recording import verify_recorded_dynamic_episode
from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_supervision import DynamicRewardConfig
from training.dynamic_canonical_training_data_pipeline import DynamicRuntimeTrainingLineage
from training.dynamic_dataset_io import VerifiedDynamicDatasetPublication, verify_dynamic_dataset_publication
from training.dynamic_dataset_publication import DynamicDatasetGovernance, EpisodeSplitPolicy
from training.dynamic_runtime_recording_dataset import publish_recorded_dynamic_runtime_dataset
from training.production_canonical_limits import assert_production_split_sequence

_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_SOURCE_LINE_BYTES = 4 * 1024 * 1024
_MAX_EPISODES = 100_000
_HEX = frozenset("0123456789abcdef")
_SOURCE_FILENAME = "episode_sources.jsonl"
_RECEIPT_FILENAME = "cohort_receipt.json"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _stream_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path, label: str) -> Mapping[str, Any]:
    if path.stat().st_size <= 0 or path.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds byte safety bound")
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain an object")
    return value


def _runtime_payload(lineage: DynamicRuntimeTrainingLineage) -> Mapping[str, Any]:
    reward = lineage.reward_config
    return {
        "source_dataset_sha256": lineage.source_dataset_sha256,
        "source_dataset_manifest_sha256": lineage.source_dataset_manifest_sha256,
        "runtime_stack_sha256": lineage.runtime_stack_sha256,
        "feature_provider_sha256": lineage.feature_provider_sha256,
        "behavior_policy_sha256": lineage.behavior_policy_sha256,
        "source_commit": lineage.source_commit,
        "reward_config": {
            "discount": reward.discount,
            "gae_lambda": reward.gae_lambda,
            "retrieval_cost": reward.retrieval_cost,
            "verification_cost": reward.verification_cost,
            "abstention_cost": reward.abstention_cost,
        },
    }


def _runtime_lineage(raw: Any) -> DynamicRuntimeTrainingLineage:
    if not isinstance(raw, Mapping):
        raise ValueError("recorded cohort runtime_lineage must be an object")
    required = {
        "source_dataset_sha256", "source_dataset_manifest_sha256", "runtime_stack_sha256",
        "feature_provider_sha256", "behavior_policy_sha256", "source_commit", "reward_config",
    }
    if set(raw) != required or not isinstance(raw["reward_config"], Mapping):
        raise ValueError("recorded cohort runtime_lineage fields are invalid")
    reward_fields = {"discount", "gae_lambda", "retrieval_cost", "verification_cost", "abstention_cost"}
    if set(raw["reward_config"]) != reward_fields:
        raise ValueError("recorded cohort reward_config fields are invalid")
    return DynamicRuntimeTrainingLineage(
        source_dataset_sha256=raw["source_dataset_sha256"],
        source_dataset_manifest_sha256=raw["source_dataset_manifest_sha256"],
        runtime_stack_sha256=raw["runtime_stack_sha256"],
        feature_provider_sha256=raw["feature_provider_sha256"],
        behavior_policy_sha256=raw["behavior_policy_sha256"],
        source_commit=raw["source_commit"],
        reward_config=DynamicRewardConfig(**dict(raw["reward_config"])),
    )


@dataclass(frozen=True)
class RecordedDynamicCohortReceipt:
    publication_receipt_path: str
    publication_receipt_sha256: str
    dataset_manifest_sha256: str
    dataset_source_set_sha256: str
    source_list_sha256: str
    episode_count: int
    record_count: int
    runtime_policy_sha256: str
    feature_provider_sha256: str
    policy_artifact_sha256: str
    policy_contract_sha256: str
    behavior_policy_sha256: str
    context_provider_sha256: str
    terminal_utility_provider_sha256: str | None
    runtime_provider_contract_sha256: str
    runtime_lineage: Mapping[str, Any]
    runtime_lineage_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        publication = safe_advanced_path(self.publication_receipt_path, label="recorded cohort publication receipt", must_exist=True, require_file=True)
        object.__setattr__(self, "publication_receipt_path", str(publication))
        for name in (
            "publication_receipt_sha256", "dataset_manifest_sha256", "dataset_source_set_sha256",
            "source_list_sha256", "runtime_policy_sha256", "feature_provider_sha256",
            "policy_artifact_sha256", "policy_contract_sha256", "behavior_policy_sha256",
            "context_provider_sha256", "runtime_provider_contract_sha256", "runtime_lineage_sha256",
            "receipt_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if self.terminal_utility_provider_sha256 is not None:
            object.__setattr__(self, "terminal_utility_provider_sha256", _sha(self.terminal_utility_provider_sha256, "terminal_utility_provider_sha256"))
        for name in ("episode_count", "record_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive")
        lineage = _runtime_lineage(self.runtime_lineage)
        payload = _runtime_payload(lineage)
        object.__setattr__(self, "runtime_lineage", payload)
        if lineage.lineage_sha256 != self.runtime_lineage_sha256:
            raise ValueError("recorded cohort runtime lineage digest mismatch")
        if _digest(self.unsigned()) != self.receipt_sha256:
            raise ValueError("recorded dynamic cohort receipt digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-recorded-dynamic-cohort-receipt/v1",
            **{name: getattr(self, name) for name in (
                "publication_receipt_path", "publication_receipt_sha256", "dataset_manifest_sha256",
                "dataset_source_set_sha256", "source_list_sha256", "episode_count", "record_count",
                "runtime_policy_sha256", "feature_provider_sha256", "policy_artifact_sha256",
                "policy_contract_sha256", "behavior_policy_sha256", "context_provider_sha256",
                "terminal_utility_provider_sha256", "runtime_provider_contract_sha256",
                "runtime_lineage", "runtime_lineage_sha256",
            )},
        }


@dataclass(frozen=True)
class VerifiedRecordedDynamicCohort:
    root: str
    dataset: VerifiedDynamicDatasetPublication
    receipt: RecordedDynamicCohortReceipt

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
        return dict(self.receipt.runtime_lineage)


def _source_record(receipt: Any) -> Mapping[str, Any]:
    return {
        "episode_id": receipt.episode_id,
        "episode_receipt_path": str(Path(receipt.output_path).parent / "episode_receipt.json"),
        "episode_receipt_sha256": receipt.receipt_sha256,
        "output_sha256": receipt.output_sha256,
        "record_count": receipt.record_count,
    }


def _common(receipts: Sequence[Any], field: str, *, optional: bool = False) -> str | None:
    values = {getattr(item, field) for item in receipts}
    if len(values) != 1:
        raise ValueError(f"recorded dynamic cohort has inconsistent {field}")
    selected = next(iter(values))
    if selected is None and optional:
        return None
    if not isinstance(selected, str) or not selected:
        raise ValueError(f"recorded dynamic cohort has invalid {field}")
    return selected


def publish_recorded_dynamic_cohort(
    episode_receipt_paths: Sequence[str | Path],
    *,
    governance: DynamicDatasetGovernance,
    split_policy: EpisodeSplitPolicy,
    source_commit: str,
    reward_config: DynamicRewardConfig = DynamicRewardConfig(),
    dataset_output_dir: str | Path,
    cohort_output_dir: str | Path,
) -> VerifiedRecordedDynamicCohort:
    recorded = publish_recorded_dynamic_runtime_dataset(
        episode_receipt_paths,
        governance=governance,
        split_policy=split_policy,
        source_commit=source_commit,
        reward_config=reward_config,
        output_dir=dataset_output_dir,
    )
    receipts = recorded.episode_receipts
    root = safe_advanced_path(cohort_output_dir, label="recorded dynamic cohort output", must_exist=False)
    if root.exists():
        raise ValueError("recorded dynamic cohort output must not already exist")
    parent = safe_advanced_path(root.parent, label="recorded dynamic cohort parent", must_exist=True, require_directory=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name or 'cohort'}-stage-", dir=parent))
    published = False
    try:
        source_path = stage / _SOURCE_FILENAME
        source_digest = hashlib.sha256()
        total_records = 0
        with source_path.open("xb") as handle:
            for receipt in receipts:
                encoded = _canonical(_source_record(receipt)) + b"\n"
                handle.write(encoded); source_digest.update(encoded); total_records += receipt.record_count
            handle.flush(); os.fsync(handle.fileno())
        publication_path = Path(recorded.dataset.receipt.manifest_path).parent / "publication_receipt.json"
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
            handle.write(_canonical(payload) + b"\n"); handle.flush(); os.fsync(handle.fileno())
        if {item.name for item in stage.iterdir()} != {_SOURCE_FILENAME, _RECEIPT_FILENAME}:
            raise RuntimeError("recorded dynamic cohort directory is not closed")
        os.replace(stage, root); published = True
        return verify_recorded_dynamic_cohort(root / _RECEIPT_FILENAME)
    except Exception:
        if published:
            shutil.rmtree(root, ignore_errors=True)
        else:
            shutil.rmtree(stage, ignore_errors=True)
        raise


def _source_set_stream_begin(digest: Any) -> None:
    digest.update(b'{"schema":"rigorousrag-dynamic-trajectory-source-set/v1","sources":[')


def _source_set_stream_item(digest: Any, *, output_sha256: str, receipt_sha256: str, first: bool) -> None:
    if not first:
        digest.update(b",")
    digest.update(_canonical({"sha256": output_sha256, "lineage_receipt_sha256": receipt_sha256}))


def verify_recorded_dynamic_cohort(path: str | Path) -> VerifiedRecordedDynamicCohort:
    raw_path = Path(path).expanduser()
    if raw_path.is_symlink():
        raise ValueError("recorded dynamic cohort receipt may not be a symlink")
    receipt_path = safe_advanced_path(raw_path, label="recorded dynamic cohort receipt", must_exist=True, require_file=True)
    root = receipt_path.parent
    if receipt_path != root / _RECEIPT_FILENAME:
        raise ValueError("recorded dynamic cohort receipt must use canonical filename")
    expected_children = {_SOURCE_FILENAME, _RECEIPT_FILENAME}
    if {item.name for item in root.iterdir()} != expected_children:
        raise ValueError("recorded dynamic cohort directory is not closed")
    if any(item.is_symlink() or not item.is_file() for item in root.iterdir()):
        raise ValueError("recorded dynamic cohort contains a non-regular child")
    raw = _strict_json(receipt_path, "recorded dynamic cohort receipt")
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
    assert_production_split_sequence(dataset.receipt.splits, label="recorded dynamic cohort dataset splits")

    source_path = root / _SOURCE_FILENAME
    if _stream_sha(source_path) != receipt.source_list_sha256:
        raise ValueError("recorded dynamic cohort source-list bytes changed")
    count = 0; records = 0; previous_episode: str | None = None
    source_set = hashlib.sha256(); _source_set_stream_begin(source_set)
    common: dict[str, Any] = {}
    with source_path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if len(line) > _MAX_SOURCE_LINE_BYTES or count >= _MAX_EPISODES:
                raise ValueError("recorded dynamic cohort source list exceeds safety bound")
            try:
                item = json.loads(line.decode("utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
            except Exception as exc:
                raise ValueError(f"recorded dynamic cohort source line {line_number} is invalid") from exc
            fields = {"episode_id", "episode_receipt_path", "episode_receipt_sha256", "output_sha256", "record_count"}
            if not isinstance(item, Mapping) or set(item) != fields:
                raise ValueError("recorded dynamic cohort source record fields are invalid")
            episode = verify_recorded_dynamic_episode(item["episode_receipt_path"])
            if episode.episode_id != item["episode_id"] or episode.receipt_sha256 != _sha(item["episode_receipt_sha256"], "episode_receipt_sha256") or episode.output_sha256 != _sha(item["output_sha256"], "output_sha256") or episode.record_count != item["record_count"]:
                raise ValueError("recorded dynamic cohort source record differs from episode authority")
            if previous_episode is not None and episode.episode_id <= previous_episode:
                raise ValueError("recorded dynamic cohort source list must be strictly episode-id sorted")
            previous_episode = episode.episode_id
            _source_set_stream_item(source_set, output_sha256=episode.output_sha256, receipt_sha256=episode.receipt_sha256, first=(count == 0))
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
            records += episode.record_count; count += 1
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
    if lineage.source_dataset_sha256 != dataset.receipt.source_set_sha256 or lineage.source_dataset_manifest_sha256 != dataset.manifest.manifest_digest or lineage.runtime_stack_sha256 != receipt.runtime_provider_contract_sha256 or lineage.feature_provider_sha256 != receipt.feature_provider_sha256 or lineage.behavior_policy_sha256 != receipt.behavior_policy_sha256:
        raise ValueError("recorded dynamic cohort runtime lineage differs from dataset/episode authority")
    return VerifiedRecordedDynamicCohort(str(root), dataset, receipt)


__all__ = [
    "RecordedDynamicCohortReceipt",
    "VerifiedRecordedDynamicCohort",
    "publish_recorded_dynamic_cohort",
    "verify_recorded_dynamic_cohort",
]
