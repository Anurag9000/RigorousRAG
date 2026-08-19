"""Publish materialized dynamic-RAG trajectories as governed train/validation datasets.

Trajectory preparation/materialization produces authoritative episode-step JSONL, but training
also requires immutable split bytes and a ``DatasetManifest``. This module closes that gap.
Episodes—not individual steps—are assigned by a deterministic hash/weight policy, preventing
an episode from leaking across train/validation splits. Source bytes and upstream lineage
receipt SHAs are bound into the resulting artifact identity.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.dataset_governance import DatasetCard, DatasetManifest, DatasetModality, DatasetTask, LicenseStatus, SplitManifest
from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_authoritative_data import LegalDynamicRagEpisodeStep, parse_authoritative_dynamic_step

_MAX_LINE_BYTES = 64 * 1024 * 1024
_MAX_RECORDS = 100_000_000
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


def _identifier(value: Any, label: str, maximum: int = 1_000) -> str:
    if not isinstance(value, str): raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected): raise ValueError(f"{label} is invalid")
    return selected


def _stream_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block: break
            digest.update(block)
    return digest.hexdigest()


def _id_digest(values: Sequence[str]) -> str:
    selected = sorted(set(values))
    return hashlib.sha256(("\n".join(selected) + ("\n" if selected else "")).encode("utf-8")).hexdigest()


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


@dataclass(frozen=True)
class DynamicTrajectorySource:
    path: str
    sha256: str
    lineage_receipt_sha256: str

    def __post_init__(self) -> None:
        source = safe_advanced_path(self.path, label="dynamic trajectory source", must_exist=True, require_file=True)
        object.__setattr__(self, "path", str(source))
        object.__setattr__(self, "sha256", _sha(self.sha256, "trajectory source sha256"))
        object.__setattr__(self, "lineage_receipt_sha256", _sha(self.lineage_receipt_sha256, "trajectory lineage receipt sha256"))
        if _stream_sha(source) != self.sha256: raise ValueError("dynamic trajectory source bytes differ from configured SHA-256")


@dataclass(frozen=True)
class EpisodeSplitPolicy:
    seed: str
    weights: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", _identifier(self.seed, "split seed", 1_000))
        normalized: dict[str, int] = {}
        for name, raw in self.weights.items():
            split = _identifier(name, "split name", 100)
            if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0: raise ValueError("split weights must be positive integers")
            normalized[split] = raw
        if len(normalized) < 2 or len(normalized) > 20: raise ValueError("split policy requires 2..20 unique splits")
        object.__setattr__(self, "weights", normalized)

    @property
    def policy_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-dynamic-episode-split-policy/v1", "seed": self.seed, "weights": dict(sorted(self.weights.items()))})

    def split_for(self, episode_id: str) -> str:
        total = sum(self.weights.values()); bucket = int(hashlib.sha256(f"{self.seed}\n{episode_id}".encode("utf-8")).hexdigest()[:16], 16) % total
        cursor = 0
        for name, weight in sorted(self.weights.items()):
            cursor += weight
            if bucket < cursor: return name
        raise RuntimeError("episode split policy failed to select a split")


@dataclass(frozen=True)
class DynamicDatasetGovernance:
    dataset_id: str
    exact_version: str
    source_locator: str
    license_identifier: str
    license_status: LicenseStatus
    license_evidence: str
    card: DatasetCard
    metadata: Mapping[str, str] = field(default_factory=dict)
    require_promotable: bool = False

    def __post_init__(self) -> None:
        for name in ("dataset_id", "exact_version", "source_locator", "license_identifier", "license_evidence"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip(): raise ValueError(f"{name} is required")
        if not isinstance(self.license_status, LicenseStatus): object.__setattr__(self, "license_status", LicenseStatus(self.license_status))
        if not isinstance(self.card, DatasetCard): raise ValueError("card must be DatasetCard")
        if not isinstance(self.require_promotable, bool): raise ValueError("require_promotable must be boolean")
        object.__setattr__(self, "metadata", {str(key): str(value) for key, value in self.metadata.items()})


@dataclass(frozen=True)
class PublishedDynamicSplit:
    name: str
    path: str
    sha256: str
    record_count: int
    record_id_sha256: str
    episode_id_sha256: str


@dataclass(frozen=True)
class DynamicDatasetPublicationReceipt:
    dataset_manifest_sha256: str
    source_set_sha256: str
    transformation_sha256: str
    split_policy_sha256: str
    manifest_path: str
    splits: tuple[PublishedDynamicSplit, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("dataset_manifest_sha256", "source_set_sha256", "transformation_sha256", "split_policy_sha256", "receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if not self.splits: raise ValueError("publication receipt requires split records")
        if _digest(self.unsigned()) != self.receipt_sha256: raise ValueError("dynamic dataset publication receipt digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {"schema": "rigorousrag-dynamic-dataset-publication-receipt/v1", "dataset_manifest_sha256": self.dataset_manifest_sha256, "source_set_sha256": self.source_set_sha256, "transformation_sha256": self.transformation_sha256, "split_policy_sha256": self.split_policy_sha256, "manifest_path": self.manifest_path, "splits": [asdict(item) for item in self.splits]}


def _strict_record(raw: bytes, label: str) -> LegalDynamicRagEpisodeStep:
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    return parse_authoritative_dynamic_step(value)


def _payload(step: LegalDynamicRagEpisodeStep, *, source_sha256: str, lineage_sha256: str) -> Mapping[str, Any]:
    metadata = dict(step.metadata); metadata["publication_source_sha256"] = source_sha256; metadata["trajectory_lineage_receipt_sha256"] = lineage_sha256
    return {
        "episode_id": step.episode_id, "step_id": step.step_id, "context": step.context, "features": dict(step.features), "action": step.action.value,
        "realized_retrieval_gain": step.realized_retrieval_gain, "behavior_action_probability": step.behavior_action_probability, "advantage": step.advantage,
        "need_spans": [asdict(span) for span in step.need_spans], "hidden_state_cache_key": step.hidden_state_cache_key, "terminal_utility": step.terminal_utility,
        "metadata": metadata, "valid_actions": [action.value for action in step.valid_actions], "value_target": step.value_target,
    }


def publish_dynamic_training_dataset(
    sources: Sequence[DynamicTrajectorySource], *, governance: DynamicDatasetGovernance, split_policy: EpisodeSplitPolicy, output_dir: str | Path,
) -> tuple[DatasetManifest, DynamicDatasetPublicationReceipt]:
    selected = tuple(sources)
    if not selected or len(selected) > 1_000 or any(not isinstance(item, DynamicTrajectorySource) for item in selected): raise ValueError("sources must be a bounded non-empty DynamicTrajectorySource sequence")
    if not isinstance(governance, DynamicDatasetGovernance) or not isinstance(split_policy, EpisodeSplitPolicy): raise ValueError("governance/split_policy have incorrect types")
    root = safe_advanced_path(output_dir, label="dynamic dataset publication output", must_exist=False)
    if root.exists() and not root.is_dir(): raise ValueError("dynamic dataset publication output must be a directory")
    root.mkdir(parents=True, exist_ok=True)
    split_names = tuple(sorted(split_policy.weights)); temporary_handles: dict[str, tuple[int, str, hashlib._Hash]] = {}; seen: set[tuple[str, str]] = set(); record_ids = {name: [] for name in split_names}; episode_ids = {name: [] for name in split_names}; counts = {name: 0 for name in split_names}
    try:
        for name in split_names:
            fd, temp = tempfile.mkstemp(prefix=f".{name}-", suffix=".tmp", dir=root); temporary_handles[name] = (fd, temp, hashlib.sha256())
        handles = {name: os.fdopen(temporary_handles[name][0], "wb") for name in split_names}
        try:
            total = 0
            for source in selected:
                with Path(source.path).open("rb") as stream:
                    for line_number, raw in enumerate(stream, start=1):
                        if not raw.strip(): continue
                        if len(raw) > _MAX_LINE_BYTES: raise ValueError(f"trajectory line {line_number} exceeds safety bound")
                        if total >= _MAX_RECORDS: raise ValueError("dynamic publication exceeds record safety bound")
                        step = _strict_record(raw, f"trajectory line {line_number}"); identity = (step.episode_id, step.step_id)
                        if identity in seen: raise ValueError(f"duplicate dynamic step identity {identity}")
                        seen.add(identity); split = split_policy.split_for(step.episode_id); encoded = _canonical(_payload(step, source_sha256=source.sha256, lineage_sha256=source.lineage_receipt_sha256)) + b"\n"
                        # Reparse canonical output before publication so serialization cannot widen schema.
                        _strict_record(encoded, "canonical dynamic publication row")
                        handles[split].write(encoded); temporary_handles[split][2].update(encoded); counts[split] += 1; record_ids[split].append(f"{step.episode_id}:{step.step_id}"); episode_ids[split].append(step.episode_id); total += 1
            for handle in handles.values(): handle.flush(); os.fsync(handle.fileno())
        finally:
            for handle in handles.values(): handle.close()
        if any(counts[name] <= 0 for name in split_names): raise ValueError(f"deterministic episode split produced an empty split: {counts}")
        published = []
        for name in split_names:
            destination = root / f"{name}.dynamic.jsonl"; os.replace(temporary_handles[name][1], destination); sha = temporary_handles[name][2].hexdigest()
            if _stream_sha(destination) != sha: raise RuntimeError("dynamic split changed during publication")
            published.append(PublishedDynamicSplit(name, str(destination), sha, counts[name], _id_digest(record_ids[name]), _id_digest(episode_ids[name])))
    finally:
        for _, temp, _ in temporary_handles.values():
            if os.path.exists(temp): os.unlink(temp)
    # Episode sets must be disjoint by construction; prove it explicitly before manifest creation.
    episode_sets = {name: set(episode_ids[name]) for name in split_names}
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            if episode_sets[left] & episode_sets[right]: raise RuntimeError("episode leakage detected across published splits")
    source_set = _digest({"schema": "rigorousrag-dynamic-trajectory-source-set/v1", "sources": [{"sha256": item.sha256, "lineage_receipt_sha256": item.lineage_receipt_sha256} for item in selected]})
    transformation = _digest({"schema": "rigorousrag-dynamic-dataset-publication/v1", "source_set_sha256": source_set, "split_policy_sha256": split_policy.policy_sha256})
    manifest = DatasetManifest(
        dataset_id=governance.dataset_id, exact_version=governance.exact_version, source_locator=governance.source_locator, artifact_sha256=source_set,
        license_identifier=governance.license_identifier, license_status=governance.license_status, license_evidence=governance.license_evidence,
        loader_name="training.dynamic_dataset_publication", loader_version="1", transformation_sha256=transformation,
        splits=tuple(SplitManifest(name=item.name, content_sha256=item.sha256, record_count=item.record_count, record_id_sha256=item.record_id_sha256, source_group_sha256=item.episode_id_sha256) for item in published),
        tasks=(DatasetTask.DOMAIN_SPECIFIC,), modalities=(DatasetModality.TEXT,), card=governance.card,
        metadata={**governance.metadata, "canonical_record_kind": "dynamic_rag_episode", "episode_split_policy_sha256": split_policy.policy_sha256},
    )
    if governance.require_promotable: manifest.assert_promotable()
    manifest_path = root / "dataset_manifest.json"; _atomic(manifest_path, _canonical({"schema": "rigorousrag-dataset-manifest/v1", "manifest": asdict(manifest), "manifest_sha256": manifest.manifest_digest}) + b"\n")
    unsigned = {"schema": "rigorousrag-dynamic-dataset-publication-receipt/v1", "dataset_manifest_sha256": manifest.manifest_digest, "source_set_sha256": source_set, "transformation_sha256": transformation, "split_policy_sha256": split_policy.policy_sha256, "manifest_path": str(manifest_path), "splits": [asdict(item) for item in published]}
    receipt = DynamicDatasetPublicationReceipt(manifest.manifest_digest, source_set, transformation, split_policy.policy_sha256, str(manifest_path), tuple(published), _digest(unsigned))
    _atomic(root / "publication_receipt.json", _canonical({**unsigned, "receipt_sha256": receipt.receipt_sha256}) + b"\n")
    return manifest, receipt


__all__ = ["DynamicDatasetGovernance", "DynamicDatasetPublicationReceipt", "DynamicTrajectorySource", "EpisodeSplitPolicy", "PublishedDynamicSplit", "publish_dynamic_training_dataset"]
