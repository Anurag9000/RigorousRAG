"""Strict read-side verification for governed dynamic-RAG dataset publication."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.dataset_governance import DatasetCard, DatasetManifest, DatasetModality, DatasetTask, LicenseStatus, SplitManifest
from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_authoritative_data import ManifestBoundAuthoritativeJsonlDataset
from training.dynamic_dataset_publication import DynamicDatasetPublicationReceipt, DynamicTrajectorySource, PublishedDynamicSplit

_MAX_BYTES = 64 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected): raise ValueError(f"{label} must be SHA-256")
    return selected


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _id_digest(values: Sequence[str]) -> str:
    selected = sorted(set(values)); return hashlib.sha256(("\n".join(selected) + ("\n" if selected else "")).encode("utf-8")).hexdigest()


def _read(path: str | Path, label: str) -> Mapping[str, Any]:
    source = safe_advanced_path(path, label=label, must_exist=True, require_file=True)
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_BYTES: raise ValueError(f"{label} exceeds byte safety bound")
    try: value = json.loads(source.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
    except Exception as exc: raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping): raise ValueError(f"{label} must contain an object")
    return value


def _manifest(raw: Any) -> DatasetManifest:
    if not isinstance(raw, Mapping): raise ValueError("dynamic manifest must be an object")
    expected = {"dataset_id", "exact_version", "source_locator", "artifact_sha256", "license_identifier", "license_status", "license_evidence", "loader_name", "loader_version", "transformation_sha256", "splits", "tasks", "modalities", "card", "metadata"}
    if set(raw) != expected: raise ValueError("dynamic manifest fields differ from DatasetManifest")
    card_raw = raw["card"]
    if not isinstance(card_raw, Mapping): raise ValueError("dynamic dataset card must be an object")
    card = DatasetCard(summary=card_raw["summary"], intended_uses=tuple(card_raw["intended_uses"]), forbidden_uses=tuple(card_raw["forbidden_uses"]), populations_or_domains=tuple(card_raw["populations_or_domains"]), languages=tuple(card_raw["languages"]), pii_notes=card_raw["pii_notes"], safety_notes=card_raw["safety_notes"], source_citation=card_raw["source_citation"], known_limitations=tuple(card_raw["known_limitations"]))
    split_fields = {"name", "content_sha256", "record_count", "record_id_sha256", "source_group_sha256", "query_id_sha256", "document_id_sha256"}
    splits_raw = raw["splits"]
    if not isinstance(splits_raw, list): raise ValueError("dynamic manifest splits must be an array")
    splits = []
    for item in splits_raw:
        if not isinstance(item, Mapping) or set(item) != split_fields: raise ValueError("dynamic manifest split fields are invalid")
        splits.append(SplitManifest(**dict(item)))
    metadata = raw["metadata"]
    if not isinstance(metadata, Mapping): raise ValueError("dynamic manifest metadata must be an object")
    return DatasetManifest(dataset_id=raw["dataset_id"], exact_version=raw["exact_version"], source_locator=raw["source_locator"], artifact_sha256=raw["artifact_sha256"], license_identifier=raw["license_identifier"], license_status=LicenseStatus(raw["license_status"]), license_evidence=raw["license_evidence"], loader_name=raw["loader_name"], loader_version=raw["loader_version"], transformation_sha256=raw["transformation_sha256"], splits=tuple(splits), tasks=tuple(DatasetTask(item) for item in raw["tasks"]), modalities=tuple(DatasetModality(item) for item in raw["modalities"]), card=card, metadata={str(key): str(value) for key, value in metadata.items()})


@dataclass(frozen=True)
class VerifiedDynamicDatasetPublication:
    manifest: DatasetManifest
    receipt: DynamicDatasetPublicationReceipt

    def split(self, name: str) -> ManifestBoundAuthoritativeJsonlDataset:
        matches = [item for item in self.receipt.splits if item.name == name]
        if len(matches) != 1: raise ValueError(f"unknown dynamic split {name!r}")
        item = matches[0]
        return ManifestBoundAuthoritativeJsonlDataset(item.path, expected_sha256=item.sha256, dataset_manifest_sha256=self.manifest.manifest_digest, split_name=item.name, record_kind="dynamic_rag_episode", expected_record_count=item.record_count)


def verify_dynamic_dataset_publication(receipt_path: str | Path, *, sources: Sequence[DynamicTrajectorySource] | None = None, require_promotable: bool = False) -> VerifiedDynamicDatasetPublication:
    raw = _read(receipt_path, "dynamic dataset publication receipt")
    required = {"schema", "dataset_manifest_sha256", "source_set_sha256", "transformation_sha256", "split_policy_sha256", "manifest_path", "splits", "receipt_sha256"}
    if set(raw) != required or raw.get("schema") != "rigorousrag-dynamic-dataset-publication-receipt/v1": raise ValueError("unsupported dynamic dataset publication receipt schema")
    split_raw = raw["splits"]
    if not isinstance(split_raw, list) or not split_raw: raise ValueError("dynamic publication receipt requires splits")
    split_fields = {"name", "path", "sha256", "record_count", "record_id_sha256", "episode_id_sha256"}; splits = []
    for item in split_raw:
        if not isinstance(item, Mapping) or set(item) != split_fields: raise ValueError("dynamic publication split receipt fields are invalid")
        splits.append(PublishedDynamicSplit(**dict(item)))
    receipt = DynamicDatasetPublicationReceipt(raw["dataset_manifest_sha256"], raw["source_set_sha256"], raw["transformation_sha256"], raw["split_policy_sha256"], raw["manifest_path"], tuple(splits), raw["receipt_sha256"])
    envelope = _read(receipt.manifest_path, "dynamic dataset manifest")
    if set(envelope) != {"schema", "manifest", "manifest_sha256"} or envelope.get("schema") != "rigorousrag-dataset-manifest/v1": raise ValueError("unsupported dynamic dataset manifest envelope")
    manifest = _manifest(envelope["manifest"])
    if manifest.manifest_digest != _sha(envelope["manifest_sha256"], "manifest_sha256") or manifest.manifest_digest != receipt.dataset_manifest_sha256: raise ValueError("dynamic dataset manifest digest differs from receipt")
    if manifest.artifact_sha256 != receipt.source_set_sha256 or manifest.transformation_sha256 != receipt.transformation_sha256: raise ValueError("dynamic manifest source/transformation identity differs from receipt")
    if require_promotable: manifest.assert_promotable()
    if sources is not None:
        selected = tuple(sources)
        if not selected: raise ValueError("sources may not be empty when supplied")
        source_set = _digest({"schema": "rigorousrag-dynamic-trajectory-source-set/v1", "sources": [{"sha256": item.sha256, "lineage_receipt_sha256": item.lineage_receipt_sha256} for item in selected]})
        if source_set != receipt.source_set_sha256: raise ValueError("supplied trajectory sources differ from publication source-set identity")
    by_name = {item.name: item for item in manifest.splits}
    if set(by_name) != {item.name for item in receipt.splits}: raise ValueError("dynamic manifest splits differ from publication receipt")
    verified = VerifiedDynamicDatasetPublication(manifest, receipt); episode_sets: dict[str, set[str]] = {}
    for item in receipt.splits:
        manifest_split = by_name[item.name]
        if manifest_split.content_sha256 != item.sha256 or manifest_split.record_count != item.record_count or manifest_split.record_id_sha256 != item.record_id_sha256 or manifest_split.source_group_sha256 != item.episode_id_sha256: raise ValueError(f"dynamic split {item.name} differs between manifest and receipt")
        dataset = verified.split(item.name); record_ids: list[str] = []; episodes: list[str] = []
        for index in range(len(dataset)):
            step = dataset[index]; record_ids.append(f"{step.episode_id}:{step.step_id}"); episodes.append(step.episode_id)
        if _id_digest(record_ids) != item.record_id_sha256 or _id_digest(episodes) != item.episode_id_sha256: raise ValueError(f"dynamic split {item.name} identity digests differ from receipt")
        episode_sets[item.name] = set(episodes)
    names = sorted(episode_sets)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if episode_sets[left] & episode_sets[right]: raise ValueError("dynamic publication leaks episode ids across splits")
    return verified


__all__ = ["VerifiedDynamicDatasetPublication", "verify_dynamic_dataset_publication"]
