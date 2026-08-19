"""Strict read-side verification for governed grounded-training imports."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from evaluation.dataset_governance import DatasetCard, DatasetManifest, DatasetModality, DatasetTask, LicenseStatus, SplitManifest
from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_authoritative_data import ManifestBoundAuthoritativeJsonlDataset
from training.governed_grounded_import import GovernedGroundedImportReceipt, GroundedSplitImportReceipt

_MAX_JSON_BYTES = 64 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _id_digest(values: list[str]) -> str:
    selected = sorted(set(values))
    return hashlib.sha256(("\n".join(selected) + ("\n" if selected else "")).encode("utf-8")).hexdigest()


def _read_json(path: str | Path, label: str) -> Mapping[str, Any]:
    source = safe_advanced_path(path, label=label, must_exist=True, require_file=True)
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds byte safety bound")
    try:
        payload = json.loads(source.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain an object")
    return payload


def _manifest(value: Any) -> DatasetManifest:
    if not isinstance(value, Mapping):
        raise ValueError("grounded dataset manifest must be an object")
    expected = {"dataset_id", "exact_version", "source_locator", "artifact_sha256", "license_identifier", "license_status", "license_evidence", "loader_name", "loader_version", "transformation_sha256", "splits", "tasks", "modalities", "card", "metadata"}
    if set(value) != expected:
        raise ValueError("grounded dataset manifest fields differ from DatasetManifest")
    card_raw = value["card"]
    if not isinstance(card_raw, Mapping):
        raise ValueError("grounded dataset card must be an object")
    card = DatasetCard(
        summary=card_raw["summary"], intended_uses=tuple(card_raw["intended_uses"]), forbidden_uses=tuple(card_raw["forbidden_uses"]),
        populations_or_domains=tuple(card_raw["populations_or_domains"]), languages=tuple(card_raw["languages"]), pii_notes=card_raw["pii_notes"],
        safety_notes=card_raw["safety_notes"], source_citation=card_raw["source_citation"], known_limitations=tuple(card_raw["known_limitations"]),
    )
    splits_raw = value["splits"]
    if not isinstance(splits_raw, list):
        raise ValueError("grounded dataset splits must be an array")
    splits = tuple(SplitManifest(**dict(item)) for item in splits_raw if isinstance(item, Mapping))
    if len(splits) != len(splits_raw):
        raise ValueError("grounded dataset split entries must be objects")
    metadata = value["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("grounded dataset metadata must be an object")
    return DatasetManifest(
        dataset_id=value["dataset_id"], exact_version=value["exact_version"], source_locator=value["source_locator"], artifact_sha256=value["artifact_sha256"],
        license_identifier=value["license_identifier"], license_status=LicenseStatus(value["license_status"]), license_evidence=value["license_evidence"],
        loader_name=value["loader_name"], loader_version=value["loader_version"], transformation_sha256=value["transformation_sha256"], splits=splits,
        tasks=tuple(DatasetTask(item) for item in value["tasks"]), modalities=tuple(DatasetModality(item) for item in value["modalities"]), card=card,
        metadata={str(key): str(item) for key, item in metadata.items()},
    )


@dataclass(frozen=True)
class VerifiedGovernedGroundedDataset:
    manifest: DatasetManifest
    receipt: GovernedGroundedImportReceipt

    def split(self, name: str) -> ManifestBoundAuthoritativeJsonlDataset:
        matches = [item for item in self.receipt.splits if item.name == name]
        if len(matches) != 1:
            raise ValueError(f"unknown grounded split: {name}")
        item = matches[0]
        return ManifestBoundAuthoritativeJsonlDataset(
            item.output_path,
            expected_sha256=item.output_sha256,
            dataset_manifest_sha256=self.manifest.manifest_digest,
            split_name=item.name,
            record_kind="grounded_generation",
            expected_record_count=item.record_count,
        )


def verify_governed_grounded_import(receipt_path: str | Path, *, require_promotable: bool = False) -> VerifiedGovernedGroundedDataset:
    raw = _read_json(receipt_path, "grounded import receipt")
    expected = {"schema", "dataset_manifest_sha256", "source_set_sha256", "transformation_sha256", "manifest_path", "splits", "receipt_sha256"}
    if set(raw) != expected or raw.get("schema") != "rigorousrag-governed-grounded-import-receipt/v1":
        raise ValueError("unsupported grounded import receipt schema")
    split_raw = raw["splits"]
    if not isinstance(split_raw, list) or not split_raw:
        raise ValueError("grounded import receipt requires split receipts")
    split_fields = {"name", "source_sha256", "output_path", "output_sha256", "record_count", "record_id_sha256", "evidence_id_sha256", "transformation_sha256"}
    receipts = []
    for item in split_raw:
        if not isinstance(item, Mapping) or set(item) != split_fields:
            raise ValueError("grounded split receipt fields are invalid")
        receipts.append(GroundedSplitImportReceipt(**dict(item)))
    receipt = GovernedGroundedImportReceipt(
        dataset_manifest_sha256=raw["dataset_manifest_sha256"], source_set_sha256=raw["source_set_sha256"], transformation_sha256=raw["transformation_sha256"],
        manifest_path=raw["manifest_path"], splits=tuple(receipts), receipt_sha256=raw["receipt_sha256"],
    )
    manifest_envelope = _read_json(receipt.manifest_path, "grounded dataset manifest")
    if set(manifest_envelope) != {"schema", "manifest", "manifest_sha256"} or manifest_envelope.get("schema") != "rigorousrag-dataset-manifest/v1":
        raise ValueError("unsupported grounded dataset manifest envelope")
    manifest = _manifest(manifest_envelope["manifest"])
    if manifest.manifest_digest != _sha(manifest_envelope["manifest_sha256"], "manifest_sha256") or manifest.manifest_digest != receipt.dataset_manifest_sha256:
        raise ValueError("grounded dataset manifest digest differs from receipt")
    if manifest.artifact_sha256 != receipt.source_set_sha256 or manifest.transformation_sha256 != receipt.transformation_sha256:
        raise ValueError("grounded dataset source/transformation identity differs from receipt")
    if require_promotable:
        manifest.assert_promotable()
    manifest_splits = {item.name: item for item in manifest.splits}
    if set(manifest_splits) != {item.name for item in receipt.splits}:
        raise ValueError("grounded dataset manifest splits differ from receipt")
    verified = VerifiedGovernedGroundedDataset(manifest, receipt)
    for item in receipt.splits:
        manifest_split = manifest_splits[item.name]
        if manifest_split.content_sha256 != item.output_sha256 or manifest_split.record_count != item.record_count or manifest_split.record_id_sha256 != item.record_id_sha256 or manifest_split.query_id_sha256 != item.record_id_sha256 or manifest_split.document_id_sha256 != item.evidence_id_sha256:
            raise ValueError(f"grounded split {item.name} differs between manifest and receipt")
        dataset = verified.split(item.name)
        record_ids: list[str] = []
        evidence_ids: list[str] = []
        for index in range(len(dataset)):
            example = dataset[index]
            record_ids.append(example.example_id)
            evidence_ids.extend(evidence.evidence_id for evidence in example.evidence)
        if _id_digest(record_ids) != item.record_id_sha256 or _id_digest(evidence_ids) != item.evidence_id_sha256:
            raise ValueError(f"grounded split {item.name} identity digests differ from receipt")
    return verified


__all__ = ["VerifiedGovernedGroundedDataset", "verify_governed_grounded_import"]
