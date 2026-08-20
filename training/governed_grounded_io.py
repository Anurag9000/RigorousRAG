"""Strict, corpus-scale read-side verification for governed grounded-training imports."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from evaluation.dataset_governance import DatasetCard, DatasetManifest, DatasetModality, DatasetTask, LicenseStatus, SplitManifest
from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_authoritative_data import ManifestBoundAuthoritativeJsonlDataset
from training.governed_grounded_import import GovernedGroundedImportReceipt, GroundedSplitImportReceipt
from training.logical_filename import logical_filename
from training.sqlite_identity_ledger import SqliteIdentityLedger

_MAX_JSON_BYTES = 64 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected): raise ValueError(f"{label} must be SHA-256")
    return selected


def _stream_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block: break
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path, label: str) -> Mapping[str, Any]:
    source = safe_advanced_path(path, label=label, must_exist=True, require_file=True)
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_JSON_BYTES: raise ValueError(f"{label} exceeds byte safety bound")
    try: payload = json.loads(source.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
    except Exception as exc: raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(payload, Mapping): raise ValueError(f"{label} must contain an object")
    return payload


def _manifest(value: Any) -> DatasetManifest:
    if not isinstance(value, Mapping): raise ValueError("grounded dataset manifest must be an object")
    expected = {"dataset_id", "exact_version", "source_locator", "artifact_sha256", "license_identifier", "license_status", "license_evidence", "loader_name", "loader_version", "transformation_sha256", "splits", "tasks", "modalities", "card", "metadata"}
    if set(value) != expected: raise ValueError("grounded dataset manifest fields differ from DatasetManifest")
    card_raw = value["card"]
    if not isinstance(card_raw, Mapping): raise ValueError("grounded dataset card must be an object")
    card = DatasetCard(summary=card_raw["summary"], intended_uses=tuple(card_raw["intended_uses"]), forbidden_uses=tuple(card_raw["forbidden_uses"]), populations_or_domains=tuple(card_raw["populations_or_domains"]), languages=tuple(card_raw["languages"]), pii_notes=card_raw["pii_notes"], safety_notes=card_raw["safety_notes"], source_citation=card_raw["source_citation"], known_limitations=tuple(card_raw["known_limitations"]))
    splits_raw = value["splits"]
    if not isinstance(splits_raw, list): raise ValueError("grounded dataset splits must be an array")
    splits = tuple(SplitManifest(**dict(item)) for item in splits_raw if isinstance(item, Mapping))
    if len(splits) != len(splits_raw): raise ValueError("grounded dataset split entries must be objects")
    metadata = value["metadata"]
    if not isinstance(metadata, Mapping): raise ValueError("grounded dataset metadata must be an object")
    return DatasetManifest(dataset_id=value["dataset_id"], exact_version=value["exact_version"], source_locator=value["source_locator"], artifact_sha256=value["artifact_sha256"], license_identifier=value["license_identifier"], license_status=LicenseStatus(value["license_status"]), license_evidence=value["license_evidence"], loader_name=value["loader_name"], loader_version=value["loader_version"], transformation_sha256=value["transformation_sha256"], splits=splits, tasks=tuple(DatasetTask(item) for item in value["tasks"]), modalities=tuple(DatasetModality(item) for item in value["modalities"]), card=card, metadata={str(key): str(item) for key, item in metadata.items()})


@dataclass(frozen=True)
class VerifiedGovernedGroundedDataset:
    manifest: DatasetManifest
    receipt: GovernedGroundedImportReceipt

    def split(self, name: str) -> ManifestBoundAuthoritativeJsonlDataset:
        matches = [item for item in self.receipt.splits if item.name == name]
        if len(matches) != 1: raise ValueError(f"unknown grounded split: {name}")
        item = matches[0]
        return ManifestBoundAuthoritativeJsonlDataset(item.output_path, expected_sha256=item.output_sha256, dataset_manifest_sha256=self.manifest.manifest_digest, split_name=item.name, record_kind="grounded_generation", expected_record_count=item.record_count)


def _verify_v2_paths(receipt_file: Path, manifest: DatasetManifest, receipt: GovernedGroundedImportReceipt) -> None:
    if manifest.loader_name != "training.authoritative_governed_grounded_import" or manifest.loader_version != "2": return
    root = receipt_file.parent
    manifest_path = safe_advanced_path(receipt.manifest_path, label="grounded dataset manifest", must_exist=True, require_file=True)
    if manifest_path != root / "dataset_manifest.json": raise ValueError("grounded v2 manifest must be canonical publication child")
    expected = {"dataset_manifest.json", "import_receipt.json"}
    for item in receipt.splits:
        split_path = safe_advanced_path(item.output_path, label=f"grounded split {item.name}", must_exist=True, require_file=True)
        filename = logical_filename(item.name, ".grounded.jsonl")
        if split_path != root / filename: raise ValueError(f"grounded split {item.name!r} does not use canonical v2 path")
        expected.add(filename)
    actual = {item.name for item in root.iterdir()}
    if actual != expected: raise ValueError(f"grounded v2 publication directory is not closed: unexpected={sorted(actual-expected)} missing={sorted(expected-actual)}")
    if any(item.is_symlink() or not item.is_file() for item in root.iterdir()): raise ValueError("grounded v2 publication contains non-regular child")


def verify_governed_grounded_import(receipt_path: str | Path, *, require_promotable: bool = False) -> VerifiedGovernedGroundedDataset:
    receipt_file = safe_advanced_path(receipt_path, label="grounded import receipt", must_exist=True, require_file=True)
    raw = _read_json(receipt_file, "grounded import receipt")
    expected = {"schema", "dataset_manifest_sha256", "source_set_sha256", "transformation_sha256", "manifest_path", "splits", "receipt_sha256"}
    if set(raw) != expected or raw.get("schema") != "rigorousrag-governed-grounded-import-receipt/v1": raise ValueError("unsupported grounded import receipt schema")
    split_raw = raw["splits"]
    if not isinstance(split_raw, list) or not split_raw: raise ValueError("grounded import receipt requires split receipts")
    split_fields = {"name", "source_sha256", "output_path", "output_sha256", "record_count", "record_id_sha256", "evidence_id_sha256", "transformation_sha256"}
    receipts = []
    for item in split_raw:
        if not isinstance(item, Mapping) or set(item) != split_fields: raise ValueError("grounded split receipt fields are invalid")
        receipts.append(GroundedSplitImportReceipt(**dict(item)))
    receipt = GovernedGroundedImportReceipt(raw["dataset_manifest_sha256"], raw["source_set_sha256"], raw["transformation_sha256"], raw["manifest_path"], tuple(receipts), raw["receipt_sha256"])
    manifest_envelope = _read_json(receipt.manifest_path, "grounded dataset manifest")
    if set(manifest_envelope) != {"schema", "manifest", "manifest_sha256"} or manifest_envelope.get("schema") != "rigorousrag-dataset-manifest/v1": raise ValueError("unsupported grounded dataset manifest envelope")
    manifest = _manifest(manifest_envelope["manifest"])
    if manifest.manifest_digest != _sha(manifest_envelope["manifest_sha256"], "manifest_sha256") or manifest.manifest_digest != receipt.dataset_manifest_sha256: raise ValueError("grounded dataset manifest digest differs from receipt")
    if manifest.artifact_sha256 != receipt.source_set_sha256 or manifest.transformation_sha256 != receipt.transformation_sha256: raise ValueError("grounded dataset source/transformation identity differs from receipt")
    if require_promotable: manifest.assert_promotable()
    manifest_splits = {item.name: item for item in manifest.splits}
    if len(manifest_splits) != len(manifest.splits) or set(manifest_splits) != {item.name for item in receipt.splits}: raise ValueError("grounded dataset manifest splits differ from receipt")
    _verify_v2_paths(receipt_file, manifest, receipt)
    verified = VerifiedGovernedGroundedDataset(manifest, receipt)

    descriptor, ledger_name = tempfile.mkstemp(prefix=".grounded-verify-", suffix=".sqlite", dir=receipt_file.parent.parent)
    os.close(descriptor); ledger_path = Path(ledger_name); ledger_path.unlink(missing_ok=True); ledger = SqliteIdentityLedger(ledger_path)
    try:
        total = 0
        for item in receipt.splits:
            manifest_split = manifest_splits[item.name]
            if manifest_split.content_sha256 != item.output_sha256 or manifest_split.record_count != item.record_count or manifest_split.record_id_sha256 != item.record_id_sha256 or manifest_split.query_id_sha256 != item.record_id_sha256 or manifest_split.document_id_sha256 != item.evidence_id_sha256: raise ValueError(f"grounded split {item.name} differs between manifest and receipt")
            split_path = safe_advanced_path(item.output_path, label=f"grounded split {item.name}", must_exist=True, require_file=True)
            if _stream_sha(split_path) != item.output_sha256: raise ValueError(f"grounded split {item.name} bytes differ from receipt")
            dataset = verified.split(item.name)
            for index in range(len(dataset)):
                example = dataset[index]
                ledger.add_unique("grounded-example", item.name, example.example_id)
                for evidence in example.evidence: ledger.add_set("grounded-evidence", item.name, evidence.evidence_id)
                total += 1
                if total % 10_000 == 0: ledger.commit()
            ledger.commit()
            if ledger.count_unique("grounded-example", scope=item.name) != item.record_count: raise ValueError(f"grounded split {item.name} record count differs")
            if ledger.digest_unique("grounded-example", scope=item.name) != item.record_id_sha256 or ledger.digest_set("grounded-evidence", scope=item.name) != item.evidence_id_sha256: raise ValueError(f"grounded split {item.name} identity digests differ from receipt")
        if total <= 0 or ledger.count_unique("grounded-example") != total: raise ValueError("grounded import global example identity authority differs")
        return verified
    finally:
        ledger.close()
        for suffix in ("", "-wal", "-shm"): Path(str(ledger_path) + suffix).unlink(missing_ok=True)


__all__ = ["VerifiedGovernedGroundedDataset", "verify_governed_grounded_import"]
