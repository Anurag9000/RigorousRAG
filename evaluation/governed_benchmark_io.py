"""Strict consumption and re-verification for governed benchmark imports.

The importer writes canonical benchmark JSONL and a self-verifying receipt.  This module is
the read side: it re-hashes the receipt, manifest and every canonical split, reconstructs the
real :class:`evaluation.dataset_governance.DatasetManifest`, validates split identities, and
exposes ``BenchmarkExample`` iterators that plug directly into ``run_benchmark_suite``.

Split verification is streaming: detailed examples are never accumulated merely to verify an
import. Identifier collections are retained only for the deterministic split-identity digests.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from evaluation.dataset_governance import DatasetCard, DatasetManifest, DatasetModality, DatasetTask, LicenseStatus, SplitManifest
from evaluation.governed_benchmark_import import GovernedBenchmarkImportReceipt, ImportedSplitReceipt
from tools.benchmark_adapters import BenchmarkExample

_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_LINE_BYTES = 64 * 1024 * 1024
_MAX_RECORDS = 100_000_000
_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_HEX = frozenset("0123456789abcdef")


def _safe_file(value: str | os.PathLike[str], label: str) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    for component in (path, *path.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or bool(int(getattr(metadata, "st_file_attributes", 0)) & _WINDOWS_REPARSE_POINT):
            raise ValueError(f"{label} may not traverse a symlink or reparse point")
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path


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
    size = path.stat().st_size
    if size <= 0 or size > _MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds JSON byte safety bound")
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _identifier(value: Any, label: str, maximum: int = 10_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _id_digest(values: list[str]) -> str:
    normalized = sorted({_identifier(value, "identifier") for value in values})
    return hashlib.sha256(("\n".join(normalized) + ("\n" if normalized else "")).encode("utf-8")).hexdigest()


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > 1_000_000:
        raise ValueError(f"{label} must be a bounded JSON array")
    return tuple(_identifier(item, label, 8_000_000) for item in value)


def _benchmark_example(value: Any, *, line_number: int) -> BenchmarkExample:
    if not isinstance(value, Mapping):
        raise ValueError(f"canonical benchmark line {line_number} must be an object")
    expected = {"schema", "example_id", "query", "answers", "relevant_ids", "contexts", "metadata"}
    if set(value) != expected or value.get("schema") != "rigorousrag-benchmark-example/v1":
        raise ValueError(f"canonical benchmark line {line_number} has an unsupported schema")
    metadata = value.get("metadata")
    if not isinstance(metadata, Mapping) or len(metadata) > 10_000:
        raise ValueError(f"canonical benchmark line {line_number} metadata must be a bounded object")
    return BenchmarkExample(
        example_id=_identifier(value.get("example_id"), "example_id"),
        query=_identifier(value.get("query"), "query", 8_000_000),
        answers=_string_tuple(value.get("answers"), "answers"),
        relevant_ids=_string_tuple(value.get("relevant_ids"), "relevant_ids"),
        contexts=_string_tuple(value.get("contexts"), "contexts"),
        metadata=dict(metadata),
    )


def iter_canonical_benchmark_jsonl(path: str | Path, *, expected_sha256: str | None = None) -> Iterator[BenchmarkExample]:
    """Yield canonical imported examples after optional whole-file digest verification."""
    source = _safe_file(path, "canonical benchmark split")
    if expected_sha256 is not None and _stream_sha(source) != _sha(expected_sha256, "expected_sha256"):
        raise ValueError("canonical benchmark split digest differs from expected immutable bytes")
    seen: set[str] = set()
    with source.open("rb") as handle:
        count = 0
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            if len(raw) > _MAX_LINE_BYTES:
                raise ValueError(f"canonical benchmark line {line_number} exceeds byte safety bound")
            if count >= _MAX_RECORDS:
                raise ValueError("canonical benchmark split exceeds record safety bound")
            try:
                value = json.loads(raw.decode("utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
            except Exception as exc:
                raise ValueError(f"canonical benchmark line {line_number} is not strict JSON") from exc
            example = _benchmark_example(value, line_number=line_number)
            if example.example_id in seen:
                raise ValueError(f"canonical benchmark contains duplicate example id {example.example_id!r}")
            seen.add(example.example_id)
            count += 1
            yield example


def _card(value: Any) -> DatasetCard:
    if not isinstance(value, Mapping):
        raise ValueError("dataset manifest card must be an object")
    required = {"summary", "intended_uses", "forbidden_uses", "populations_or_domains", "languages", "pii_notes", "safety_notes", "source_citation", "known_limitations"}
    if set(value) != required:
        raise ValueError("dataset manifest card fields differ from DatasetCard")
    return DatasetCard(
        summary=value["summary"], intended_uses=tuple(value["intended_uses"]), forbidden_uses=tuple(value["forbidden_uses"]),
        populations_or_domains=tuple(value["populations_or_domains"]), languages=tuple(value["languages"]), pii_notes=value["pii_notes"],
        safety_notes=value["safety_notes"], source_citation=value["source_citation"], known_limitations=tuple(value["known_limitations"]),
    )


def _manifest(value: Any) -> DatasetManifest:
    if not isinstance(value, Mapping):
        raise ValueError("dataset manifest payload must be an object")
    expected = {"dataset_id", "exact_version", "source_locator", "artifact_sha256", "license_identifier", "license_status", "license_evidence", "loader_name", "loader_version", "transformation_sha256", "splits", "tasks", "modalities", "card", "metadata"}
    if set(value) != expected:
        raise ValueError("dataset manifest fields differ from DatasetManifest")
    splits_raw = value["splits"]
    if not isinstance(splits_raw, list):
        raise ValueError("dataset manifest splits must be an array")
    split_fields = {"name", "content_sha256", "record_count", "record_id_sha256", "source_group_sha256", "query_id_sha256", "document_id_sha256"}
    splits = []
    for item in splits_raw:
        if not isinstance(item, Mapping) or set(item) != split_fields:
            raise ValueError("dataset manifest split fields differ from SplitManifest")
        splits.append(SplitManifest(**dict(item)))
    metadata = value["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("dataset manifest metadata must be an object")
    return DatasetManifest(
        dataset_id=value["dataset_id"], exact_version=value["exact_version"], source_locator=value["source_locator"], artifact_sha256=value["artifact_sha256"],
        license_identifier=value["license_identifier"], license_status=LicenseStatus(value["license_status"]), license_evidence=value["license_evidence"],
        loader_name=value["loader_name"], loader_version=value["loader_version"], transformation_sha256=value["transformation_sha256"], splits=tuple(splits),
        tasks=tuple(DatasetTask(item) for item in value["tasks"]), modalities=tuple(DatasetModality(item) for item in value["modalities"]), card=_card(value["card"]),
        metadata={str(key): str(item) for key, item in metadata.items()},
    )


@dataclass(frozen=True)
class VerifiedGovernedBenchmark:
    manifest: DatasetManifest
    receipt: GovernedBenchmarkImportReceipt

    def split(self, name: str) -> Iterator[BenchmarkExample]:
        selected = _identifier(name, "split name", 200)
        matches = [item for item in self.receipt.split_receipts if item.name == selected]
        if len(matches) != 1:
            raise ValueError(f"unknown governed benchmark split: {selected}")
        item = matches[0]
        return iter_canonical_benchmark_jsonl(item.output_path, expected_sha256=item.output_sha256)


def verify_governed_benchmark_import(receipt_path: str | Path, *, require_promotable: bool = False) -> VerifiedGovernedBenchmark:
    """Re-verify a persisted import receipt, manifest, and all canonical split bytes."""
    receipt_file = _safe_file(receipt_path, "governed benchmark import receipt")
    raw = _strict_json(receipt_file, "governed benchmark import receipt")
    expected_receipt_fields = {"schema", "dataset_manifest_sha256", "dataset_artifact_sha256", "transformation_sha256", "manifest_path", "split_receipts", "receipt_sha256"}
    if set(raw) != expected_receipt_fields or raw.get("schema") != "rigorousrag-governed-benchmark-import-receipt/v1":
        raise ValueError("unsupported governed benchmark import receipt schema")
    split_raw = raw.get("split_receipts")
    if not isinstance(split_raw, list) or not split_raw:
        raise ValueError("governed benchmark import receipt requires split receipts")
    split_fields = {"name", "source_sha256", "output_path", "output_sha256", "record_count", "record_id_sha256", "query_id_sha256", "document_id_sha256", "source_group_sha256", "transformation_component_sha256"}
    receipts = []
    for item in split_raw:
        if not isinstance(item, Mapping) or set(item) != split_fields:
            raise ValueError("import split receipt fields differ from ImportedSplitReceipt")
        receipts.append(ImportedSplitReceipt(**dict(item)))
    receipt = GovernedBenchmarkImportReceipt(
        dataset_manifest_sha256=raw["dataset_manifest_sha256"], dataset_artifact_sha256=raw["dataset_artifact_sha256"], transformation_sha256=raw["transformation_sha256"],
        manifest_path=raw["manifest_path"], split_receipts=tuple(receipts), receipt_sha256=raw["receipt_sha256"],
    )
    manifest_file = _safe_file(receipt.manifest_path, "governed dataset manifest")
    manifest_raw = _strict_json(manifest_file, "governed dataset manifest")
    if set(manifest_raw) != {"schema", "manifest", "manifest_sha256"} or manifest_raw.get("schema") != "rigorousrag-dataset-manifest/v1":
        raise ValueError("unsupported governed dataset manifest envelope")
    manifest = _manifest(manifest_raw["manifest"])
    if manifest.manifest_digest != _sha(manifest_raw["manifest_sha256"], "manifest_sha256") or manifest.manifest_digest != receipt.dataset_manifest_sha256:
        raise ValueError("dataset manifest digest differs from import receipt")
    if manifest.artifact_sha256 != receipt.dataset_artifact_sha256 or manifest.transformation_sha256 != receipt.transformation_sha256:
        raise ValueError("dataset manifest artifact/transformation identity differs from import receipt")
    if require_promotable:
        manifest.assert_promotable()
    split_manifest_by_name = {item.name: item for item in manifest.splits}
    if set(split_manifest_by_name) != {item.name for item in receipt.split_receipts}:
        raise ValueError("dataset manifest splits differ from import receipt splits")
    for item in receipt.split_receipts:
        split_manifest = split_manifest_by_name[item.name]
        if split_manifest.content_sha256 != item.output_sha256 or split_manifest.record_count != item.record_count or split_manifest.record_id_sha256 != item.record_id_sha256 or split_manifest.query_id_sha256 != item.query_id_sha256 or split_manifest.document_id_sha256 != item.document_id_sha256 or split_manifest.source_group_sha256 != item.source_group_sha256:
            raise ValueError(f"dataset manifest split {item.name} differs from import receipt")
        record_ids: list[str] = []
        document_ids: list[str] = []
        source_groups: list[str] = []
        count = 0
        for example in iter_canonical_benchmark_jsonl(item.output_path, expected_sha256=item.output_sha256):
            count += 1
            record_ids.append(example.example_id)
            document_ids.extend(example.relevant_ids)
            source_group = example.metadata.get("source_group_id") if isinstance(example.metadata, Mapping) else None
            if isinstance(source_group, str) and source_group.strip():
                source_groups.append(source_group.strip())
        if count != item.record_count:
            raise ValueError(f"canonical split {item.name} record count differs from receipt")
        if _id_digest(record_ids) != item.record_id_sha256 or _id_digest(record_ids) != item.query_id_sha256:
            raise ValueError(f"canonical split {item.name} record/query identity digest differs from receipt")
        actual_document = _id_digest(document_ids) if document_ids else None
        actual_source_group = _id_digest(source_groups) if source_groups else None
        if actual_document != item.document_id_sha256 or actual_source_group != item.source_group_sha256:
            raise ValueError(f"canonical split {item.name} document/source-group identity differs from receipt")
    return VerifiedGovernedBenchmark(manifest=manifest, receipt=receipt)


__all__ = ["VerifiedGovernedBenchmark", "iter_canonical_benchmark_jsonl", "verify_governed_benchmark_import"]
