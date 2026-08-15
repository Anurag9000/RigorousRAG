"""Fail-closed bridge from acquisition manifests to benchmark examples.

The acquisition manifest proves which local bytes are allowed to participate in an
experiment.  Benchmark adapters normalize those bytes into examples.  This module
binds the two contracts so an evaluation cannot silently parse an unverified file,
use the wrong dataset version, or lose record-level provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from evaluation.dataset_manifest import (
    DatasetAcquisitionManifest,
    DatasetFileManifest,
    DatasetVerificationReport,
    verify_dataset_manifest,
)
from tools.benchmark_adapters import BenchmarkExample, adapt_record

_MAX_JSON_BYTES = 512 * 1024 * 1024
_MAX_EXAMPLES = 2_000_000
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_link_or_reparse(info: os.stat_result) -> bool:
    attributes = int(getattr(info, "st_file_attributes", 0) or 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


@dataclass(frozen=True)
class VerifiedBenchmarkDataset:
    dataset_name: str
    version: str
    revision: str
    manifest_digest: str
    execution_digest: str
    verification: DatasetVerificationReport
    examples: tuple[BenchmarkExample, ...]

    def __post_init__(self) -> None:
        for field_name in ("dataset_name", "version", "revision"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty.")
        for field_name in ("manifest_digest", "execution_digest"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(ch not in "0123456789abcdef" for ch in value.lower())
            ):
                raise ValueError(f"{field_name} must be a SHA-256 digest.")
        if not isinstance(self.verification, DatasetVerificationReport) or not self.verification.verified:
            raise ValueError("verification must be a successful DatasetVerificationReport.")
        if self.verification.manifest_digest != self.manifest_digest:
            raise ValueError("verification does not belong to this manifest.")
        if not isinstance(self.examples, tuple):
            raise ValueError("examples must be an immutable tuple.")


def _safe_candidate(root: Path, entry: DatasetFileManifest) -> tuple[Path, tuple[int, int]]:
    candidate = root.joinpath(*PurePosixPath(entry.path).parts)
    current = root
    for part in PurePosixPath(entry.path).parts:
        current = current / part
        try:
            info = os.lstat(current)
        except OSError as exc:
            raise RuntimeError(f"dataset member {entry.path!r} is missing or unreadable.") from exc
        if _is_link_or_reparse(info):
            raise RuntimeError(f"dataset member {entry.path!r} may not traverse links or reparse points.")
    before = os.lstat(candidate)
    if not stat.S_ISREG(before.st_mode) or before.st_size != entry.bytes:
        raise RuntimeError(f"dataset member {entry.path!r} changed after manifest verification.")
    return candidate, (int(before.st_dev), int(before.st_ino))


def _open_verified_descriptor(root: Path, entry: DatasetFileManifest) -> tuple[int, tuple[int, int]]:
    candidate, identity = _safe_candidate(root, entry)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise RuntimeError(f"dataset member {entry.path!r} could not be opened safely.") from exc
    opened = os.fstat(descriptor)
    opened_identity = (int(opened.st_dev), int(opened.st_ino))
    if not stat.S_ISREG(opened.st_mode) or opened_identity != identity or opened.st_size != entry.bytes:
        os.close(descriptor)
        raise RuntimeError(f"dataset member {entry.path!r} changed before parsing.")
    return descriptor, identity


def _validate_finished_descriptor(
    descriptor: int,
    identity: tuple[int, int],
    entry: DatasetFileManifest,
    digest: str,
    total: int,
) -> None:
    opened = os.fstat(descriptor)
    if (
        (int(opened.st_dev), int(opened.st_ino)) != identity
        or opened.st_size != entry.bytes
        or total != entry.bytes
        or digest != entry.sha256
    ):
        raise RuntimeError(f"dataset member {entry.path!r} changed while it was parsed.")


def _parse_jsonl(root: Path, entry: DatasetFileManifest) -> list[tuple[int, Mapping[str, Any], str]]:
    descriptor, identity = _open_verified_descriptor(root, entry)
    rows: list[tuple[int, Mapping[str, Any], str]] = []
    hasher = hashlib.sha256()
    total = 0
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                total += len(raw_line)
                if total > entry.bytes:
                    raise RuntimeError(f"dataset member {entry.path!r} grew while it was parsed.")
                hasher.update(raw_line)
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"{entry.path}:{line_number} is not valid UTF-8 JSON.") from exc
                if not isinstance(row, Mapping):
                    raise ValueError(f"{entry.path}:{line_number} must contain a JSON object.")
                rows.append((line_number, row, _canonical_digest(row)))
                if len(rows) > _MAX_EXAMPLES:
                    raise ValueError("benchmark dataset exceeds the example limit.")
        _validate_finished_descriptor(descriptor, identity, entry, hasher.hexdigest(), total)
    finally:
        os.close(descriptor)
    if entry.records is not None and len(rows) != entry.records:
        raise RuntimeError(f"dataset member {entry.path!r} record count does not match the manifest.")
    return rows


def _json_records(payload: Any, path: str) -> Iterable[tuple[int, Mapping[str, Any]]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, Mapping):
        selected = None
        for key in ("data", "items", "examples", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                selected = value
                break
        if selected is None:
            records = [payload]
        else:
            records = selected
    else:
        raise ValueError(f"{path} must contain a JSON object or array of objects.")
    if len(records) > _MAX_EXAMPLES:
        raise ValueError("benchmark dataset exceeds the example limit.")
    for index, row in enumerate(records, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}:record:{index} must contain a JSON object.")
        yield index, row


def _parse_json(root: Path, entry: DatasetFileManifest) -> list[tuple[int, Mapping[str, Any], str]]:
    if entry.bytes > _MAX_JSON_BYTES:
        raise ValueError("JSON benchmark member is too large for bounded in-memory parsing; use JSONL.")
    descriptor, identity = _open_verified_descriptor(root, entry)
    hasher = hashlib.sha256()
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max(entry.bytes - total, 1)))
            if not chunk:
                break
            total += len(chunk)
            if total > entry.bytes or total > _MAX_JSON_BYTES:
                raise RuntimeError(f"dataset member {entry.path!r} grew while it was parsed.")
            hasher.update(chunk)
            chunks.append(chunk)
        _validate_finished_descriptor(descriptor, identity, entry, hasher.hexdigest(), total)
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{entry.path} is not valid UTF-8 JSON.") from exc
    rows = [(index, row, _canonical_digest(row)) for index, row in _json_records(payload, entry.path)]
    if entry.records is not None and len(rows) != entry.records:
        raise RuntimeError(f"dataset member {entry.path!r} record count does not match the manifest.")
    return rows


def _load_rows(root: Path, entry: DatasetFileManifest) -> list[tuple[int, Mapping[str, Any], str]]:
    suffix = PurePosixPath(entry.path).suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return _parse_jsonl(root, entry)
    if suffix == ".json":
        return _parse_json(root, entry)
    raise ValueError(f"unsupported benchmark manifest member format: {entry.path!r}")


def load_verified_benchmark_dataset(
    root: str | os.PathLike[str],
    manifest: DatasetAcquisitionManifest,
    *,
    expected_dataset: str | None = None,
    expected_version: str | None = None,
) -> VerifiedBenchmarkDataset:
    """Verify local bytes and normalize only manifest-declared records.

    The manifest is verified once before parsing and each member is then reopened with
    descriptor identity, byte count, and SHA-256 checks.  This closes the normal
    verify-then-open race and makes record provenance reproducible.
    """

    if not isinstance(manifest, DatasetAcquisitionManifest):
        raise ValueError("manifest must be DatasetAcquisitionManifest.")
    if expected_dataset is not None and manifest.dataset_name != expected_dataset.strip().lower():
        raise RuntimeError("dataset name does not match the expected benchmark contract.")
    if expected_version is not None and manifest.version != expected_version.strip():
        raise RuntimeError("dataset version does not match the expected benchmark contract.")

    verification = verify_dataset_manifest(root, manifest)
    safe_root = Path(os.path.abspath(os.fspath(root)))
    examples: list[BenchmarkExample] = []
    fingerprints: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()

    for entry in manifest.files:
        for position, row, row_digest in _load_rows(safe_root, entry):
            try:
                example = adapt_record(manifest.dataset_name, row)
            except Exception as exc:
                raise ValueError(f"failed to adapt {entry.path}:record:{position}: {exc}") from exc
            if not example.example_id:
                raise ValueError(f"{entry.path}:record:{position} produced an empty example id.")
            if example.example_id in seen_ids:
                raise ValueError(f"duplicate benchmark example id: {example.example_id!r}")
            seen_ids.add(example.example_id)
            metadata = dict(example.metadata)
            metadata.update(
                {
                    "dataset": manifest.dataset_name,
                    "dataset_version": manifest.version,
                    "dataset_revision": manifest.revision,
                    "dataset_manifest_sha256": manifest.manifest_digest,
                    "source_path": entry.path,
                    "source_record": position,
                    "source_record_sha256": row_digest,
                }
            )
            enriched = BenchmarkExample(
                example_id=example.example_id,
                query=example.query,
                answers=example.answers,
                relevant_ids=example.relevant_ids,
                contexts=example.contexts,
                metadata=metadata,
            )
            examples.append(enriched)
            fingerprints.append(
                {
                    "example_id": enriched.example_id,
                    "source_path": entry.path,
                    "source_record": position,
                    "source_record_sha256": row_digest,
                }
            )
            if len(examples) > _MAX_EXAMPLES:
                raise ValueError("benchmark dataset exceeds the example limit.")

    if not examples:
        raise ValueError("verified benchmark dataset contains no examples.")
    execution_digest = _canonical_digest(
        {
            "contract": "rigorousrag-verified-benchmark-v1",
            "manifest_sha256": manifest.manifest_digest,
            "records": fingerprints,
        }
    )
    return VerifiedBenchmarkDataset(
        dataset_name=manifest.dataset_name,
        version=manifest.version,
        revision=manifest.revision,
        manifest_digest=manifest.manifest_digest,
        execution_digest=execution_digest,
        verification=verification,
        examples=tuple(examples),
    )


__all__ = ["VerifiedBenchmarkDataset", "load_verified_benchmark_dataset"]
