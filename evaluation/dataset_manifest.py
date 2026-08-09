"""Immutable acquisition manifests for reproducible evaluation datasets.

The static dataset registry describes *what* to evaluate. This module records exactly
which external corpus release was acquired and verifies the local bytes before an
experiment can claim provenance. It performs no network access and never treats the
presence of a catalog entry as evidence that a dataset has been downloaded.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from evaluation.dataset_registry import get_dataset_spec

_MAX_FILES = 10_000
_MAX_FILE_BYTES = 20_000_000_000
_MAX_TOTAL_BYTES = 100_000_000_000
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400


def _text(value: Any, label: str, maximum: int = 2_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a bounded string.")
    selected = value.strip()
    if (
        not selected
        or len(selected) > maximum
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected)
    ):
        raise ValueError(f"{label} is invalid.")
    return selected


def _digest(value: Any, label: str) -> str:
    selected = _text(value, label, 64).lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return selected


def _relative_path(value: Any) -> str:
    selected = _text(value, "dataset file path", 1_000).replace("\\", "/")
    path = PurePosixPath(selected)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("dataset file path must be a safe relative path.")
    return path.as_posix()


def _is_link_or_reparse(info: os.stat_result) -> bool:
    attributes = int(getattr(info, "st_file_attributes", 0) or 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DatasetFileManifest:
    path: str
    sha256: str
    bytes: int
    records: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path))
        object.__setattr__(self, "sha256", _digest(self.sha256, "file sha256"))
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or not 0 <= self.bytes <= _MAX_FILE_BYTES:
            raise ValueError("dataset file bytes are invalid or exceed the limit.")
        if self.records is not None and (
            isinstance(self.records, bool)
            or not isinstance(self.records, int)
            or not 0 <= self.records <= 2**63 - 1
        ):
            raise ValueError("dataset file records are invalid.")


@dataclass(frozen=True)
class DatasetAcquisitionManifest:
    dataset_name: str
    version: str
    revision: str
    source_uri: str
    license_id: str
    license_sha256: str
    files: tuple[DatasetFileManifest, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        dataset = _text(self.dataset_name, "dataset_name", 200).lower()
        get_dataset_spec(dataset)
        object.__setattr__(self, "dataset_name", dataset)
        for name, maximum in (
            ("version", 200),
            ("revision", 500),
            ("source_uri", 2_000),
            ("license_id", 200),
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name, maximum))
        object.__setattr__(
            self,
            "license_sha256",
            _digest(self.license_sha256, "license_sha256"),
        )
        if isinstance(self.files, (str, bytes, bytearray)):
            raise ValueError("files must be a bounded sequence.")
        files = tuple(self.files)
        if not files or len(files) > _MAX_FILES or any(
            not isinstance(item, DatasetFileManifest) for item in files
        ):
            raise ValueError("files must contain between 1 and 10000 file manifests.")
        if len({item.path for item in files}) != len(files):
            raise ValueError("dataset manifest file paths must be unique.")
        if sum(item.bytes for item in files) > _MAX_TOTAL_BYTES:
            raise ValueError("dataset manifest exceeds the total byte limit.")
        object.__setattr__(self, "files", tuple(sorted(files, key=lambda item: item.path)))
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("unsupported dataset acquisition manifest schema_version.")

    @property
    def manifest_digest(self) -> str:
        payload = asdict(self)
        payload["contract"] = "rigorousrag-dataset-acquisition-manifest-v1"
        return _canonical_digest(payload)


@dataclass(frozen=True)
class DatasetVerificationReport:
    manifest_digest: str
    verified_files: int
    verified_bytes: int
    verified: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "manifest_digest",
            _digest(self.manifest_digest, "manifest_digest"),
        )
        if (
            isinstance(self.verified_files, bool)
            or not isinstance(self.verified_files, int)
            or not 0 <= self.verified_files <= _MAX_FILES
        ):
            raise ValueError("verified_files is invalid.")
        if (
            isinstance(self.verified_bytes, bool)
            or not isinstance(self.verified_bytes, int)
            or not 0 <= self.verified_bytes <= _MAX_TOTAL_BYTES
        ):
            raise ValueError("verified_bytes is invalid.")
        if not isinstance(self.verified, bool):
            raise ValueError("verified must be boolean.")


def _safe_root(root: str | os.PathLike[str]) -> Path:
    try:
        rendered = os.fspath(root)
    except TypeError as exc:
        raise ValueError("dataset root must be a filesystem path.") from exc
    if not isinstance(rendered, str) or not rendered or len(rendered) > 4096:
        raise ValueError("dataset root is invalid.")
    path = Path(os.path.abspath(rendered))
    for component in (path, *path.parents):
        try:
            info = os.lstat(component)
        except FileNotFoundError:
            if component == path:
                raise ValueError("dataset root does not exist.")
            continue
        except OSError as exc:
            raise ValueError("dataset root could not be inspected safely.") from exc
        if _is_link_or_reparse(info):
            raise ValueError("dataset root may not contain links or reparse points.")
    try:
        root_info = os.lstat(path)
    except OSError as exc:
        raise ValueError("dataset root could not be inspected safely.") from exc
    if not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("dataset root must be a directory.")
    return path


def _verify_file(root: Path, entry: DatasetFileManifest) -> None:
    candidate = root.joinpath(*PurePosixPath(entry.path).parts)
    current = root
    for part in PurePosixPath(entry.path).parts:
        current = current / part
        try:
            info = os.lstat(current)
        except OSError as exc:
            raise RuntimeError("dataset manifest member is missing or unreadable.") from exc
        if _is_link_or_reparse(info):
            raise RuntimeError("dataset manifest member may not traverse links or reparse points.")
    try:
        before = os.lstat(candidate)
    except OSError as exc:
        raise RuntimeError("dataset manifest member is missing or unreadable.") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_size != entry.bytes:
        raise RuntimeError("dataset manifest member size/type does not match the manifest.")
    identity = (int(before.st_dev), int(before.st_ino))
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise RuntimeError("dataset manifest member could not be opened safely.") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (int(opened.st_dev), int(opened.st_ino)) != identity or opened.st_size != entry.bytes:
            raise RuntimeError("dataset manifest member identity changed before verification.")
        hasher = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > entry.bytes:
                raise RuntimeError("dataset manifest member grew during verification.")
            hasher.update(chunk)
        after_open = os.fstat(descriptor)
        if (int(after_open.st_dev), int(after_open.st_ino)) != identity or after_open.st_size != entry.bytes:
            raise RuntimeError("dataset manifest member changed during verification.")
    finally:
        os.close(descriptor)
    try:
        after = os.lstat(candidate)
    except OSError as exc:
        raise RuntimeError("dataset manifest member disappeared during verification.") from exc
    if _is_link_or_reparse(after) or (int(after.st_dev), int(after.st_ino)) != identity or after.st_size != entry.bytes:
        raise RuntimeError("dataset manifest member identity changed during verification.")
    if total != entry.bytes or hasher.hexdigest() != entry.sha256:
        raise RuntimeError("dataset manifest member digest does not match the manifest.")


def verify_dataset_manifest(
    root: str | os.PathLike[str],
    manifest: DatasetAcquisitionManifest,
) -> DatasetVerificationReport:
    if not isinstance(manifest, DatasetAcquisitionManifest):
        raise ValueError("manifest must be DatasetAcquisitionManifest.")
    safe_root = _safe_root(root)
    verified_bytes = 0
    for entry in manifest.files:
        _verify_file(safe_root, entry)
        verified_bytes += entry.bytes
    return DatasetVerificationReport(
        manifest_digest=manifest.manifest_digest,
        verified_files=len(manifest.files),
        verified_bytes=verified_bytes,
        verified=True,
    )


__all__ = [
    "DatasetAcquisitionManifest",
    "DatasetFileManifest",
    "DatasetVerificationReport",
    "verify_dataset_manifest",
]
