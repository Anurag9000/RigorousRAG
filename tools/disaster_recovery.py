"""Checksum-verified local backup/restore and canary rollback policy primitives."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    name = str(value).strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("backup entry names must be simple file names.")
    return name


@dataclass(frozen=True)
class BackupEntry:
    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class BackupManifest:
    schema: str
    generation: str
    entries: tuple[BackupEntry, ...]
    encryption_key_id: str | None


@dataclass(frozen=True)
class RestoreReport:
    restored: tuple[str, ...]
    manifest_sha256: str


@dataclass(frozen=True)
class CanaryPolicy:
    max_error_rate: float = 0.01
    max_p95_latency_ratio: float = 1.20
    min_quality_delta: float = -0.005


@dataclass(frozen=True)
class CanaryObservation:
    requests: int
    errors: int
    baseline_p95_latency_ms: float
    canary_p95_latency_ms: float
    quality_delta: float


@dataclass(frozen=True)
class CanaryDecision:
    promote: bool
    rollback: bool
    reason_codes: tuple[str, ...]


def manifest_sha256(manifest: BackupManifest) -> str:
    payload = json.dumps(asdict(manifest), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_backup(
    *,
    sources: Iterable[str | Path],
    destination: str | Path,
    generation: str,
    encryption_key_id: str | None = None,
) -> BackupManifest:
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    entries: list[BackupEntry] = []
    seen: set[str] = set()
    for source_value in sources:
        source = Path(source_value)
        if not source.is_file():
            raise ValueError(f"backup source is not a file: {source}")
        name = _safe_name(source.name)
        if name in seen:
            raise ValueError("backup source names must be unique.")
        seen.add(name)
        copied = target / name
        shutil.copyfile(source, copied)
        entries.append(BackupEntry(name, copied.stat().st_size, _sha256_file(copied)))
    entries.sort(key=lambda item: item.name)
    manifest = BackupManifest(
        schema="rigorousrag-backup/v1",
        generation=str(generation),
        entries=tuple(entries),
        encryption_key_id=encryption_key_id,
    )
    (target / "manifest.json").write_text(
        json.dumps(asdict(manifest), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_backup(*, source: str | Path, manifest: BackupManifest) -> bool:
    root = Path(source)
    for entry in manifest.entries:
        path = root / _safe_name(entry.name)
        if not path.is_file() or path.stat().st_size != entry.size_bytes:
            return False
        if _sha256_file(path) != entry.sha256:
            return False
    return True


def restore_backup(
    *, source: str | Path, destination: str | Path, manifest: BackupManifest
) -> RestoreReport:
    backup_root = Path(source)
    if not verify_backup(source=backup_root, manifest=manifest):
        raise ValueError("backup verification failed.")
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="rigorousrag-restore-", dir=str(target.parent)))
    restored: list[str] = []
    try:
        for entry in manifest.entries:
            name = _safe_name(entry.name)
            shutil.copyfile(backup_root / name, staging / name)
        for entry in manifest.entries:
            name = _safe_name(entry.name)
            os.replace(staging / name, target / name)
            restored.append(name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return RestoreReport(tuple(sorted(restored)), manifest_sha256(manifest))


def evaluate_canary(
    observation: CanaryObservation, policy: CanaryPolicy | None = None
) -> CanaryDecision:
    selected = policy or CanaryPolicy()
    if observation.requests < 1 or observation.errors < 0 or observation.errors > observation.requests:
        raise ValueError("canary request/error counts are invalid.")
    if observation.baseline_p95_latency_ms < 0 or observation.canary_p95_latency_ms < 0:
        raise ValueError("latencies must be non-negative.")
    reasons: list[str] = []
    error_rate = observation.errors / observation.requests
    latency_ratio = (
        1.0
        if observation.baseline_p95_latency_ms == observation.canary_p95_latency_ms == 0
        else float("inf")
        if observation.baseline_p95_latency_ms == 0
        else observation.canary_p95_latency_ms / observation.baseline_p95_latency_ms
    )
    if error_rate > selected.max_error_rate:
        reasons.append("canary_error_rate_exceeded")
    if latency_ratio > selected.max_p95_latency_ratio:
        reasons.append("canary_latency_regression")
    if observation.quality_delta < selected.min_quality_delta:
        reasons.append("canary_quality_regression")
    return CanaryDecision(promote=not reasons, rollback=bool(reasons), reason_codes=tuple(reasons))


__all__ = [
    "BackupEntry",
    "BackupManifest",
    "CanaryDecision",
    "CanaryObservation",
    "CanaryPolicy",
    "RestoreReport",
    "create_backup",
    "evaluate_canary",
    "manifest_sha256",
    "restore_backup",
    "verify_backup",
]
