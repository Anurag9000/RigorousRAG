"""Restart-verifiable bundles for expensive canonical advanced-RAG supervision outputs.

Canonical grounded/dynamic builders may execute admitted teacher/generator providers and write
large strict caches. Their useful state must survive process loss without serializing model
objects. This module persists only immutable authority data: final dataset manifest, split
paths/SHAs/counts, canonical receipt payload, and full strict-cache identity/root/contract.
Read-side verification re-hashes every split and reconstructs/re-seals each cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_strict_cache import AuthoritativeSafetensorSupervisionCache
from training.advanced_rag_supervision import SupervisionCacheIdentity
from training.dynamic_canonical_training_data_pipeline import CanonicalDynamicTrainingDataResult
from training.grounded_canonical_training_data_pipeline import CanonicalGroundedTrainingDataResult

_HEX = frozenset("0123456789abcdef")
_MAX_BYTES = 64 * 1024 * 1024


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
    if path.stat().st_size <= 0 or path.stat().st_size > _MAX_BYTES:
        raise ValueError(f"{label} exceeds byte safety bound")
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain an object")
    return value


def _atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical(payload) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _manifest_sha(path: Path) -> str:
    envelope = _strict_json(path, "canonical dataset manifest")
    if set(envelope) != {"schema", "manifest", "manifest_sha256"} or envelope.get("schema") != "rigorousrag-dataset-manifest/v1":
        raise ValueError("unsupported canonical dataset manifest envelope")
    expected = _sha(envelope["manifest_sha256"], "manifest_sha256")
    manifest = envelope["manifest"]
    if not isinstance(manifest, Mapping):
        raise ValueError("canonical dataset manifest payload must be an object")
    # DatasetManifest.manifest_digest is canonical_digest(asdict(self)); enum values are
    # str-enum subclasses and therefore serialize as their string values in this envelope.
    actual = _digest(dict(manifest))
    if actual != expected:
        raise ValueError("canonical dataset manifest payload digest mismatch")
    return expected


@dataclass(frozen=True)
class CanonicalSplitDescriptor:
    name: str
    path: str
    sha256: str
    record_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("split name is required")
        source = safe_advanced_path(self.path, label=f"canonical split {self.name}", must_exist=True, require_file=True)
        object.__setattr__(self, "path", str(source))
        object.__setattr__(self, "sha256", _sha(self.sha256, "split sha256"))
        if isinstance(self.record_count, bool) or not isinstance(self.record_count, int) or self.record_count <= 0:
            raise ValueError("split record_count must be positive")
        if _stream_sha(source) != self.sha256:
            raise ValueError(f"canonical split {self.name} bytes differ from bundle SHA-256")


@dataclass(frozen=True)
class CanonicalCacheDescriptor:
    role: str
    root: str
    identity: SupervisionCacheIdentity
    contract_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("cache role is required")
        if not isinstance(self.identity, SupervisionCacheIdentity):
            raise ValueError("cache identity must be SupervisionCacheIdentity")
        root = safe_advanced_path(self.root, label=f"canonical {self.role} cache root", must_exist=True, require_directory=True)
        object.__setattr__(self, "root", str(root))
        object.__setattr__(self, "contract_sha256", _sha(self.contract_sha256, "cache contract_sha256"))

    def reopen(self) -> AuthoritativeSafetensorSupervisionCache:
        cache = AuthoritativeSafetensorSupervisionCache(self.root, self.identity)
        if cache.contract_sha256 != self.contract_sha256:
            raise ValueError(f"canonical {self.role} cache content contract changed")
        return cache


@dataclass(frozen=True)
class CanonicalTrainingDataBundle:
    kind: str
    dataset_manifest_path: str
    dataset_manifest_sha256: str
    dataset_receipt_path: str
    dataset_receipt_sha256: str
    canonical_receipt: Mapping[str, Any]
    canonical_receipt_sha256: str
    splits: tuple[CanonicalSplitDescriptor, ...]
    caches: tuple[CanonicalCacheDescriptor, ...]
    bundle_sha256: str

    def __post_init__(self) -> None:
        if self.kind not in {"grounded_generation", "dynamic_rag_policy"}:
            raise ValueError("unsupported canonical training-data bundle kind")
        manifest_path = safe_advanced_path(self.dataset_manifest_path, label="canonical dataset manifest", must_exist=True, require_file=True)
        receipt_path = safe_advanced_path(self.dataset_receipt_path, label="canonical dataset receipt", must_exist=True, require_file=True)
        object.__setattr__(self, "dataset_manifest_path", str(manifest_path))
        object.__setattr__(self, "dataset_receipt_path", str(receipt_path))
        for name in ("dataset_manifest_sha256", "dataset_receipt_sha256", "canonical_receipt_sha256", "bundle_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if _manifest_sha(manifest_path) != self.dataset_manifest_sha256:
            raise ValueError("canonical bundle dataset manifest identity mismatch")
        if not isinstance(self.canonical_receipt, Mapping) or _digest(dict(self.canonical_receipt)) != self.canonical_receipt_sha256:
            raise ValueError("canonical receipt payload digest mismatch")
        splits = tuple(self.splits); caches = tuple(self.caches)
        if not splits or len({item.name for item in splits}) != len(splits):
            raise ValueError("canonical bundle splits must be non-empty and unique")
        if len({item.role for item in caches}) != len(caches):
            raise ValueError("canonical bundle cache roles must be unique")
        object.__setattr__(self, "splits", splits); object.__setattr__(self, "caches", caches)
        if _digest(self.unsigned()) != self.bundle_sha256:
            raise ValueError("canonical training-data bundle digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-canonical-training-data-bundle/v1",
            "kind": self.kind,
            "dataset_manifest_path": self.dataset_manifest_path,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "dataset_receipt_path": self.dataset_receipt_path,
            "dataset_receipt_sha256": self.dataset_receipt_sha256,
            "canonical_receipt": dict(self.canonical_receipt),
            "canonical_receipt_sha256": self.canonical_receipt_sha256,
            "splits": [asdict(item) for item in self.splits],
            "caches": [{"role": item.role, "root": item.root, "identity": asdict(item.identity), "contract_sha256": item.contract_sha256} for item in self.caches],
        }

    def reopened_caches(self) -> Mapping[str, AuthoritativeSafetensorSupervisionCache]:
        return {item.role: item.reopen() for item in self.caches}


def _receipt_file_sha(path: Path) -> str:
    return _stream_sha(path)


def write_dynamic_canonical_bundle(path: str | Path, result: CanonicalDynamicTrainingDataResult) -> CanonicalTrainingDataBundle:
    if not isinstance(result, CanonicalDynamicTrainingDataResult):
        raise ValueError("result must be CanonicalDynamicTrainingDataResult")
    manifest_path = Path(result.dataset.receipt.manifest_path)
    dataset_receipt_path = manifest_path.parent / "publication_receipt.json"
    splits = tuple(CanonicalSplitDescriptor(item.name, item.path, item.sha256, item.record_count) for item in result.dataset.receipt.splits)
    cache = CanonicalCacheDescriptor("hidden_state", str(result.hidden_cache.root), result.hidden_cache.identity, result.hidden_cache.contract_sha256)
    canonical_receipt = {**result.receipt.unsigned(), "receipt_sha256": result.receipt.receipt_sha256}
    unsigned = {
        "schema": "rigorousrag-canonical-training-data-bundle/v1",
        "kind": "dynamic_rag_policy",
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": result.dataset.manifest.manifest_digest,
        "dataset_receipt_path": str(dataset_receipt_path),
        "dataset_receipt_sha256": _receipt_file_sha(dataset_receipt_path),
        "canonical_receipt": canonical_receipt,
        "canonical_receipt_sha256": _digest(canonical_receipt),
        "splits": [asdict(item) for item in splits],
        "caches": [{"role": cache.role, "root": cache.root, "identity": asdict(cache.identity), "contract_sha256": cache.contract_sha256}],
    }
    bundle = CanonicalTrainingDataBundle("dynamic_rag_policy", str(manifest_path), result.dataset.manifest.manifest_digest, str(dataset_receipt_path), unsigned["dataset_receipt_sha256"], canonical_receipt, unsigned["canonical_receipt_sha256"], splits, (cache,), _digest(unsigned))
    destination = safe_advanced_path(path, label="canonical dynamic bundle output", must_exist=False)
    _atomic(destination, {**bundle.unsigned(), "bundle_sha256": bundle.bundle_sha256})
    return bundle


def write_grounded_canonical_bundle(path: str | Path, result: CanonicalGroundedTrainingDataResult) -> CanonicalTrainingDataBundle:
    if not isinstance(result, CanonicalGroundedTrainingDataResult):
        raise ValueError("result must be CanonicalGroundedTrainingDataResult")
    if not result.splits:
        raise ValueError("canonical grounded result has no splits")
    root = Path(result.splits[0].path).parent
    manifest_path = root / "dataset_manifest.json"; dataset_receipt_path = root / "canonical_receipt.json"
    splits = tuple(CanonicalSplitDescriptor(item.name, item.path, item.sha256, item.record_count) for item in result.splits)
    cache_items = []
    for role, cache in (("teacher", result.teacher_cache), ("reference", result.reference_cache), ("retriever_utility", result.retriever_utility_cache)):
        if cache is not None:
            cache_items.append(CanonicalCacheDescriptor(role, str(cache.root), cache.identity, cache.contract_sha256))
    canonical_receipt = {**result.receipt.unsigned(), "receipt_sha256": result.receipt.receipt_sha256}
    unsigned = {
        "schema": "rigorousrag-canonical-training-data-bundle/v1",
        "kind": "grounded_generation",
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": result.manifest.manifest_digest,
        "dataset_receipt_path": str(dataset_receipt_path),
        "dataset_receipt_sha256": _receipt_file_sha(dataset_receipt_path),
        "canonical_receipt": canonical_receipt,
        "canonical_receipt_sha256": _digest(canonical_receipt),
        "splits": [asdict(item) for item in splits],
        "caches": [{"role": item.role, "root": item.root, "identity": asdict(item.identity), "contract_sha256": item.contract_sha256} for item in cache_items],
    }
    bundle = CanonicalTrainingDataBundle("grounded_generation", str(manifest_path), result.manifest.manifest_digest, str(dataset_receipt_path), unsigned["dataset_receipt_sha256"], canonical_receipt, unsigned["canonical_receipt_sha256"], splits, tuple(cache_items), _digest(unsigned))
    destination = safe_advanced_path(path, label="canonical grounded bundle output", must_exist=False)
    _atomic(destination, {**bundle.unsigned(), "bundle_sha256": bundle.bundle_sha256})
    return bundle


def read_canonical_training_data_bundle(path: str | Path) -> CanonicalTrainingDataBundle:
    source = safe_advanced_path(path, label="canonical training-data bundle", must_exist=True, require_file=True)
    raw = _strict_json(source, "canonical training-data bundle")
    required = {"schema", "kind", "dataset_manifest_path", "dataset_manifest_sha256", "dataset_receipt_path", "dataset_receipt_sha256", "canonical_receipt", "canonical_receipt_sha256", "splits", "caches", "bundle_sha256"}
    if set(raw) != required or raw.get("schema") != "rigorousrag-canonical-training-data-bundle/v1":
        raise ValueError("unsupported canonical training-data bundle schema")
    if _stream_sha(safe_advanced_path(raw["dataset_receipt_path"], label="canonical dataset receipt", must_exist=True, require_file=True)) != _sha(raw["dataset_receipt_sha256"], "dataset_receipt_sha256"):
        raise ValueError("canonical dataset receipt bytes differ from bundle")
    split_raw = raw["splits"]; cache_raw = raw["caches"]
    if not isinstance(split_raw, list) or not isinstance(cache_raw, list):
        raise ValueError("canonical bundle splits/caches must be arrays")
    splits = tuple(CanonicalSplitDescriptor(**dict(item)) for item in split_raw if isinstance(item, Mapping))
    if len(splits) != len(split_raw):
        raise ValueError("canonical bundle split entries must be objects")
    caches = []
    for item in cache_raw:
        if not isinstance(item, Mapping) or set(item) != {"role", "root", "identity", "contract_sha256"} or not isinstance(item["identity"], Mapping):
            raise ValueError("canonical cache descriptor is invalid")
        identity = SupervisionCacheIdentity(**dict(item["identity"]))
        caches.append(CanonicalCacheDescriptor(item["role"], item["root"], identity, item["contract_sha256"]))
    bundle = CanonicalTrainingDataBundle(raw["kind"], raw["dataset_manifest_path"], raw["dataset_manifest_sha256"], raw["dataset_receipt_path"], raw["dataset_receipt_sha256"], raw["canonical_receipt"], raw["canonical_receipt_sha256"], splits, tuple(caches), raw["bundle_sha256"])
    bundle.reopened_caches()  # prove every exact cache content contract before returning
    return bundle


__all__ = ["CanonicalCacheDescriptor", "CanonicalSplitDescriptor", "CanonicalTrainingDataBundle", "read_canonical_training_data_bundle", "write_dynamic_canonical_bundle", "write_grounded_canonical_bundle"]
