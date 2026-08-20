"""Bridge authoritative canonical-data v2 publications into neutral training bundles.

The bundle container remains schema v1 because it is only a restart-verifiable transport of
manifest/split/cache identities. Authority generation is proven before emission by reopening the
Grounded/Dynamic v2 canonical receipt. Cache descriptors use the existing training-config type,
whose reader recognizes both historical v1 and disk-backed v2 cache authorities.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_supervision import SupervisionCacheIdentity
from training.authoritative_dynamic_canonical_training_data import (
    VerifiedAuthoritativeDynamicCanonicalData,
    verify_authoritative_dynamic_canonical_training_data,
)
from training.authoritative_grounded_canonical_training_data import (
    VerifiedAuthoritativeGroundedCanonicalData,
    verify_authoritative_grounded_canonical_training_data,
)
from training.canonical_training_data_bundle import (
    CanonicalCacheDescriptor,
    CanonicalSplitDescriptor,
    CanonicalTrainingDataBundle,
    read_canonical_training_data_bundle,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _stream_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic(path: Path, payload: Mapping[str, Any]) -> None:
    destination = safe_advanced_path(path, label="authoritative canonical bundle output", must_exist=False)
    if destination.exists():
        raise ValueError("authoritative canonical bundle output must not already exist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _cache_payload(item: CanonicalCacheDescriptor) -> Mapping[str, Any]:
    return {
        "role": item.role,
        "root": item.root,
        "identity": asdict(item.identity),
        "contract_sha256": item.contract_sha256,
    }


def _write_bundle(path: str | Path, unsigned: Mapping[str, Any]) -> CanonicalTrainingDataBundle:
    bundle = CanonicalTrainingDataBundle(
        kind=unsigned["kind"],
        dataset_manifest_path=unsigned["dataset_manifest_path"],
        dataset_manifest_sha256=unsigned["dataset_manifest_sha256"],
        dataset_receipt_path=unsigned["dataset_receipt_path"],
        dataset_receipt_sha256=unsigned["dataset_receipt_sha256"],
        canonical_receipt=unsigned["canonical_receipt"],
        canonical_receipt_sha256=unsigned["canonical_receipt_sha256"],
        splits=tuple(CanonicalSplitDescriptor(**dict(item)) for item in unsigned["splits"]),
        caches=tuple(
            CanonicalCacheDescriptor(
                item["role"],
                item["root"],
                SupervisionCacheIdentity(**dict(item["identity"])),
                item["contract_sha256"],
            )
            for item in unsigned["caches"]
        ),
        bundle_sha256=_digest(unsigned),
    )
    destination = safe_advanced_path(path, label="authoritative canonical bundle output", must_exist=False)
    _atomic(destination, {**bundle.unsigned(), "bundle_sha256": bundle.bundle_sha256})
    verified = read_canonical_training_data_bundle(destination)
    if verified.bundle_sha256 != bundle.bundle_sha256:
        raise RuntimeError("authoritative canonical training bundle changed during read-back")
    return verified


def write_authoritative_grounded_canonical_bundle(
    path: str | Path,
    canonical_receipt_path: str | Path,
) -> CanonicalTrainingDataBundle:
    verified: VerifiedAuthoritativeGroundedCanonicalData = verify_authoritative_grounded_canonical_training_data(canonical_receipt_path)
    root = Path(verified.root)
    canonical_receipt_file = root / "canonical_receipt.json"
    canonical_payload = {**verified.receipt.unsigned(), "receipt_sha256": verified.receipt.receipt_sha256}
    splits = tuple(
        CanonicalSplitDescriptor(item.name, str(root / item.filename), item.sha256, item.record_count)
        for item in verified.receipt.splits
    )
    role_map = {
        "teacher_logits": "teacher",
        "reference_policy_log_probs": "reference",
        "document_lm_utility": "retriever_utility",
    }
    caches = tuple(
        CanonicalCacheDescriptor(
            role_map[binding.kind],
            str(root / binding.relative_root),
            binding.identity(),
            binding.contract_sha256,
        )
        for binding in verified.receipt.caches
    )
    manifest_path = root / "dataset_manifest.json"
    unsigned = {
        "schema": "rigorousrag-canonical-training-data-bundle/v1",
        "kind": "grounded_generation",
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": verified.manifest.manifest_digest,
        "dataset_receipt_path": str(canonical_receipt_file),
        "dataset_receipt_sha256": _stream_sha(canonical_receipt_file),
        "canonical_receipt": canonical_payload,
        "canonical_receipt_sha256": _digest(canonical_payload),
        "splits": [asdict(item) for item in splits],
        "caches": [_cache_payload(item) for item in caches],
    }
    return _write_bundle(path, unsigned)


def write_authoritative_dynamic_canonical_bundle(
    path: str | Path,
    canonical_receipt_path: str | Path,
) -> CanonicalTrainingDataBundle:
    verified: VerifiedAuthoritativeDynamicCanonicalData = verify_authoritative_dynamic_canonical_training_data(canonical_receipt_path)
    root = Path(verified.root)
    canonical_receipt_file = root / "canonical_receipt.json"
    canonical_payload = {**verified.receipt.unsigned(), "receipt_sha256": verified.receipt.receipt_sha256}
    splits = tuple(
        CanonicalSplitDescriptor(item.name, item.path, item.sha256, item.record_count)
        for item in verified.dataset.receipt.splits
    )
    cache = CanonicalCacheDescriptor(
        "hidden_state",
        str(root / "hidden_cache"),
        verified.receipt.hidden_identity(),
        verified.receipt.hidden_cache_contract_sha256,
    )
    manifest_path = Path(verified.dataset.receipt.manifest_path)
    publication_receipt_path = manifest_path.parent / "publication_receipt.json"
    unsigned = {
        "schema": "rigorousrag-canonical-training-data-bundle/v1",
        "kind": "dynamic_rag_policy",
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": verified.dataset.manifest.manifest_digest,
        "dataset_receipt_path": str(publication_receipt_path),
        "dataset_receipt_sha256": _stream_sha(publication_receipt_path),
        "canonical_receipt": canonical_payload,
        "canonical_receipt_sha256": _digest(canonical_payload),
        "splits": [asdict(item) for item in splits],
        "caches": [_cache_payload(cache)],
    }
    return _write_bundle(path, unsigned)


__all__ = [
    "write_authoritative_dynamic_canonical_bundle",
    "write_authoritative_grounded_canonical_bundle",
]
