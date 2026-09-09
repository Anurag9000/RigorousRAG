#!/usr/bin/env python3
"""Immutable bootstrap for exhaustive repository workload orchestration v35.

v35 is a compatibility-only layer above the immutable v34 bootstrap. It fixes one
historical controller-host map inconsistency in the v25 ancestry: host commit
26c833284c2be2e80af149b1a2d8902b1a5b43ee contains a newer semantic scanner blob,
while that immutable layer correctly expects the older d6185607... scanner. A
clean controller cache therefore failed before v34 could run.

This layer pre-seeds exactly that historical cache slot from a separately retained,
Git-blob-verified copy, then delegates to v34 unchanged. It contains no resource
scheduler and changes no scientific, resume, early-stopping, DAG, source-closure or
resource-pressure semantics. GPU-first admission, concurrency, RAM/VRAM/swap
hysteresis, pause/resume, persisted state, retries, CUDA-OOM CPU fallback,
polling/backoff, device selection and all other scheduling behavior remain the
literal byte-pinned OPF_ADP implementation reached through v34.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import urllib.request
from pathlib import Path

V34_BOOTSTRAP_REPOSITORY = "Anurag9000/RigorousRAG"
V34_BOOTSTRAP_COMMIT = "142587b31ea4097e237cbe4ad756bb595c1591f7"
V34_BOOTSTRAP_BLOB = "0c7c6307ca97fbc502b511f845059f345712fc7d"
V34_BOOTSTRAP_PATH = "tools/universal_training_controller_entry.py"

BROKEN_LEGACY_HOST_COMMIT = "26c833284c2be2e80af149b1a2d8902b1a5b43ee"
LEGACY_SCANNER_EXPECTED_BLOB = "d6185607102353f8d18b993427065832f3fb374b"
LEGACY_SCANNER_CACHE_NAME = "training_surface_semantic_scan.py"
COMPAT_SOURCE_REPOSITORY = "Anurag9000/RigorousRAG"
COMPAT_SOURCE_COMMIT = "2b39c1fb680e608424ba77ed0f7245b0a558f256"
COMPAT_SOURCE_PATH = "tools/training_surface_semantic_scan_legacy_v20.py"


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "opf-exhaustive-training-controller/38"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def verified_fetch(repository: str, commit: str, relative: str, expected: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{repository}/{commit}/{relative}"
    data = fetch(url)
    actual = git_blob_sha(data)
    if actual != expected:
        raise RuntimeError(f"Pinned controller blob mismatch for {relative}: {actual} != {expected}")
    return data


def _preseed_legacy_semantic_scanner(root: Path) -> None:
    destination = (
        root
        / ".training_control"
        / "controller_host"
        / BROKEN_LEGACY_HOST_COMMIT
        / LEGACY_SCANNER_CACHE_NAME
    )
    if destination.is_file() and git_blob_sha(destination.read_bytes()) == LEGACY_SCANNER_EXPECTED_BLOB:
        return

    local = root / COMPAT_SOURCE_PATH
    if local.is_file() and git_blob_sha(local.read_bytes()) == LEGACY_SCANNER_EXPECTED_BLOB:
        data = local.read_bytes()
    else:
        data = verified_fetch(
            COMPAT_SOURCE_REPOSITORY,
            COMPAT_SOURCE_COMMIT,
            COMPAT_SOURCE_PATH,
            LEGACY_SCANNER_EXPECTED_BLOB,
        )
    atomic_write(destination, data)
    actual = git_blob_sha(destination.read_bytes())
    if actual != LEGACY_SCANNER_EXPECTED_BLOB:
        raise RuntimeError(
            f"Failed to pre-seed verified legacy semantic scanner: {actual} != {LEGACY_SCANNER_EXPECTED_BLOB}"
        )


def _load_v34_bootstrap(root: Path):
    cached = (
        root
        / ".training_control"
        / "bootstrap"
        / V34_BOOTSTRAP_COMMIT
        / "universal_training_controller_entry.py"
    )
    if not cached.is_file() or git_blob_sha(cached.read_bytes()) != V34_BOOTSTRAP_BLOB:
        atomic_write(
            cached,
            verified_fetch(
                V34_BOOTSTRAP_REPOSITORY,
                V34_BOOTSTRAP_COMMIT,
                V34_BOOTSTRAP_PATH,
                V34_BOOTSTRAP_BLOB,
            ),
        )
    spec = importlib.util.spec_from_file_location("_training_control_bootstrap_v34", cached)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pinned v34 bootstrap {cached}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    root = Path(os.environ.get("TRAINING_CONTROL_REPO_ROOT") or Path.cwd()).resolve()
    _preseed_legacy_semantic_scanner(root)
    v34 = _load_v34_bootstrap(root)
    return int(v34.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
