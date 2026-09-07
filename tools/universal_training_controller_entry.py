#!/usr/bin/env python3
"""Immutable bootstrap for exhaustive repository workload orchestration v27.

v27 is layered immutably on the v26 bootstrap. It preserves the exact pinned
OPF_ADP scheduler and the complete v26 lifecycle/source/scientific contracts,
then adds fail-closed coverage for repository-authored compatible scientific
combinations (model/dataset/task/loss/optimizer/pipeline/etc.) without inventing
Cartesian products.

This bootstrap contains no resource scheduling implementation. GPU-first
admission, concurrency, live controls, RAM/VRAM/swap pressure hysteresis,
pause/resume, persisted state, retries, CUDA-OOM CPU fallback, polling/backoff,
device selection and all other scheduler mechanisms remain the literal pinned
OPF_ADP implementation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

V26_BOOTSTRAP_REPOSITORY = "Anurag9000/RigorousRAG"
V26_BOOTSTRAP_COMMIT = "a0222d8753cddddb87e24c57175487d38158b663"
V26_BOOTSTRAP_BLOB = "a31a729adf4a2ce926cc4277eee06a0b7c7a2f1e"
V26_BOOTSTRAP_PATH = "tools/universal_training_controller_entry.py"

V27_HOST_COMMIT = "24144db827d49fcc668f6e6acc419e37e30d39ea"
V27_FILES = {
    "tools/universal_training_controller_combination_closure_v27.py": "b5d23bf38596a112482be5ecfa3835af92d552d0",
    "tools/universal_training_controller_v27.py": "b4b026302f8bde1e408e3fe62632c8cc0a007685",
}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "opf-exhaustive-training-controller/30"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def verified_fetch(repository: str, commit: str, relative: str, expected: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{repository}/{commit}/{relative}"
    data = fetch(url)
    actual = git_blob_sha(data)
    if actual != expected:
        raise RuntimeError(f"Pinned controller blob mismatch for {relative}: {actual} != {expected}")
    return data


def load_v26_bootstrap(root: Path):
    cached = root / ".training_control" / "bootstrap" / V26_BOOTSTRAP_COMMIT / "universal_training_controller_entry.py"
    if not cached.is_file() or git_blob_sha(cached.read_bytes()) != V26_BOOTSTRAP_BLOB:
        atomic_write(
            cached,
            verified_fetch(V26_BOOTSTRAP_REPOSITORY, V26_BOOTSTRAP_COMMIT, V26_BOOTSTRAP_PATH, V26_BOOTSTRAP_BLOB),
        )
    spec = importlib.util.spec_from_file_location("_training_control_bootstrap_v26", cached)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pinned v26 bootstrap {cached}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_controller_host(root: Path, v26) -> tuple[Path, object]:
    v25 = v26.load_v25_bootstrap(root)
    legacy = v25.load_legacy_entry(root)
    cache = v25.prepare_controller_host(root, legacy)
    # Preserve the full v26 selector/scientific stack before adding v27 files.
    for relative, expected in v26.V26_FILES.items():
        destination = cache / Path(relative).name
        valid = destination.is_file() and git_blob_sha(destination.read_bytes()) == expected
        if valid:
            continue
        local = root / relative
        if local.is_file() and git_blob_sha(local.read_bytes()) == expected:
            data = local.read_bytes()
        else:
            data = verified_fetch("Anurag9000/RigorousRAG", v26.V26_HOST_COMMIT, relative, expected)
        atomic_write(destination, data)
    for relative, expected in V27_FILES.items():
        destination = cache / Path(relative).name
        valid = destination.is_file() and git_blob_sha(destination.read_bytes()) == expected
        if valid:
            continue
        local = root / relative
        if local.is_file() and git_blob_sha(local.read_bytes()) == expected:
            data = local.read_bytes()
        else:
            data = verified_fetch("Anurag9000/RigorousRAG", V27_HOST_COMMIT, relative, expected)
        atomic_write(destination, data)
    return cache, legacy


def main() -> int:
    root = Path(os.environ.get("TRAINING_CONTROL_REPO_ROOT") or Path.cwd()).resolve()
    argv = list(sys.argv[1:])
    v26 = load_v26_bootstrap(root)
    cache, legacy = prepare_controller_host(root, v26)

    if not legacy.diagnostic_only(argv):
        legacy.prepare_reference_cache(root, legacy.OPF_COMMIT, legacy.OPF_FILES)
        if os.environ.get("TRAINING_CONTROL_PREPARE_LEGACY_OPF", "").strip().lower() in {"1", "true", "yes", "on"}:
            legacy.prepare_reference_cache(root, legacy.LEGACY_OPF_COMMIT, legacy.LEGACY_OPF_FILES)

    env = os.environ.copy()
    env["TRAINING_CONTROL_REPO_ROOT"] = str(root)
    return subprocess.call(
        [sys.executable, str(cache / "universal_training_controller_v27.py"), *legacy.canonical_argv(argv)],
        cwd=root,
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
