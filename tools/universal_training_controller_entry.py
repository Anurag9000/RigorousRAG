#!/usr/bin/env python3
"""Immutable bootstrap for exhaustive repository workload orchestration v29.

v29 layers immutably on the v28 bootstrap.  It preserves the exact pinned
OPF_ADP scheduler and every v28 lifecycle/source/scientific contract, then
extends scientific closure/materialization to model/architecture/network/learner
families, model/architecture/network types, variants, capabilities,
functionalities and regularization families.

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

V28_BOOTSTRAP_REPOSITORY = "Anurag9000/RigorousRAG"
V28_BOOTSTRAP_COMMIT = "1239f040b7b96dba77f056d16e1b4da9723a42c9"
V28_BOOTSTRAP_BLOB = "4ba01878e41c736c09503040c5498f66beffffa0"
V28_BOOTSTRAP_PATH = "tools/universal_training_controller_entry.py"

V29_HOST_COMMIT = "c5a13e9573deaa0028a5e44418e5d8e02874be6d"
V29_FILES = {
    "tools/universal_training_controller_scientific_ontology_v29.py": "ff27586e040f98b150b50e6a7b8ee0e97423f4c3",
    "tools/universal_training_controller_v29.py": "78edf24d2c2c24b508bf7aa263332be7d76981fa",
}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "opf-exhaustive-training-controller/32"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def verified_fetch(repository: str, commit: str, relative: str, expected: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{repository}/{commit}/{relative}"
    data = fetch(url)
    actual = git_blob_sha(data)
    if actual != expected:
        raise RuntimeError(f"Pinned controller blob mismatch for {relative}: {actual} != {expected}")
    return data


def load_v28_bootstrap(root: Path):
    cached = root / ".training_control" / "bootstrap" / V28_BOOTSTRAP_COMMIT / "universal_training_controller_entry.py"
    if not cached.is_file() or git_blob_sha(cached.read_bytes()) != V28_BOOTSTRAP_BLOB:
        atomic_write(
            cached,
            verified_fetch(V28_BOOTSTRAP_REPOSITORY, V28_BOOTSTRAP_COMMIT, V28_BOOTSTRAP_PATH, V28_BOOTSTRAP_BLOB),
        )
    spec = importlib.util.spec_from_file_location("_training_control_bootstrap_v28", cached)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pinned v28 bootstrap {cached}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_controller_host(root: Path, v28) -> tuple[Path, object]:
    v27 = v28.load_v27_bootstrap(root)
    cache, legacy = v28.prepare_controller_host(root, v27)
    for relative, expected in V29_FILES.items():
        destination = cache / Path(relative).name
        valid = destination.is_file() and git_blob_sha(destination.read_bytes()) == expected
        if valid:
            continue
        local = root / relative
        if local.is_file() and git_blob_sha(local.read_bytes()) == expected:
            data = local.read_bytes()
        else:
            data = verified_fetch("Anurag9000/RigorousRAG", V29_HOST_COMMIT, relative, expected)
        atomic_write(destination, data)
    return cache, legacy


def main() -> int:
    root = Path(os.environ.get("TRAINING_CONTROL_REPO_ROOT") or Path.cwd()).resolve()
    argv = list(sys.argv[1:])
    v28 = load_v28_bootstrap(root)
    cache, legacy = prepare_controller_host(root, v28)

    if not legacy.diagnostic_only(argv):
        legacy.prepare_reference_cache(root, legacy.OPF_COMMIT, legacy.OPF_FILES)
        if os.environ.get("TRAINING_CONTROL_PREPARE_LEGACY_OPF", "").strip().lower() in {"1", "true", "yes", "on"}:
            legacy.prepare_reference_cache(root, legacy.LEGACY_OPF_COMMIT, legacy.LEGACY_OPF_FILES)

    env = os.environ.copy()
    env["TRAINING_CONTROL_REPO_ROOT"] = str(root)
    return subprocess.call(
        [sys.executable, str(cache / "universal_training_controller_v29.py"), *legacy.canonical_argv(argv)],
        cwd=root,
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
