#!/usr/bin/env python3
"""Immutable bootstrap for exhaustive repository workload orchestration v28.

v28 is layered immutably on the v27 bootstrap.  It preserves the exact pinned
OPF_ADP scheduler and every v27 lifecycle/source/scientific contract, then adds
execution materialization for concrete repository-authored scientific
combinations whenever an existing trainer exposes all required selectors.
Ambiguous combinations continue to fail closed for repo-specific source repair.

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

V27_BOOTSTRAP_REPOSITORY = "Anurag9000/RigorousRAG"
V27_BOOTSTRAP_COMMIT = "a5caedea0618ecce1cef38e4e9b7d4fe14fe76ab"
V27_BOOTSTRAP_BLOB = "3acfd37ae22e887b5a7bcd42b04c9a288593abae"
V27_BOOTSTRAP_PATH = "tools/universal_training_controller_entry.py"

V28_HOST_COMMIT = "9dc0fdc23420eb5918966712feec9607a5fe4aa0"
V28_FILES = {
    "tools/universal_training_controller_declared_combination_materializer_v28.py": "249d49953b7ea42359b14eb2177500920e363afe",
    "tools/universal_training_controller_v28.py": "3ccf609957cc4bb9f6242518b768ccd39717ac14",
}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "opf-exhaustive-training-controller/31"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def verified_fetch(repository: str, commit: str, relative: str, expected: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{repository}/{commit}/{relative}"
    data = fetch(url)
    actual = git_blob_sha(data)
    if actual != expected:
        raise RuntimeError(f"Pinned controller blob mismatch for {relative}: {actual} != {expected}")
    return data


def load_v27_bootstrap(root: Path):
    cached = root / ".training_control" / "bootstrap" / V27_BOOTSTRAP_COMMIT / "universal_training_controller_entry.py"
    if not cached.is_file() or git_blob_sha(cached.read_bytes()) != V27_BOOTSTRAP_BLOB:
        atomic_write(
            cached,
            verified_fetch(V27_BOOTSTRAP_REPOSITORY, V27_BOOTSTRAP_COMMIT, V27_BOOTSTRAP_PATH, V27_BOOTSTRAP_BLOB),
        )
    spec = importlib.util.spec_from_file_location("_training_control_bootstrap_v27", cached)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pinned v27 bootstrap {cached}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_controller_host(root: Path, v27) -> tuple[Path, object]:
    v26 = v27.load_v26_bootstrap(root)
    cache, legacy = v27.prepare_controller_host(root, v26)
    for relative, expected in V28_FILES.items():
        destination = cache / Path(relative).name
        valid = destination.is_file() and git_blob_sha(destination.read_bytes()) == expected
        if valid:
            continue
        local = root / relative
        if local.is_file() and git_blob_sha(local.read_bytes()) == expected:
            data = local.read_bytes()
        else:
            data = verified_fetch("Anurag9000/RigorousRAG", V28_HOST_COMMIT, relative, expected)
        atomic_write(destination, data)
    return cache, legacy


def main() -> int:
    root = Path(os.environ.get("TRAINING_CONTROL_REPO_ROOT") or Path.cwd()).resolve()
    argv = list(sys.argv[1:])
    v27 = load_v27_bootstrap(root)
    cache, legacy = prepare_controller_host(root, v27)

    if not legacy.diagnostic_only(argv):
        legacy.prepare_reference_cache(root, legacy.OPF_COMMIT, legacy.OPF_FILES)
        if os.environ.get("TRAINING_CONTROL_PREPARE_LEGACY_OPF", "").strip().lower() in {"1", "true", "yes", "on"}:
            legacy.prepare_reference_cache(root, legacy.LEGACY_OPF_COMMIT, legacy.LEGACY_OPF_FILES)

    env = os.environ.copy()
    env["TRAINING_CONTROL_REPO_ROOT"] = str(root)
    return subprocess.call(
        [sys.executable, str(cache / "universal_training_controller_v28.py"), *legacy.canonical_argv(argv)],
        cwd=root,
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
