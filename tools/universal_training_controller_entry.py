#!/usr/bin/env python3
"""Immutable bootstrap for exhaustive repository workload orchestration v26.

v26 is layered immutably on the v25 bootstrap.  It preserves the exact pinned
OPF_ADP scheduler and the complete v25 lifecycle/source/scientific contracts,
then adds selector/config closure for CLI choices, Enum/Literal aliases,
registration APIs, registry subscript assignment, branch/match dispatch and
scientific component config groups.

This bootstrap contains no resource scheduling implementation.  GPU-first
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

V25_BOOTSTRAP_REPOSITORY = "Anurag9000/RigorousRAG"
V25_BOOTSTRAP_COMMIT = "5e76048950a8c5531f213a5a7cb3ed71fceda48a"
V25_BOOTSTRAP_BLOB = "e538519e846ed4bb9af10fd1b3ad9afa1f40825c"
V25_BOOTSTRAP_PATH = "tools/universal_training_controller_entry.py"

V26_HOST_COMMIT = "cf6c96793f0d2daf2a26abe3da63668b91d6ddd1"
V26_FILES = {
    "tools/universal_training_controller_selector_closure_v26.py": "1e0d3d15bb577e658b646fb5a689076f7cb0e058",
    "tools/universal_training_controller_v26.py": "e545b9a72ddd360cde45e843198483203a6d4ed7",
}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "opf-exhaustive-training-controller/29"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def verified_fetch(repository: str, commit: str, relative: str, expected: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{repository}/{commit}/{relative}"
    data = fetch(url)
    actual = git_blob_sha(data)
    if actual != expected:
        raise RuntimeError(f"Pinned controller blob mismatch for {relative}: {actual} != {expected}")
    return data


def load_v25_bootstrap(root: Path):
    cached = root / ".training_control" / "bootstrap" / V25_BOOTSTRAP_COMMIT / "universal_training_controller_entry.py"
    if not cached.is_file() or git_blob_sha(cached.read_bytes()) != V25_BOOTSTRAP_BLOB:
        atomic_write(
            cached,
            verified_fetch(V25_BOOTSTRAP_REPOSITORY, V25_BOOTSTRAP_COMMIT, V25_BOOTSTRAP_PATH, V25_BOOTSTRAP_BLOB),
        )
    spec = importlib.util.spec_from_file_location("_training_control_bootstrap_v25", cached)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pinned v25 bootstrap {cached}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_controller_host(root: Path, v25) -> tuple[Path, object]:
    legacy = v25.load_legacy_entry(root)
    cache = v25.prepare_controller_host(root, legacy)
    for relative, expected in V26_FILES.items():
        destination = cache / Path(relative).name
        valid = destination.is_file() and git_blob_sha(destination.read_bytes()) == expected
        if valid:
            continue
        local = root / relative
        if local.is_file() and git_blob_sha(local.read_bytes()) == expected:
            data = local.read_bytes()
        else:
            data = verified_fetch("Anurag9000/RigorousRAG", V26_HOST_COMMIT, relative, expected)
        atomic_write(destination, data)
    return cache, legacy


def main() -> int:
    root = Path(os.environ.get("TRAINING_CONTROL_REPO_ROOT") or Path.cwd()).resolve()
    argv = list(sys.argv[1:])
    v25 = load_v25_bootstrap(root)
    cache, legacy = prepare_controller_host(root, v25)

    if not legacy.diagnostic_only(argv):
        legacy.prepare_reference_cache(root, legacy.OPF_COMMIT, legacy.OPF_FILES)
        if os.environ.get("TRAINING_CONTROL_PREPARE_LEGACY_OPF", "").strip().lower() in {"1", "true", "yes", "on"}:
            legacy.prepare_reference_cache(root, legacy.LEGACY_OPF_COMMIT, legacy.LEGACY_OPF_FILES)

    env = os.environ.copy()
    env["TRAINING_CONTROL_REPO_ROOT"] = str(root)
    return subprocess.call(
        [sys.executable, str(cache / "universal_training_controller_v26.py"), *legacy.canonical_argv(argv)],
        cwd=root,
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
