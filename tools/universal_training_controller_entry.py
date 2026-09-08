#!/usr/bin/env python3
"""Immutable bootstrap for exhaustive repository workload orchestration v31.

v31 layers immutably on the v30 bootstrap. It preserves the exact pinned
OPF_ADP scheduler and every v30 lifecycle/source/scientific contract, then adds
fail-closed accounting for extended architecture, adaptation, compression,
retrieval, generative/RL, continual-learning, diffusion, graph, data-protocol
and post-processing scientific component selectors.

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

V30_BOOTSTRAP_REPOSITORY = "Anurag9000/RigorousRAG"
V30_BOOTSTRAP_COMMIT = "df5cd8e3b78451c3e9a134cb685d2f90902d8925"
V30_BOOTSTRAP_BLOB = "ecf809be32092aac6e585e15edb6b5a6f90798bc"
V30_BOOTSTRAP_PATH = "tools/universal_training_controller_entry.py"

V31_HOST_COMMIT = "ba0e71784b0702ebb3208f7ca7362b60eac04dcd"
V31_FILES = {
    "tools/universal_training_controller_scientific_ontology_v31.py": "6ae33fbaf4a55e534232bdc9eca3a5727ae4c370",
    "tools/universal_training_controller_v31.py": "7c3e61a8924a7f88049b54be3654c3e086606bf1",
}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "opf-exhaustive-training-controller/34"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def verified_fetch(repository: str, commit: str, relative: str, expected: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{repository}/{commit}/{relative}"
    data = fetch(url)
    actual = git_blob_sha(data)
    if actual != expected:
        raise RuntimeError(f"Pinned controller blob mismatch for {relative}: {actual} != {expected}")
    return data


def load_v30_bootstrap(root: Path):
    cached = root / ".training_control" / "bootstrap" / V30_BOOTSTRAP_COMMIT / "universal_training_controller_entry.py"
    if not cached.is_file() or git_blob_sha(cached.read_bytes()) != V30_BOOTSTRAP_BLOB:
        atomic_write(cached, verified_fetch(V30_BOOTSTRAP_REPOSITORY, V30_BOOTSTRAP_COMMIT, V30_BOOTSTRAP_PATH, V30_BOOTSTRAP_BLOB))
    spec = importlib.util.spec_from_file_location("_training_control_bootstrap_v30", cached)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pinned v30 bootstrap {cached}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_controller_host(root: Path, v30) -> tuple[Path, object]:
    v29 = v30.load_v29_bootstrap(root)
    cache, legacy = v30.prepare_controller_host(root, v29)
    for relative, expected in V31_FILES.items():
        destination = cache / Path(relative).name
        valid = destination.is_file() and git_blob_sha(destination.read_bytes()) == expected
        if valid:
            continue
        local = root / relative
        if local.is_file() and git_blob_sha(local.read_bytes()) == expected:
            data = local.read_bytes()
        else:
            data = verified_fetch("Anurag9000/RigorousRAG", V31_HOST_COMMIT, relative, expected)
        atomic_write(destination, data)
    return cache, legacy


def main() -> int:
    root = Path(os.environ.get("TRAINING_CONTROL_REPO_ROOT") or Path.cwd()).resolve()
    argv = list(sys.argv[1:])
    v30 = load_v30_bootstrap(root)
    cache, legacy = prepare_controller_host(root, v30)

    if not legacy.diagnostic_only(argv):
        legacy.prepare_reference_cache(root, legacy.OPF_COMMIT, legacy.OPF_FILES)
        if os.environ.get("TRAINING_CONTROL_PREPARE_LEGACY_OPF", "").strip().lower() in {"1", "true", "yes", "on"}:
            legacy.prepare_reference_cache(root, legacy.LEGACY_OPF_COMMIT, legacy.LEGACY_OPF_FILES)

    env = os.environ.copy()
    env["TRAINING_CONTROL_REPO_ROOT"] = str(root)
    return subprocess.call(
        [sys.executable, str(cache / "universal_training_controller_v31.py"), *legacy.canonical_argv(argv)],
        cwd=root,
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
