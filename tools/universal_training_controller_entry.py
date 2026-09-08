#!/usr/bin/env python3
"""Immutable bootstrap for exhaustive repository workload orchestration v32.

v32 layers immutably on the v31 bootstrap. It preserves the exact pinned
OPF_ADP scheduler and every v31 lifecycle/source/scientific contract, then adds
fail-closed accounting for the remaining scientifically material choice families:
regularization/gradient controls, probabilistic/Bayesian components, federated
and meta-learning controls, sweep/ablation/search dimensions, data pipeline and
provenance components, decoding/selection policies, kernels/latent spaces,
robustness/corruption/imputation, modality and teacher/student components.

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

V31_BOOTSTRAP_REPOSITORY = "Anurag9000/RigorousRAG"
V31_BOOTSTRAP_COMMIT = "6bbb504893699aac7ab461767f8170b4ee1797fc"
V31_BOOTSTRAP_BLOB = "8ee1a31f2368f3fa0fad7b0b2896fe6da4613bd3"
V31_BOOTSTRAP_PATH = "tools/universal_training_controller_entry.py"

V32_HOST_COMMIT = "b69834bc5f46efaa781976881c147eed806adcbf"
V32_FILES = {
    "tools/universal_training_controller_scientific_ontology_v32.py": "a0d4f27d0ba7f94f836ef60e10983257d3028236",
    "tools/universal_training_controller_v32.py": "99b0afbeb284700c0f7cbdb89dd09c6c60668c23",
}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "opf-exhaustive-training-controller/35"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def verified_fetch(repository: str, commit: str, relative: str, expected: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{repository}/{commit}/{relative}"
    data = fetch(url)
    actual = git_blob_sha(data)
    if actual != expected:
        raise RuntimeError(f"Pinned controller blob mismatch for {relative}: {actual} != {expected}")
    return data


def load_v31_bootstrap(root: Path):
    cached = root / ".training_control" / "bootstrap" / V31_BOOTSTRAP_COMMIT / "universal_training_controller_entry.py"
    if not cached.is_file() or git_blob_sha(cached.read_bytes()) != V31_BOOTSTRAP_BLOB:
        atomic_write(cached, verified_fetch(V31_BOOTSTRAP_REPOSITORY, V31_BOOTSTRAP_COMMIT, V31_BOOTSTRAP_PATH, V31_BOOTSTRAP_BLOB))
    spec = importlib.util.spec_from_file_location("_training_control_bootstrap_v31", cached)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pinned v31 bootstrap {cached}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_controller_host(root: Path, v31) -> tuple[Path, object]:
    v30 = v31.load_v30_bootstrap(root)
    cache, legacy = v31.prepare_controller_host(root, v30)
    for relative, expected in V32_FILES.items():
        destination = cache / Path(relative).name
        valid = destination.is_file() and git_blob_sha(destination.read_bytes()) == expected
        if valid:
            continue
        local = root / relative
        if local.is_file() and git_blob_sha(local.read_bytes()) == expected:
            data = local.read_bytes()
        else:
            data = verified_fetch("Anurag9000/RigorousRAG", V32_HOST_COMMIT, relative, expected)
        atomic_write(destination, data)
    return cache, legacy


def main() -> int:
    root = Path(os.environ.get("TRAINING_CONTROL_REPO_ROOT") or Path.cwd()).resolve()
    argv = list(sys.argv[1:])
    v31 = load_v31_bootstrap(root)
    cache, legacy = prepare_controller_host(root, v31)

    if not legacy.diagnostic_only(argv):
        legacy.prepare_reference_cache(root, legacy.OPF_COMMIT, legacy.OPF_FILES)
        if os.environ.get("TRAINING_CONTROL_PREPARE_LEGACY_OPF", "").strip().lower() in {"1", "true", "yes", "on"}:
            legacy.prepare_reference_cache(root, legacy.LEGACY_OPF_COMMIT, legacy.LEGACY_OPF_FILES)

    env = os.environ.copy()
    env["TRAINING_CONTROL_REPO_ROOT"] = str(root)
    return subprocess.call(
        [sys.executable, str(cache / "universal_training_controller_v32.py"), *legacy.canonical_argv(argv)],
        cwd=root,
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
