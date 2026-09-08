#!/usr/bin/env python3
"""Immutable bootstrap for exhaustive repository workload orchestration v33.

v33 layers immutably on the v32 bootstrap. It preserves the exact pinned
OPF_ADP scheduler and every v32 lifecycle/source/scientific contract, then adds
fail-closed role/paradigm/protocol accounting for scientific choices commonly
named as classifier/regressor/detector roles, open-world and adaptation methods,
supervision paradigms, ensemble algorithms, data sampling/mining/transforms,
RL learning components, stopping/checkpoint policies, evaluation protocols,
decision rules and deployment/export alternatives.

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

V32_BOOTSTRAP_REPOSITORY = "Anurag9000/RigorousRAG"
V32_BOOTSTRAP_COMMIT = "e16946686d09a2b2c3afdba5a18162d6f0aaeaf9"
V32_BOOTSTRAP_BLOB = "e794a90b6d681a7b275f92deadce82861f303bbf"
V32_BOOTSTRAP_PATH = "tools/universal_training_controller_entry.py"

V33_HOST_COMMIT = "d36e02933ab66e01c7ec3926448235f02dcc8c92"
V33_FILES = {
    "tools/universal_training_controller_scientific_ontology_v33.py": "a10ca7e562107af67d1ca30168cfc042a5f0bc3c",
    "tools/universal_training_controller_v33.py": "768ae59decf400171ebda9385b7318adb871c449",
}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "opf-exhaustive-training-controller/36"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def verified_fetch(repository: str, commit: str, relative: str, expected: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{repository}/{commit}/{relative}"
    data = fetch(url)
    actual = git_blob_sha(data)
    if actual != expected:
        raise RuntimeError(f"Pinned controller blob mismatch for {relative}: {actual} != {expected}")
    return data


def load_v32_bootstrap(root: Path):
    cached = root / ".training_control" / "bootstrap" / V32_BOOTSTRAP_COMMIT / "universal_training_controller_entry.py"
    if not cached.is_file() or git_blob_sha(cached.read_bytes()) != V32_BOOTSTRAP_BLOB:
        atomic_write(cached, verified_fetch(V32_BOOTSTRAP_REPOSITORY, V32_BOOTSTRAP_COMMIT, V32_BOOTSTRAP_PATH, V32_BOOTSTRAP_BLOB))
    spec = importlib.util.spec_from_file_location("_training_control_bootstrap_v32", cached)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pinned v32 bootstrap {cached}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_controller_host(root: Path, v32) -> tuple[Path, object]:
    v31 = v32.load_v31_bootstrap(root)
    cache, legacy = v32.prepare_controller_host(root, v31)
    for relative, expected in V33_FILES.items():
        destination = cache / Path(relative).name
        valid = destination.is_file() and git_blob_sha(destination.read_bytes()) == expected
        if valid:
            continue
        local = root / relative
        if local.is_file() and git_blob_sha(local.read_bytes()) == expected:
            data = local.read_bytes()
        else:
            data = verified_fetch("Anurag9000/RigorousRAG", V33_HOST_COMMIT, relative, expected)
        atomic_write(destination, data)
    return cache, legacy


def main() -> int:
    root = Path(os.environ.get("TRAINING_CONTROL_REPO_ROOT") or Path.cwd()).resolve()
    argv = list(sys.argv[1:])
    v32 = load_v32_bootstrap(root)
    cache, legacy = prepare_controller_host(root, v32)

    if not legacy.diagnostic_only(argv):
        legacy.prepare_reference_cache(root, legacy.OPF_COMMIT, legacy.OPF_FILES)
        if os.environ.get("TRAINING_CONTROL_PREPARE_LEGACY_OPF", "").strip().lower() in {"1", "true", "yes", "on"}:
            legacy.prepare_reference_cache(root, legacy.LEGACY_OPF_COMMIT, legacy.LEGACY_OPF_FILES)

    env = os.environ.copy()
    env["TRAINING_CONTROL_REPO_ROOT"] = str(root)
    return subprocess.call(
        [sys.executable, str(cache / "universal_training_controller_v33.py"), *legacy.canonical_argv(argv)],
        cwd=root,
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
