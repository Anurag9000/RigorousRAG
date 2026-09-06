#!/usr/bin/env python3
"""Immutable bootstrap for exhaustive repository workload orchestration v22.

The bootstrap reuses the previously pinned universal controller stack and exact
OPF_ADP runtime, then adds only source-orchestration layers. Resource admission
and scheduling are never reimplemented here: the literal byte-pinned OPF runner
remains solely responsible for pressure-aware/GPU-first/fixed scheduling,
concurrency, memory pressure gates, pause/resume, retries, OOM fallback, device
selection, logging and persistent process state.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

LEGACY_ENTRY_REPOSITORY = "Anurag9000/RigorousRAG"
LEGACY_ENTRY_COMMIT = "7b9ceb12d6c5fdef33eefd73eaea4c027b941737"
LEGACY_ENTRY_BLOB = "4ecb86674c3baa91c88ff57a8699decce26c528d"
LEGACY_ENTRY_PATH = "tools/universal_training_controller_entry.py"

V22_HOST_COMMIT = "d741e571dbee8052a7d6d0179ab624b8e5d4cf42"
V22_FILES = {
    "tools/universal_training_controller_lifecycle_exhaustive.py": "8e05a0ab48264cc3a3ffd8834bdf6ba40a9cf460",
    "tools/universal_training_controller_config_matrix.py": "cfdc58f5dea0e5351871bb0adca7b1f39444d756",
    "tools/universal_training_controller_early_stopping_wiring.py": "fdd00d7b406a77933c827237fb45db9d083001dc",
    "tools/universal_training_controller_lifecycle_affinity.py": "31e896f6ab10cc902d594e0c21c324d0f4092e8c",
    "tools/universal_training_controller_workload_closure.py": "622e0a2e3d45df8988613d9275c9478b600b89dc",
    "tools/universal_training_controller_dag_slicing.py": "b5ecd8fbb152c1da40107cd023af19d1e86ab3e4",
    "tools/universal_training_controller_metrics_v2.py": "26bac973df28fa37f21c6464031fec2e3ece9908",
    "tools/universal_training_controller_v22.py": "bb1720e9ba82e8f9977ae52cee6bebb33f86a876",
}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "opf-exhaustive-training-controller/25"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def verified_fetch(repository: str, commit: str, relative: str, expected: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{repository}/{commit}/{relative}"
    data = fetch(url)
    actual = git_blob_sha(data)
    if actual != expected:
        raise RuntimeError(f"Pinned controller blob mismatch for {relative}: {actual} != {expected}")
    return data


def load_legacy_entry(root: Path):
    cached = root / ".training_control" / "bootstrap" / LEGACY_ENTRY_COMMIT / "universal_training_controller_entry.py"
    if not cached.is_file() or git_blob_sha(cached.read_bytes()) != LEGACY_ENTRY_BLOB:
        atomic_write(
            cached,
            verified_fetch(LEGACY_ENTRY_REPOSITORY, LEGACY_ENTRY_COMMIT, LEGACY_ENTRY_PATH, LEGACY_ENTRY_BLOB),
        )
    spec = importlib.util.spec_from_file_location("_training_control_legacy_entry_v25", cached)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pinned legacy controller bootstrap {cached}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_controller_host(root: Path, legacy) -> Path:
    cache = root / ".training_control" / "controller_host" / legacy.HOST_COMMIT
    for relative, expected in legacy.FILES.items():
        destination = cache / Path(relative).name
        valid = destination.is_file() and legacy.git_blob_sha(destination.read_bytes()) == expected
        if valid:
            continue
        data = legacy.verified_host_local(root, relative, expected)
        if data is None:
            data = legacy.fetch(
                f"https://raw.githubusercontent.com/{legacy.HOST_REPO}/{legacy.HOST_COMMIT}/{relative}"
            )
        actual = legacy.git_blob_sha(data)
        if actual != expected:
            raise RuntimeError(f"Legacy controller blob mismatch for {relative}: {actual} != {expected}")
        legacy.atomic_write(destination, data)

    for relative, expected in V22_FILES.items():
        destination = cache / Path(relative).name
        valid = destination.is_file() and git_blob_sha(destination.read_bytes()) == expected
        if valid:
            continue
        local = root / relative
        if local.is_file() and git_blob_sha(local.read_bytes()) == expected:
            data = local.read_bytes()
        else:
            data = verified_fetch("Anurag9000/RigorousRAG", V22_HOST_COMMIT, relative, expected)
        atomic_write(destination, data)
    return cache


def main() -> int:
    root = Path(os.environ.get("TRAINING_CONTROL_REPO_ROOT") or Path.cwd()).resolve()
    argv = list(sys.argv[1:])
    legacy = load_legacy_entry(root)
    cache = prepare_controller_host(root, legacy)

    if not legacy.diagnostic_only(argv):
        legacy.prepare_reference_cache(root, legacy.OPF_COMMIT, legacy.OPF_FILES)
        if os.environ.get("TRAINING_CONTROL_PREPARE_LEGACY_OPF", "").strip().lower() in {"1", "true", "yes", "on"}:
            legacy.prepare_reference_cache(root, legacy.LEGACY_OPF_COMMIT, legacy.LEGACY_OPF_FILES)

    env = os.environ.copy()
    env["TRAINING_CONTROL_REPO_ROOT"] = str(root)
    return subprocess.call(
        [sys.executable, str(cache / "universal_training_controller_v22.py"), *legacy.canonical_argv(argv)],
        cwd=root,
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
