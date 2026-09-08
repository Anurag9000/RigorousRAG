#!/usr/bin/env python3
"""Immutable bootstrap for exhaustive repository workload orchestration v30.

v30 layers immutably on the v29 bootstrap. It preserves the exact pinned
OPF_ADP scheduler and every v29 lifecycle/source/scientific contract, then adds
fail-closed accounting for typed scientific selectors, static registry merges
and mutations, structured config choice surfaces, Hydra conf component groups,
and concrete scientific implementation declarations.

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

V29_BOOTSTRAP_REPOSITORY = "Anurag9000/RigorousRAG"
V29_BOOTSTRAP_COMMIT = "5fecd0732a6f57741a1ea61b94f196965af5411f"
V29_BOOTSTRAP_BLOB = "2c76bc3c4b70c511fc4e94aaa1bd0dd145bbadb4"
V29_BOOTSTRAP_PATH = "tools/universal_training_controller_entry.py"

V30_HOST_COMMIT = "b300f8357aaf955a7d03fd2b23523b4ee8f2384a"
V30_FILES = {
    "tools/universal_training_controller_declaration_closure_v30.py": "02873be0cca8bf21f739fcb3a4576e5e810df256",
    "tools/universal_training_controller_v30.py": "a55f4119eaa141a656aa431a7d3fa4767a3d124c",
}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "opf-exhaustive-training-controller/33"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def verified_fetch(repository: str, commit: str, relative: str, expected: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{repository}/{commit}/{relative}"
    data = fetch(url)
    actual = git_blob_sha(data)
    if actual != expected:
        raise RuntimeError(f"Pinned controller blob mismatch for {relative}: {actual} != {expected}")
    return data


def load_v29_bootstrap(root: Path):
    cached = root / ".training_control" / "bootstrap" / V29_BOOTSTRAP_COMMIT / "universal_training_controller_entry.py"
    if not cached.is_file() or git_blob_sha(cached.read_bytes()) != V29_BOOTSTRAP_BLOB:
        atomic_write(cached, verified_fetch(V29_BOOTSTRAP_REPOSITORY, V29_BOOTSTRAP_COMMIT, V29_BOOTSTRAP_PATH, V29_BOOTSTRAP_BLOB))
    spec = importlib.util.spec_from_file_location("_training_control_bootstrap_v29", cached)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pinned v29 bootstrap {cached}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_controller_host(root: Path, v29) -> tuple[Path, object]:
    v28 = v29.load_v28_bootstrap(root)
    cache, legacy = v29.prepare_controller_host(root, v28)
    for relative, expected in V30_FILES.items():
        destination = cache / Path(relative).name
        valid = destination.is_file() and git_blob_sha(destination.read_bytes()) == expected
        if valid:
            continue
        local = root / relative
        if local.is_file() and git_blob_sha(local.read_bytes()) == expected:
            data = local.read_bytes()
        else:
            data = verified_fetch("Anurag9000/RigorousRAG", V30_HOST_COMMIT, relative, expected)
        atomic_write(destination, data)
    return cache, legacy


def main() -> int:
    root = Path(os.environ.get("TRAINING_CONTROL_REPO_ROOT") or Path.cwd()).resolve()
    argv = list(sys.argv[1:])
    v29 = load_v29_bootstrap(root)
    cache, legacy = prepare_controller_host(root, v29)

    if not legacy.diagnostic_only(argv):
        legacy.prepare_reference_cache(root, legacy.OPF_COMMIT, legacy.OPF_FILES)
        if os.environ.get("TRAINING_CONTROL_PREPARE_LEGACY_OPF", "").strip().lower() in {"1", "true", "yes", "on"}:
            legacy.prepare_reference_cache(root, legacy.LEGACY_OPF_COMMIT, legacy.LEGACY_OPF_FILES)

    env = os.environ.copy()
    env["TRAINING_CONTROL_REPO_ROOT"] = str(root)
    return subprocess.call(
        [sys.executable, str(cache / "universal_training_controller_v30.py"), *legacy.canonical_argv(argv)],
        cwd=root,
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
