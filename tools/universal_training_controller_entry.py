#!/usr/bin/env python3
"""Immutable bootstrap for exhaustive repository workload orchestration v34.

v34 layers immutably on the v33 bootstrap. It preserves the exact pinned
OPF_ADP scheduler and every v33 lifecycle/source/scientific contract, then adds
fail-closed retained-trainable-source reachability: a real trainer/model source
cannot disappear from the central workload merely because a repository labels it
ignored, dynamic, manual, reference or research. Narrow reasoned exclusions are
reserved for genuinely user-removed/non-trainable/vendor/generated surfaces.

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

V33_BOOTSTRAP_REPOSITORY = "Anurag9000/RigorousRAG"
V33_BOOTSTRAP_COMMIT = "985379f540edc92517955f07c3f81ef942364917"
V33_BOOTSTRAP_BLOB = "f100cf8363b9ebb45a4ae53e7260a3924e20e134"
V33_BOOTSTRAP_PATH = "tools/universal_training_controller_entry.py"

V34_HOST_COMMIT = "0d6eb1fed761e876aa7ad00e0af998d5926fb946"
V34_FILES = {
    "tools/universal_training_controller_retained_training_closure_v34.py": "afeb885dcbef6d20fe3294600e6ec9a7e191b5ed",
    "tools/universal_training_controller_v34.py": "90ce5e0425d23c86e8a2b13c21c45b75c0e9ba40",
}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "opf-exhaustive-training-controller/37"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def verified_fetch(repository: str, commit: str, relative: str, expected: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{repository}/{commit}/{relative}"
    data = fetch(url)
    actual = git_blob_sha(data)
    if actual != expected:
        raise RuntimeError(f"Pinned controller blob mismatch for {relative}: {actual} != {expected}")
    return data


def load_v33_bootstrap(root: Path):
    cached = root / ".training_control" / "bootstrap" / V33_BOOTSTRAP_COMMIT / "universal_training_controller_entry.py"
    if not cached.is_file() or git_blob_sha(cached.read_bytes()) != V33_BOOTSTRAP_BLOB:
        atomic_write(cached, verified_fetch(V33_BOOTSTRAP_REPOSITORY, V33_BOOTSTRAP_COMMIT, V33_BOOTSTRAP_PATH, V33_BOOTSTRAP_BLOB))
    spec = importlib.util.spec_from_file_location("_training_control_bootstrap_v33", cached)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pinned v33 bootstrap {cached}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_controller_host(root: Path, v33) -> tuple[Path, object]:
    v32 = v33.load_v32_bootstrap(root)
    cache, legacy = v33.prepare_controller_host(root, v32)
    for relative, expected in V34_FILES.items():
        destination = cache / Path(relative).name
        valid = destination.is_file() and git_blob_sha(destination.read_bytes()) == expected
        if valid:
            continue
        local = root / relative
        if local.is_file() and git_blob_sha(local.read_bytes()) == expected:
            data = local.read_bytes()
        else:
            data = verified_fetch("Anurag9000/RigorousRAG", V34_HOST_COMMIT, relative, expected)
        atomic_write(destination, data)
    return cache, legacy


def main() -> int:
    root = Path(os.environ.get("TRAINING_CONTROL_REPO_ROOT") or Path.cwd()).resolve()
    argv = list(sys.argv[1:])
    v33 = load_v33_bootstrap(root)
    cache, legacy = prepare_controller_host(root, v33)

    if not legacy.diagnostic_only(argv):
        legacy.prepare_reference_cache(root, legacy.OPF_COMMIT, legacy.OPF_FILES)
        if os.environ.get("TRAINING_CONTROL_PREPARE_LEGACY_OPF", "").strip().lower() in {"1", "true", "yes", "on"}:
            legacy.prepare_reference_cache(root, legacy.LEGACY_OPF_COMMIT, legacy.LEGACY_OPF_FILES)

    env = os.environ.copy()
    env["TRAINING_CONTROL_REPO_ROOT"] = str(root)
    return subprocess.call(
        [sys.executable, str(cache / "universal_training_controller_v34.py"), *legacy.canonical_argv(argv)],
        cwd=root,
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
