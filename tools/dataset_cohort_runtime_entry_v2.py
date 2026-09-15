"""Verified loader for the transactional dataset-cohort runtime v2.

The v2 executor extends, rather than forks, the immutable v1 runtime.  This entry
first materializes/imports v1 through its existing blob-verified loader, then
materializes v2 by immutable Git blob identity.  Preloading v1 under its canonical
module name means v2 never takes its standalone network fallback on the supported
estate path.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import urllib.request
from pathlib import Path
from types import ModuleType

from tools.dataset_cohort_runtime_entry import (
    RUNTIME_BLOB as V1_BLOB,
    RUNTIME_COMMIT as V1_COMMIT,
    load_runtime as load_v1_runtime,
)

RUNTIME_REPOSITORY = "Anurag9000/RigorousRAG"
RUNTIME_COMMIT = "0124a824d4fff32f4b427812a27cf7937d7c0891"
RUNTIME_PATH = "training/dataset_cohort_runtime_v2.py"
RUNTIME_BLOB = "9364847538404151966b3719b10a1a9dc5f0f400"
RUNTIME_URL = f"https://raw.githubusercontent.com/{RUNTIME_REPOSITORY}/{RUNTIME_COMMIT}/{RUNTIME_PATH}"


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _root() -> Path:
    return Path(os.environ.get("TRAINING_CONTROL_REPO_ROOT") or Path.cwd()).resolve()


def materialize_runtime(root: Path | None = None) -> Path:
    repository_root = (root or _root()).resolve()
    target = (
        repository_root
        / ".training_control"
        / "dataset_cohort"
        / RUNTIME_COMMIT
        / "dataset_cohort_runtime_v2.py"
    )
    if target.is_file() and git_blob_sha(target.read_bytes()) == RUNTIME_BLOB:
        return target
    request = urllib.request.Request(
        RUNTIME_URL,
        headers={"User-Agent": "opf-dataset-cohort-runtime-v2/2"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    actual = git_blob_sha(payload)
    if actual != RUNTIME_BLOB:
        raise RuntimeError(
            f"Pinned dataset-cohort v2 checksum mismatch: {actual} != {RUNTIME_BLOB}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, target)
    return target


def load_runtime(root: Path | None = None) -> ModuleType:
    repository_root = (root or _root()).resolve()
    base = load_v1_runtime(repository_root)
    expected_base_name = f"_opf_dataset_cohort_runtime_{V1_COMMIT[:12]}"
    if sys.modules.get(expected_base_name) is not base:
        raise RuntimeError("verified v1 runtime was not registered under its canonical module name")
    path = materialize_runtime(repository_root)
    name = f"_opf_dataset_cohort_runtime_v2_{RUNTIME_COMMIT[:12]}"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import transactional dataset-cohort runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if getattr(module, "BASE_COMMIT", None) != V1_COMMIT:
        raise RuntimeError("transactional runtime v2 is bound to an unexpected v1 commit")
    if V1_BLOB != "1ffb1af0f70812d3418a6f0959ae88c7a145939e":
        raise RuntimeError("local v1 loader pin drifted from the v2-reviewed base blob")
    return module


__all__ = [
    "RUNTIME_BLOB",
    "RUNTIME_COMMIT",
    "RUNTIME_PATH",
    "RUNTIME_REPOSITORY",
    "load_runtime",
    "materialize_runtime",
]
