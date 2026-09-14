"""Byte-pinned loader for the canonical dataset-cohort runtime.

Repository-specific scientific authorities should import ``load_runtime`` from a
copy of this file rather than fork the runtime.  The runtime is immutable by Git
blob identity; changing it requires a deliberate repin and estate re-audit.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import urllib.request
from pathlib import Path
from types import ModuleType

RUNTIME_REPOSITORY = "Anurag9000/RigorousRAG"
RUNTIME_COMMIT = "0c50adb23e5ba58b7c49b18401950f0bf7e5b736"
RUNTIME_PATH = "training/dataset_cohort_runtime.py"
RUNTIME_BLOB = "1ffb1af0f70812d3418a6f0959ae88c7a145939e"
RUNTIME_URL = f"https://raw.githubusercontent.com/{RUNTIME_REPOSITORY}/{RUNTIME_COMMIT}/{RUNTIME_PATH}"


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _root() -> Path:
    return Path(os.environ.get("TRAINING_CONTROL_REPO_ROOT") or Path.cwd()).resolve()


def materialize_runtime(root: Path | None = None) -> Path:
    repository_root = (root or _root()).resolve()
    target = repository_root / ".training_control" / "dataset_cohort" / RUNTIME_COMMIT / "dataset_cohort_runtime.py"
    if target.is_file() and git_blob_sha(target.read_bytes()) == RUNTIME_BLOB:
        return target
    request = urllib.request.Request(RUNTIME_URL, headers={"User-Agent": "opf-dataset-cohort-runtime/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    actual = git_blob_sha(payload)
    if actual != RUNTIME_BLOB:
        raise RuntimeError(f"Pinned dataset-cohort runtime checksum mismatch: {actual} != {RUNTIME_BLOB}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, target)
    return target


def load_runtime(root: Path | None = None) -> ModuleType:
    path = materialize_runtime(root)
    name = f"_opf_dataset_cohort_runtime_{RUNTIME_COMMIT[:12]}"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import dataset-cohort runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


__all__ = ["RUNTIME_BLOB", "RUNTIME_COMMIT", "RUNTIME_PATH", "RUNTIME_REPOSITORY", "load_runtime", "materialize_runtime"]
