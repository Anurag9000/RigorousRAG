"""Immutable-pinned loader for dataset-cohort runtime v5.

v5 preserves the transactional state schema while making the outer scheduler's
CPU/GPU backend admission authoritative and requiring executable CUDA proof.
Historical v1-v4 loaders and cache identities remain unchanged.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import urllib.request
from types import ModuleType

REPOSITORY = "Anurag9000/RigorousRAG"
BASE_COMMIT = "3f0fbefb7f6525c71a2ecd6f0fdf512b8b2fc75a"
BASE_PATH = "training/dataset_cohort_runtime.py"
BASE_BLOB = "8d92fbc258b1b5f024923bc051f021f08453dd0c"
RUNTIME_COMMIT = "120f6fd3e934f5febf3e78c4f3c8d1ca43bf05d3"
RUNTIME_PATH = "training/dataset_cohort_runtime_v4.py"
RUNTIME_BLOB = "3a6c4d9573e570a15232bbaea141e449c67ce5ba"


def _blob(payload: bytes) -> str:
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


def _root() -> Path:
    return Path(os.environ.get("TRAINING_CONTROL_REPO_ROOT") or Path.cwd()).resolve()


def _materialize(root: Path, commit: str, source: str, blob: str) -> Path:
    target = root / ".training_control" / "dataset_cohort" / commit / Path(source).name
    if target.is_file() and _blob(target.read_bytes()) == blob:
        return target
    request = urllib.request.Request(
        f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/{source}",
        headers={"User-Agent": "opf-dataset-cohort-runtime-v5"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    actual = _blob(payload)
    if actual != blob:
        raise RuntimeError(f"pinned cohort runtime blob mismatch: {source}: {actual} != {blob}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, target)
    return target


def _import_verified(name: str, path: Path, expected_blob: str) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        origin = Path(str(getattr(existing, "__file__", "") or ""))
        if not origin.is_file() or origin.resolve() != path.resolve() or _blob(origin.read_bytes()) != expected_blob:
            raise RuntimeError(f"cohort runtime module identity mismatch: {name}")
        return existing
    if _blob(path.read_bytes()) != expected_blob:
        raise RuntimeError(f"cohort runtime source changed before import: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned cohort runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def materialize_runtime(root: Path | None = None) -> Path:
    directory = (root or _root()).resolve()
    _materialize(directory, BASE_COMMIT, BASE_PATH, BASE_BLOB)
    return _materialize(directory, RUNTIME_COMMIT, RUNTIME_PATH, RUNTIME_BLOB)


def load_runtime(root: Path | None = None) -> ModuleType:
    directory = (root or _root()).resolve()
    base_path = _materialize(directory, BASE_COMMIT, BASE_PATH, BASE_BLOB)
    base = _import_verified(
        f"_opf_dataset_cohort_runtime_{BASE_COMMIT[:12]}", base_path, BASE_BLOB
    )
    runtime_path = _materialize(directory, RUNTIME_COMMIT, RUNTIME_PATH, RUNTIME_BLOB)
    runtime = _import_verified(
        f"_opf_dataset_cohort_runtime_v4_{RUNTIME_COMMIT[:12]}",
        runtime_path,
        RUNTIME_BLOB,
    )
    if runtime.BASE_COMMIT != BASE_COMMIT or runtime.base is not base:
        raise RuntimeError("cohort runtime v5 is bound to an unexpected scheduler-admission base")
    if runtime.STATE_SCHEMA != "opf-dataset-cohort-state/v2":
        raise RuntimeError("cohort runtime v5 changed the persisted exact-resume state schema")
    return runtime


__all__ = [
    "BASE_COMMIT", "BASE_BLOB", "RUNTIME_COMMIT", "RUNTIME_BLOB",
    "RUNTIME_PATH", "load_runtime", "materialize_runtime",
]
