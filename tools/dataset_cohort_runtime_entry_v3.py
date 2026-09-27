"""Immutable-pinned loader for the CUDA-policy-corrected cohort runtime v3.

The corrected base and v3 transactional layer are independently SHA-verified.
Historical v1/v2 caches and their exact-resume archives remain untouched.
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
BASE_COMMIT = "955a092e4a3e2cc04adfd8007206acd6d1341dce"
BASE_PATH = "training/dataset_cohort_runtime.py"
BASE_BLOB = "8afef6ade42b9885878102d42e26ec3ab2a18bf3"
RUNTIME_COMMIT = "0160deb303f6a4d2a48b8453244dc01ceaae1295"
RUNTIME_PATH = "training/dataset_cohort_runtime_v3.py"
RUNTIME_BLOB = "e6455266b63bed08691cfa76e620060f99eca35a"


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
        headers={"User-Agent": "opf-dataset-cohort-runtime-v3"},
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


def _import_verified(name: str, path: Path) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        origin = getattr(existing, "__file__", None)
        if origin is None or Path(origin).resolve() != path.resolve():
            raise RuntimeError(f"cohort runtime module identity mismatch: {name}")
        return existing
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
    base = _import_verified(f"_opf_dataset_cohort_runtime_{BASE_COMMIT[:12]}", base_path)
    runtime_path = _materialize(directory, RUNTIME_COMMIT, RUNTIME_PATH, RUNTIME_BLOB)
    runtime = _import_verified(f"_opf_dataset_cohort_runtime_v3_{RUNTIME_COMMIT[:12]}", runtime_path)
    if runtime.BASE_COMMIT != BASE_COMMIT or runtime.base is not base:
        raise RuntimeError("cohort runtime v3 is bound to an unexpected CUDA-policy base")
    if runtime.STATE_SCHEMA != "opf-dataset-cohort-state/v2":
        raise RuntimeError("cohort runtime v3 changed the persisted exact-resume state schema")
    return runtime


__all__ = ["BASE_COMMIT", "BASE_BLOB", "RUNTIME_COMMIT", "RUNTIME_BLOB", "load_runtime", "materialize_runtime"]
