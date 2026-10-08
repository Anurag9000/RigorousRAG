"""Versioned cohort runtime v5: executable Git identity and offline loading.

Historical v1-v4 source blobs are immutable. v5 corrects v4's accidental
printable backslash-zero Git-header hash while retaining the v2 state schema.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from tools import dataset_cohort_runtime_entry_v4 as v4
from tools import dataset_cohort_runtime_entry_v5 as v5

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / v5.BASE_PATH
RUNTIME = ROOT / v5.RUNTIME_PATH


def blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def test_new_v5_manifest_pins_correct_bytes_without_rewriting_v4():
    assert v5.BASE_COMMIT == v4.BASE_COMMIT
    assert v5.BASE_BLOB == v4.BASE_BLOB == blob(BASE.read_bytes())
    assert v5.RUNTIME_BLOB == blob(RUNTIME.read_bytes())
    assert v5.RUNTIME_COMMIT == "120f6fd3e934f5febf3e78c4f3c8d1ca43bf05d3"
    assert v5.RUNTIME_BLOB == "3a6c4d9573e570a15232bbaea141e449c67ce5ba"
    assert v5.RUNTIME_PATH.endswith("dataset_cohort_runtime_v5.py")
    assert v4.RUNTIME_BLOB == "47f058e9357a2ee8ec9e9d97c545d2ce91b243b9"


def test_v5_preloaded_base_accepts_exact_runtime_file_and_state_schema():
    from training import dataset_cohort_runtime as base
    name = f"_opf_dataset_cohort_runtime_{v5.BASE_COMMIT[:12]}"
    spec = importlib.util.spec_from_file_location("_cohort_v5_preload_test", RUNTIME)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {name: base}):
        spec.loader.exec_module(module)
    assert module.base is base
    assert module.detect_backend is base.detect_backend
    assert module.subprocess_environment is base.subprocess_environment
    assert module.STATE_SCHEMA == "opf-dataset-cohort-state/v2"


def test_v5_offline_pinned_cache_materializes_and_imports_exact_bytes(tmp_path):
    for commit, relative, source in (
        (v5.BASE_COMMIT, v5.BASE_PATH, BASE),
        (v5.RUNTIME_COMMIT, v5.RUNTIME_PATH, RUNTIME),
    ):
        cache = tmp_path / ".training_control" / "dataset_cohort" / commit / Path(relative).name
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(source.read_bytes())

    base_name = f"_opf_dataset_cohort_runtime_{v5.BASE_COMMIT[:12]}"
    runtime_name = f"_opf_dataset_cohort_runtime_v5_{v5.RUNTIME_COMMIT[:12]}"
    with mock.patch.dict(sys.modules):
        sys.modules.pop(base_name, None)
        sys.modules.pop(runtime_name, None)
        result = v5.load_runtime(tmp_path)
        assert result.STATE_SCHEMA == "opf-dataset-cohort-state/v2"
        assert result.BASE_COMMIT == v5.BASE_COMMIT
        assert result.base is sys.modules[base_name]
        assert result.CohortExecutor is not result.base.CohortExecutor


def test_v5_rejects_tampered_preloaded_base_even_with_same_module_name(tmp_path):
    forged = tmp_path / "dataset_cohort_runtime.py"
    forged.write_text("x = 1\n", encoding="utf-8")
    name = f"_opf_dataset_cohort_runtime_{v5.BASE_COMMIT[:12]}"
    spec = importlib.util.spec_from_file_location("_cohort_v5_tamper_test", RUNTIME)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {name: SimpleNamespace(__file__=str(forged))}):
        with pytest.raises(RuntimeError, match="failed blob verification"):
            spec.loader.exec_module(module)
