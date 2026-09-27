"""v3 immutable CUDA-policy release; no network or physical GPU required."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "training" / "dataset_cohort_runtime.py"
V3 = ROOT / "training" / "dataset_cohort_runtime_v3.py"
BASE_COMMIT = "955a092e4a3e2cc04adfd8007206acd6d1341dce"


def _blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def test_release_has_pinned_corrected_base_and_preserves_v2_state():
    source = V3.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignment = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"BASE_COMMIT", "STATE_SCHEMA_V2"}
    }
    assert assignment["BASE_COMMIT"] == BASE_COMMIT
    assert assignment["STATE_SCHEMA_V2"] == "opf-dataset-cohort-state/v2"
    assert 'OPF_ADP_DISABLE_GPU_ACCELERATORS' in BASE.read_text(encoding="utf-8")


def test_v3_reexports_corrected_base_policy_without_fetch():
    from training import dataset_cohort_runtime as base
    key = f"_opf_dataset_cohort_runtime_{BASE_COMMIT[:12]}"
    spec = importlib.util.spec_from_file_location("_v3_cuda_pin_test", V3)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {key: base}):
        spec.loader.exec_module(module)
    assert module.base is base
    assert module.detect_backend is base.detect_backend
    assert module.subprocess_environment is base.subprocess_environment
    with mock.patch.dict(os.environ, {"OPF_ADP_DISABLE_GPU_ACCELERATORS": "1",
                                      "CUDA_VISIBLE_DEVICES": "0"}, clear=False):
        assert module.detect_backend("auto").selected_device == "cpu"
        with pytest.raises(module.BackendUnavailable):
            module.subprocess_environment("gpu")
