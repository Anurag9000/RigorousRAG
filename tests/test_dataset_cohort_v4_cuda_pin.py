"""Immutable dataset-cohort v4 scheduler/CUDA release tests."""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "training/dataset_cohort_runtime.py"
V4 = ROOT / "training/dataset_cohort_runtime_v4.py"
BASE_COMMIT = "3f0fbefb7f6525c71a2ecd6f0fdf512b8b2fc75a"
BASE_BLOB = "8d92fbc258b1b5f024923bc051f021f08453dd0c"


def _constants(path: Path, names: set[str]) -> dict:
    found = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in names:
            found[target.id] = ast.literal_eval(node.value)
    assert set(found) == names
    return found


def test_v4_pins_corrected_base_and_preserves_v2_state_schema():
    values = _constants(V4, {"BASE_COMMIT", "BASE_BLOB", "STATE_SCHEMA_V2"})
    assert values["BASE_COMMIT"] == BASE_COMMIT
    assert values["BASE_BLOB"] == BASE_BLOB
    assert values["STATE_SCHEMA_V2"] == "opf-dataset-cohort-state/v2"

    import hashlib
    payload = BASE.read_bytes()
    actual = hashlib.sha1(
        f"blob {len(payload)}\0".encode("ascii") + payload
    ).hexdigest()
    assert actual == BASE_BLOB

    source = BASE.read_text(encoding="utf-8")
    assert 'TRAINING_CONTROL_BACKEND' in source
    assert 'torch.cuda.synchronize("cuda:0")' in source
    assert 'cupy.cuda.runtime.deviceSynchronize()' in source


def test_v4_reuses_only_blob_verified_preloaded_base(monkeypatch):
    from training import dataset_cohort_runtime as base

    key = f"_opf_dataset_cohort_runtime_{BASE_COMMIT[:12]}"
    spec = importlib.util.spec_from_file_location("_v4_cuda_pin_test", V4)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {key: base}):
        spec.loader.exec_module(module)
    assert module.base is base
    assert module.detect_backend is base.detect_backend
    assert module.subprocess_environment is base.subprocess_environment


def test_v4_rejects_preloaded_base_with_wrong_source_blob(tmp_path):
    fake_source = tmp_path / "dataset_cohort_runtime.py"
    fake_source.write_text("x = 1\n", encoding="utf-8")
    fake = SimpleNamespace(__file__=str(fake_source))
    key = f"_opf_dataset_cohort_runtime_{BASE_COMMIT[:12]}"
    spec = importlib.util.spec_from_file_location("_v4_tamper_test", V4)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {key: fake}):
        with pytest.raises(RuntimeError, match="failed blob verification"):
            spec.loader.exec_module(module)


def test_v4_declared_cpu_backend_is_authoritative_without_optional_flags():
    from training import dataset_cohort_runtime as base

    with mock.patch.dict(os.environ, {
        "TRAINING_CONTROL_BACKEND": "cpu",
        "CUDA_VISIBLE_DEVICES": "0",
    }, clear=True), mock.patch.object(
        base, "_safe_import", side_effect=AssertionError("accelerator imported")
    ), mock.patch.object(
        base, "_nvidia_smi", side_effect=AssertionError("hardware probed")
    ):
        probe = base.detect_backend("auto")
        assert probe.selected_device == "cpu"
        assert not probe.gpu_available
