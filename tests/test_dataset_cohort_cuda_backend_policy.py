"""CPU-safe backend policy regressions for shared dataset-cohort workers."""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from training import dataset_cohort_runtime as runtime


def test_visible_hardware_without_usable_framework_remains_cpu():
    with mock.patch.object(runtime, "_safe_import", return_value=None), \
         mock.patch.object(runtime, "_nvidia_smi", return_value=True), \
         mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False):
        result = runtime.detect_backend("auto")
        assert result.selected_device == "cpu"
        assert not result.gpu_available
        assert result.nvidia_smi
        with pytest.raises(runtime.BackendUnavailable):
            runtime.detect_backend("gpu")


def test_unusable_cupy_allocation_does_not_enable_gpu():
    cp = SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 1)),
        uint8=object(),
        empty=mock.Mock(side_effect=RuntimeError("driver unusable")),
    )
    with mock.patch.object(runtime, "_safe_import", side_effect=lambda name: cp if name == "cupy" else None), \
         mock.patch.object(runtime, "_nvidia_smi", return_value=False), \
         mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False):
        result = runtime.detect_backend("auto")
        assert result.selected_device == "cpu"
        assert not result.cupy_cuda


def test_torch_only_gpu_uses_numpy_without_cupy():
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
        empty=lambda *args, **kwargs: object(),
    )
    with mock.patch.object(runtime, "_safe_import", side_effect=lambda name: torch if name == "torch" else np if name == "numpy" else None), \
         mock.patch.object(runtime, "_nvidia_smi", return_value=False), \
         mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2", "TRAINING_CONTROL_GPU_INDEX": "2"}, clear=False):
        assert runtime.detect_backend("auto").selected_device == "cuda:0"
        assert runtime.array_namespace() is np


def test_cpu_mask_is_never_unmasked_by_auto_subprocess():
    env = runtime.subprocess_environment("auto", base={"CUDA_VISIBLE_DEVICES": "", "GPU_DEVICE_INDEX": "0"})
    assert env["CUDA_VISIBLE_DEVICES"] == ""


def test_gpu_assignment_remains_owned_by_outer_scheduler():
    env = runtime.subprocess_environment("gpu", base={"CUDA_VISIBLE_DEVICES": "2", "GPU_DEVICE_INDEX": "0"})
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    with pytest.raises(runtime.BackendUnavailable):
        runtime.subprocess_environment("gpu", gpu_index="3", base={"CUDA_VISIBLE_DEVICES": "2"})


def test_opf_cpu_policy_blocks_all_backend_probes_and_gpu_requests():
    with mock.patch.dict(os.environ, {"OPF_ADP_DISABLE_GPU_ACCELERATORS": "1",
                                      "CUDA_VISIBLE_DEVICES": "0"}, clear=False), \
         mock.patch.object(runtime, "_safe_import", side_effect=AssertionError("GPU import during CPU admission")):
        probe = runtime.detect_backend("auto")
        assert probe.selected_device == "cpu"
        assert not probe.gpu_available
        with pytest.raises(runtime.BackendUnavailable, match="central CPU policy"):
            runtime.detect_backend("gpu")
        assert runtime.subprocess_environment("auto")["CUDA_VISIBLE_DEVICES"] == ""
        with pytest.raises(runtime.BackendUnavailable, match="central CPU admission"):
            runtime.subprocess_environment("gpu")


def test_torch_cuda_reports_available_but_allocation_fails():
    fake = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
        empty=mock.Mock(side_effect=RuntimeError("CUDA allocation failed")),
    )
    with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0",
                                      "OPF_ADP_DISABLE_GPU_ACCELERATORS": "0",
                                      "CPU_ONLY": "0", "TRAINING_CONTROL_CPU_ONLY": "0"}, clear=False), \
         mock.patch.object(runtime, "_safe_import", side_effect=lambda name: fake if name == "torch" else None), \
         mock.patch.object(runtime, "_nvidia_smi", return_value=True):
        probe = runtime.detect_backend("auto")
        assert probe.selected_device == "cpu"
        assert not probe.torch_cuda
        with pytest.raises(runtime.BackendUnavailable):
            runtime.detect_backend("gpu")


def test_cpu_environment_blocks_cuda_and_optional_library_acceleration():
    env = runtime.subprocess_environment("cpu", base={"CUDA_VISIBLE_DEVICES": "3"})
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["OPF_ADP_DISABLE_GPU_ACCELERATORS"] == "1"
    assert env["JAX_PLATFORMS"] == "cpu"
    assert env["TRAINING_CONTROL_CPU_ONLY"] == "1"
