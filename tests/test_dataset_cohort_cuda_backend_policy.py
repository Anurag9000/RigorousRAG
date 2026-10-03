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
    probe_tensor = SimpleNamespace(fill_=mock.Mock())
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            synchronize=mock.Mock(),
        ),
        empty=mock.Mock(return_value=probe_tensor),
    )
    with mock.patch.object(runtime, "_safe_import", side_effect=lambda name: torch if name == "torch" else np if name == "numpy" else None), \
         mock.patch.object(runtime, "_nvidia_smi", return_value=False), \
         mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2", "TRAINING_CONTROL_GPU_INDEX": "2"}, clear=False):
        assert runtime.detect_backend("auto").selected_device == "cuda:0"
        assert runtime.array_namespace() is np
    probe_tensor.fill_.assert_called()
    torch.cuda.synchronize.assert_called_with("cuda:0")


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


def test_declared_cpu_backend_blocks_all_framework_probes():
    with mock.patch.dict(os.environ, {
        "TRAINING_CONTROL_BACKEND": "cpu",
        "CUDA_VISIBLE_DEVICES": "0",
    }, clear=True), \
         mock.patch.object(runtime, "_safe_import",
                           side_effect=AssertionError("accelerator import during CPU admission")), \
         mock.patch.object(runtime, "_nvidia_smi",
                           side_effect=AssertionError("nvidia-smi during CPU admission")):
        probe = runtime.detect_backend("auto")
        assert probe.selected_device == "cpu"
        assert not probe.gpu_available
        with pytest.raises(runtime.BackendUnavailable, match="central CPU policy"):
            runtime.detect_backend("gpu")


def test_declared_gpu_backend_cannot_silently_fallback_to_cpu():
    with mock.patch.dict(os.environ, {
        "TRAINING_CONTROL_BACKEND": "gpu",
        "CUDA_VISIBLE_DEVICES": "0",
    }, clear=True), \
         mock.patch.object(runtime, "_safe_import", return_value=None), \
         mock.patch.object(runtime, "_nvidia_smi", return_value=True):
        with pytest.raises(runtime.BackendUnavailable, match="no usable GPU"):
            runtime.detect_backend("auto")
        with pytest.raises(runtime.BackendUnavailable, match="cannot silently select CPU"):
            runtime.detect_backend("cpu")


def test_conflicting_declared_gpu_and_cpu_flags_fail_before_probe():
    with mock.patch.dict(os.environ, {
        "TRAINING_CONTROL_BACKEND": "gpu",
        "TRAINING_CONTROL_CPU_ONLY": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }, clear=True), \
         mock.patch.object(runtime, "_safe_import") as imported:
        with pytest.raises(runtime.BackendUnavailable, match="conflicting central CPU and GPU"):
            runtime.detect_backend("auto")
        imported.assert_not_called()


def test_subprocess_environment_preserves_declared_backend_authority():
    cpu = runtime.subprocess_environment(
        "auto",
        base={"TRAINING_CONTROL_BACKEND": "cpu", "CUDA_VISIBLE_DEVICES": "0"},
    )
    assert cpu["TRAINING_CONTROL_BACKEND"] == "cpu"
    assert cpu["CUDA_VISIBLE_DEVICES"] == ""
    with pytest.raises(runtime.BackendUnavailable, match="GPU-admitted parent"):
        runtime.subprocess_environment(
            "cpu",
            base={"TRAINING_CONTROL_BACKEND": "gpu", "CUDA_VISIBLE_DEVICES": "3"},
        )


def test_torch_probe_requires_completed_operation_and_synchronization():
    tensor = SimpleNamespace(fill_=mock.Mock())
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            synchronize=mock.Mock(side_effect=RuntimeError("kernel failed")),
        ),
        empty=mock.Mock(return_value=tensor),
    )
    with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=True), \
         mock.patch.object(runtime, "_safe_import",
                           side_effect=lambda name: torch if name == "torch" else None), \
         mock.patch.object(runtime, "_nvidia_smi", return_value=True):
        probe = runtime.detect_backend("auto")
    assert not probe.torch_cuda
    assert probe.selected_device == "cpu"
    tensor.fill_.assert_called_once_with(1)
    torch.cuda.synchronize.assert_called_once_with("cuda:0")


def test_cupy_probe_requires_completed_operation_and_synchronization():
    array = SimpleNamespace(fill=mock.Mock())
    cp = SimpleNamespace(
        uint8=object(),
        empty=mock.Mock(return_value=array),
        cuda=SimpleNamespace(runtime=SimpleNamespace(
            getDeviceCount=lambda: 1,
            deviceSynchronize=mock.Mock(side_effect=RuntimeError("kernel failed")),
        )),
    )
    with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=True), \
         mock.patch.object(runtime, "_safe_import",
                           side_effect=lambda name: cp if name == "cupy" else None), \
         mock.patch.object(runtime, "_nvidia_smi", return_value=True):
        probe = runtime.detect_backend("auto")
    assert not probe.cupy_cuda
    assert probe.selected_device == "cpu"
    array.fill.assert_called_once_with(1)
    cp.cuda.runtime.deviceSynchronize.assert_called_once()


def test_cpu_environment_blocks_cuda_and_optional_library_acceleration():
    env = runtime.subprocess_environment("cpu", base={"CUDA_VISIBLE_DEVICES": "3"})
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["OPF_ADP_DISABLE_GPU_ACCELERATORS"] == "1"
    assert env["JAX_PLATFORMS"] == "cpu"
    assert env["TRAINING_CONTROL_CPU_ONLY"] == "1"
