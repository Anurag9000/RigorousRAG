from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import universal_training_controller as base
import universal_training_controller_opf_reference_v2 as reference


def _literal_scheduler(tmp_path: Path):
    reference.install()
    cache = base._prepare_opf_runtime(tmp_path)
    scheduler = base._import_opf_scheduler(cache)
    assert base._git_blob_sha((cache / "utils" / "opf_massive_suite_runner.py").read_bytes()) == (
        reference.OPF_RUNTIME_BLOBS["utils/opf_massive_suite_runner.py"]
    )
    return scheduler, cache


def _runtime(scheduler, tmp_path: Path):
    spec = scheduler.JobSpec(
        job_id="job-1",
        phase="training",
        family="test",
        case_name="case",
        repeat_index=0,
        depth=1,
        width=1,
        seed=1,
        results_dir=str(tmp_path / "results"),
        command=(sys.executable, "train.py"),
        device_capable=True,
    )
    return scheduler.JobRuntime(
        spec=spec,
        state_path=tmp_path / "job_state.json",
        log_path=tmp_path / "job.log",
    )


def _args(**overrides):
    values = {
        "cpu_only": False,
        "host_ram_resume_avail_mib": 1280.0,
        "gpu_resume_avail_mib": 500.0,
        "max_active_gpu_jobs": 0,
        "max_active_jobs": 0,
        "scheduler": "gpu_first",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_pressure_resume_requires_every_paused_resource_domain(tmp_path: Path) -> None:
    scheduler, _ = _literal_scheduler(tmp_path)
    kwargs = dict(
        pause_reason="gpu_memory_pressure+host_ram_pressure",
        device_mode="gpu",
        host_available_mib=400.0,
        gpu_free_mib=900.0,
        host_used_mib=15000.0,
        gpu_used_mib=7000.0,
        host_resume_avail_mib=1280.0,
        gpu_resume_avail_mib=500.0,
        host_peak_used_mib=15000.0,
        gpu_peak_used_mib=7000.0,
        external_drop_threshold_mib=500.0,
    )
    assert scheduler.pressure_pause_reopened(**kwargs) is False
    kwargs["host_available_mib"] = 1600.0
    assert scheduler.pressure_pause_reopened(**kwargs) is True


def test_external_process_memory_drop_reopens_matching_pressure_domain(tmp_path: Path) -> None:
    scheduler, _ = _literal_scheduler(tmp_path)
    assert scheduler.pressure_pause_reopened(
        pause_reason="host_ram_pressure",
        device_mode="cpu",
        host_available_mib=200.0,
        gpu_free_mib=None,
        host_used_mib=15000.0,
        gpu_used_mib=None,
        host_resume_avail_mib=1280.0,
        gpu_resume_avail_mib=500.0,
        host_peak_used_mib=15000.0,
        gpu_peak_used_mib=0.0,
        external_drop_threshold_mib=500.0,
        external_host_drop_mib=600.0,
    ) is True
    assert scheduler.pressure_pause_reopened(
        pause_reason="gpu_memory_pressure",
        device_mode="gpu",
        host_available_mib=4000.0,
        gpu_free_mib=100.0,
        host_used_mib=12000.0,
        gpu_used_mib=7900.0,
        host_resume_avail_mib=1280.0,
        gpu_resume_avail_mib=500.0,
        host_peak_used_mib=12000.0,
        gpu_peak_used_mib=7900.0,
        external_drop_threshold_mib=500.0,
        external_gpu_drop_mib=600.0,
    ) is True


def test_gpu_first_admission_and_cuda_oom_cpu_fallback_are_literal(tmp_path: Path) -> None:
    scheduler, _ = _literal_scheduler(tmp_path)
    runtime = _runtime(scheduler, tmp_path)
    host = scheduler.MemorySample(
        total_mib=16000,
        available_mib=6000,
        used_pct=62.5,
        swap_total_mib=0,
        swap_used_mib=0,
        swap_used_pct=0.0,
    )
    gpu = scheduler.GpuMemorySample(total_mib=8000, used_mib=2000, free_mib=6000, used_pct=25.0)

    assert scheduler.select_launch_device(runtime, _args(), host, gpu, 0, 0) == "gpu"
    assert scheduler.select_launch_device(
        runtime,
        _args(),
        host,
        gpu,
        0,
        0,
        gpu_admission_enabled=False,
    ) == "cpu"

    runtime.force_cpu_after_cuda_oom = True
    assert scheduler.select_launch_device(runtime, _args(), host, gpu, 0, 0) == "cpu"


def test_retry_budget_is_bounded(tmp_path: Path) -> None:
    scheduler, _ = _literal_scheduler(tmp_path)
    runtime = _runtime(scheduler, tmp_path)
    runtime.retry_count = 0
    assert scheduler.should_retry(runtime, max_retries=1) is True
    runtime.retry_count = 1
    assert scheduler.should_retry(runtime, max_retries=1) is False


def test_resume_launch_failure_is_persisted_as_paused_resume_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler, _ = _literal_scheduler(tmp_path)
    runtime = _runtime(scheduler, tmp_path)
    runtime.paused = True
    runtime.pause_reason = "gpu_memory_pressure"
    captured: dict = {}

    def fail_launch(*_args, **_kwargs):
        raise OSError("synthetic relaunch failure")

    monkeypatch.setattr(scheduler, "launch_job", fail_launch)
    monkeypatch.setattr(scheduler, "save_job_state", lambda _path, payload: captured.update(payload))
    monkeypatch.setattr(scheduler, "write_compat_state_views", lambda *_args, **_kwargs: None)

    with pytest.raises(OSError, match="synthetic relaunch failure"):
        scheduler._resume_runtime_from_pause(
            runtime,
            SimpleNamespace(),
            1,
            tmp_path,
            device_mode="gpu",
            active=[],
        )

    assert runtime.paused is True
    assert runtime.pause_reason == "gpu_memory_pressure"
    assert captured["status"] == "paused"
    assert captured["reason"] == "resume_launch_error"
    assert captured["pid"] is None


def test_literal_inline_gate_close_and_external_reopen_logic_remains_present(tmp_path: Path) -> None:
    _, cache = _literal_scheduler(tmp_path)
    text = (cache / "utils" / "opf_massive_suite_runner.py").read_text(encoding="utf-8")
    required = (
        "gpu_child_pressure = (last_child_gpu_vram_mib + gpu_buffer) > float(gpu_free_effective)",
        "host_child_pressure = (last_child_host_rss_mib + host_buffer) > float(host_available_effective)",
        "if not gpu_gate_closed and (gpu_pressure or gpu_child_pressure):",
        "if not host_gate_closed and (host_pressure or swap_pressure or host_child_pressure):",
        "if gpu_gate_closed and gpu_external_drop:",
        "if host_gate_closed and host_external_drop:",
    )
    for snippet in required:
        assert snippet in text
