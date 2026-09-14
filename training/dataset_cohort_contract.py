"""Fail-closed scientific contract for dataset-cohort training.

The contract converts repository-owned training-job metadata into the canonical
runtime's ``ModelSpec`` records. It is intentionally strict: a retained training
job may not become cohort-scheduled unless its dataset identity, model family,
view contract, physical batch size, backend capabilities and exact-resume adapter
are explicit. Non-training jobs are left to the outer OPF DAG unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.dataset_cohort_runtime_entry import load_runtime

SCHEMA = "dataset-cohort-scientific-contract/v1"
REQUIRED_PROFILE_CONTROLS = (
    "require_dataset_cohort_execution",
    "require_cpu_gpu_backend_variants",
    "require_shared_batch_views",
    "require_uniform_cohort_batch_size",
    "require_cohort_exact_resume",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _strings(value: Any, label: str) -> tuple[str, ...]:
    raw = [value] if isinstance(value, str) else value
    if not isinstance(raw, Sequence) or isinstance(raw, (bytes, bytearray)):
        raise ValueError(f"{label} must be a string or string array")
    result = tuple(dict.fromkeys(str(item).strip() for item in raw if str(item).strip()))
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        selected = int(value)
    except Exception as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if selected <= 0:
        raise ValueError(f"{label} must be positive")
    return selected


def training_jobs(jobs: Iterable[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    return tuple(row for row in jobs if row.get("is_training_job") is True or str(row.get("phase") or "").lower() == "training")


def model_spec_from_job(job: Mapping[str, Any]) -> Any:
    runtime = load_runtime()
    job_id = str(job.get("id") or job.get("job_id") or job.get("name") or "").strip()
    if not job_id:
        raise ValueError("training job has no stable id")
    family = str(job.get("model_family") or job.get("family") or job.get("architecture") or "").strip()
    if not family:
        raise ValueError(f"{job_id}: model family is missing")
    datasets = _strings(job.get("datasets", job.get("dataset")), f"{job_id}.datasets")
    view_key = str(job.get("cohort_view_key") or "").strip()
    if not view_key:
        raise ValueError(f"{job_id}: cohort_view_key is required; raw dataset identity alone cannot prove tensor compatibility")
    batch_size = _positive_int(job.get("cohort_batch_size", job.get("batch_size")), f"{job_id}.cohort_batch_size")
    adapter = str(job.get("cohort_adapter") or "").strip()
    if not adapter:
        raise ValueError(f"{job_id}: cohort_adapter is required")
    if job.get("cpu_capable") is not True:
        raise ValueError(f"{job_id}: explicit cpu_capable=true is required")
    device_capable = bool(job.get("device_capable", False))
    gpu_capable = bool(job.get("gpu_capable", device_capable))
    if device_capable and not gpu_capable:
        raise ValueError(f"{job_id}: device-capable training requires gpu_capable=true")
    checkpoint = job.get("checkpoint_contract")
    exact = bool(job.get("exact_resume", False)) or (
        isinstance(checkpoint, Mapping) and checkpoint.get("exact_resume") is True
    ) or str(job.get("resume_strategy") or "") in {"exact_checkpoint", "native_transactional_state"}
    if not exact:
        raise ValueError(f"{job_id}: cohort training requires exact resume")
    metadata = {
        "task": job.get("task"),
        "architecture": job.get("architecture"),
        "model": job.get("model"),
        "recipe": job.get("recipe"),
        "recipe_config": job.get("recipe_config") or job.get("config") or job.get("config_path"),
        "source": job.get("entrypoint_source"),
        "command": list(job.get("command") or ()),
        "repeat_index": job.get("repeat_index"),
        "restart_exact_monolithic": bool(job.get("cohort_restart_exact_monolithic", False)),
    }
    return runtime.ModelSpec(
        job_id=job_id,
        model_family=family,
        datasets=datasets,
        view_key=view_key,
        batch_size=batch_size,
        adapter=adapter,
        device_capable=device_capable,
        cpu_capable=True,
        gpu_capable=gpu_capable or not device_capable,
        exact_resume=True,
        metadata=metadata,
    )


def compile_plan(jobs: Iterable[Mapping[str, Any]], *, batch_size_override: int | None = None) -> Any:
    runtime = load_runtime()
    specs = tuple(model_spec_from_job(job) for job in training_jobs(jobs))
    return runtime.build_cohort_plan(specs, batch_size_override=batch_size_override)


def validate_profile(profile: Mapping[str, Any], jobs: Iterable[Mapping[str, Any]], registry: Any) -> dict[str, Any]:
    errors: list[str] = []
    for key in REQUIRED_PROFILE_CONTROLS:
        if profile.get(key) is not True:
            errors.append(f"strict control absent/disabled: {key}")
    try:
        selected = training_jobs(jobs)
        plan = compile_plan(selected)
        audit = registry.audit(tuple(model_spec_from_job(job) for job in selected), plan)
        errors.extend(str(value) for value in audit.get("errors", ()))
    except Exception as exc:
        plan = None
        audit = None
        errors.append(str(exc))
    return {
        "schema": SCHEMA,
        "pass": not errors,
        "errors": sorted(set(errors)),
        "training_job_count": len(training_jobs(jobs)),
        "plan": None if plan is None else plan.to_dict(),
        "adapter_audit": audit,
    }


def write_certificate(path: str | Path, report: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload["certificate_sha256"] = _digest(report)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(_canonical(payload) + b"\n")
    temporary.replace(destination)
    return destination


__all__ = ["REQUIRED_PROFILE_CONTROLS", "SCHEMA", "compile_plan", "model_spec_from_job", "training_jobs", "validate_profile", "write_certificate"]
