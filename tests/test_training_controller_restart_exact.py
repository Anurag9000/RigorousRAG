from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import universal_training_controller_restart_exact as restart_exact


def _base_evidence(root: Path, job: dict, reachability: dict) -> dict:
    return {
        "job_id": job["id"],
        "resume_strategy": job.get("resume_strategy") or "unproven",
        "native_resume_proven": False,
        "exact_resume_proven": False,
        "early_stopping_present": False,
        "checkpoint_contract": {},
        "declared_checkpoint_contract": job.get("checkpoint_contract"),
    }


def _evaluate(tmp_path: Path, job: dict) -> dict:
    original = restart_exact._ORIGINAL
    try:
        restart_exact._ORIGINAL = _base_evidence
        return restart_exact._resume_evidence(tmp_path, job, {})
    finally:
        restart_exact._ORIGINAL = original


def test_restart_exact_allows_only_explicit_deterministic_data_jobs(tmp_path: Path) -> None:
    job = {
        "id": "dataset",
        "phase": "dataset",
        "resume_strategy": "restart_exact",
        "deterministic": True,
        "idempotent": True,
        "atomic_outputs": True,
        "checkpoint_contract": {"exact_resume": True},
    }
    evidence = _evaluate(tmp_path, job)
    assert evidence["restart_exact_proven"] is True
    assert evidence["native_resume_proven"] is True
    assert evidence["exact_resume_proven"] is True
    assert evidence["resume_kind"] == "deterministic_restart"


def test_restart_exact_is_forbidden_for_training(tmp_path: Path) -> None:
    job = {
        "id": "trainer",
        "phase": "training",
        "resume_strategy": "restart_exact",
        "deterministic": True,
        "idempotent": True,
        "atomic_outputs": True,
        "checkpoint_contract": {"exact_resume": True},
    }
    evidence = _evaluate(tmp_path, job)
    assert evidence["restart_exact_proven"] is False
    assert evidence["restart_exact_training_forbidden"] is True
    assert evidence["exact_resume_proven"] is False


def test_restart_exact_requires_all_output_safety_properties(tmp_path: Path) -> None:
    job = {
        "id": "features",
        "phase": "features",
        "resume_strategy": "restart_exact",
        "deterministic": True,
        "idempotent": True,
        "atomic_outputs": False,
        "checkpoint_contract": {"exact_resume": True},
    }
    evidence = _evaluate(tmp_path, job)
    assert evidence["restart_exact_proven"] is False
    assert evidence["exact_resume_proven"] is False
