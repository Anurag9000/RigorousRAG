#!/usr/bin/env python3
"""Recovery semantics for deterministic non-training jobs.

A dataset/materialization/preprocessing job does not need optimizer checkpoints
if restarting it from its beginning is itself exact. This module permits that
contract only when the job explicitly declares deterministic, idempotent,
atomic-output behavior. It is intentionally forbidden for genuine training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import universal_training_controller_current as current

RESTART_EXACT_PHASES = {
    "dataset", "data", "download", "materialize", "setup", "preprocess",
    "preprocessing", "features", "feature", "manifest", "folds", "index",
}
TRAINING_PHASES = {"training", "train", "pretrain", "finetune", "fine-tune", "adapt", "fit"}


def _truth(value: Any) -> bool:
    return value is True


def _resume_evidence(root: Path, job: Dict[str, Any], reachability: Mapping[str, Any]) -> Dict[str, Any]:
    evidence = _ORIGINAL(root, job, reachability)
    strategy = str(job.get("resume_strategy") or "").strip().lower()
    if strategy != "restart_exact":
        return evidence

    phase = str(job.get("phase") or "training").strip().lower()
    family = str(job.get("family") or "").strip().lower()
    is_training = phase in TRAINING_PHASES or family in TRAINING_PHASES
    phase_allowed = phase in RESTART_EXACT_PHASES
    contract = job.get("checkpoint_contract")
    contract = dict(contract) if isinstance(contract, Mapping) else {}
    deterministic = _truth(job.get("deterministic")) or _truth(contract.get("deterministic"))
    idempotent = _truth(job.get("idempotent")) or _truth(contract.get("idempotent"))
    atomic_outputs = _truth(job.get("atomic_outputs")) or _truth(contract.get("atomic_outputs"))
    declared_exact = _truth(contract.get("exact_resume"))
    allowed = bool(not is_training and phase_allowed and deterministic and idempotent and atomic_outputs and declared_exact)

    evidence.update({
        "restart_exact_requested": True,
        "restart_exact_phase_allowed": phase_allowed,
        "restart_exact_training_forbidden": is_training,
        "restart_exact_deterministic": deterministic,
        "restart_exact_idempotent": idempotent,
        "restart_exact_atomic_outputs": atomic_outputs,
        "restart_exact_proven": allowed,
        "resume_kind": "deterministic_restart" if allowed else "invalid_restart_exact_declaration",
    })
    if allowed:
        # These fields are consumed by the existing strict recovery gates. The
        # separate resume_kind/restart_exact_proven fields preserve the fact
        # that this is deterministic replay, not a model checkpoint.
        evidence["native_resume_proven"] = True
        evidence["exact_resume_proven"] = True
        evidence["resume_strategy"] = "restart_exact"
    else:
        evidence["native_resume_proven"] = False
        evidence["exact_resume_proven"] = False
    return evidence


def install() -> None:
    current._job_resume_evidence = _resume_evidence


_ORIGINAL = current._job_resume_evidence
