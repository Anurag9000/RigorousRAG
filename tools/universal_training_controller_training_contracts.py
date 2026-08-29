#!/usr/bin/env python3
"""Phase-aware strict contracts for universal training jobs.

This layer changes no OPF_ADP scheduling behavior.  It only tightens the
repository-specific audit so a project may require interruption-exact resume and
early stopping for *training* jobs without incorrectly demanding early stopping
from dataset, audit, materialization, aggregation, or preprocessing jobs.

A training job may be exempted from early stopping only by declaring both
``early_stopping_applicable=False`` and a non-empty
``early_stopping_exception_reason``.  This keeps exemptions explicit and
auditable instead of silently treating missing evidence as acceptable.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Sequence

import universal_training_controller_current as current

TRAINING_CONTRACT_SCHEMA = 1
TRAINING_PHASES = {
    "training", "train", "pretrain", "pretraining", "finetune", "fine-tune",
    "fine_tune", "adapt", "fit", "rl", "reinforcement-learning",
    "reinforcement_learning", "policy-training", "policy_training",
}


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _is_training_job(job: Mapping[str, Any]) -> bool:
    explicit = job.get("is_training_job")
    if explicit is not None:
        return explicit is True
    phase = str(job.get("phase") or "").strip().lower()
    family = str(job.get("family") or "").strip().lower()
    return phase in TRAINING_PHASES or family in TRAINING_PHASES


def _job_index(jobs: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {str(job.get("id")): job for job in jobs}


def install() -> None:
    original_report = current._enhanced_coverage_report

    def coverage_report(root, profile: Dict[str, Any], jobs):
        report = original_report(root, profile, jobs)
        by_id = _job_index(jobs)
        resume_rows = {
            str(row.get("job_id")): row
            for row in (report.get("resume_audit") or [])
            if isinstance(row, Mapping)
        }

        training_ids = [
            str(job.get("id")) for job in jobs if _is_training_job(job)
        ]
        exact_missing = [
            job_id for job_id in training_ids
            if not bool((resume_rows.get(job_id) or {}).get("exact_resume_proven"))
        ]

        early_missing = []
        early_exemptions = []
        malformed_exemptions = []
        for job_id in training_ids:
            job = by_id[job_id]
            row = resume_rows.get(job_id) or {}
            if bool(row.get("early_stopping_present")):
                continue
            applicable = job.get("early_stopping_applicable")
            reason = str(job.get("early_stopping_exception_reason") or "").strip()
            if applicable is False and reason:
                early_exemptions.append({"job_id": job_id, "reason": reason})
            elif applicable is False and not reason:
                malformed_exemptions.append(job_id)
            else:
                early_missing.append(job_id)

        require_training_exact = bool(profile.get("require_training_exact_resume", False)) or _truthy_env(
            "TRAINING_CONTROL_REQUIRE_TRAINING_EXACT_RESUME"
        )
        require_training_early = bool(profile.get("require_training_early_stopping", False)) or _truthy_env(
            "TRAINING_CONTROL_REQUIRE_TRAINING_EARLY_STOPPING"
        )
        require_no_malformed_exemptions = bool(
            profile.get("require_well_formed_training_exemptions", True)
        )

        training_contracts_ok = (
            (not require_training_exact or not exact_missing)
            and (not require_training_early or not early_missing)
            and (not require_no_malformed_exemptions or not malformed_exemptions)
        )
        controls = dict(report.get("strict_controls") or {})
        controls.update(
            {
                "require_training_exact_resume": require_training_exact,
                "require_training_early_stopping": require_training_early,
                "require_well_formed_training_exemptions": require_no_malformed_exemptions,
            }
        )
        report.update(
            {
                "training_contract_schema": TRAINING_CONTRACT_SCHEMA,
                "training_job_ids": training_ids,
                "training_job_count": len(training_ids),
                "training_jobs_without_exact_resume": exact_missing,
                "training_jobs_without_early_stopping": early_missing,
                "training_early_stopping_exemptions": early_exemptions,
                "malformed_training_early_stopping_exemptions": malformed_exemptions,
                "strict_training_contracts_pass": training_contracts_ok,
                "strict_controls": controls,
                "opf_scheduler_binding": {
                    "implementation": "literal_pinned_OPF_ADP",
                    "repository": current.base.OPF_REFERENCE_REPOSITORY,
                    "commit": current.OPF_REFERENCE_COMMIT,
                    "runtime_blobs": dict(current.OPF_RUNTIME_BLOBS),
                    "scheduler_source_modified": False,
                },
            }
        )
        report["coverage_ok"] = bool(report.get("coverage_ok", False)) and training_contracts_ok
        return report

    current._enhanced_coverage_report = coverage_report
