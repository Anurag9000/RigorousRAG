#!/usr/bin/env python3
"""Estate-wide dataset-cohort migration certificate v39.

v39 preserves every v38 scientific-authority, main-only topology, exact-resume,
early-stopping and literal-OPF requirement and adds an explicit migration state for
every owner repository except OPF_ADP.

A copied runtime loader is never treated as completion.  Every repository must be
exactly one of:

* ACTIVE: its root scientific authority physically schedules dataset-cohort jobs and
  requires the shared-batch CPU/GPU/exact-resume controls;
* PINNED_PENDING_ADAPTER: the immutable cohort runtime is present, but the native
  trainer/sample-stream adapter has not yet been promoted into the root authority;
* N/A: a source-backed certificate proves there is no local optimizer-training
  transaction to cohort; or
* BLOCKED: migration is explicitly blocked and therefore remains a certificate
  failure rather than disappearing from the estate report.

PINNED_PENDING_ADAPTER, BLOCKED and UNCLASSIFIED are hard failures.  This file is a
source/configuration audit only; it never claims that model training was executed.
"""
from __future__ import annotations

import re
from typing import Any

import account_wide_training_control_audit as base
import account_wide_training_control_audit_v38 as v38

CERTIFICATE_SCHEMA = 39
_V38_REMOTE = v38._remote

_BLOCKED: dict[str, str] = {
    "Anurag9000/existential-coordination-games": (
        "repository write was blocked by the connected GitHub action safety gate; "
        "no bypass is permitted, so the cohort migration remains explicitly open"
    ),
    "Anurag9000/IndoDocFusion": (
        "transactional v2 runtime repin was blocked by the connected GitHub action "
        "safety gate; no bypass is permitted, so its adapter promotion remains open"
    ),
}

_ACTIVE_TRUE = re.compile(
    r"[\"']require_dataset_cohort_execution[\"']\s*:\s*True\b"
)
_REQUIRED_ACTIVE_CONTROLS = (
    "require_dataset_cohort_execution",
    "require_cpu_gpu_backend_variants",
    "require_shared_batch_views",
    "require_uniform_cohort_batch_size",
    "require_cohort_exact_resume",
)
_LOADER_PATHS = (
    "training_control/dataset_cohort_runtime_entry.py",
    "tools/dataset_cohort_runtime_entry.py",
    "tools/dataset_cohort_runtime_entry_v2.py",
)
_NA_PATH = "training_control/dataset_cohort_not_applicable_v1.py"


def _try_raw(full_name: str, path: str) -> str | None:
    try:
        return base._raw(full_name, path, "main").decode("utf-8", errors="replace")
    except Exception:
        return None


def _loader_evidence(full_name: str) -> tuple[str | None, str | None, str | None]:
    for path in _LOADER_PATHS:
        text = _try_raw(full_name, path)
        if text is None:
            continue
        if "dataset_cohort_runtime_v2.py" in text or "RUNTIME_SCHEMA_V2" in text:
            generation = "v2"
        elif "0c50adb23e5ba58b7c49b18401950f0bf7e5b736" in text:
            generation = "v1"
        else:
            generation = "unknown"
        return path, generation, text
    return None, None, None


def _na_evidence(full_name: str) -> tuple[bool, str | None]:
    text = _try_raw(full_name, _NA_PATH)
    if text is None:
        return False, None
    valid = (
        'SCHEMA = "opf-dataset-cohort-not-applicable/v1"' in text
        and "APPLICABLE = False" in text
        and "require_literal_opf_mechanism_parity" in text
        and "require_all_retained_trainable_source_reachability" in text
    )
    return valid, text


def _remote(repo_row: dict[str, Any]) -> dict[str, Any]:
    row = _V38_REMOTE(repo_row)
    full_name = str(repo_row.get("full_name") or "")
    errors = list(row.get("errors") or [])
    launcher = _try_raw(full_name, "run_all_training.py") or ""
    loader_path, runtime_generation, _loader = _loader_evidence(full_name)
    na_valid, _na = _na_evidence(full_name)

    if full_name in _BLOCKED:
        status = "BLOCKED"
        reason = _BLOCKED[full_name]
        errors.append(f"dataset-cohort migration BLOCKED: {reason}")
    elif na_valid:
        status = "N/A"
        reason = "source-backed certificate proves no local optimizer-training surface"
        if loader_path:
            errors.append(
                "repository declares dataset-cohort N/A but also retains a cohort runtime loader; "
                "remove the contradictory migration surface or activate a trainer"
            )
    elif launcher and _ACTIVE_TRUE.search(launcher):
        status = "ACTIVE"
        missing = [key for key in _REQUIRED_ACTIVE_CONTROLS if key not in launcher]
        reason = "root scientific authority physically requires dataset-cohort execution"
        if missing:
            errors.append("active cohort authority lacks controls: " + ", ".join(missing))
        if loader_path is None:
            errors.append("active cohort authority has no immutable dataset-cohort runtime loader")
        if "cohort" not in launcher.lower():
            errors.append("active cohort authority does not visibly bind a physical cohort catalog/worker")
    elif loader_path:
        status = "PINNED_PENDING_ADAPTER"
        reason = (
            "canonical runtime is pinned but repository-native sample-stream/model adapters "
            "are not promoted into the root scientific authority"
        )
        errors.append("dataset-cohort migration remains PINNED_PENDING_ADAPTER")
    else:
        status = "UNCLASSIFIED"
        reason = "no active cohort authority, immutable loader, or valid N/A certificate was found"
        errors.append("dataset-cohort migration is UNCLASSIFIED")

    row.update(
        {
            "dataset_cohort_migration_status": status,
            "dataset_cohort_migration_reason": reason,
            "dataset_cohort_loader": loader_path,
            "dataset_cohort_runtime_generation": runtime_generation,
            "dataset_cohort_na_certificate": _NA_PATH if na_valid else None,
            "dataset_cohort_migration_complete": status in {"ACTIVE", "N/A"},
            "certificate_schema": CERTIFICATE_SCHEMA,
        }
    )
    row["errors"] = sorted(set(errors))
    row["pass"] = not row["errors"]
    return row


def main() -> int:
    v38.CERTIFICATE_SCHEMA = CERTIFICATE_SCHEMA
    base.SCHEMA = CERTIFICATE_SCHEMA
    v38._remote = _remote
    return v38.main()


if __name__ == "__main__":
    raise SystemExit(main())
