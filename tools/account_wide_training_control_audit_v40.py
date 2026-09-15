#!/usr/bin/env python3
"""Estate-wide transactional dataset-cohort migration certificate v40.

v40 strengthens v39 without weakening any earlier OPF/scientific-authority gate.
An ACTIVE repository must use the exact blob-pinned transactional v2 runtime, keep
all five shared-batch controls in its root authority, and expose physical cohort
execution from that authority.  Repositories with native adapters that are not yet
root-promoted remain failures, but v40 records that partial implementation instead
of conflating it with a loader-only pin.

This remains source/configuration evidence only.  It does not claim model training
or benchmark execution occurred.
"""
from __future__ import annotations

from typing import Any

import account_wide_training_control_audit as base
import account_wide_training_control_audit_v38 as v38
import account_wide_training_control_audit_v39 as v39

CERTIFICATE_SCHEMA = 40
RUNTIME_V2_COMMIT = "0124a824d4fff32f4b427812a27cf7937d7c0891"
RUNTIME_V2_BLOB = "9364847538404151966b3719b10a1a9dc5f0f400"
RUNTIME_V1_COMMIT = "0c50adb23e5ba58b7c49b18401950f0bf7e5b736"
RUNTIME_V1_BLOB = "1ffb1af0f70812d3418a6f0959ae88c7a145939e"
_V39_REMOTE = v39._remote

# Native implementations completed during the migration.  These are *evidence*,
# not completion overrides: unless the root authority becomes ACTIVE the repo still
# fails the estate certificate.
_NATIVE_EVIDENCE: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "Anurag9000/RigorousRAG": (
        (
            "training/rigorousrag_retrieval_cohort.py",
            ("build_registry", "prepare_view", "save_checkpoint", "is_complete"),
        ),
        (
            "training/run_rigorousrag_retrieval_cohort.py",
            ("CohortExecutor", "run_plan", "--backend"),
        ),
    ),
    "Anurag9000/VaaniEventClean": (
        (
            "training_control/vaani_eventclean_enhancer_cohort_v1.py",
            ("build_registry", "prepare_view", "save_checkpoint", "is_complete"),
        ),
        (
            "training_control/run_vaani_eventclean_enhancer_cohort_v1.py",
            ("CohortExecutor", "run_plan", "--backend"),
        ),
    ),
    "Anurag9000/VaaniNoise-SED": (
        (
            "training_control/vaaninoise_prepared_cohort_v1.py",
            (
                "LOGICAL_DATASET",
                "build_registry",
                "prepare_view",
                "save_checkpoint",
                "is_complete",
                "transactional_batch_safe = True",
            ),
        ),
        (
            "training_control/run_vaaninoise_prepared_cohort_v1.py",
            ("CohortExecutor", "run_plan", "--backend"),
        ),
    ),
}


def _native_evidence(full_name: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path, markers in _NATIVE_EVIDENCE.get(full_name, ()):
        text = v39._try_raw(full_name, path)
        present = text is not None
        missing = [] if text is None else [marker for marker in markers if marker not in text]
        if not present:
            errors.append(f"missing declared native cohort surface: {path}")
        elif missing:
            errors.append(f"native cohort surface {path} lacks markers: {', '.join(missing)}")
        rows.append({"path": path, "present": present, "missing_markers": missing})
    return {
        "declared": bool(rows),
        "pass": bool(rows) and not errors,
        "rows": rows,
        "errors": errors,
    }


def _remote(repo_row: dict[str, Any]) -> dict[str, Any]:
    row = _V39_REMOTE(repo_row)
    full_name = str(repo_row.get("full_name") or "")
    errors = list(row.get("errors") or [])
    status = str(row.get("dataset_cohort_migration_status") or "UNCLASSIFIED")
    loader_path = row.get("dataset_cohort_loader")
    loader_text = v39._try_raw(full_name, str(loader_path)) if loader_path else None
    pin_exact = bool(
        loader_text
        and RUNTIME_V2_COMMIT in loader_text
        and RUNTIME_V2_BLOB in loader_text
        and RUNTIME_V1_COMMIT in loader_text
        and RUNTIME_V1_BLOB in loader_text
    )
    native = _native_evidence(full_name)

    if status == "ACTIVE":
        if row.get("dataset_cohort_runtime_generation") != "v2":
            errors.append("ACTIVE cohort authority is not using transactional runtime v2")
        if not pin_exact:
            errors.append("ACTIVE cohort runtime loader is not pinned to the exact v1+v2 commit/blob chain")
        launcher = v39._try_raw(full_name, "run_all_training.py") or ""
        if "require_dataset_cohort_execution" not in launcher:
            errors.append("ACTIVE root authority lost require_dataset_cohort_execution")
        if not any(token in launcher.lower() for token in ("cohort", "shared_batch", "shared-batch")):
            errors.append("ACTIVE root authority does not expose physical cohort execution")
    elif status == "PINNED_PENDING_ADAPTER":
        if row.get("dataset_cohort_runtime_generation") == "v2" and not pin_exact:
            errors.append("pending transactional runtime pin does not match the canonical v1+v2 commit/blob chain")
        if native.get("declared") and not native.get("pass"):
            errors.extend(str(value) for value in native.get("errors", []))

    if status == "N/A" and native.get("declared"):
        errors.append("N/A repository unexpectedly declares native cohort implementation evidence")

    implementation_state = (
        "ROOT_ACTIVE"
        if status == "ACTIVE"
        else "NATIVE_ADAPTER_PARTIAL"
        if native.get("pass")
        else "RUNTIME_PIN_ONLY"
        if status == "PINNED_PENDING_ADAPTER"
        else status
    )
    row.update(
        {
            "certificate_schema": CERTIFICATE_SCHEMA,
            "dataset_cohort_runtime_v2_exact_pin": pin_exact,
            "dataset_cohort_native_adapter_evidence": native,
            "dataset_cohort_implementation_state": implementation_state,
        }
    )
    row["errors"] = sorted(set(errors))
    row["pass"] = not row["errors"]
    return row


def main() -> int:
    v38.CERTIFICATE_SCHEMA = CERTIFICATE_SCHEMA
    v39.CERTIFICATE_SCHEMA = CERTIFICATE_SCHEMA
    base.SCHEMA = CERTIFICATE_SCHEMA
    v38._remote = _remote
    return v38.main()


if __name__ == "__main__":
    raise SystemExit(main())
