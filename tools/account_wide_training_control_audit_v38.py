#!/usr/bin/env python3
"""Account-wide static source certificate for exhaustive training-control v38.

v38 inherits the complete v37/v34 retained-trainable scientific closure and adds a
new estate invariant: a canonical controller pin is not sufficient by itself.
Every live owner repository except OPF_ADP must expose a first-class repository-
owned scientific authority from its root launcher.  That authority may be:

* a trainable scientific/job DAG;
* a deterministic non-optimizer scientific lifecycle; or
* a fail-closed no-trainable-surface certificate.

Opaque preservation-adapter roots are rejected.  This prevents a repository from
appearing estate-complete merely because it can recover some historical launcher.
The launcher must name its current scientific authority explicitly while still
pinning the canonical v37 controller and satisfying all inherited local source,
model, dataset/task/loss/optimizer/scheduler/sampler/augmentation/config/registry/
combination/ensemble/workflow and exact-resume/early-stopping contracts.

This remains a source/configuration certificate. It does not execute model training
or claim empirical/runtime correctness.
"""
from __future__ import annotations

from typing import Any

import account_wide_training_control_audit as base
import account_wide_training_control_audit_v34 as retained

CANONICAL_BOOTSTRAP_COMMIT = "fd34a95d18892df7fb14d1efbb99076a7810fb91"
CANONICAL_BOOTSTRAP_BLOB = "05ef472b29933f18e956c69dfb7e543921ddaff5"
CANONICAL_ADAPTER_COMMIT = "ab4ad585a3cfd0b0350a163983ead9ad7a51d559"
CANONICAL_ADAPTER_BLOB = "046ef85de71ac2d9852cd43d615d4db8c8ee5f1c"
CERTIFICATE_SCHEMA = 38

_AUTHORITY_MARKERS = (
    '"scientific_authority"',
    "'scientific_authority'",
    "SCIENTIFIC_AUTHORITY",
)
_FORBIDDEN_OPAQUE_ROOT_MARKERS = (
    "TRAINING_LAUNCHER_BASE_REPOSITORY",
    "TRAINING_LAUNCHER_BASE_COMMIT",
    "repo_training_launcher_adapter.py",
)


def _remote(repo_row: dict[str, Any]) -> dict[str, Any]:
    retained.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    retained.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    retained.CANONICAL_ADAPTER_COMMIT = CANONICAL_ADAPTER_COMMIT
    retained.CANONICAL_ADAPTER_BLOB = CANONICAL_ADAPTER_BLOB
    row = retained._remote(repo_row)
    full_name = str(repo_row.get("full_name") or "")
    errors = list(row.get("errors") or [])
    authority_kind = "unknown"
    launcher_text = ""
    try:
        launcher_text = base._raw(full_name, "run_all_training.py", "main").decode(
            "utf-8", errors="replace"
        )
    except Exception as exc:
        errors.append(f"v38 cannot inspect root scientific authority: {exc}")

    if launcher_text:
        if not any(marker in launcher_text for marker in _AUTHORITY_MARKERS):
            errors.append("root launcher does not declare a repository-owned scientific_authority")
        opaque = [marker for marker in _FORBIDDEN_OPAQUE_ROOT_MARKERS if marker in launcher_text]
        if opaque:
            errors.append(
                "root launcher still delegates through opaque preservation adapter markers: "
                + ", ".join(opaque)
            )

        lower = launcher_text.lower()
        if "no_trainable" in lower or "no-trainable" in lower or "no authored optimizer" in lower:
            authority_kind = "no_trainable_surface"
        elif "is_training_job" in launcher_text or "job_catalog" in launcher_text or "training" in lower:
            authority_kind = "scientific_dag"
        else:
            authority_kind = "non_optimizer_scientific_lifecycle"

        # Strict roots must not turn the repository-owned authority into generic
        # auto-discovery again. Explicit catalog/authority logic remains primary.
        if '"strict_coverage": True' not in launcher_text and "'strict_coverage': True" not in launcher_text:
            errors.append("root scientific authority does not visibly enable strict_coverage")
        if "require_literal_opf_mechanism_parity" not in launcher_text:
            errors.append("root scientific authority does not require literal OPF mechanism parity")
        if "require_all_retained_trainable_source_reachability" not in launcher_text:
            errors.append("root scientific authority does not require retained trainable-source reachability")

    row["scientific_authority_kind"] = authority_kind
    row["first_class_scientific_authority"] = bool(launcher_text) and not errors
    row["opaque_preservation_adapter_root"] = bool(
        launcher_text
        and any(marker in launcher_text for marker in _FORBIDDEN_OPAQUE_ROOT_MARKERS)
    )
    row["errors"] = sorted(set(errors))
    row["pass"] = not row["errors"]
    row["certificate_schema"] = CERTIFICATE_SCHEMA
    return row


def _strict(report: dict[str, Any]) -> list[str]:
    # Retain the complete v34 local retained-source scientific closure unchanged.
    return retained._strict(report)


def main() -> int:
    base.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    base.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    base.SCHEMA = CERTIFICATE_SCHEMA
    base._launcher_remote_audit = _remote
    base._strict_local_report = _strict
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
