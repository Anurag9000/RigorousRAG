#!/usr/bin/env python3
"""Account-wide static source certificate for exhaustive training-control v38.

v38 inherits the complete v37/v34 retained-trainable scientific closure and adds
three estate invariants:

1. a canonical controller pin is not sufficient by itself; every live owner
   repository except OPF_ADP must expose a first-class repository-owned scientific
   authority from its root launcher;
2. the development topology must match the direct-to-main contract: ``main`` is the
   default branch and no additional live branch refs remain; and
3. the literal OPF_ADP scheduler reference must still be the live ``main`` head.
   Any future OPF scheduler change therefore invalidates this certificate until the
   shared controller is deliberately reviewed, repinned and revalidated.

A scientific authority may be a trainable scientific/job DAG, a deterministic
non-optimizer scientific lifecycle, or a fail-closed no-trainable-surface
certificate. Opaque preservation-adapter roots are rejected. This prevents a
repository from appearing estate-complete merely because it can recover some
historical launcher.

The launcher must name its current scientific authority explicitly while still
pinning canonical v37 and satisfying all inherited local source, model,
dataset/task/loss/optimizer/scheduler/sampler/augmentation/config/registry/
combination/ensemble/workflow and exact-resume/early-stopping contracts.

This remains a source/configuration certificate. It does not execute model training
or claim empirical/runtime correctness.
"""
from __future__ import annotations

import re
import sys
from typing import Any

import account_wide_training_control_audit as base
import account_wide_training_control_audit_v34 as retained

CANONICAL_BOOTSTRAP_COMMIT = "fd34a95d18892df7fb14d1efbb99076a7810fb91"
CANONICAL_BOOTSTRAP_BLOB = "05ef472b29933f18e956c69dfb7e543921ddaff5"
CANONICAL_ADAPTER_COMMIT = "ab4ad585a3cfd0b0350a163983ead9ad7a51d559"
CANONICAL_ADAPTER_BLOB = "046ef85de71ac2d9852cd43d615d4db8c8ee5f1c"
LITERAL_OPF_REFERENCE_COMMIT = "1d1dfbbf7521ac40ee60c1f78f84956bf5f70598"
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
_STRICT_TRUE = re.compile(r"[\"']strict_coverage[\"']\s*:\s*True\b")


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

    # The requested estate topology is main-only. The base audit records these
    # fields but older schemas treated them as informational; v38 makes them hard
    # certificate conditions.
    default_branch = str(row.get("default_branch") or repo_row.get("default_branch") or "")
    if default_branch != "main":
        errors.append(f"default branch is not main: {default_branch!r}")
    extra_branches = [str(value) for value in (row.get("extra_branches") or []) if str(value)]
    if extra_branches:
        errors.append("non-main branch refs remain: " + ", ".join(sorted(set(extra_branches))))

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
        # auto-discovery again.  Accept normal Python formatting, compact literals,
        # and either quote style while still requiring a literal True value.
        if _STRICT_TRUE.search(launcher_text) is None:
            errors.append("root scientific authority does not visibly enable strict_coverage=True")
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
    row["main_only_topology"] = default_branch == "main" and not extra_branches
    row["errors"] = sorted(set(errors))
    row["pass"] = not row["errors"]
    row["certificate_schema"] = CERTIFICATE_SCHEMA
    return row


def _strict(report: dict[str, Any]) -> list[str]:
    # Retain the complete v34 local retained-source scientific closure unchanged.
    errors = list(retained._strict(report))
    controls = report.get("strict_controls") or {}
    if not isinstance(controls, dict):
        errors.append("strict_controls missing/not an object")
    else:
        for key in (
            "require_literal_opf_mechanism_parity",
            "require_all_retained_trainable_source_reachability",
            "require_source_proven_training_exact_resume",
            "require_source_proven_training_early_stopping",
            "require_full_scientific_choice_accounting",
            "require_role_paradigm_protocol_accounting",
        ):
            if controls.get(key) is not True:
                errors.append(f"strict control absent/disabled: {key}")
    return sorted(set(errors))


def main() -> int:
    # Estate certification is always relative to the live literal OPF scheduler.
    # A new OPF commit intentionally makes v38 fail until the shared authority is
    # reviewed and repinned; this prevents a stale "ditto" claim across 40 repos.
    live_opf = base._main_sha(f"{base.OWNER}/{base.REFERENCE_REPO}")
    if live_opf != LITERAL_OPF_REFERENCE_COMMIT:
        print(
            "v38 refuses stale OPF parity: "
            f"live OPF_ADP/main={live_opf!r}, pinned={LITERAL_OPF_REFERENCE_COMMIT}",
            file=sys.stderr,
        )
        return 2

    base.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    base.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    base.SCHEMA = CERTIFICATE_SCHEMA
    base._launcher_remote_audit = _remote
    base._strict_local_report = _strict
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
