#!/usr/bin/env python3
"""Hardened execution policy for deterministic post-producer job expansion.

This module reuses the v1 materialization/fingerprinting primitives but corrects
normal-run producer semantics: every real invocation re-enters the complete
non-training prerequisite closure so restart-exact producers can verify that
upstream inputs still match their published artifacts.  Audit, list and dry-run
modes remain side-effect-free with respect to producers and frozen expansion
state.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import universal_training_controller as base
import universal_training_controller_dag as dag
import universal_training_controller_deferred as core


def _ephemeral_materialize(
    root: Path,
    descriptors: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Materialize jobs for audit/list/dry-run without freezing repository state."""
    records: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for descriptor in descriptors:
        row = core._materialize_one(root, descriptor)
        generated = list(row.pop("records"))
        for record in generated:
            job_id = str(record.get("id"))
            if job_id in seen:
                raise SystemExit(f"duplicate generated job id across deferred expanders: {job_id}")
            seen.add(job_id)
            records.append(record)
        rows.append(row)
    return records, rows


def _pending_report(
    root: Path,
    profile: Dict[str, Any],
    static_records: Sequence[Mapping[str, Any]],
    descriptors: Sequence[Mapping[str, Any]],
    producer_ids: Sequence[str],
) -> Dict[str, Any]:
    report = base._coverage_report(root, profile, static_records)
    report = core._report_with_deferred(
        report,
        rows=[{"id": descriptor["id"]} for descriptor in descriptors],
        pending=True,
    )
    report["deferred_producer_job_ids"] = list(producer_ids)
    report["deferred_execution_note"] = (
        "Enumeration artifacts are absent. Audit/list/dry-run never execute producers; "
        "run the controller normally to materialize them."
    )
    core._atomic_json(root / ".training_control" / "coverage_report.json", report)
    return report


def _final_report(
    root: Path,
    profile: Dict[str, Any],
    full_records: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    producer_ids: Sequence[str],
) -> Dict[str, Any]:
    report = base._coverage_report(root, profile, full_records)
    report = core._report_with_deferred(report, rows=rows, pending=False)
    report["deferred_producer_job_ids"] = list(producer_ids)
    core._atomic_json(root / ".training_control" / "coverage_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    root = base._repo_root()
    profile = base._load_profile()
    descriptors = core._descriptors(profile)
    if not descriptors:
        return dag.main(argv)

    dag._install()
    flags, forwarded = base._custom_flags(list(sys.argv[1:] if argv is None else argv))
    if flags["skip_setup"]:
        os.environ["TRAINING_CONTROL_SKIP_SETUP"] = "1"

    static_records = base._job_records(root, profile)
    producer_ids = core._producer_closure(static_records, descriptors)
    core._validate_producers(static_records, producer_ids)
    producer_set = set(producer_ids)
    producer_records = [
        dict(record)
        for record in static_records
        if str(record.get("id")) in producer_set
    ]

    audit_only = bool(flags["audit"])
    list_only = bool(flags["list_jobs"])
    dry_run = dag._has_arg(forwarded, "--dry-run")
    diagnostic_mode = audit_only or list_only or dry_run

    if diagnostic_mode:
        if not core._artifacts_ready(root, descriptors):
            report = _pending_report(root, profile, static_records, descriptors, producer_ids)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 2 if audit_only else 0

        generated_records, rows = _ephemeral_materialize(root, descriptors)
        full_records = core._combine_records(static_records, generated_records)
        report = _final_report(root, profile, full_records, rows, producer_ids)
        print(json.dumps(report, indent=2, sort_keys=True))
        if audit_only:
            return 0 if report.get("coverage_ok") else 2
        if list_only:
            return 0
        strict = bool(profile.get("strict_coverage", True)) and not flags["allow_uncovered"]
        if strict and not report.get("coverage_ok"):
            return 2
        remaining = core._without_completed(full_records, producer_ids)
        if not remaining:
            return 0
        # Dry-run enumerates the expanded OPF jobs but never executes producer/setup code.
        return dag._execute(root, profile, remaining, core._stage_forwarded(forwarded, "expanded_dry_run"))

    # A real run always re-enters restart-exact producers.  This is intentional:
    # their own fingerprints are the authoritative check that already-published
    # enumeration artifacts still correspond to current upstream inputs/config.
    base._run_setup(root, profile)
    if producer_records:
        result = dag._execute(
            root,
            profile,
            producer_records,
            core._stage_forwarded(forwarded, "producers"),
        )
        if result != 0:
            return int(result)
    if not core._artifacts_ready(root, descriptors):
        raise SystemExit(
            "Deferred producers completed but required enumeration artifacts are still missing"
        )

    generated_records, rows = core._freeze_materializations(root, profile, descriptors)
    full_records = core._combine_records(static_records, generated_records)
    report = _final_report(root, profile, full_records, rows, producer_ids)
    strict = bool(profile.get("strict_coverage", True)) and not flags["allow_uncovered"]
    if strict and not report.get("coverage_ok"):
        raise SystemExit(
            f"Expanded training coverage audit failed. See {root / '.training_control' / 'coverage_report.json'}. "
            "Only non-training enumeration producers have run; no training job was launched."
        )

    remaining = core._without_completed(full_records, producer_ids)
    execution_state = {
        "schema": core.DEFERRED_SCHEMA,
        "repository": profile.get("repository") or root.name,
        "producer_job_ids": producer_ids,
        "generated_job_ids": [str(record.get("id")) for record in generated_records],
        "remaining_job_count": len(remaining),
        "coverage_ok": bool(report.get("coverage_ok")),
        "producer_reverified_this_invocation": True,
    }
    core._atomic_json(
        root / ".training_control" / core.EXECUTION_STATE_NAME,
        execution_state,
    )
    if not remaining:
        print("[training-control] deferred producer/expansion graph is complete; nothing remains")
        return 0
    return dag._execute(
        root,
        profile,
        remaining,
        core._stage_forwarded(forwarded, "expanded"),
    )
