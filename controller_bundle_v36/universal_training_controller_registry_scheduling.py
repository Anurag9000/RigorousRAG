#!/usr/bin/env python3
"""Correct strict registry scheduling for curated and auto-generated jobs.

Older registry layers only subtracted compiled jobs when auto-materialization was
enabled. That incorrectly failed a repository that deliberately expands one
registered training surface into multiple curated jobs (for example one job per
cross-validation fold). This layer changes no discovery and no scheduling; it
only evaluates the strict registry scheduling invariant as ``expected -
scheduled`` regardless of how the jobs were constructed.

A parent console command whose actual training authority is a scheduled argparse
subcommand is already covered by that child.  Such a parent remains visible in
the registry as ``satisfied_by_subcommand`` but is not independently required as
a duplicate executable job.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import universal_training_controller_current as current

REGISTRY_SCHEDULING_SCHEMA = 2


def _active_expected_console(report: Mapping[str, Any]) -> set[str]:
    registry = report.get("console_registry") or {}
    entries = registry.get("training_entries", []) if isinstance(registry, Mapping) else []
    expected: set[str] = set()
    for item in entries:
        if not isinstance(item, Mapping) or item.get("ignored") or not item.get("source"):
            continue
        if item.get("satisfied_by_subcommand"):
            continue
        if item.get("configured"):
            expected.add(str(item.get("name")))
    return expected


def _active_expected_subcommands(report: Mapping[str, Any]) -> set[str]:
    registry = report.get("console_subcommand_registry") or {}
    entries = registry.get("training_subcommands", []) if isinstance(registry, Mapping) else []
    expected: set[str] = set()
    for item in entries:
        if not isinstance(item, Mapping) or item.get("ignored") or not item.get("source"):
            continue
        if item.get("configured"):
            expected.add(str(item.get("key")))
    return expected


def _scheduled_console(jobs: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(job.get("console_script"))
        for job in jobs
        if job.get("console_script") and not job.get("console_subcommand")
    }


def _scheduled_subcommands(jobs: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        f"{job.get('console_script')}:{job.get('console_subcommand')}"
        for job in jobs
        if job.get("console_script") and job.get("console_subcommand")
    }


def install() -> None:
    original_report = current._enhanced_coverage_report

    def coverage_report(root, profile: Dict[str, Any], jobs):
        require_console = bool(profile.get("require_registered_training_scheduling", False))
        require_subcommands = bool(
            profile.get("require_registered_training_subcommand_scheduling", False)
        )
        # Evaluate every pre-existing strict invariant, but suppress only the two
        # legacy scheduling checks whose set-subtraction semantics are corrected
        # below. Auto-materialization remains untouched.
        relaxed = dict(profile)
        relaxed["require_registered_training_scheduling"] = False
        relaxed["require_registered_training_subcommand_scheduling"] = False
        report = original_report(root, relaxed, jobs)

        expected_console = _active_expected_console(report)
        expected_subcommands = _active_expected_subcommands(report)
        scheduled_console = _scheduled_console(jobs)
        scheduled_subcommands = _scheduled_subcommands(jobs)
        missing_console = sorted(expected_console - scheduled_console) if require_console else []
        missing_subcommands = (
            sorted(expected_subcommands - scheduled_subcommands)
            if require_subcommands
            else []
        )

        console_registry = report.get("console_registry") or {}
        sub_registry = report.get("console_subcommand_registry") or {}
        console_unresolved = list(console_registry.get("unresolved_targets", [])) if isinstance(console_registry, Mapping) else []
        console_unconfigured = list(console_registry.get("unconfigured_training_entrypoints", [])) if isinstance(console_registry, Mapping) else []
        sub_unresolved = list(sub_registry.get("unresolved_targets", [])) if isinstance(sub_registry, Mapping) else []
        sub_unconfigured = list(sub_registry.get("unconfigured_training_subcommands", [])) if isinstance(sub_registry, Mapping) else []

        console_ok = not console_unresolved and not console_unconfigured and not missing_console
        subcommands_ok = not sub_unresolved and not sub_unconfigured and not missing_subcommands
        strict_console = bool(profile.get("require_registered_training_entrypoints", True))
        strict_subcommands = bool(profile.get("require_registered_training_subcommands", True))

        # The relaxed inner report is authoritative for every non-registry-
        # scheduling blocker. Reintroduce only the corrected registry conditions.
        coverage_ok = bool(report.get("coverage_ok", False))
        if strict_console and not console_ok:
            coverage_ok = False
        if strict_subcommands and not subcommands_ok:
            coverage_ok = False

        controls = dict(report.get("strict_controls") or {})
        controls.update(
            {
                "require_registered_training_scheduling": require_console,
                "require_registered_training_subcommand_scheduling": require_subcommands,
            }
        )
        report.update(
            {
                "registry_scheduling_schema": REGISTRY_SCHEDULING_SCHEMA,
                "compiled_console_training_jobs": sorted(scheduled_console),
                "compiled_console_training_subcommands": sorted(scheduled_subcommands),
                "missing_console_job_materialization": missing_console,
                "missing_console_subcommand_job_materialization": missing_subcommands,
                "strict_registered_training_entrypoints_pass": console_ok,
                "strict_registered_training_subcommands_pass": subcommands_ok,
                "strict_controls": controls,
                "coverage_ok": coverage_ok,
            }
        )
        return report

    current._enhanced_coverage_report = coverage_report
