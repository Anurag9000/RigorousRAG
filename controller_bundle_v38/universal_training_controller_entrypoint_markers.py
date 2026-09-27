#!/usr/bin/env python3
"""Expose generated console-job source paths to the core reachability scanner.

PEP-621/argparse training jobs are invoked through a small ``python -c`` import
trampoline so package-relative imports behave exactly like the installed console
entrypoint.  The core controller intentionally discovers source roots from
literal command tokens, therefore the trampoline also carries its exact
repository ``entrypoint_source`` as a marker argument.  The marker is removed
from ``sys.argv`` before the real CLI is called, so user-visible argument
semantics are unchanged while the generic reachability/resume auditor sees the
actual source file without monkey-patching core graph functions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import universal_training_controller_subcommands as subcommands

MARKER_SCHEMA = 1
_MARKER_PREFIX = "import sys;sys.argv.pop(1);"


def _mark_job(root: Path, job: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(job)
    if not result.get("console_subcommand"):
        return result
    rel = str(result.get("entrypoint_source") or "").replace("\\", "/").strip("/")
    if not rel or not (root / rel).is_file():
        return result
    command = [str(x) for x in result.get("command", [])]
    if len(command) < 3 or command[1] != "-c":
        return result
    # Idempotent if another catalog layer already applied the marker.
    if len(command) >= 4 and command[3] == rel and command[2].startswith(_MARKER_PREFIX):
        return result
    command[2] = _MARKER_PREFIX + command[2]
    command.insert(3, rel)
    result["command"] = command
    result["entrypoint_marker_schema"] = MARKER_SCHEMA
    return result


def _jobs(root: Path, profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [_mark_job(root, job) for job in _ORIGINAL_JOBS(root, profile)]


def install() -> None:
    subcommands._jobs = _jobs


_ORIGINAL_JOBS = subcommands._jobs

__all__ = ["MARKER_SCHEMA", "_mark_job", "install"]
