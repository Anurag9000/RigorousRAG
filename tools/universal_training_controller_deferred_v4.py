#!/usr/bin/env python3
"""Repository-root execution boundary for deferred job expanders.

Deferred descriptors are repository configuration, so relative arguments passed to
an expander must have repository-root semantics regardless of the process working
directory used to launch the controller.  v4 wraps the existing immutable
materializer instead of modifying the published v18 layers: it temporarily runs
materialization from the repository root and restores the caller's CWD in a
``finally`` block, including when an expander raises.

No OPF scheduling or resource-admission behavior is changed here.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import universal_training_controller_deferred as core
import universal_training_controller_deferred_v3 as strict


def _repository_root_materialize(
    root: Path,
    descriptor: Mapping[str, Any],
):
    original_cwd = Path.cwd()
    resolved_root = Path(root).resolve()
    try:
        os.chdir(resolved_root)
        return _ORIGINAL_MATERIALIZE(resolved_root, descriptor)
    finally:
        os.chdir(original_cwd)


_ORIGINAL_MATERIALIZE = core._materialize_one


def main() -> int:
    original = core._materialize_one
    core._materialize_one = _repository_root_materialize
    try:
        return strict.main()
    finally:
        core._materialize_one = original


if __name__ == "__main__":
    raise SystemExit(main())
