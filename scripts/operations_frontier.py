#!/usr/bin/env python3
"""Compatibility entrypoint for release, canary, and disaster-recovery operations."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from tools.operations_cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
