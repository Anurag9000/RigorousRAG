#!/usr/bin/env python3
"""Compatibility entrypoint for release, canary, and disaster-recovery operations."""

from tools.operations_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
