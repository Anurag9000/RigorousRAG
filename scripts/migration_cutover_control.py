#!/usr/bin/env python3
"""Operator entrypoint for preparation-only migration cutover control."""

from tools.migration_cutover_control_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
