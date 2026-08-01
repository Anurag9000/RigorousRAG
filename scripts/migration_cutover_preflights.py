#!/usr/bin/env python3
"""Operator entrypoint for non-mutating migration cutover preflights."""

from tools.migration_cutover_preflight_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
