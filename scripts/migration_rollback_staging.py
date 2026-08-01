#!/usr/bin/env python3
"""Operator entrypoint for non-authoritative rollback staging verification."""

from tools.migration_rollback_staging_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
