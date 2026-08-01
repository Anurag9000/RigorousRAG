#!/usr/bin/env python3
"""Operator entrypoint for non-mutating migration promotion reports."""

from tools.migration_promotion_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
