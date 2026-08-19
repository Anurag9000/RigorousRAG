"""Compatibility shim for the authoritative governed grounded import CLI.

The previous module contained an independent publication implementation.  Keeping two
executable writers weakens path and identity authority, so the public module now delegates to
the final staged/path-safe implementation in
:mod:`training.authoritative_governed_grounded_import_cli`.
"""
from __future__ import annotations

from training.authoritative_governed_grounded_import_cli import main, run_import_config


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "run_import_config"]
