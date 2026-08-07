"""Public restore executor with custody preflight for mutating commands."""

from __future__ import annotations

import sys
from typing import Sequence

from tools.evidence_graph_set_signed_retirement_restore_custody_runtime import (
    get_signed_retirement_restore_custody_store,
)
from tools.evidence_graph_set_signed_retirement_restore_execute_cli import (
    main as _main,
)

_MUTATING_COMMANDS = frozenset({"seed", "execute", "reconcile-one"})


def _value(argv: Sequence[str], option: str) -> str | None:
    try:
        index = argv.index(option)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        return None
    return argv[index + 1]


def _preflight(argv: Sequence[str]) -> None:
    if not argv or argv[0] not in _MUTATING_COMMANDS:
        return
    target = _value(argv, "--target-db-path")
    if target is None:
        raise ValueError("target database path is required for mutation.")
    # Resolve the custody store before the CLI can perform any target mutation.
    # The runtime getter enforces target/custody path separation and stable identity.
    get_signed_retirement_restore_custody_store(target_db_path=target)


def main(argv: Sequence[str] | None = None) -> int:
    selected = list(sys.argv[1:] if argv is None else argv)
    _preflight(selected)
    return _main(selected)


if __name__ == "__main__":
    raise SystemExit(main())
