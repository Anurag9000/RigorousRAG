from __future__ import annotations

import sys
from typing import Sequence

from tools.evidence_graph_set_signed_retirement_restore_custody_boundary import (
    verify_pre_restore_backup_receipt,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_runtime import (
    get_signed_retirement_restore_custody_store,
)
from tools.evidence_graph_set_signed_retirement_restore_execute_cli import (
    main as _main,
)
from tools.evidence_graph_set_signed_retirement_restore_mutation import (
    target_path_digest,
)


def _value(argv: Sequence[str], option: str) -> str | None:
    try:
        index = argv.index(option)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        return None
    return argv[index + 1]


def _preflight(argv: Sequence[str]) -> None:
    if not argv or argv[0] not in {"seed", "execute", "reconcile-one"}:
        return
    target = _value(argv, "--target-db-path")
    receipt_path = _value(argv, "--pre-receipt")
    backup_path = _value(argv, "--backup")
    if target is None or receipt_path is None or backup_path is None:
        raise ValueError(
            "target, pre-restore receipt, and backup are required."
        )
    receipt = verify_pre_restore_backup_receipt(
        receipt_path=receipt_path,
        backup_path=backup_path,
    )
    if receipt.target_path_digest != target_path_digest(target):
        raise RuntimeError(
            "pre-restore custody evidence differs from target path."
        )
    get_signed_retirement_restore_custody_store(
        target_db_path=target
    )


def main(argv: Sequence[str] | None = None) -> int:
    selected = list(sys.argv[1:] if argv is None else argv)
    _preflight(selected)
    return _main(selected)


if __name__ == "__main__":
    raise SystemExit(main())
