"""Compatibility seam for non-mutating restore-deletion marker checks."""

from __future__ import annotations

from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_deletion_mutation import (
    assert_restore_not_under_deletion as _assert_concrete,
)


def assert_restore_not_under_deletion(
    restore_journal: Any,
    restore_id: str,
) -> None:
    """Use the concrete marker table or an explicit adapter marker check.

    Production mutation and operator runtimes use the concrete SQLite journal. The
    adapter method keeps read-only test doubles and future non-SQLite readers from
    depending on private SQLite attributes; actual deletion execution still requires
    the concrete mutation boundary.
    """

    checker = getattr(restore_journal, "assert_not_under_deletion", None)
    if callable(checker):
        checker(restore_id)
        return
    if hasattr(restore_journal, "_lock") and callable(
        getattr(restore_journal, "_connect", None)
    ):
        _assert_concrete(restore_journal, restore_id)


__all__ = ["assert_restore_not_under_deletion"]
