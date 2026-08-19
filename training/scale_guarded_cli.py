"""Shared adapter for config-first authoritative import CLIs."""
from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

from training.authoritative_input_scale import assert_authoritative_input_scale


def run_scale_guarded(
    delegate: Callable[[Sequence[str] | None], int],
    argv: Sequence[str] | None = None,
) -> int:
    selected = list(sys.argv[1:] if argv is None else argv)
    if not selected:
        # Preserve the downstream CLI's normal argparse usage/error behavior.
        return delegate(selected)
    first = selected[0]
    if isinstance(first, str) and first and not first.startswith("-"):
        assert_authoritative_input_scale(first)
    return delegate(selected)


__all__ = ["run_scale_guarded"]
