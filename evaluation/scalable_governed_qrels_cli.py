"""Scale-guarded production entry point for disk-backed governed qrels."""
from __future__ import annotations

from collections.abc import Sequence

from evaluation.governed_qrels_cli import main as _delegate
from training.scale_guarded_cli import run_scale_guarded


def main(argv: Sequence[str] | None = None) -> int:
    return run_scale_guarded(_delegate, argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
