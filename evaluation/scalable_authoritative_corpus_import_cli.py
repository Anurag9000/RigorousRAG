"""Scale-guarded production entry point for authoritative corpus import."""
from __future__ import annotations

from collections.abc import Sequence

from evaluation.authoritative_governed_benchmark_corpus_cli import main as _delegate
from training.scale_guarded_cli import run_scale_guarded


def main(argv: Sequence[str] | None = None) -> int:
    return run_scale_guarded(_delegate, argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
