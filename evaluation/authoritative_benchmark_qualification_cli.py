"""Strict CLI for v2 governed-benchmark leakage qualification.

This command closes operator glue between an authoritative benchmark import and the persisted
leakage receipt required by evaluation-cohort/retrieval composition.  It re-verifies the v2
benchmark bytes, independently computes leakage, atomically persists the receipt through the
existing writer, then re-reads and independently recomputes it before returning success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from evaluation.authoritative_benchmark_leakage_io import (
    verify_authoritative_benchmark_leakage_receipt,
)
from evaluation.authoritative_governed_benchmark_io import (
    verify_authoritative_governed_benchmark_import,
)
from evaluation.authoritative_governed_benchmark_qualification import (
    qualify_authoritative_governed_benchmark_leakage,
    write_authoritative_governed_benchmark_leakage_receipt,
)
from training.advanced_path_authority import safe_advanced_path

_DEFAULT_BLOCKING = ("record_id", "query_id", "source_group_id")
_MAX_KINDS = 100


def qualify_benchmark(
    import_receipt_path: str | Path,
    *,
    output_path: str | Path,
    blocking_key_kinds: Sequence[str] = _DEFAULT_BLOCKING,
) -> Mapping[str, object]:
    selected = tuple(str(item).strip() for item in blocking_key_kinds)
    if (
        not selected
        or len(selected) > _MAX_KINDS
        or any(not item for item in selected)
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("blocking_key_kinds must be a unique bounded non-empty sequence")
    output = safe_advanced_path(
        output_path,
        label="authoritative benchmark leakage receipt output",
        must_exist=False,
    )
    if output.exists():
        raise ValueError("authoritative benchmark leakage receipt output must not already exist")
    benchmark = verify_authoritative_governed_benchmark_import(
        import_receipt_path,
        require_promotable=True,
    )
    receipt = qualify_authoritative_governed_benchmark_leakage(
        benchmark,
        blocking_key_kinds=selected,
        require_pass=True,
    )
    write_authoritative_governed_benchmark_leakage_receipt(output, receipt)
    verified = verify_authoritative_benchmark_leakage_receipt(
        output,
        benchmark=benchmark,
        require_pass=True,
    )
    if verified.receipt_sha256 != receipt.receipt_sha256:
        raise RuntimeError("benchmark leakage receipt changed during publication")
    return {
        "benchmark_id": benchmark.manifest.dataset_id,
        "benchmark_manifest_sha256": benchmark.manifest.manifest_digest,
        "import_receipt_sha256": benchmark.receipt.receipt_sha256,
        "blocking_key_kinds": list(verified.blocking_key_kinds),
        "finding_count": len(verified.findings),
        "passed": verified.passed,
        "leakage_receipt_sha256": verified.receipt_sha256,
        "output": str(
            safe_advanced_path(
                output,
                label="authoritative benchmark leakage receipt",
                must_exist=True,
                require_file=True,
            )
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rigorousrag-benchmark-qualify",
        description="Qualify an authoritative v2 governed benchmark for split leakage",
    )
    parser.add_argument("--benchmark-import-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--blocking-key-kind",
        action="append",
        dest="blocking_key_kinds",
        help="repeat to override the default blocking key kinds",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    kinds = (
        tuple(args.blocking_key_kinds)
        if args.blocking_key_kinds is not None
        else _DEFAULT_BLOCKING
    )
    result = qualify_benchmark(
        args.benchmark_import_receipt,
        output_path=args.output,
        blocking_key_kinds=kinds,
    )
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "qualify_benchmark"]
