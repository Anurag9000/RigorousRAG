"""Strict CLI for composing and persisting authoritative retrieval benchmark v3.

Inputs are already-local, independently governed artifacts: authoritative v2 query benchmark,
its recomputed leakage receipt, disk-backed qrels receipt, and authoritative v2 corpus receipt.
The command reconstructs/verifies every input, builds the v3 exact query/qrels/corpus contract,
persists a component-file-bound composite receipt, reconstructs that receipt from scratch, and
closes all temporary qrels stores before returning success.
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
from evaluation.authoritative_governed_qrels import close_authoritative_governed_qrels
from evaluation.authoritative_governed_retrieval_benchmark import (
    build_authoritative_governed_retrieval_benchmark,
)
from evaluation.authoritative_governed_retrieval_io import (
    build_authoritative_retrieval_benchmark_receipt,
    close_reconstructed_authoritative_retrieval_benchmark,
    reconstruct_authoritative_retrieval_benchmark,
    write_authoritative_retrieval_benchmark_receipt,
)
from evaluation.governed_qrels_io import load_governed_qrels_from_receipt
from training.advanced_path_authority import safe_advanced_path


def compose_authoritative_retrieval_benchmark(
    *,
    benchmark_import_receipt_path: str | Path,
    leakage_receipt_path: str | Path,
    qrels_receipt_path: str | Path,
    corpus_receipt_path: str | Path,
    output_path: str | Path,
) -> Mapping[str, object]:
    output = safe_advanced_path(
        output_path,
        label="authoritative retrieval benchmark receipt output",
        must_exist=False,
    )
    if output.exists():
        raise ValueError("authoritative retrieval benchmark receipt output must not already exist")

    queries = verify_authoritative_governed_benchmark_import(
        benchmark_import_receipt_path,
        require_promotable=True,
    )
    leakage = verify_authoritative_benchmark_leakage_receipt(
        leakage_receipt_path,
        benchmark=queries,
        require_pass=True,
    )
    qrels = load_governed_qrels_from_receipt(qrels_receipt_path)
    reconstructed = None
    try:
        benchmark = build_authoritative_governed_retrieval_benchmark(
            queries,
            leakage=leakage,
            qrels=qrels,
            corpus_receipt_path=corpus_receipt_path,
        )
        receipt = build_authoritative_retrieval_benchmark_receipt(
            benchmark,
            query_import_receipt_path=benchmark_import_receipt_path,
            leakage_receipt_path=leakage_receipt_path,
            qrels_receipt_path=qrels_receipt_path,
            corpus_receipt_path=corpus_receipt_path,
        )
        write_authoritative_retrieval_benchmark_receipt(output, receipt)
        reconstructed, parsed = reconstruct_authoritative_retrieval_benchmark(output)
        if (
            parsed.receipt_sha256 != receipt.receipt_sha256
            or reconstructed.contract_sha256 != benchmark.contract_sha256
            or reconstructed.query_universe_sha256 != benchmark.query_universe_sha256
            or reconstructed.qrels_coverage_sha256 != benchmark.qrels_coverage_sha256
        ):
            raise RuntimeError(
                "authoritative retrieval benchmark changed during persistence/reconstruction"
            )
        return {
            "benchmark_id": benchmark.queries.manifest.dataset_id,
            "query_manifest_sha256": benchmark.queries.manifest.manifest_digest,
            "query_universe_sha256": benchmark.query_universe_sha256,
            "qrels_receipt_sha256": benchmark.qrels.receipt.receipt_sha256,
            "corpus_receipt_sha256": benchmark.corpus.receipt_sha256,
            "qrels_coverage_sha256": benchmark.qrels_coverage_sha256,
            "retrieval_contract_sha256": benchmark.contract_sha256,
            "retrieval_receipt_sha256": receipt.receipt_sha256,
            "output": str(
                safe_advanced_path(
                    output,
                    label="authoritative retrieval benchmark receipt",
                    must_exist=True,
                    require_file=True,
                )
            ),
        }
    finally:
        close_authoritative_governed_qrels(qrels)
        if reconstructed is not None:
            close_reconstructed_authoritative_retrieval_benchmark(reconstructed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rigorousrag-retrieval-benchmark",
        description="Compose authoritative v3 retrieval benchmark from governed local artifacts",
    )
    parser.add_argument("--benchmark-import-receipt", required=True)
    parser.add_argument("--leakage-receipt", required=True)
    parser.add_argument("--qrels-receipt", required=True)
    parser.add_argument("--corpus-receipt", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = compose_authoritative_retrieval_benchmark(
        benchmark_import_receipt_path=args.benchmark_import_receipt,
        leakage_receipt_path=args.leakage_receipt,
        qrels_receipt_path=args.qrels_receipt,
        corpus_receipt_path=args.corpus_receipt,
        output_path=args.output,
    )
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["compose_authoritative_retrieval_benchmark", "main"]
