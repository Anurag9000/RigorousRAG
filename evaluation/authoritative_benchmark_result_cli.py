"""Materialize already-produced local benchmark results as promotion-grade v2 evidence.

This command does not execute a benchmark.  It accepts a strict local ``rows + aggregate``
representation of ``BenchmarkSuiteResult``, re-verifies the persisted authoritative evaluation
cohort, proves the supplied example-ID universe equals the cohort sample universe, reconstructs
the exact benchmark manifest from the cohort authority receipt, and publishes/re-verifies the
streaming v2 result artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evaluation.authoritative_benchmark_run_evidence import (
    materialize_authoritative_benchmark_run_evidence,
)
from evaluation.authoritative_evaluation_cohort import (
    AuthoritativeEvaluationCohortContract,
    assert_result_receipt_matches_cohort,
    verify_authoritative_evaluation_cohort,
)
from evaluation.authoritative_governed_benchmark_io import (
    verify_authoritative_governed_benchmark_import,
)
from evaluation.authoritative_governed_retrieval_io import (
    close_reconstructed_authoritative_retrieval_benchmark,
    reconstruct_authoritative_retrieval_benchmark,
)
from evaluation.benchmark_run_evidence import _normalize_row
from evaluation.benchmark_suite import BenchmarkRow, BenchmarkSuiteResult
from training.advanced_path_authority import safe_advanced_path

_MAX_INPUT_BYTES = 2 * 1024 * 1024 * 1024
_MAX_ROWS = 100_000_000
_MAX_LINE_TEXT = 16 * 1024 * 1024
_MAX_SAMPLE = 100


def _strict_json(path: str | Path) -> Mapping[str, Any]:
    source = safe_advanced_path(
        path,
        label="benchmark suite result input",
        must_exist=True,
        require_file=True,
    )
    size = source.stat().st_size
    if size <= 0 or size > _MAX_INPUT_BYTES:
        raise ValueError("benchmark suite result input exceeds byte safety bound")
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except Exception as exc:
        raise ValueError("benchmark suite result input is not strict JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("benchmark suite result input must contain an object")
    return raw


def _suite_result(path: str | Path) -> BenchmarkSuiteResult:
    raw = _strict_json(path)
    if (
        set(raw) != {"schema", "rows", "aggregate"}
        or raw.get("schema") != "rigorousrag-benchmark-suite-result-input/v1"
        or not isinstance(raw.get("rows"), list)
        or not isinstance(raw.get("aggregate"), Mapping)
    ):
        raise ValueError(
            "result input must be rigorousrag-benchmark-suite-result-input/v1"
        )
    rows_raw = raw["rows"]
    if not rows_raw or len(rows_raw) > _MAX_ROWS:
        raise ValueError("benchmark result rows must be bounded and non-empty")
    expected = {
        "example_id",
        "retrieval_metrics",
        "retrieval_latency_ms",
        "generated_answer",
        "generation_latency_ms",
        "generation_metrics",
    }
    rows = []
    for index, item in enumerate(rows_raw):
        if not isinstance(item, Mapping) or set(item) != expected:
            raise ValueError(f"benchmark result row {index} fields are invalid")
        answer = item["generated_answer"]
        if answer is not None and (
            not isinstance(answer, str)
            or len(answer.encode("utf-8")) > _MAX_LINE_TEXT
        ):
            raise ValueError(f"benchmark result row {index} generated_answer is invalid")
        row = BenchmarkRow(
            example_id=item["example_id"],
            retrieval_metrics=item["retrieval_metrics"],
            retrieval_latency_ms=item["retrieval_latency_ms"],
            generated_answer=answer,
            generation_latency_ms=item["generation_latency_ms"],
            generation_metrics=item["generation_metrics"],
        )
        _normalize_row(row)
        rows.append(row)
    return BenchmarkSuiteResult(rows=tuple(rows), aggregate=raw["aggregate"])


def _receipt_binding(
    cohort: AuthoritativeEvaluationCohortContract,
    role: str,
) -> str:
    matches = [item.path for item in cohort.authority_receipts if item.role == role]
    if len(matches) != 1:
        raise ValueError(f"evaluation cohort lacks unique {role} receipt binding")
    return matches[0]


def _benchmark_manifest(cohort: AuthoritativeEvaluationCohortContract) -> tuple[Any, Any | None]:
    if cohort.authority_kind == "governed_benchmark_v2":
        benchmark = verify_authoritative_governed_benchmark_import(
            _receipt_binding(cohort, "benchmark_import"),
            require_promotable=True,
        )
        return benchmark.manifest, None
    if cohort.authority_kind == "retrieval_benchmark_v3":
        benchmark, _ = reconstruct_authoritative_retrieval_benchmark(
            _receipt_binding(cohort, "retrieval_benchmark")
        )
        return benchmark.queries.manifest, benchmark
    raise ValueError("unsupported authoritative cohort kind")


def _expected_ids(cohort: AuthoritativeEvaluationCohortContract) -> Iterable[str]:
    if cohort.authority_kind == "governed_benchmark_v2":
        benchmark = verify_authoritative_governed_benchmark_import(
            _receipt_binding(cohort, "benchmark_import"),
            require_promotable=True,
        )
        for split in cohort.selected_splits:
            for example in benchmark.split(split):
                yield example.example_id
        return
    benchmark, _ = reconstruct_authoritative_retrieval_benchmark(
        _receipt_binding(cohort, "retrieval_benchmark")
    )
    try:
        for split in cohort.selected_splits:
            for example in benchmark.queries.split(split):
                yield example.example_id
    finally:
        close_reconstructed_authoritative_retrieval_benchmark(benchmark)


def _assert_result_object_universe(
    result: BenchmarkSuiteResult,
    cohort: AuthoritativeEvaluationCohortContract,
) -> None:
    descriptor, raw_database = tempfile.mkstemp(
        prefix="rigorousrag-result-input-universe-",
        suffix=".sqlite3",
    )
    os.close(descriptor)
    try:
        connection = sqlite3.connect(raw_database)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("CREATE TABLE expected(id TEXT PRIMARY KEY) WITHOUT ROWID")
            connection.execute("CREATE TABLE actual(id TEXT PRIMARY KEY) WITHOUT ROWID")
            expected_count = 0
            for example_id in _expected_ids(cohort):
                try:
                    connection.execute("INSERT INTO expected(id) VALUES (?)", (example_id,))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"cohort sample id {example_id!r} is duplicated") from exc
                expected_count += 1
                if expected_count % 20_000 == 0:
                    connection.commit()
            actual_count = 0
            for row in result.rows:
                example_id = str(_normalize_row(row)["example_id"])
                try:
                    connection.execute("INSERT INTO actual(id) VALUES (?)", (example_id,))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"result input repeats example id {example_id!r}") from exc
                actual_count += 1
                if actual_count % 20_000 == 0:
                    connection.commit()
            connection.commit()
            missing = [
                str(item[0])
                for item in connection.execute(
                    "SELECT e.id FROM expected e LEFT JOIN actual a ON e.id=a.id "
                    "WHERE a.id IS NULL ORDER BY e.id COLLATE BINARY LIMIT ?",
                    (_MAX_SAMPLE,),
                )
            ]
            extra = [
                str(item[0])
                for item in connection.execute(
                    "SELECT a.id FROM actual a LEFT JOIN expected e ON a.id=e.id "
                    "WHERE e.id IS NULL ORDER BY a.id COLLATE BINARY LIMIT ?",
                    (_MAX_SAMPLE,),
                )
            ]
            if (
                expected_count != cohort.sample_count
                or actual_count != cohort.sample_count
                or missing
                or extra
            ):
                raise ValueError(
                    "result input/cohort universes differ; "
                    f"expected_count={expected_count} actual_count={actual_count} "
                    f"missing_sample={missing} extra_sample={extra}"
                )
            digest = hashlib.sha256()
            for (value,) in connection.execute(
                "SELECT id FROM actual ORDER BY id COLLATE BINARY"
            ):
                digest.update(str(value).encode("utf-8"))
                digest.update(b"\n")
            if digest.hexdigest() != cohort.sample_universe_sha256:
                raise ValueError("result input sample-universe digest differs from cohort")
        finally:
            connection.close()
    finally:
        try:
            os.unlink(raw_database)
        except FileNotFoundError:
            pass


def materialize_result(
    *,
    cohort_contract_path: str | Path,
    result_input_path: str | Path,
    seed: int,
    repeat_index: int,
    output_dir: str | Path,
) -> Mapping[str, object]:
    for label, value in (("seed", seed), ("repeat_index", repeat_index)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
    cohort = verify_authoritative_evaluation_cohort(cohort_contract_path)
    result = _suite_result(result_input_path)
    _assert_result_object_universe(result, cohort)
    manifest, retrieval_benchmark = _benchmark_manifest(cohort)
    try:
        run, receipt = materialize_authoritative_benchmark_run_evidence(
            result,
            benchmark_manifest=manifest,
            evaluator_contract_sha256=cohort.evaluator_contract_sha256,
            seed=seed,
            repeat_index=repeat_index,
            output_dir=output_dir,
        )
    finally:
        if retrieval_benchmark is not None:
            close_reconstructed_authoritative_retrieval_benchmark(retrieval_benchmark)
    receipt_path = Path(output_dir) / "result_receipt.json"
    cohort_run = assert_result_receipt_matches_cohort(receipt_path, cohort=cohort)
    if cohort_run.run_sha256 != run.run_sha256:
        raise RuntimeError("result evidence changed during cohort re-verification")
    return {
        "benchmark_id": run.benchmark_id,
        "benchmark_manifest_sha256": run.benchmark_manifest_sha256,
        "evaluator_contract_sha256": run.evaluator_contract_sha256,
        "sample_count": run.sample_count,
        "seed": run.seed,
        "repeat_index": run.repeat_index,
        "result_artifact_sha256": run.result_artifact_sha256,
        "run_sha256": run.run_sha256,
        "result_receipt_sha256": receipt.receipt_sha256,
        "output_dir": str(
            safe_advanced_path(
                output_dir,
                label="authoritative benchmark result output",
                must_exist=True,
                require_directory=True,
            )
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rigorousrag-benchmark-result",
        description="Materialize an already-produced local benchmark result as cohort-bound v2 evidence",
    )
    parser.add_argument("--cohort-contract", required=True)
    parser.add_argument("--result-input", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--repeat-index", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = materialize_result(
        cohort_contract_path=args.cohort_contract,
        result_input_path=args.result_input,
        seed=args.seed,
        repeat_index=args.repeat_index,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "materialize_result"]
