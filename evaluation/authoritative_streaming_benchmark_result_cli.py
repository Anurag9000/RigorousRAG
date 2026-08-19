"""Streaming CLI for promotion-grade benchmark result evidence.

Input is an already-produced local JSONL result stream, not a benchmark execution request:

* one header: ``{"schema":"rigorousrag-benchmark-suite-result-jsonl/v1","record_type":"header"}``;
* canonical benchmark row objects with ``record_type="row"``;
* one footer with ``record_type="footer"`` and an ``aggregate`` metric mapping.

The source bytes must match an operator-supplied SHA-256. Rows are validated and copied into
the canonical v2 result artifact without retaining the corpus in Python, metrics/example IDs
are tracked in SQLite, the supplied aggregate is checked against a disk-backed recomputation,
and the final receipt is proved against the exact authoritative evaluation cohort. Any failed
cohort verification removes the newly published result directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.advanced_rag_receipts import AdvancedEvaluationRun
from evaluation.authoritative_benchmark_run_evidence import AuthoritativeBenchmarkResultReceipt
from evaluation.authoritative_evaluation_cohort import (
    assert_result_receipt_matches_cohort,
    verify_authoritative_evaluation_cohort,
)
from evaluation.benchmark_run_evidence import _metric_map, _normalize_row
from evaluation.benchmark_suite import BenchmarkRow
from training.advanced_path_authority import safe_advanced_path

_MAX_LINE_BYTES = 128 * 1024 * 1024
_MAX_ROWS = 100_000_000
_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _strict(raw: bytes, label: str) -> Mapping[str, Any]:
    if len(raw) > _MAX_LINE_BYTES:
        raise ValueError(f"{label} exceeds line safety bound")
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain an object")
    return value


def _open_ledger(stage: Path) -> tuple[sqlite3.Connection, Path]:
    descriptor, raw_path = tempfile.mkstemp(prefix=".result-ledger-", suffix=".sqlite3", dir=stage)
    os.close(descriptor)
    path = Path(raw_path)
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("CREATE TABLE examples(id TEXT PRIMARY KEY) WITHOUT ROWID")
    connection.execute("CREATE TABLE metric_values(kind TEXT NOT NULL,name TEXT NOT NULL,ordinal INTEGER NOT NULL,value REAL NOT NULL,PRIMARY KEY(kind,name,ordinal)) WITHOUT ROWID")
    return connection, path


def _record_metrics(connection: sqlite3.Connection, normalized: Mapping[str, Any], ordinal: int, retrieval_names: set[str] | None) -> set[str]:
    example_id = str(normalized["example_id"])
    try:
        connection.execute("INSERT INTO examples(id) VALUES (?)", (example_id,))
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"benchmark result contains duplicate example id {example_id!r}") from exc
    retrieval = dict(normalized["retrieval_metrics"])
    names = set(retrieval)
    if retrieval_names is not None and names != retrieval_names:
        raise ValueError("retrieval metric names differ across result rows")
    for name, value in retrieval.items():
        connection.execute("INSERT INTO metric_values(kind,name,ordinal,value) VALUES ('retrieval',?,?,?)", (name, ordinal, float(value)))
    for name, value in dict(normalized["generation_metrics"]).items():
        connection.execute("INSERT INTO metric_values(kind,name,ordinal,value) VALUES ('generation',?,?,?)", (name, ordinal, float(value)))
    connection.execute("INSERT INTO metric_values(kind,name,ordinal,value) VALUES ('latency','retrieval_latency_ms',?,?)", (ordinal, float(normalized["retrieval_latency_ms"])))
    connection.execute("INSERT INTO metric_values(kind,name,ordinal,value) VALUES ('latency','generation_latency_ms',?,?)", (ordinal, float(normalized["generation_latency_ms"])))
    return names


def _aggregate(connection: sqlite3.Connection) -> dict[str, float]:
    result: dict[str, float] = {}
    groups = connection.execute("SELECT kind,name,COUNT(*) FROM metric_values GROUP BY kind,name ORDER BY kind,name").fetchall()
    for kind, name, raw_count in groups:
        count = int(raw_count)
        if count <= 0:
            raise RuntimeError("result metric ledger contains empty group")
        values = connection.execute("SELECT value FROM metric_values WHERE kind=? AND name=? ORDER BY ordinal", (kind, name))
        mean = math.fsum(float(row[0]) for row in values) / count
        selected = str(name)
        if selected in result:
            raise ValueError(f"aggregate metric name {selected!r} collides across metric families")
        result[selected] = float(mean)
    if not result:
        raise ValueError("benchmark result has no aggregate metrics")
    return result


def _validate_row(value: Mapping[str, Any], line_number: int) -> Mapping[str, Any]:
    expected = {"record_type", "example_id", "retrieval_metrics", "retrieval_latency_ms", "generated_answer", "generation_latency_ms", "generation_metrics"}
    if set(value) != expected or value.get("record_type") != "row":
        raise ValueError(f"result input row {line_number} fields are invalid")
    row = BenchmarkRow(
        example_id=value["example_id"],
        retrieval_metrics=value["retrieval_metrics"],
        retrieval_latency_ms=value["retrieval_latency_ms"],
        generated_answer=value["generated_answer"],
        generation_latency_ms=value["generation_latency_ms"],
        generation_metrics=value["generation_metrics"],
    )
    return _normalize_row(row)


def materialize_streaming_result(
    *,
    cohort_contract_path: str | Path,
    result_input_path: str | Path,
    result_input_sha256: str,
    seed: int,
    repeat_index: int,
    output_dir: str | Path,
) -> Mapping[str, object]:
    for label, value in (("seed", seed), ("repeat_index", repeat_index)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
    source = safe_advanced_path(result_input_path, label="benchmark result JSONL input", must_exist=True, require_file=True)
    expected_source_sha = _sha(result_input_sha256, "result_input_sha256")
    if _file_sha(source) != expected_source_sha:
        raise ValueError("benchmark result input bytes differ from configured SHA-256")
    cohort = verify_authoritative_evaluation_cohort(cohort_contract_path)
    root = safe_advanced_path(output_dir, label="authoritative benchmark result output", must_exist=False)
    if root.exists():
        raise ValueError("authoritative benchmark result output must not already exist")
    parent = safe_advanced_path(root.parent, label="authoritative benchmark result parent", must_exist=True, require_directory=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name or 'result'}-stage-", dir=parent))
    connection, ledger_path = _open_ledger(stage)
    published = False
    try:
        artifact = stage / "result.jsonl"
        retrieval_names: set[str] | None = None
        count = 0
        supplied_aggregate: Mapping[str, float] | None = None
        with source.open("rb") as input_handle, artifact.open("xb") as output_handle:
            header_raw = input_handle.readline()
            header = _strict(header_raw, "result input header")
            if set(header) != {"schema", "record_type"} or header.get("schema") != "rigorousrag-benchmark-suite-result-jsonl/v1" or header.get("record_type") != "header":
                raise ValueError("result input must begin with rigorousrag-benchmark-suite-result-jsonl/v1 header")
            output_header = {
                "record_type": "header",
                "schema": "rigorousrag-authoritative-benchmark-result/v2",
                "benchmark_id": cohort.benchmark_id,
                "benchmark_manifest_sha256": cohort.benchmark_manifest_sha256,
                "evaluator_contract_sha256": cohort.evaluator_contract_sha256,
                "seed": seed,
                "repeat_index": repeat_index,
            }
            output_handle.write(_canonical(output_header) + b"\n")
            footer_seen = False
            for line_number, raw in enumerate(input_handle, start=2):
                if not raw.strip():
                    continue
                value = _strict(raw, f"result input line {line_number}")
                if value.get("record_type") == "footer":
                    if footer_seen or set(value) != {"record_type", "aggregate"} or not isinstance(value.get("aggregate"), Mapping):
                        raise ValueError("result input footer is malformed or repeated")
                    footer_seen = True
                    supplied_aggregate = _metric_map(value["aggregate"], "result input aggregate")
                    for trailing in input_handle:
                        if trailing.strip():
                            raise ValueError("result input contains records after footer")
                    break
                if footer_seen:
                    raise ValueError("result input contains row after footer")
                if count >= _MAX_ROWS:
                    raise ValueError("benchmark result input exceeds row safety bound")
                normalized = _validate_row(value, line_number)
                retrieval_names = _record_metrics(connection, normalized, count, retrieval_names)
                output_handle.write(_canonical({"record_type": "row", **normalized}) + b"\n")
                count += 1
                if count % 10_000 == 0:
                    connection.commit()
            connection.commit()
            if not footer_seen or supplied_aggregate is None:
                raise ValueError("result input footer is missing")
            if count <= 0:
                raise ValueError("result input requires at least one row")
            aggregate = _aggregate(connection)
            if set(aggregate) != set(supplied_aggregate) or any(not math.isclose(aggregate[name], supplied_aggregate[name], rel_tol=1e-12, abs_tol=1e-12) for name in aggregate):
                raise ValueError("result input aggregate differs from streaming row recomputation")
            metrics_sha = _digest(dict(aggregate))
            output_handle.write(_canonical({"record_type": "footer", "sample_count": count, "metrics": dict(aggregate), "metrics_sha256": metrics_sha}) + b"\n")
            output_handle.flush(); os.fsync(output_handle.fileno())
        connection.close()
        ledger_path.unlink()
        artifact_sha = _file_sha(artifact)
        final_artifact = root / "result.jsonl"
        unsigned = {
            "schema": "rigorousrag-authoritative-benchmark-result-receipt/v2",
            "benchmark_id": cohort.benchmark_id,
            "benchmark_manifest_sha256": cohort.benchmark_manifest_sha256,
            "evaluator_contract_sha256": cohort.evaluator_contract_sha256,
            "seed": seed,
            "repeat_index": repeat_index,
            "sample_count": count,
            "result_artifact_path": str(final_artifact),
            "result_artifact_sha256": artifact_sha,
            "metrics_sha256": metrics_sha,
        }
        receipt_sha = _digest(unsigned)
        receipt_path = stage / "result_receipt.json"
        with receipt_path.open("xb") as handle:
            handle.write(_canonical({**unsigned, "receipt_sha256": receipt_sha}) + b"\n"); handle.flush(); os.fsync(handle.fileno())
        if {item.name for item in stage.iterdir()} != {"result.jsonl", "result_receipt.json"}:
            raise RuntimeError("authoritative result staging directory is not closed")
        os.replace(stage, root)
        published = True
        receipt = AuthoritativeBenchmarkResultReceipt(
            benchmark_id=cohort.benchmark_id,
            benchmark_manifest_sha256=cohort.benchmark_manifest_sha256,
            evaluator_contract_sha256=cohort.evaluator_contract_sha256,
            seed=seed,
            repeat_index=repeat_index,
            sample_count=count,
            result_artifact_path=str(final_artifact),
            result_artifact_sha256=artifact_sha,
            metrics_sha256=metrics_sha,
            receipt_sha256=receipt_sha,
        )
        run = AdvancedEvaluationRun(
            benchmark_id=cohort.benchmark_id,
            benchmark_manifest_sha256=cohort.benchmark_manifest_sha256,
            evaluator_contract_sha256=cohort.evaluator_contract_sha256,
            seed=seed,
            repeat_index=repeat_index,
            sample_count=count,
            metrics=aggregate,
            result_artifact_sha256=artifact_sha,
        )
        cohort_run = assert_result_receipt_matches_cohort(root / "result_receipt.json", cohort=cohort)
        if cohort_run.run_sha256 != run.run_sha256:
            raise RuntimeError("result evidence changed during cohort verification")
        return {
            "benchmark_id": run.benchmark_id,
            "benchmark_manifest_sha256": run.benchmark_manifest_sha256,
            "evaluator_contract_sha256": run.evaluator_contract_sha256,
            "sample_count": run.sample_count,
            "seed": run.seed,
            "repeat_index": run.repeat_index,
            "result_input_sha256": expected_source_sha,
            "result_artifact_sha256": run.result_artifact_sha256,
            "run_sha256": run.run_sha256,
            "result_receipt_sha256": receipt.receipt_sha256,
            "output_dir": str(root),
        }
    except Exception:
        try:
            connection.close()
        except Exception:
            pass
        if published:
            shutil.rmtree(root, ignore_errors=True)
        else:
            shutil.rmtree(stage, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rigorousrag-benchmark-result",
        description="Stream an already-produced local result JSONL into cohort-bound v2 evidence",
    )
    parser.add_argument("--cohort-contract", required=True)
    parser.add_argument("--result-input", required=True)
    parser.add_argument("--result-input-sha256", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--repeat-index", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = materialize_streaming_result(
        cohort_contract_path=args.cohort_contract,
        result_input_path=args.result_input,
        result_input_sha256=args.result_input_sha256,
        seed=args.seed,
        repeat_index=args.repeat_index,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "materialize_streaming_result"]
