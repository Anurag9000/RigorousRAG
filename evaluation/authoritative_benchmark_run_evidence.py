"""Streaming v2 benchmark-result evidence for promotion-grade evaluation.

The legacy result artifact stores every row inside one JSON object and requires whole-file JSON
materialization during verification.  This authority writes a closed two-file directory with a
JSONL result stream plus a self-verifying receipt.  Duplicate example IDs and metric samples
are tracked in temporary SQLite, aggregate metrics use ``math.fsum`` over database cursors,
and verification is streaming and content-addressed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from evaluation.advanced_rag_receipts import AdvancedEvaluationRun
from evaluation.benchmark_run_evidence import _metric_map, _normalize_row
from evaluation.benchmark_suite import BenchmarkSuiteResult
from evaluation.dataset_governance import DatasetManifest
from training.advanced_path_authority import safe_advanced_path

_MAX_ROWS = 100_000_000
_MAX_LINE_BYTES = 128 * 1024 * 1024
_MAX_RECEIPT_BYTES = 16 * 1024 * 1024
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
            if not block: break
            digest.update(block)
    return digest.hexdigest()


def _strict_line(raw: bytes, label: str) -> Mapping[str, Any]:
    if len(raw) > _MAX_LINE_BYTES:
        raise ValueError(f"{label} exceeds line safety bound")
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain an object")
    return value


def _open_ledger() -> tuple[sqlite3.Connection, Path]:
    descriptor, raw_path = tempfile.mkstemp(prefix="rigorousrag-result-metrics-", suffix=".sqlite3")
    os.close(descriptor)
    path = Path(raw_path)
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("CREATE TABLE examples (id TEXT PRIMARY KEY) WITHOUT ROWID")
    connection.execute("CREATE TABLE metric_values (kind TEXT NOT NULL, name TEXT NOT NULL, ordinal INTEGER NOT NULL, value REAL NOT NULL, PRIMARY KEY(kind,name,ordinal)) WITHOUT ROWID")
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
        raise ValueError("retrieval metric names differ across benchmark result rows")
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
    for kind, name, count_raw in groups:
        count = int(count_raw)
        if count <= 0:
            raise RuntimeError("metric ledger contains an empty group")
        cursor = connection.execute("SELECT value FROM metric_values WHERE kind=? AND name=? ORDER BY ordinal", (kind, name))
        mean = math.fsum(float(row[0]) for row in cursor) / count
        metric_name = str(name)
        if kind == "generation" and metric_name in result:
            raise ValueError(f"generation metric {metric_name!r} collides with another aggregate metric")
        if kind == "retrieval" and metric_name in result:
            raise ValueError(f"retrieval metric {metric_name!r} collides with another aggregate metric")
        if kind == "latency" and metric_name in result:
            raise ValueError(f"latency metric {metric_name!r} collides with another aggregate metric")
        result[metric_name] = float(mean)
    if not result:
        raise ValueError("benchmark result has no aggregate metrics")
    return result


def _assert_supplied(result: BenchmarkSuiteResult, aggregate: Mapping[str, float]) -> None:
    supplied = _metric_map(result.aggregate, "benchmark aggregate")
    if set(supplied) != set(aggregate):
        raise ValueError("benchmark aggregate metric names differ from streaming recomputation")
    mismatches = [name for name in supplied if not math.isclose(float(supplied[name]), float(aggregate[name]), rel_tol=1e-12, abs_tol=1e-12)]
    if mismatches:
        raise ValueError(f"benchmark aggregate differs from streaming row recomputation: {mismatches[:40]}")


@dataclass(frozen=True)
class AuthoritativeBenchmarkResultReceipt:
    benchmark_id: str
    benchmark_manifest_sha256: str
    evaluator_contract_sha256: str
    seed: int
    repeat_index: int
    sample_count: int
    result_artifact_path: str
    result_artifact_sha256: str
    metrics_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        path = safe_advanced_path(self.result_artifact_path, label="authoritative benchmark result", must_exist=True, require_file=True)
        object.__setattr__(self, "result_artifact_path", str(path))
        if not isinstance(self.benchmark_id, str) or not self.benchmark_id.strip():
            raise ValueError("benchmark_id must be non-empty")
        for name in ("benchmark_manifest_sha256", "evaluator_contract_sha256", "result_artifact_sha256", "metrics_sha256", "receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        for name in ("seed", "repeat_index", "sample_count"):
            value = getattr(self, name); minimum = 1 if name == "sample_count" else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be integer >= {minimum}")
        if _digest(self.unsigned()) != self.receipt_sha256:
            raise ValueError("authoritative benchmark result receipt digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {"schema": "rigorousrag-authoritative-benchmark-result-receipt/v2", "benchmark_id": self.benchmark_id, "benchmark_manifest_sha256": self.benchmark_manifest_sha256, "evaluator_contract_sha256": self.evaluator_contract_sha256, "seed": self.seed, "repeat_index": self.repeat_index, "sample_count": self.sample_count, "result_artifact_path": self.result_artifact_path, "result_artifact_sha256": self.result_artifact_sha256, "metrics_sha256": self.metrics_sha256}


def materialize_authoritative_benchmark_run_evidence(
    result: BenchmarkSuiteResult,
    *,
    benchmark_manifest: DatasetManifest,
    evaluator_contract_sha256: str,
    seed: int,
    repeat_index: int,
    output_dir: str | Path,
) -> tuple[AdvancedEvaluationRun, AuthoritativeBenchmarkResultReceipt]:
    if not isinstance(result, BenchmarkSuiteResult) or not isinstance(benchmark_manifest, DatasetManifest):
        raise ValueError("result/benchmark_manifest types are invalid")
    for label, value in (("seed", seed), ("repeat_index", repeat_index)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} must be a non-negative integer")
    evaluator_sha = _sha(evaluator_contract_sha256, "evaluator_contract_sha256")
    root = safe_advanced_path(output_dir, label="authoritative benchmark result output", must_exist=False)
    if root.exists():
        raise ValueError("authoritative benchmark result output must not already exist")
    parent = safe_advanced_path(root.parent, label="authoritative benchmark result parent", must_exist=True, require_directory=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name or 'result'}-stage-", dir=parent))
    ledger, ledger_path = _open_ledger()
    try:
        artifact = stage / "result.jsonl"; count = 0; retrieval_names: set[str] | None = None
        with artifact.open("xb") as handle:
            header = {"record_type": "header", "schema": "rigorousrag-authoritative-benchmark-result/v2", "benchmark_id": benchmark_manifest.dataset_id, "benchmark_manifest_sha256": benchmark_manifest.manifest_digest, "evaluator_contract_sha256": evaluator_sha, "seed": seed, "repeat_index": repeat_index}
            handle.write(_canonical(header) + b"\n")
            for row in result.rows:
                if count >= _MAX_ROWS: raise ValueError("benchmark evidence exceeds row safety bound")
                normalized = _normalize_row(row); retrieval_names = _record_metrics(ledger, normalized, count, retrieval_names)
                handle.write(_canonical({"record_type": "row", **normalized}) + b"\n"); count += 1
                if count % 10_000 == 0: ledger.commit()
            ledger.commit()
            if count <= 0: raise ValueError("benchmark evidence requires at least one result row")
            aggregate = _aggregate(ledger); _assert_supplied(result, aggregate)
            footer = {"record_type": "footer", "sample_count": count, "metrics": dict(aggregate), "metrics_sha256": _digest(dict(aggregate))}
            handle.write(_canonical(footer) + b"\n"); handle.flush(); os.fsync(handle.fileno())
        artifact_sha = _file_sha(artifact); metrics_sha = _digest(dict(aggregate)); final_artifact = root / "result.jsonl"
        unsigned = {"schema": "rigorousrag-authoritative-benchmark-result-receipt/v2", "benchmark_id": benchmark_manifest.dataset_id, "benchmark_manifest_sha256": benchmark_manifest.manifest_digest, "evaluator_contract_sha256": evaluator_sha, "seed": seed, "repeat_index": repeat_index, "sample_count": count, "result_artifact_path": str(final_artifact), "result_artifact_sha256": artifact_sha, "metrics_sha256": metrics_sha}
        receipt_sha = _digest(unsigned)
        receipt_payload = {**unsigned, "receipt_sha256": receipt_sha}
        receipt_path = stage / "result_receipt.json"
        with receipt_path.open("xb") as handle:
            handle.write(_canonical(receipt_payload) + b"\n"); handle.flush(); os.fsync(handle.fileno())
        if {item.name for item in stage.iterdir()} != {"result.jsonl", "result_receipt.json"}:
            raise RuntimeError("authoritative benchmark result staging directory is not closed")
        os.replace(stage, root)
        receipt = AuthoritativeBenchmarkResultReceipt(benchmark_manifest.dataset_id, benchmark_manifest.manifest_digest, evaluator_sha, seed, repeat_index, count, str(final_artifact), artifact_sha, metrics_sha, receipt_sha)
        run = AdvancedEvaluationRun(benchmark_id=benchmark_manifest.dataset_id, benchmark_manifest_sha256=benchmark_manifest.manifest_digest, evaluator_contract_sha256=evaluator_sha, seed=seed, repeat_index=repeat_index, sample_count=count, metrics=aggregate, result_artifact_sha256=artifact_sha)
        return run, receipt
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise
    finally:
        ledger.close()
        try: ledger_path.unlink()
        except FileNotFoundError: pass


def verify_authoritative_benchmark_result_receipt(path: str | Path) -> tuple[AdvancedEvaluationRun, AuthoritativeBenchmarkResultReceipt]:
    receipt_path = safe_advanced_path(path, label="authoritative benchmark result receipt", must_exist=True, require_file=True)
    root = receipt_path.parent
    if receipt_path != root / "result_receipt.json": raise ValueError("authoritative result receipt must use canonical filename")
    if receipt_path.stat().st_size <= 0 or receipt_path.stat().st_size > _MAX_RECEIPT_BYTES: raise ValueError("authoritative result receipt exceeds safety bound")
    raw = _strict_line(receipt_path.read_bytes(), "authoritative result receipt")
    required = {"schema", "benchmark_id", "benchmark_manifest_sha256", "evaluator_contract_sha256", "seed", "repeat_index", "sample_count", "result_artifact_path", "result_artifact_sha256", "metrics_sha256", "receipt_sha256"}
    if set(raw) != required or raw.get("schema") != "rigorousrag-authoritative-benchmark-result-receipt/v2": raise ValueError("unsupported authoritative result receipt schema")
    receipt = AuthoritativeBenchmarkResultReceipt(**{key: value for key, value in raw.items() if key != "schema"})
    artifact = safe_advanced_path(receipt.result_artifact_path, label="authoritative benchmark result", must_exist=True, require_file=True)
    if artifact != root / "result.jsonl" or {item.name for item in root.iterdir()} != {"result.jsonl", "result_receipt.json"}: raise ValueError("authoritative result directory is not closed/canonical")
    if _file_sha(artifact) != receipt.result_artifact_sha256: raise ValueError("authoritative result bytes differ from receipt")
    ledger, ledger_path = _open_ledger()
    try:
        with artifact.open("rb") as handle:
            header_raw = handle.readline(); header = _strict_line(header_raw, "result header")
            required_header = {"record_type", "schema", "benchmark_id", "benchmark_manifest_sha256", "evaluator_contract_sha256", "seed", "repeat_index"}
            if set(header) != required_header or header.get("record_type") != "header" or header.get("schema") != "rigorousrag-authoritative-benchmark-result/v2": raise ValueError("invalid authoritative result header")
            checks = (header["benchmark_id"] == receipt.benchmark_id, header["benchmark_manifest_sha256"] == receipt.benchmark_manifest_sha256, header["evaluator_contract_sha256"] == receipt.evaluator_contract_sha256, header["seed"] == receipt.seed, header["repeat_index"] == receipt.repeat_index)
            if not all(checks): raise ValueError("authoritative result header differs from receipt")
            count = 0; retrieval_names: set[str] | None = None; footer: Mapping[str, Any] | None = None
            for line_number, raw_line in enumerate(handle, start=2):
                if not raw_line.strip(): continue
                value = _strict_line(raw_line, f"result line {line_number}")
                if value.get("record_type") == "footer":
                    if footer is not None: raise ValueError("authoritative result contains multiple footers")
                    footer = value
                    # Footer must be the last non-empty record.
                    for remaining in handle:
                        if remaining.strip(): raise ValueError("authoritative result has rows after footer")
                    break
                expected_row = {"record_type", "example_id", "retrieval_metrics", "retrieval_latency_ms", "generated_answer", "generation_latency_ms", "generation_metrics"}
                if set(value) != expected_row or value.get("record_type") != "row": raise ValueError(f"invalid result row at line {line_number}")
                normalized = {key: value[key] for key in expected_row if key != "record_type"}
                # Reuse canonical validation by passing a shape-equivalent mapping into the ledger after strict metric normalization.
                normalized["retrieval_metrics"] = _metric_map(normalized["retrieval_metrics"], "retrieval metrics")
                normalized["generation_metrics"] = _metric_map(normalized["generation_metrics"], "generation metrics") if normalized["generation_metrics"] else {}
                normalized["retrieval_latency_ms"] = float(normalized["retrieval_latency_ms"]); normalized["generation_latency_ms"] = float(normalized["generation_latency_ms"])
                if not math.isfinite(normalized["retrieval_latency_ms"]) or not math.isfinite(normalized["generation_latency_ms"]): raise ValueError("result latency is non-finite")
                retrieval_names = _record_metrics(ledger, normalized, count, retrieval_names); count += 1
                if count > _MAX_ROWS: raise ValueError("authoritative result exceeds row safety bound")
                if count % 10_000 == 0: ledger.commit()
            ledger.commit()
        if footer is None: raise ValueError("authoritative result footer is missing")
        required_footer = {"record_type", "sample_count", "metrics", "metrics_sha256"}
        if set(footer) != required_footer or footer.get("record_type") != "footer": raise ValueError("invalid authoritative result footer")
        aggregate = _aggregate(ledger); supplied = _metric_map(footer["metrics"], "result footer metrics")
        if count != receipt.sample_count or footer["sample_count"] != receipt.sample_count: raise ValueError("authoritative result sample count differs from receipt")
        if set(aggregate) != set(supplied) or any(not math.isclose(aggregate[name], supplied[name], rel_tol=1e-12, abs_tol=1e-12) for name in aggregate): raise ValueError("authoritative result footer metrics differ from rows")
        if _digest(dict(aggregate)) != receipt.metrics_sha256 or footer["metrics_sha256"] != receipt.metrics_sha256: raise ValueError("authoritative result metrics digest differs")
        return AdvancedEvaluationRun(benchmark_id=receipt.benchmark_id, benchmark_manifest_sha256=receipt.benchmark_manifest_sha256, evaluator_contract_sha256=receipt.evaluator_contract_sha256, seed=receipt.seed, repeat_index=receipt.repeat_index, sample_count=receipt.sample_count, metrics=aggregate, result_artifact_sha256=receipt.result_artifact_sha256), receipt
    finally:
        ledger.close()
        try: ledger_path.unlink()
        except FileNotFoundError: pass


__all__ = ["AuthoritativeBenchmarkResultReceipt", "materialize_authoritative_benchmark_run_evidence", "verify_authoritative_benchmark_result_receipt"]
