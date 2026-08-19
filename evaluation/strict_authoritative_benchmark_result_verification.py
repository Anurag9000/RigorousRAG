"""Symmetric restart verification for authoritative v2 benchmark result artifacts.

The underlying v2 verifier proves content addressing, aggregate metrics and receipt lineage.
This wrapper additionally replays every persisted row through the exact canonical
``BenchmarkRow -> _normalize_row`` validation used during publication, eliminating any
publication/restart schema or safety asymmetry without loading the result corpus into memory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from evaluation.authoritative_benchmark_run_evidence import (
    AuthoritativeBenchmarkResultReceipt,
    verify_authoritative_benchmark_result_receipt,
)
from evaluation.benchmark_run_evidence import _normalize_row
from evaluation.benchmark_suite import BenchmarkRow
from training.advanced_path_authority import safe_advanced_path

_MAX_LINE_BYTES = 128 * 1024 * 1024
_MAX_ROWS = 100_000_000


def _strict(raw: bytes, label: str) -> Mapping[str, Any]:
    if len(raw) > _MAX_LINE_BYTES:
        raise ValueError(f"{label} exceeds line safety bound")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain an object")
    return value


def verify_strict_authoritative_benchmark_result_receipt(
    path: str | Path,
) -> tuple[Any, AuthoritativeBenchmarkResultReceipt]:
    run, receipt = verify_authoritative_benchmark_result_receipt(path)
    artifact = safe_advanced_path(
        receipt.result_artifact_path,
        label="authoritative benchmark result artifact",
        must_exist=True,
        require_file=True,
    )
    count = 0
    footer_seen = False
    with artifact.open("rb") as handle:
        header = _strict(handle.readline(), "authoritative result header")
        if header.get("record_type") != "header":
            raise ValueError("authoritative result does not begin with a header")
        for line_number, raw in enumerate(handle, start=2):
            if not raw.strip():
                continue
            value = _strict(raw, f"authoritative result line {line_number}")
            record_type = value.get("record_type")
            if record_type == "footer":
                if footer_seen:
                    raise ValueError("authoritative result contains multiple footers")
                footer_seen = True
                continue
            if footer_seen:
                raise ValueError("authoritative result contains rows after footer")
            if record_type != "row":
                raise ValueError(f"authoritative result line {line_number} is not a row/footer")
            expected = {
                "record_type",
                "example_id",
                "retrieval_metrics",
                "retrieval_latency_ms",
                "generated_answer",
                "generation_latency_ms",
                "generation_metrics",
            }
            if set(value) != expected:
                raise ValueError(
                    f"authoritative result row {line_number} fields differ from v2 schema"
                )
            row = BenchmarkRow(
                example_id=value["example_id"],
                retrieval_metrics=value["retrieval_metrics"],
                retrieval_latency_ms=value["retrieval_latency_ms"],
                generated_answer=value["generated_answer"],
                generation_latency_ms=value["generation_latency_ms"],
                generation_metrics=value["generation_metrics"],
            )
            normalized = _normalize_row(row)
            persisted = {key: value[key] for key in expected if key != "record_type"}
            if persisted != normalized:
                raise ValueError(
                    f"authoritative result row {line_number} is not canonical under publication-time normalization"
                )
            count += 1
            if count > _MAX_ROWS:
                raise ValueError("authoritative result exceeds row safety bound")
    if not footer_seen:
        raise ValueError("authoritative result footer is missing")
    if count != receipt.sample_count or count != run.sample_count:
        raise ValueError("strict result row count differs from verified receipt/run")
    return run, receipt


__all__ = ["verify_strict_authoritative_benchmark_result_receipt"]
