"""Streaming metric-schema consistency checks for promotion-grade result artifacts.

A cohort aggregate must not silently mix different denominators because some rows omit a
per-sample metric. This verifier requires every canonical result row to expose the same
retrieval-metric key set and the same generation-metric key set. Retrieval-only or
no-generation evaluations remain valid when the corresponding set is consistently empty.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from evaluation.authoritative_benchmark_run_evidence import AuthoritativeBenchmarkResultReceipt
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


def verify_homogeneous_result_metric_schema(
    receipt: AuthoritativeBenchmarkResultReceipt,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(receipt, AuthoritativeBenchmarkResultReceipt):
        raise ValueError("receipt must be AuthoritativeBenchmarkResultReceipt")
    artifact = safe_advanced_path(
        receipt.result_artifact_path,
        label="authoritative benchmark result artifact",
        must_exist=True,
        require_file=True,
    )
    retrieval_names: tuple[str, ...] | None = None
    generation_names: tuple[str, ...] | None = None
    count = 0
    footer_seen = False
    with artifact.open("rb") as handle:
        header = _strict(handle.readline(), "result header")
        if header.get("record_type") != "header":
            raise ValueError("authoritative result does not begin with a header")
        for line_number, raw in enumerate(handle, start=2):
            if not raw.strip():
                continue
            value = _strict(raw, f"result line {line_number}")
            record_type = value.get("record_type")
            if record_type == "footer":
                if footer_seen:
                    raise ValueError("authoritative result contains multiple footers")
                footer_seen = True
                continue
            if footer_seen:
                raise ValueError("authoritative result contains rows after footer")
            if record_type != "row":
                raise ValueError(f"result line {line_number} is not a canonical row")
            retrieval = value.get("retrieval_metrics")
            generation = value.get("generation_metrics")
            if not isinstance(retrieval, Mapping) or not isinstance(generation, Mapping):
                raise ValueError(f"result line {line_number} metric fields must be objects")
            current_retrieval = tuple(sorted(str(key) for key in retrieval))
            current_generation = tuple(sorted(str(key) for key in generation))
            if retrieval_names is None:
                retrieval_names = current_retrieval
            elif current_retrieval != retrieval_names:
                raise ValueError(
                    "retrieval metric key set differs across result rows; "
                    f"line={line_number} expected={retrieval_names} actual={current_retrieval}"
                )
            if generation_names is None:
                generation_names = current_generation
            elif current_generation != generation_names:
                raise ValueError(
                    "generation metric key set differs across result rows; "
                    f"line={line_number} expected={generation_names} actual={current_generation}"
                )
            count += 1
            if count > _MAX_ROWS:
                raise ValueError("authoritative result exceeds row safety bound")
    if not footer_seen:
        raise ValueError("authoritative result footer is missing")
    if count != receipt.sample_count:
        raise ValueError("result metric-schema row count differs from receipt sample_count")
    return retrieval_names or (), generation_names or ()


__all__ = ["verify_homogeneous_result_metric_schema"]
