"""Executable benchmark orchestration across retrieval, generation, latency, and evidence metrics."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

from evaluation import EvaluationQuery, RetrievalResult, evaluate_query
from evaluation.generation_metrics import best_reference_score, chrf, rouge_l
from tools.benchmark_adapters import BenchmarkExample


Retriever = Callable[[BenchmarkExample, int], Sequence[tuple[str, float] | RetrievalResult]]
Generator = Callable[[BenchmarkExample, Sequence[RetrievalResult]], str]


@dataclass(frozen=True)
class BenchmarkRow:
    example_id: str
    retrieval_metrics: Mapping[str, float]
    retrieval_latency_ms: float
    generated_answer: str = ""
    generation_latency_ms: float = 0.0
    generation_metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkSuiteResult:
    rows: tuple[BenchmarkRow, ...]
    aggregate: Mapping[str, float]


def _normalize_results(
    raw: Sequence[tuple[str, float] | RetrievalResult],
    *,
    top_k: int,
) -> tuple[RetrievalResult, ...]:
    output = []
    seen = set()
    for position, item in enumerate(raw, start=1):
        if isinstance(item, RetrievalResult):
            result = item
        else:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
            ):
                raise ValueError("retrievers must return RetrievalResult or (document_id, score).")
            result = RetrievalResult(item[0], float(item[1]), position)
        if result.document_id in seen:
            continue
        seen.add(result.document_id)
        output.append(
            RetrievalResult(
                result.document_id,
                result.score,
                len(output) + 1,
                metadata=result.metadata,
            )
        )
        if len(output) >= top_k:
            break
    return tuple(output)


def run_benchmark_suite(
    examples: Iterable[BenchmarkExample],
    retriever: Retriever,
    *,
    generator: Generator | None = None,
    top_k: int = 10,
    metric_ks: Sequence[int] = (1, 3, 5, 10),
) -> BenchmarkSuiteResult:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 1000:
        raise ValueError("top_k must be an integer between 1 and 1000.")
    if not metric_ks or any(
        isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in metric_ks
    ):
        raise ValueError("metric_ks must contain positive integers.")

    rows = []
    for example in examples:
        if not isinstance(example, BenchmarkExample):
            raise TypeError("examples must contain BenchmarkExample instances.")
        started = time.perf_counter()
        retrieval = _normalize_results(retriever(example, top_k), top_k=top_k)
        retrieval_ms = (time.perf_counter() - started) * 1000.0
        query = EvaluationQuery(
            example.example_id or f"query-{len(rows) + 1}",
            example.query,
            {identifier: 1.0 for identifier in example.relevant_ids},
            metadata=dict(example.metadata),
        )
        retrieval_metrics = evaluate_query(query, retrieval, ks=metric_ks)

        answer = ""
        generation_ms = 0.0
        generation_metrics: dict[str, float] = {}
        if generator is not None:
            generation_started = time.perf_counter()
            answer = str(generator(example, retrieval))
            generation_ms = (time.perf_counter() - generation_started) * 1000.0
            if example.answers:
                generation_metrics = {
                    "rouge_l": best_reference_score(answer, example.answers, rouge_l),
                    "chrf": best_reference_score(answer, example.answers, chrf),
                }
        rows.append(
            BenchmarkRow(
                example_id=query.query_id,
                retrieval_metrics=retrieval_metrics,
                retrieval_latency_ms=retrieval_ms,
                generated_answer=answer,
                generation_latency_ms=generation_ms,
                generation_metrics=generation_metrics,
            )
        )

    aggregate: dict[str, float] = {}
    retrieval_metric_names = sorted(
        {name for row in rows for name in row.retrieval_metrics}
    )
    generation_metric_names = sorted(
        {name for row in rows for name in row.generation_metrics}
    )
    for name in retrieval_metric_names:
        aggregate[name] = fmean(row.retrieval_metrics[name] for row in rows)
    for name in generation_metric_names:
        values = [row.generation_metrics[name] for row in rows if name in row.generation_metrics]
        aggregate[name] = fmean(values)
    if rows:
        aggregate["retrieval_latency_ms"] = fmean(row.retrieval_latency_ms for row in rows)
        aggregate["generation_latency_ms"] = fmean(row.generation_latency_ms for row in rows)
    return BenchmarkSuiteResult(tuple(rows), aggregate)
