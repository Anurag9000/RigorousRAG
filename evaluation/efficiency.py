"""Latency, throughput, token, cost, and resource accounting helpers."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class LatencySummary:
    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    minimum_ms: float
    maximum_ms: float


def summarize_latencies(values_ms: Iterable[float]) -> LatencySummary:
    values = sorted(float(value) for value in values_ms)
    if any((not math.isfinite(value) or value < 0) for value in values):
        raise ValueError("latencies must be finite and non-negative.")
    if not values:
        return LatencySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def percentile(q: float) -> float:
        index = (len(values) - 1) * q
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        fraction = index - lower
        return values[lower] * (1.0 - fraction) + values[upper] * fraction

    return LatencySummary(
        len(values),
        statistics.fmean(values),
        statistics.median(values),
        percentile(0.95),
        percentile(0.99),
        values[0],
        values[-1],
    )


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def estimate_cost(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    prompt_usd_per_million: float,
    completion_usd_per_million: float,
) -> Usage:
    for value in (prompt_tokens, completion_tokens):
        if isinstance(value, bool) or int(value) != value or value < 0:
            raise ValueError("token counts must be non-negative integers.")
    for value in (prompt_usd_per_million, completion_usd_per_million):
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError("token prices must be finite and non-negative.")
    cost = (
        prompt_tokens * prompt_usd_per_million
        + completion_tokens * completion_usd_per_million
    ) / 1_000_000.0
    return Usage(int(prompt_tokens), int(completion_tokens), cost)


def tokens_per_second(tokens: int, duration_seconds: float) -> float:
    if tokens < 0 or not math.isfinite(float(duration_seconds)) or duration_seconds <= 0:
        raise ValueError("tokens must be non-negative and duration_seconds positive.")
    return tokens / duration_seconds


def throughput(completed_requests: int, duration_seconds: float) -> float:
    if completed_requests < 0 or not math.isfinite(float(duration_seconds)) or duration_seconds <= 0:
        raise ValueError("completed_requests must be non-negative and duration_seconds positive.")
    return completed_requests / duration_seconds


class Stopwatch:
    def __init__(self) -> None:
        self.elapsed_seconds: Optional[float] = None
        self._started: Optional[float] = None

    def __enter__(self) -> "Stopwatch":
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._started is not None
        self.elapsed_seconds = time.perf_counter() - self._started

    @property
    def elapsed_ms(self) -> float:
        return 0.0 if self.elapsed_seconds is None else self.elapsed_seconds * 1000.0
