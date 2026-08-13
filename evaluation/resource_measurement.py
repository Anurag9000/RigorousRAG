"""Real process resource measurement for benchmark and route evaluations.

The meter reports wall time, process CPU time, Python allocation peaks, and process
peak RSS where the standard library exposes it. Provider token/cost accounting is
explicitly supplied by the caller; GPU power and energy are intentionally not faked.
"""

from __future__ import annotations

import math
import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return parsed


def process_peak_rss_bytes() -> int | None:
    """Return process peak RSS using ``resource`` when the platform provides it."""

    try:
        import resource

        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value) or value < 0.0:
        return None
    # macOS reports bytes; Linux and the common BSD resource ABI report KiB.
    multiplier = 1 if sys.platform == "darwin" else 1024
    return int(value * multiplier)


@dataclass(frozen=True)
class ProviderUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_units: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_tokens", _integer(self.prompt_tokens, "prompt_tokens"))
        object.__setattr__(self, "completion_tokens", _integer(self.completion_tokens, "completion_tokens"))
        object.__setattr__(self, "cost_units", _finite(self.cost_units, "cost_units"))


@dataclass(frozen=True)
class ResourceUsage:
    wall_ms: float
    cpu_ms: float
    python_peak_allocated_bytes: int
    process_peak_rss_bytes: int | None
    provider: ProviderUsage


class ResourceMeter:
    """Context manager for real standard-library resource counters."""

    def __init__(self, *, provider: ProviderUsage | None = None) -> None:
        self._provider = provider or ProviderUsage()
        if not isinstance(self._provider, ProviderUsage):
            raise ValueError("provider must be ProviderUsage.")
        self._entered = False
        self._owns_tracemalloc = False
        self._wall_start = 0
        self._cpu_start = 0
        self._memory_start = 0
        self.usage: ResourceUsage | None = None

    def __enter__(self) -> "ResourceMeter":
        if self._entered:
            raise RuntimeError("ResourceMeter instances are single-use.")
        self._entered = True
        self._owns_tracemalloc = not tracemalloc.is_tracing()
        if self._owns_tracemalloc:
            tracemalloc.start()
        self._memory_start = tracemalloc.get_traced_memory()[0]
        self._wall_start = time.perf_counter_ns()
        self._cpu_start = time.process_time_ns()
        return self

    def set_provider_usage(self, usage: ProviderUsage) -> None:
        if not isinstance(usage, ProviderUsage):
            raise ValueError("usage must be ProviderUsage.")
        self._provider = usage

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        cpu_end = time.process_time_ns()
        wall_end = time.perf_counter_ns()
        _current, peak = tracemalloc.get_traced_memory()
        python_peak = max(0, int(peak - self._memory_start))
        if self._owns_tracemalloc:
            tracemalloc.stop()
        self.usage = ResourceUsage(
            wall_ms=max(0.0, (wall_end - self._wall_start) / 1_000_000.0),
            cpu_ms=max(0.0, (cpu_end - self._cpu_start) / 1_000_000.0),
            python_peak_allocated_bytes=python_peak,
            process_peak_rss_bytes=process_peak_rss_bytes(),
            provider=self._provider,
        )


@dataclass(frozen=True)
class MeasuredResult(Generic[T]):
    value: T
    usage: ResourceUsage


def measure_call(
    function: Callable[..., T],
    *args: Any,
    provider_usage: ProviderUsage | None = None,
    **kwargs: Any,
) -> MeasuredResult[T]:
    if not callable(function):
        raise ValueError("function must be callable.")
    meter = ResourceMeter(provider=provider_usage)
    with meter:
        value = function(*args, **kwargs)
    if meter.usage is None:
        raise RuntimeError("resource meter failed to finalize.")
    return MeasuredResult(value, meter.usage)


__all__ = [
    "MeasuredResult",
    "ProviderUsage",
    "ResourceMeter",
    "ResourceUsage",
    "measure_call",
    "process_peak_rss_bytes",
]
