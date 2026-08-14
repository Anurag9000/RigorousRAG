"""Multi-window SLO burn-rate alerts over canonical service observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tools.service_slo import StageObservation


@dataclass(frozen=True)
class BurnRatePolicy:
    availability_target: float = 0.99
    short_window_requests: int = 20
    long_window_requests: int = 100
    short_burn_threshold: float = 14.4
    long_burn_threshold: float = 6.0

    def __post_init__(self) -> None:
        if not 0.0 < float(self.availability_target) < 1.0:
            raise ValueError("availability_target must be in (0, 1).")
        for name in ("short_window_requests", "long_window_requests"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.short_window_requests > self.long_window_requests:
            raise ValueError("short_window_requests must not exceed long_window_requests.")
        for name in ("short_burn_threshold", "long_burn_threshold"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive.")


@dataclass(frozen=True)
class BurnRateReport:
    short_request_count: int
    long_request_count: int
    short_error_rate: float
    long_error_rate: float
    short_burn_rate: float
    long_burn_rate: float
    alert: bool


def _error_rate(values: Sequence[StageObservation]) -> float:
    if not values:
        return 0.0
    return sum(1 for item in values if not item.success) / len(values)


def evaluate_burn_rate(
    observations: Sequence[StageObservation], policy: BurnRatePolicy | None = None
) -> BurnRateReport:
    selected = policy or BurnRatePolicy()
    short = list(observations)[-selected.short_window_requests :]
    long = list(observations)[-selected.long_window_requests :]
    short_error = _error_rate(short)
    long_error = _error_rate(long)
    budget_rate = 1.0 - selected.availability_target
    short_burn = short_error / budget_rate
    long_burn = long_error / budget_rate
    enough_data = len(short) >= selected.short_window_requests and len(long) >= selected.long_window_requests
    alert = (
        enough_data
        and short_burn >= selected.short_burn_threshold
        and long_burn >= selected.long_burn_threshold
    )
    return BurnRateReport(
        short_request_count=len(short),
        long_request_count=len(long),
        short_error_rate=short_error,
        long_error_rate=long_error,
        short_burn_rate=short_burn,
        long_burn_rate=long_burn,
        alert=alert,
    )


__all__ = ["BurnRatePolicy", "BurnRateReport", "evaluate_burn_rate"]
