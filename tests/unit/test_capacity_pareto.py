import pytest

from evaluation.capacity_pareto import (
    ArchitecturePoint,
    CapacityGate,
    LoadSample,
    pareto_frontier,
    summarize_capacity,
)


def test_capacity_summary_tracks_throughput_tail_latency_errors_and_rejections():
    summary = summarize_capacity(
        [
            LoadSample(0.10, True),
            LoadSample(0.20, True),
            LoadSample(0.30, False),
            LoadSample(0.40, False, rejected=True),
        ],
        wall_seconds=2.0,
    )
    assert summary.requests == 4
    assert summary.throughput_rps == pytest.approx(2.0)
    assert summary.success_rate == pytest.approx(0.5)
    assert summary.error_rate == pytest.approx(0.5)
    assert summary.rejection_rate == pytest.approx(0.25)
    assert summary.p50_latency_seconds == pytest.approx(0.20)
    assert summary.p95_latency_seconds == pytest.approx(0.40)
    assert summary.p99_latency_seconds == pytest.approx(0.40)


def test_capacity_gate_reports_every_failed_dimension():
    summary = summarize_capacity(
        [LoadSample(1.0, True), LoadSample(2.0, False, rejected=True)],
        wall_seconds=2.0,
    )
    passed, reasons = CapacityGate(
        minimum_throughput_rps=2.0,
        minimum_success_rate=0.9,
        maximum_p95_latency_seconds=1.0,
        maximum_rejection_rate=0.1,
    ).evaluate(summary)
    assert passed is False
    assert reasons == ("throughput", "success_rate", "p95_latency", "rejection_rate")


def test_capacity_gate_passes_a_healthy_envelope():
    summary = summarize_capacity([LoadSample(0.1, True) for _ in range(20)], wall_seconds=1.0)
    passed, reasons = CapacityGate(
        minimum_throughput_rps=10,
        minimum_success_rate=0.99,
        maximum_p95_latency_seconds=0.2,
        maximum_rejection_rate=0,
    ).evaluate(summary)
    assert passed is True
    assert reasons == ()


def test_pareto_frontier_removes_strictly_dominated_architectures():
    points = [
        ArchitecturePoint("fast-good", quality=0.90, latency_seconds=0.10, cost=0.02, memory_bytes=100),
        ArchitecturePoint("dominated", quality=0.85, latency_seconds=0.20, cost=0.03, memory_bytes=120),
        ArchitecturePoint("high-quality", quality=0.95, latency_seconds=0.30, cost=0.05, memory_bytes=150),
        ArchitecturePoint("cheap", quality=0.80, latency_seconds=0.08, cost=0.005, memory_bytes=80),
    ]
    frontier = pareto_frontier(points)
    assert [point.name for point in frontier] == ["high-quality", "fast-good", "cheap"]


def test_pareto_frontier_rejects_ambiguous_duplicate_names():
    point = ArchitecturePoint("same", 0.5, 1.0, 1.0)
    with pytest.raises(ValueError, match="unique"):
        pareto_frontier([point, ArchitecturePoint("same", 0.6, 0.9, 0.8)])


def test_capacity_inputs_fail_closed():
    with pytest.raises(ValueError):
        summarize_capacity([], wall_seconds=0)
    with pytest.raises(ValueError):
        LoadSample(float("nan"), True)
    with pytest.raises(ValueError):
        LoadSample(1, True, rejected=True)
    with pytest.raises(ValueError):
        CapacityGate(minimum_success_rate=1.1)
