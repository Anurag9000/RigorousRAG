from __future__ import annotations

import pytest

from tools.dead_letter import DeadLetterStore
from tools.service_resilience import (
    BackpressureSnapshot,
    CircuitBreakerState,
    SLOObservation,
    backpressure_decision,
    circuit_breaker_transition,
    compute_slo_report,
)

D = "a" * 64
R = "b" * 64


def test_dead_letter_is_digest_only_idempotent_and_owner_scoped(tmp_path):
    store = DeadLetterStore(tmp_path / "dlq.sqlite3")
    first = store.enqueue(
        owner_id="alice",
        job_id="job-1",
        job_type="ingestion",
        payload_digest=D,
        failure_type="RuntimeError",
        delivery_attempts=3,
        now=1.0,
    )
    second = store.enqueue(
        owner_id="alice",
        job_id="job-1",
        job_type="ingestion",
        payload_digest=D,
        failure_type="RuntimeError",
        delivery_attempts=3,
        now=2.0,
    )
    assert second == first
    assert first.state == "queued"
    assert first.fencing_token == 0
    assert len(first.audit_digest) == 64
    assert store.list(owner_id="alice") == (first,)
    assert store.list(owner_id="bob") == ()
    assert "payload" not in first.__dict__


def test_dead_letter_replay_requires_current_fence_and_receipt(tmp_path):
    store = DeadLetterStore(tmp_path / "dlq.sqlite3")
    item = store.enqueue(
        owner_id="alice",
        job_id="job-1",
        job_type="ingestion",
        payload_digest=D,
        failure_type="RuntimeError",
        delivery_attempts=3,
        now=1.0,
    )
    stale = store.claim(item.dead_letter_id, worker_id="worker", lease_seconds=5, now=2.0)
    current = store.claim(item.dead_letter_id, worker_id="worker", lease_seconds=5, now=8.0)
    assert stale.fencing_token == 1
    assert current.fencing_token == 2
    with pytest.raises(RuntimeError, match="fence"):
        store.mark_replayed(
            item.dead_letter_id,
            worker_id="worker",
            fencing_token=stale.fencing_token,
            replay_receipt_digest=R,
            now=9.0,
        )
    replayed = store.mark_replayed(
        item.dead_letter_id,
        worker_id="worker",
        fencing_token=current.fencing_token,
        replay_receipt_digest=R,
        now=9.0,
    )
    assert replayed.state == "replayed"
    assert replayed.replay_count == 1
    assert replayed.replay_receipt_digest == R


def test_dead_letter_release_and_exact_abandon_confirmation(tmp_path):
    store = DeadLetterStore(tmp_path / "dlq.sqlite3")
    item = store.enqueue(
        owner_id="alice",
        job_id="job-2",
        job_type="graph",
        payload_digest=D,
        failure_type="OSError",
        delivery_attempts=2,
        now=1.0,
    )
    leased = store.claim(item.dead_letter_id, worker_id="worker", now=2.0)
    queued = store.release(
        item.dead_letter_id,
        worker_id="worker",
        fencing_token=leased.fencing_token,
        now=3.0,
    )
    assert queued.state == "queued"
    with pytest.raises(ValueError, match="exactly"):
        store.abandon(
            item.dead_letter_id,
            confirm_dead_letter_id="9" * 64,
            now=4.0,
        )
    abandoned = store.abandon(
        item.dead_letter_id,
        confirm_dead_letter_id=item.dead_letter_id,
        now=4.0,
    )
    assert abandoned.state == "abandoned"


def test_backpressure_admits_defers_and_sheds_on_distinct_pressure_levels():
    admit = backpressure_decision(
        BackpressureSnapshot(2, 10, 5, 100, 0.01, 50, 100)
    )
    defer = backpressure_decision(
        BackpressureSnapshot(10, 10, 85, 100, 0.10, 120, 100)
    )
    shed = backpressure_decision(
        BackpressureSnapshot(10, 10, 100, 100, 0.60, 250, 100)
    )
    circuit = backpressure_decision(
        BackpressureSnapshot(1, 10, 1, 100, 0.0, 10, 100, circuit_open=True)
    )
    assert admit.action == "admit"
    assert defer.action == "defer"
    assert "workers_saturated" in defer.reasons
    assert shed.action == "shed"
    assert "queue_full" in shed.reasons
    assert circuit.action == "shed" and circuit.reasons == ("circuit_open",)


def test_circuit_breaker_opens_cools_probes_and_closes_after_successes():
    state = CircuitBreakerState()
    state = circuit_breaker_transition(state, success=False, now=1, failure_threshold=2)
    assert state.state == "closed" and state.consecutive_failures == 1
    state = circuit_breaker_transition(state, success=False, now=2, failure_threshold=2)
    assert state.state == "open"
    assert circuit_breaker_transition(
        state, success=None, now=20, failure_threshold=2, cooldown_seconds=30
    ).state == "open"
    state = circuit_breaker_transition(
        state, success=None, now=32, failure_threshold=2, cooldown_seconds=30
    )
    assert state.state == "half_open"
    state = circuit_breaker_transition(state, success=True, now=33, half_open_successes=2)
    assert state.state == "half_open"
    state = circuit_breaker_transition(state, success=True, now=34, half_open_successes=2)
    assert state.state == "closed"


def test_slo_report_exposes_latency_quantiles_compliance_and_error_budget_burn():
    report = compute_slo_report(
        [
            SLOObservation(True, 50),
            SLOObservation(True, 70),
            SLOObservation(True, 90),
            SLOObservation(False, 300),
        ],
        latency_slo_ms=100,
        availability_slo=0.99,
    )
    assert report.observations == 4
    assert report.availability == pytest.approx(0.75)
    assert report.error_rate == pytest.approx(0.25)
    assert report.latency_slo_fraction == pytest.approx(0.75)
    assert report.p50_latency_ms <= report.p95_latency_ms <= report.p99_latency_ms
    assert report.error_budget_burn_rate > 1.0
