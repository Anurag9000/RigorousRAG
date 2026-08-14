from __future__ import annotations

import pytest

from tools.deployment_canary import (
    DeploymentCandidate,
    DeploymentJournal,
    evaluate_deployment_canary,
)
from tools.fault_injection import FaultInjector, InjectedFault
from tools.recovery_catalog import RecoveryAsset, RecoveryCatalog, plan_restore
from tools.recovery_control import CanaryAction, CanaryObservation, CanaryThresholds
from tools.service_slo import SLOReport, StageObservation
from tools.slo_alerts import BurnRateReport
from tools.sql_durable_queue import SQLiteDurableQueue
from tools.telemetry_export import (
    OpenTelemetrySpanSink,
    TelemetryExportPolicy,
    prometheus_slo_text,
)


class Clock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_sqlite_queue_persists_idempotency_and_visibility_across_workers(tmp_path) -> None:
    clock = Clock()
    path = tmp_path / "queue.sqlite3"
    first_worker = SQLiteDurableQueue(path, max_attempts=2, clock=clock)
    second_worker = SQLiteDurableQueue(path, max_attempts=2, clock=clock)
    message_id = first_worker.enqueue({"source": "doc-1"}, idempotency_key="doc-1")
    assert second_worker.enqueue({"source": "ignored"}, idempotency_key="doc-1") == message_id

    first = first_worker.claim("worker-a", visibility_timeout=5)
    assert first is not None and first.message_id == message_id and first.attempts == 1
    assert second_worker.claim("worker-b", visibility_timeout=5) is None
    clock.advance(6)
    second = second_worker.claim("worker-b", visibility_timeout=5)
    assert second is not None and second.message_id == message_id and second.attempts == 2
    second_worker.ack(second.receipt)
    assert first_worker.claim("worker-c", visibility_timeout=5) is None


def test_sqlite_queue_dead_letters_after_visibility_exhaustion(tmp_path) -> None:
    clock = Clock()
    queue = SQLiteDurableQueue(tmp_path / "queue.sqlite3", max_attempts=1, clock=clock)
    message_id = queue.enqueue({"kind": "parse"}, idempotency_key="parse-1")
    claimed = queue.claim("worker", visibility_timeout=2)
    assert claimed is not None
    clock.advance(3)
    dead = queue.dead_letters()
    assert len(dead) == 1 and dead[0].message_id == message_id and dead[0].attempts == 1


def test_recovery_catalog_orders_dependencies_and_fails_closed() -> None:
    digest = "a" * 64
    catalog = RecoveryCatalog(
        "generation-7",
        (
            RecoveryAsset("source", "source.snapshot", digest),
            RecoveryAsset("bm25", "bm25.snapshot", digest, ("source",)),
            RecoveryAsset("vector", "vector.snapshot", digest, ("source",)),
            RecoveryAsset("graph", "graph.snapshot", digest, ("vector",)),
            RecoveryAsset("model", "model.snapshot", digest, ("source",)),
            RecoveryAsset("adapter", "adapter.snapshot", digest, ("model",)),
            RecoveryAsset("policy", "policy.snapshot", digest, ("adapter", "graph")),
        ),
    )
    available = {asset.artifact: digest for asset in catalog.assets}
    plan = plan_restore(catalog, available)
    assert plan.ready
    positions = {name: index for index, name in enumerate(plan.restore_order)}
    assert positions["source"] < positions["vector"] < positions["graph"] < positions["policy"]
    assert positions["source"] < positions["model"] < positions["adapter"] < positions["policy"]

    corrupted = plan_restore(catalog, {**available, "graph.snapshot": "b" * 64})
    assert not corrupted.ready
    assert corrupted.reason_codes == ("restore_checksum_mismatch",)
    assert corrupted.checksum_mismatches == ("graph.snapshot",)

    cyclic = RecoveryCatalog(
        "bad",
        (
            RecoveryAsset("a", "a.snapshot", digest, ("b",)),
            RecoveryAsset("b", "b.snapshot", digest, ("a",)),
        ),
    )
    cycle_plan = plan_restore(cyclic, {"a.snapshot": digest, "b.snapshot": digest})
    assert not cycle_plan.ready and cycle_plan.cyclic_assets == ("a", "b")


def test_telemetry_export_drops_private_attributes_and_bounds_cardinality() -> None:
    class Span:
        def __init__(self) -> None:
            self.attributes: dict[str, object] = {}
            self.ended = False

        def set_attribute(self, key: str, value: object) -> None:
            self.attributes[key] = value

        def end(self) -> None:
            self.ended = True

    class Tracer:
        def __init__(self) -> None:
            self.spans: list[Span] = []

        def start_span(self, _name: str) -> Span:
            span = Span()
            self.spans.append(span)
            return span

    tracer = Tracer()
    sink = OpenTelemetrySpanSink(
        tracer,
        policy=TelemetryExportPolicy(
            allowed_attribute_keys=("model",),
            max_value_length=8,
            max_distinct_values_per_key=1,
        ),
    )
    sink.emit_span(
        StageObservation(
            "trace-1",
            "retrieve",
            10.0,
            True,
            attributes={"model": "abcdefghijk", "query": "private user text"},
        )
    )
    sink.emit_span(
        StageObservation("trace-2", "retrieve", 11.0, True, attributes={"model": "different"})
    )
    assert tracer.spans[0].attributes["rigorousrag.model"] == "abcdefgh"
    assert "rigorousrag.query" not in tracer.spans[0].attributes
    assert tracer.spans[1].attributes["rigorousrag.model"] == "_other"
    assert all(span.ended for span in tracer.spans)

    metrics = prometheus_slo_text(
        SLOReport(100, 0.99, 0.96, 120.0, 1.0, 1.0, 0.0, True),
        BurnRateReport(20, 100, 0.0, 0.01, 0.0, 1.0, False),
    )
    assert "rigorousrag_error_budget_remaining 0.000000000000" in metrics
    assert "rigorousrag_burn_rate_alert 0" in metrics


def test_deployment_canary_binds_rollback_target_and_hash_chain(tmp_path) -> None:
    candidate = DeploymentCandidate(
        "candidate-9",
        (("api", "a" * 64), ("retriever", "b" * 64)),
        (("api", "c" * 64), ("retriever", "d" * 64)),
        evidence_generated_at=100.0,
    )
    thresholds = CanaryThresholds(
        min_samples=10,
        max_error_rate=0.02,
        max_p95_latency_ms=500.0,
        min_quality_score=0.8,
    )
    decision = evaluate_deployment_canary(
        candidate,
        CanaryObservation(10, 0.1, 700.0, 0.7),
        thresholds=thresholds,
        now=110.0,
    )
    assert decision.action == CanaryAction.ROLLBACK
    assert decision.rollback_artifacts == candidate.known_good_artifacts
    journal = DeploymentJournal(tmp_path / "deployments.sqlite3", clock=lambda: 111.0)
    first = journal.append(decision, actor="release-controller")
    assert journal.append(decision, actor="different-replay-actor") == first
    assert journal.verify_chain()

    stale = evaluate_deployment_canary(
        candidate,
        CanaryObservation(100, 0.0, 100.0, 1.0),
        thresholds=thresholds,
        now=1000.0,
        max_evidence_age_seconds=60.0,
    )
    assert stale.action == CanaryAction.HOLD
    assert stale.reason_codes == ("canary_evidence_stale",)


def test_fault_injector_is_deterministic_and_auditable() -> None:
    clock = Clock()
    injector = FaultInjector({"after_backup_copy": (2,)}, clock=clock)
    injector.checkpoint("after_backup_copy")
    clock.advance(1)
    with pytest.raises(InjectedFault) as exc_info:
        injector.checkpoint("after_backup_copy")
    assert exc_info.value.invocation == 2
    injector.checkpoint("after_backup_copy")
    assert [event.injected for event in injector.events()] == [False, True, False]
    assert injector.count("after_backup_copy") == 3
