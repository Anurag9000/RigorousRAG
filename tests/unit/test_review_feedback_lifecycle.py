from __future__ import annotations

import pytest

from evaluation.drift import DriftThresholds, evaluate_drift
from evaluation.fault_injection import FaultInjector, FaultRule, InjectedFault
from tools.feedback_store import FeedbackStore
from tools.governed_answer_flow import AnswerMaterial, run_governed_answer_flow
from tools.lease_coordination import LeaseCoordinator
from tools.review_routing import route_for_review
from tools.review_store import ReviewStore


def test_fencing_tokens_survive_release_and_expiry(tmp_path) -> None:
    coordinator = LeaseCoordinator(tmp_path / "leases.db")
    first = coordinator.acquire(owner_id="alice", resource_id="index-a", holder_id="worker-1", ttl_seconds=10, now=100)
    assert first is not None and first.fencing_token == 1
    assert coordinator.acquire(owner_id="alice", resource_id="index-a", holder_id="worker-2", ttl_seconds=10, now=101) is None
    renewed = coordinator.renew(first, ttl_seconds=20, now=102)
    assert renewed is not None and renewed.fencing_token == first.fencing_token
    assert coordinator.validate_fence(renewed, now=103) is True
    assert coordinator.release(renewed) is True
    second = coordinator.acquire(owner_id="alice", resource_id="index-a", holder_id="worker-2", ttl_seconds=5, now=104)
    assert second is not None and second.fencing_token == 2
    assert coordinator.validate_fence(first, now=104) is False
    expired_replacement = coordinator.acquire(owner_id="alice", resource_id="index-a", holder_id="worker-3", ttl_seconds=5, now=110)
    assert expired_replacement is not None and expired_replacement.fencing_token == 3


def test_persistent_review_claim_reclaim_and_stale_resolution(tmp_path) -> None:
    store = ReviewStore(tmp_path / "reviews.db")
    decision = route_for_review(aggregate_uncertainty=0.5, evidence_conflict=0.4)
    assert decision.route == "human_review"
    first = store.enqueue(
        owner_id="alice",
        request_id="req-1",
        decision=decision,
        query="sensitive raw query",
        metadata={"route": "corpus-hybrid"},
        now=100,
    )
    replay = store.enqueue(owner_id="alice", request_id="req-1", decision=decision, query="sensitive raw query", now=101)
    assert replay.request_id == first.request_id
    assert replay.query_sha256 == first.query_sha256
    assert "sensitive raw query" not in repr(first)
    claimed = store.claim_next(owner_id="alice", reviewer_id="reviewer-a", ttl_seconds=5, now=102)
    assert claimed is not None and claimed.state == "claimed" and claimed.lease_token == 1
    assert store.resolve(claimed, resolution="approved", now=108) is False
    reclaimed = store.claim_next(owner_id="alice", reviewer_id="reviewer-b", ttl_seconds=5, now=108)
    assert reclaimed is not None and reclaimed.lease_token == 2
    assert store.resolve(claimed, resolution="stale", now=109) is False
    assert store.resolve(reclaimed, resolution="approved", now=109) is True
    resolved = store.get(owner_id="alice", request_id="req-1")
    assert resolved is not None and resolved.state == "resolved" and resolved.resolution == "approved"


def test_feedback_is_idempotent_hashed_and_exportable(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "feedback.db")
    event = store.put(
        owner_id="alice",
        event_id="event-1",
        kind="route_preference",
        subject_id="route-comparison-1",
        query="raw private query",
        evidence="raw evidence passage",
        weight=2.0,
        metadata={"preferred": "corpus-sparse", "rejected": "dense"},
        created_at=100,
    )
    replay = store.put(
        owner_id="alice",
        event_id="event-1",
        kind="route_preference",
        subject_id="route-comparison-1",
        query="raw private query",
        evidence="raw evidence passage",
        weight=2.0,
        metadata={"preferred": "corpus-sparse", "rejected": "dense"},
        created_at=100,
    )
    assert replay == event
    assert event.query_sha256 is not None and len(event.query_sha256) == 64
    assert event.evidence_sha256 is not None and len(event.evidence_sha256) == 64
    assert "raw private query" not in repr(event)
    exported = store.export_active_learning(owner_id="alice")
    assert len(exported) == 1
    assert exported[0].metadata["preferred"] == "corpus-sparse"


def test_drift_report_detects_score_route_calibration_latency_and_cost_shift() -> None:
    report = evaluate_drift(
        reference_scores=(0.1, 0.1, 0.2, 0.2),
        current_scores=(0.8, 0.8, 0.9, 0.9),
        score_edges=(0.0, 0.5, 1.0),
        reference_routes={"dense": 100, "web": 1},
        current_routes={"dense": 1, "web": 100},
        reference_calibration_error=0.05,
        current_calibration_error=0.20,
        reference_latency=(100, 100),
        current_latency=(160, 160),
        reference_cost=(1.0, 1.0),
        current_cost=(1.8, 1.8),
        thresholds=DriftThresholds(
            score_psi=0.1,
            route_jsd=0.1,
            calibration_shift=0.1,
            latency_relative=0.2,
            cost_relative=0.2,
        ),
    )
    assert set(report.alerts) == {
        "score_distribution_drift",
        "route_mix_drift",
        "calibration_drift",
        "latency_drift",
        "cost_drift",
    }


def test_fault_injection_is_deterministic_and_resettable() -> None:
    injector = FaultInjector((FaultRule("retrieve", 2, "boom"),))
    assert injector.call("retrieve", lambda: 1) == 1
    with pytest.raises(InjectedFault) as error:
        injector.call("retrieve", lambda: 2)
    assert error.value.stage == "retrieve"
    assert error.value.occurrence == 2
    assert injector.events[-1].injected is True
    injector.reset()
    assert injector.counts == {}
    assert injector.call("retrieve", lambda: 3) == 3


def test_governed_flow_automatic_answer_can_cache(tmp_path) -> None:
    cached: list[str] = []
    events: list[str] = []

    def retrieve(_query, _route, _context):
        return AnswerMaterial(
            "supported answer",
            0.98,
            0.97,
            proof_completeness=1.0,
            independent_sources=2,
            evidence_ids=("ev-1", "ev-2"),
        )

    result = run_governed_answer_flow(
        "Explain this evidence",
        owner_id="alice",
        request_id="req-auto",
        retrieve=retrieve,
        cache_store=lambda _q, _r, _c, material: cached.append(material.answer),
        event_hook=lambda event, _payload: events.append(event),
    )
    assert result.decision.route == "automatic"
    assert result.answer == "supported answer"
    assert cached == ["supported answer"]
    assert "cache_store" in events


def test_governed_flow_review_persists_and_never_caches(tmp_path) -> None:
    store = ReviewStore(tmp_path / "flow-reviews.db")
    cached: list[str] = []

    def retrieve(_query, _route, _context):
        return AnswerMaterial(
            "uncertain answer",
            0.55,
            0.55,
            evidence_conflict=0.55,
            proof_completeness=0.6,
            independent_sources=1,
            evidence_ids=("ev-1",),
        )

    result = run_governed_answer_flow(
        "Review this claim",
        owner_id="alice",
        request_id="req-review",
        retrieve=retrieve,
        review_store=store,
        cache_store=lambda _q, _r, _c, material: cached.append(material.answer),
    )
    assert result.decision.route == "human_review"
    assert result.review_record is not None
    assert result.review_record.query_sha256 is not None
    assert cached == []
    assert store.get(owner_id="alice", request_id="req-review") is not None


def test_security_block_bypasses_all_external_query_processors_and_cache() -> None:
    calls: list[str] = []

    def forbidden(*_args, **_kwargs):
        calls.append("called")
        raise AssertionError("blocked requests must not reach this callback")

    result = run_governed_answer_flow(
        "blocked query",
        owner_id="alice",
        request_id="req-block",
        retrieve=forbidden,
        classifier=forbidden,
        classifier_version="classifier-v1",
        entity_resolver=forbidden,
        temporal_parser=forbidden,
        cache_lookup=forbidden,
        cache_store=forbidden,
        security_check=lambda _owner, _query: False,
    )
    assert result.decision.route == "block"
    assert result.material is None
    assert result.cache_hit is False
    assert calls == []
