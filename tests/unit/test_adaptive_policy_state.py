from __future__ import annotations

import hashlib

import pytest

from tools.adaptive_policy_governance import (
    AdaptivePolicyComparison,
    AdaptivePolicyDecision,
    AdaptivePolicyGate,
)
from tools.adaptive_policy_state import AdaptivePolicyStateStore


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def comparison(
    *,
    baseline: str = "policy-v1",
    candidate: str = "policy-v2",
    success_delta: float = 0.05,
) -> AdaptivePolicyComparison:
    return AdaptivePolicyComparison(
        baseline_policy_id=baseline,
        candidate_policy_id=candidate,
        case_count=100,
        success_rate_delta=success_delta,
        route_accuracy_delta=0.01,
        regret_delta=-0.02,
        cost_delta=1.0,
        latency_delta_ms=5.0,
        route_shift_jsd=0.01,
    )


def eligible(value: AdaptivePolicyComparison) -> AdaptivePolicyDecision:
    return AdaptivePolicyDecision(
        comparison=value,
        gate=AdaptivePolicyGate(),
        decision="eligible",
        reasons=(),
    )


def rollback(value: AdaptivePolicyComparison) -> AdaptivePolicyDecision:
    return AdaptivePolicyDecision(
        comparison=value,
        gate=AdaptivePolicyGate(),
        decision="rollback",
        reasons=("success_rate_regressed",),
    )


def test_shadow_promotion_and_rollback_restore_exact_baseline(tmp_path):
    path = tmp_path / "policy-state.sqlite3"
    store = AdaptivePolicyStateStore(path)
    baseline = store.bootstrap_promoted(
        owner_id="alice",
        policy_id="policy-v1",
        policy_digest=digest("policy-v1"),
        now=1.0,
    )
    assert baseline.revision == 1
    shadow = store.start_shadow(
        owner_id="alice",
        policy_id="policy-v2",
        policy_digest=digest("policy-v2"),
        now=2.0,
    )
    assert shadow.revision == 2
    assert shadow.baseline_policy_id == "policy-v1"

    offline = comparison()
    recorded = store.record_shadow_evidence(
        owner_id="alice",
        revision=shadow.revision,
        comparison=offline,
        shadow_metrics_digest=digest("shadow-metrics"),
        now=3.0,
    )
    assert recorded.comparison_digest == offline.comparison_digest
    promoted = store.promote(
        owner_id="alice",
        revision=shadow.revision,
        decision=eligible(offline),
        now=4.0,
    )
    assert promoted.state == "promoted"
    assert store.current_promoted("alice").policy_id == "policy-v2"
    assert store.get("alice", baseline.revision).state == "superseded"

    online = comparison(success_delta=-0.20)
    rolled_back = store.rollback(
        owner_id="alice",
        revision=promoted.revision,
        decision=rollback(online),
        now=5.0,
    )
    assert rolled_back.state == "rolled_back"
    assert rolled_back.comparison_digest == online.comparison_digest
    assert store.current_promoted("alice").policy_id == "policy-v1"

    store.close()
    reopened = AdaptivePolicyStateStore(path)
    assert reopened.current_promoted("alice").revision == baseline.revision
    assert reopened.get("alice", promoted.revision).state == "rolled_back"


def test_promotion_requires_recorded_exact_evidence_and_eligible_decision(tmp_path):
    store = AdaptivePolicyStateStore(tmp_path / "state.sqlite3")
    store.bootstrap_promoted(
        owner_id="alice",
        policy_id="policy-v1",
        policy_digest=digest("v1"),
        now=1.0,
    )
    shadow = store.start_shadow(
        owner_id="alice",
        policy_id="policy-v2",
        policy_digest=digest("v2"),
        now=2.0,
    )
    offline = comparison()
    with pytest.raises(RuntimeError, match="recorded shadow evidence"):
        store.promote(
            owner_id="alice",
            revision=shadow.revision,
            decision=eligible(offline),
            now=3.0,
        )
    store.record_shadow_evidence(
        owner_id="alice",
        revision=shadow.revision,
        comparison=offline,
        shadow_metrics_digest=digest("metrics"),
        now=3.0,
    )
    mismatched = comparison(candidate="policy-v3")
    with pytest.raises(RuntimeError, match="does not match"):
        store.promote(
            owner_id="alice",
            revision=shadow.revision,
            decision=eligible(mismatched),
            now=4.0,
        )
    held = AdaptivePolicyDecision(
        comparison=offline,
        gate=AdaptivePolicyGate(),
        decision="hold",
        reasons=("latency_increased",),
    )
    with pytest.raises(ValueError, match="eligible"):
        store.promote(
            owner_id="alice",
            revision=shadow.revision,
            decision=held,
            now=4.0,
        )


def test_shadow_evidence_is_immutable_within_revision(tmp_path):
    store = AdaptivePolicyStateStore(tmp_path / "state.sqlite3")
    store.bootstrap_promoted(
        owner_id="alice",
        policy_id="policy-v1",
        policy_digest=digest("v1"),
        now=1.0,
    )
    shadow = store.start_shadow(
        owner_id="alice",
        policy_id="policy-v2",
        policy_digest=digest("v2"),
        now=2.0,
    )
    offline = comparison()
    first = store.record_shadow_evidence(
        owner_id="alice",
        revision=shadow.revision,
        comparison=offline,
        shadow_metrics_digest=digest("metrics-a"),
        now=3.0,
    )
    second = store.record_shadow_evidence(
        owner_id="alice",
        revision=shadow.revision,
        comparison=offline,
        shadow_metrics_digest=digest("metrics-a"),
        now=3.5,
    )
    assert first.comparison_digest == second.comparison_digest
    with pytest.raises(RuntimeError, match="collision"):
        store.record_shadow_evidence(
            owner_id="alice",
            revision=shadow.revision,
            comparison=offline,
            shadow_metrics_digest=digest("metrics-b"),
            now=4.0,
        )


def test_owner_isolation_and_stale_revision_transitions_fail_closed(tmp_path):
    store = AdaptivePolicyStateStore(tmp_path / "state.sqlite3")
    alice = store.bootstrap_promoted(
        owner_id="alice",
        policy_id="policy-a1",
        policy_digest=digest("a1"),
        now=1.0,
    )
    bob = store.bootstrap_promoted(
        owner_id="bob",
        policy_id="policy-b1",
        policy_digest=digest("b1"),
        now=1.0,
    )
    assert alice.revision == bob.revision == 1
    assert store.current_promoted("alice").policy_id == "policy-a1"
    assert store.current_promoted("bob").policy_id == "policy-b1"
    with pytest.raises(RuntimeError, match="shadow registration"):
        empty = AdaptivePolicyStateStore(tmp_path / "empty.sqlite3")
        empty.start_shadow(
            owner_id="carol",
            policy_id="policy-c2",
            policy_digest=digest("c2"),
            now=2.0,
        )
