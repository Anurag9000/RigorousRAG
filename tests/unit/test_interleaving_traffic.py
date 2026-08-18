from __future__ import annotations

import hashlib

import pytest

from evaluation.interleaving_traffic import InterleavingTrafficPolicy, TrafficEligibilityContext, assign_interleaving_traffic
from evaluation.retrieval_interleaving import InterleavingSpec
from orchestration.interleaving_traffic_journal import SQLiteInterleavingTrafficJournal


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def spec(tag: str = "one") -> InterleavingSpec:
    return InterleavingSpec(sha(f"experiment:{tag}"), sha("baseline"), sha(f"candidate:{tag}"), max_positions=10)


def policy(*, interleaving=1.0, baseline=0.0, group="retrieval-policy"):
    return InterleavingTrafficPolicy(
        assignment_salt_sha256=sha("assignment-salt"),
        exclusion_group_id=group,
        interleaving_fraction=interleaving,
        baseline_holdout_fraction=baseline,
        allowed_domain_ids=("science",),
        allowed_route_ids=("hybrid",),
    )


def context(*, owner="alice", unit="unit", query="query", permitted=True, safety=False, domain="science", route="hybrid"):
    return TrafficEligibilityContext(owner, sha(unit), sha(query), domain, route, permitted, safety)


def test_assignment_is_sticky_to_randomization_unit_across_queries() -> None:
    first = assign_interleaving_traffic(spec(), policy(interleaving=0.4, baseline=0.3), context(query="q1"))
    second = assign_interleaving_traffic(spec(), policy(interleaving=0.4, baseline=0.3), context(query="q2"))
    assert first.bucket == second.bucket
    assert first.arm == second.arm
    assert first.assignment_sha256 != second.assignment_sha256


def test_permission_safety_domain_and_route_eligibility_fail_closed() -> None:
    for ctx, reason in (
        (context(permitted=False), "experiment_permission_missing"),
        (context(safety=True), "safety_blocked"),
        (context(domain="finance"), "domain_not_eligible"),
        (context(route="web"), "route_not_eligible"),
    ):
        assignment = assign_interleaving_traffic(spec(), policy(), ctx)
        assert assignment.arm == "ineligible"
        assert reason in assignment.reason_codes


def test_interleaving_exposure_blocks_another_experiment_in_same_exclusion_group(tmp_path) -> None:
    journal = SQLiteInterleavingTrafficJournal(tmp_path / "traffic.sqlite3")
    first = assign_interleaving_traffic(spec("one"), policy(), context(unit="same"))
    second = assign_interleaving_traffic(spec("two"), policy(), context(unit="same", query="q2"))
    assert first.arm == second.arm == "interleaving"
    journal.record_assignment(first, now=1.0)
    with pytest.raises(RuntimeError, match="another experiment|already occupied"):
        journal.record_assignment(second, now=2.0)


def test_baseline_only_assignment_does_not_consume_interleaving_exposure_slot(tmp_path) -> None:
    journal = SQLiteInterleavingTrafficJournal(tmp_path / "traffic.sqlite3")
    baseline = assign_interleaving_traffic(spec("one"), policy(interleaving=0.0, baseline=1.0), context(unit="same"))
    exposed = assign_interleaving_traffic(spec("two"), policy(interleaving=1.0, baseline=0.0), context(unit="same", query="q2"))
    assert baseline.arm == "baseline_only"
    assert exposed.arm == "interleaving"
    journal.record_assignment(baseline, now=1.0)
    journal.record_assignment(exposed, now=2.0)
    assert journal.exposed_spec(owner_id="alice", exclusion_group_id="retrieval-policy", randomization_unit_sha256=sha("same")) == spec("two").spec_sha256


def test_mutual_exclusion_is_owner_scoped(tmp_path) -> None:
    journal = SQLiteInterleavingTrafficJournal(tmp_path / "traffic.sqlite3")
    alice = assign_interleaving_traffic(spec("one"), policy(), context(owner="alice", unit="same"))
    bob = assign_interleaving_traffic(spec("two"), policy(), context(owner="bob", unit="same", query="q2"))
    journal.record_assignment(alice, now=1.0)
    journal.record_assignment(bob, now=2.0)


def test_exact_assignment_replay_is_idempotent(tmp_path) -> None:
    journal = SQLiteInterleavingTrafficJournal(tmp_path / "traffic.sqlite3")
    assignment = assign_interleaving_traffic(spec(), policy(), context())
    assert journal.record_assignment(assignment, now=1.0) == assignment.assignment_sha256
    assert journal.record_assignment(assignment, now=2.0) == assignment.assignment_sha256
