from __future__ import annotations

import hashlib

import pytest

from orchestration.multi_region_authority import (
    MultiRegionFailoverPolicy,
    RegionHealthObservation,
    SQLiteRegionAuthorityStore,
    decide_region_authority,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _health(
    region: str,
    *,
    now: float = 100.0,
    ready: bool = True,
    lag: float = 0.5,
    recovery_age: float = 1.0,
    observed_at: float | None = None,
) -> RegionHealthObservation:
    return RegionHealthObservation(
        region_id=region,
        observed_at=now if observed_at is None else observed_at,
        ready_for_reads=ready,
        ready_for_writes=ready,
        replication_lag_seconds=lag,
        recovery_point_at=now - recovery_age,
        evidence_sha256=_sha(f"{region}:{now}:{ready}:{lag}:{recovery_age}:{observed_at}"),
    )


def _policy(*, automatic_failback: bool = False) -> MultiRegionFailoverPolicy:
    return MultiRegionFailoverPolicy(
        primary_region="primary",
        failover_regions=("secondary-a", "secondary-b"),
        max_health_age_seconds=10.0,
        max_replication_lag_seconds=2.0,
        max_recovery_point_age_seconds=10.0,
        allow_automatic_failback=automatic_failback,
    )


def test_explicit_failback_allows_unhealthy_secondary_to_return_to_healthy_primary() -> None:
    observations = (
        _health("primary"),
        _health("secondary-a", ready=False),
        _health("secondary-b"),
    )
    held = decide_region_authority(
        owner_id="owner",
        service_id="retrieval",
        current_region="secondary-a",
        observations=observations,
        policy=_policy(),
        now=100.0,
        explicit_failback=False,
    )
    assert held.action == "hold"
    assert "healthy_primary_requires_failback_authorization" in held.reason_codes

    decision = decide_region_authority(
        owner_id="owner",
        service_id="retrieval",
        current_region="secondary-a",
        observations=observations,
        policy=_policy(),
        now=100.0,
        explicit_failback=True,
    )
    assert decision.action == "failback"
    assert decision.target_region == "primary"


def test_future_health_observation_is_not_treated_as_fresh() -> None:
    decision = decide_region_authority(
        owner_id="owner",
        service_id="retrieval",
        current_region=None,
        observations=(
            _health("primary", observed_at=101.0),
            _health("secondary-a", ready=False),
            _health("secondary-b", ready=False),
        ),
        policy=_policy(),
        now=100.0,
    )
    assert decision.action == "hold"
    assert decision.target_region is None


def test_failover_increments_fence_and_stale_region_is_rejected(tmp_path) -> None:
    store = SQLiteRegionAuthorityStore(tmp_path / "authority.sqlite")
    policy = _policy()
    bootstrap = decide_region_authority(
        owner_id="owner",
        service_id="retrieval",
        current_region=None,
        observations=(_health("primary"), _health("secondary-a")),
        policy=policy,
        now=100.0,
    )
    first = store.apply_decision(bootstrap, expected_revision=0, now=100.0)
    assert first.authority_region == "primary"
    assert first.fencing_token == 1

    failover = decide_region_authority(
        owner_id="owner",
        service_id="retrieval",
        current_region="primary",
        observations=(_health("primary", ready=False), _health("secondary-a")),
        policy=policy,
        now=101.0,
    )
    second = store.apply_decision(failover, expected_revision=first.revision, now=101.0)
    assert second.authority_region == "secondary-a"
    assert second.fencing_token == 2
    with pytest.raises(RuntimeError, match="stale or non-authoritative"):
        store.assert_write_authority(
            owner_id="owner",
            service_id="retrieval",
            region_id="primary",
            fencing_token=first.fencing_token,
        )
    assert store.assert_write_authority(
        owner_id="owner",
        service_id="retrieval",
        region_id="secondary-a",
        fencing_token=second.fencing_token,
    ) == second


def test_no_change_is_journaled_without_rotating_write_fence(tmp_path) -> None:
    store = SQLiteRegionAuthorityStore(tmp_path / "authority.sqlite")
    policy = _policy()
    bootstrap = decide_region_authority(
        owner_id="owner",
        service_id="retrieval",
        current_region=None,
        observations=(_health("primary"),),
        policy=policy,
        now=100.0,
    )
    first = store.apply_decision(bootstrap, expected_revision=0, now=100.0)
    no_change = decide_region_authority(
        owner_id="owner",
        service_id="retrieval",
        current_region="primary",
        observations=(_health("primary", now=101.0),),
        policy=policy,
        now=101.0,
    )
    second = store.apply_decision(no_change, expected_revision=first.revision, now=101.0)
    assert second == first
    history = store.history(owner_id="owner", service_id="retrieval")
    assert {row.action for row in history} == {"bootstrap", "no_change"}
    assert all(row.resulting_fencing_token == 1 for row in history)


def test_transition_requires_current_revision_and_current_region(tmp_path) -> None:
    store = SQLiteRegionAuthorityStore(tmp_path / "authority.sqlite")
    policy = _policy()
    bootstrap = decide_region_authority(
        owner_id="owner",
        service_id="retrieval",
        current_region=None,
        observations=(_health("primary"),),
        policy=policy,
        now=100.0,
    )
    first = store.apply_decision(bootstrap, expected_revision=0, now=100.0)
    failover = decide_region_authority(
        owner_id="owner",
        service_id="retrieval",
        current_region="primary",
        observations=(_health("primary", ready=False), _health("secondary-a")),
        policy=policy,
        now=101.0,
    )
    with pytest.raises(RuntimeError, match="transition CAS failed"):
        store.apply_decision(failover, expected_revision=first.revision + 1, now=101.0)
