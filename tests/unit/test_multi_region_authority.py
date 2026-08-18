from __future__ import annotations

import hashlib

import pytest

from orchestration.current_multi_region_authority import decide_current_region_authority
from orchestration.multi_region_authority import (
    MultiRegionFailoverPolicy,
    RegionHealthObservation,
    SQLiteRegionAuthorityStore,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def health(
    region: str,
    *,
    observed_at: float = 100.0,
    reads: bool = True,
    writes: bool = True,
    lag: float = 1.0,
    recovery_point_at: float = 99.0,
) -> RegionHealthObservation:
    return RegionHealthObservation(
        region_id=region,
        observed_at=observed_at,
        ready_for_reads=reads,
        ready_for_writes=writes,
        replication_lag_seconds=lag,
        recovery_point_at=recovery_point_at,
        evidence_sha256=sha(f"health:{region}:{observed_at}:{reads}:{writes}:{lag}:{recovery_point_at}"),
    )


def policy(**overrides) -> MultiRegionFailoverPolicy:
    values = dict(
        primary_region="india-primary",
        failover_regions=("india-secondary", "singapore-secondary"),
        max_health_age_seconds=10.0,
        max_replication_lag_seconds=5.0,
        max_recovery_point_age_seconds=10.0,
        require_primary_unhealthy_for_failover=True,
        allow_automatic_failback=False,
    )
    values.update(overrides)
    return MultiRegionFailoverPolicy(**values)


def decide(current, observations, *, explicit_failback=False, selected_policy=None):
    return decide_current_region_authority(
        owner_id="alice",
        service_id="rag-write-plane",
        current_region=current,
        observations=observations,
        policy=selected_policy or policy(),
        now=100.0,
        explicit_failback=explicit_failback,
    )


def test_bootstrap_prefers_healthy_primary() -> None:
    decision = decide(None, (health("india-primary"), health("india-secondary")))
    assert decision.action == "bootstrap"
    assert decision.target_region == "india-primary"


def test_bootstrap_uses_first_healthy_failover_when_primary_is_unhealthy() -> None:
    decision = decide(
        None,
        (
            health("india-primary", writes=False),
            health("india-secondary"),
            health("singapore-secondary"),
        ),
    )
    assert decision.action == "bootstrap"
    assert decision.target_region == "india-secondary"


def test_no_healthy_bootstrap_region_holds_authority() -> None:
    decision = decide(
        None,
        (
            health("india-primary", writes=False),
            health("india-secondary", writes=False),
        ),
    )
    assert decision.action == "hold"
    assert decision.target_region is None
    assert "no_healthy_region_available" in decision.reason_codes


def test_replication_lag_and_stale_health_make_failover_target_ineligible() -> None:
    decision = decide(
        "india-primary",
        (
            health("india-primary", writes=False),
            health("india-secondary", lag=100.0),
            health("singapore-secondary", observed_at=70.0, recovery_point_at=69.0),
        ),
    )
    assert decision.action == "hold"
    assert "no_safe_failover_target" in decision.reason_codes


def test_unhealthy_primary_fails_over_to_first_safe_secondary() -> None:
    decision = decide(
        "india-primary",
        (
            health("india-primary", writes=False),
            health("india-secondary"),
            health("singapore-secondary"),
        ),
    )
    assert decision.action == "failover"
    assert decision.target_region == "india-secondary"


def test_healthy_secondary_requires_explicit_or_automatic_failback() -> None:
    observations = (health("india-primary"), health("india-secondary"))
    held = decide("india-secondary", observations)
    assert held.action == "no_change"
    assert held.target_region == "india-secondary"

    explicit = decide("india-secondary", observations, explicit_failback=True)
    assert explicit.action == "failback"
    assert explicit.target_region == "india-primary"

    automatic = decide(
        "india-secondary",
        observations,
        selected_policy=policy(allow_automatic_failback=True),
    )
    assert automatic.action == "failback"
    assert automatic.target_region == "india-primary"


def test_unhealthy_secondary_can_explicitly_fail_back_to_healthy_primary() -> None:
    decision = decide(
        "india-secondary",
        (health("india-primary"), health("india-secondary", writes=False)),
        explicit_failback=True,
    )
    assert decision.action == "failback"
    assert decision.target_region == "india-primary"


def test_store_bootstrap_and_failover_increment_monotonic_fence(tmp_path) -> None:
    store = SQLiteRegionAuthorityStore(tmp_path / "authority.sqlite3")
    bootstrap = decide(None, (health("india-primary"), health("india-secondary")))
    primary = store.apply_decision(bootstrap, expected_revision=None, now=100.0)
    assert primary.authority_region == "india-primary"
    assert primary.revision == 1
    assert primary.fencing_token == 1
    store.assert_write_authority(
        owner_id="alice",
        service_id="rag-write-plane",
        region_id="india-primary",
        fencing_token=1,
    )

    failover = decide(
        "india-primary",
        (health("india-primary", writes=False), health("india-secondary")),
    )
    secondary = store.apply_decision(failover, expected_revision=primary.revision, now=101.0)
    assert secondary.authority_region == "india-secondary"
    assert secondary.revision == 2
    assert secondary.fencing_token == 2

    with pytest.raises(RuntimeError, match="stale|non-authoritative"):
        store.assert_write_authority(
            owner_id="alice",
            service_id="rag-write-plane",
            region_id="india-primary",
            fencing_token=1,
        )


def test_stale_revision_cannot_apply_region_transition(tmp_path) -> None:
    store = SQLiteRegionAuthorityStore(tmp_path / "authority.sqlite3")
    primary = store.apply_decision(
        decide(None, (health("india-primary"), health("india-secondary"))),
        expected_revision=None,
        now=100.0,
    )
    failover = decide(
        "india-primary",
        (health("india-primary", writes=False), health("india-secondary")),
    )
    with pytest.raises(RuntimeError, match="CAS failed"):
        store.apply_decision(failover, expected_revision=primary.revision + 1, now=101.0)


def test_region_authority_is_owner_and_service_scoped(tmp_path) -> None:
    store = SQLiteRegionAuthorityStore(tmp_path / "authority.sqlite3")
    alice = store.apply_decision(
        decide(None, (health("india-primary"), health("india-secondary"))),
        expected_revision=None,
        now=100.0,
    )
    assert store.get(owner_id="alice", service_id="rag-write-plane") == alice
    assert store.get(owner_id="bob", service_id="rag-write-plane") is None
    assert store.get(owner_id="alice", service_id="graph-write-plane") is None
