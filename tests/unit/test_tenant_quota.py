from __future__ import annotations

import pytest

from tools.tenant_quota import (
    QuotaExceededError,
    TenantQuotaConfig,
    TenantQuotaStore,
)


def configure(
    store: TenantQuotaStore,
    owner: str = "alice",
    *,
    requests: int = 2,
    units: float = 5.0,
    inflight: int = 2,
    window: int = 60,
    lease: int = 10,
):
    return store.configure(
        owner,
        TenantQuotaConfig(
            request_limit=requests,
            unit_limit=units,
            inflight_limit=inflight,
            window_seconds=window,
            lease_seconds=lease,
        ),
    )


def test_reserve_commit_release_and_snapshot_are_atomic(tmp_path):
    path = tmp_path / "quota.sqlite3"
    store = TenantQuotaStore(path)
    assert configure(store) == 1
    first = store.reserve(
        owner_id="alice",
        reservation_id="req-1",
        units=2.0,
        now=1.0,
    )
    second = store.reserve(
        owner_id="alice",
        reservation_id="req-2",
        units=2.0,
        now=2.0,
    )
    snapshot = store.snapshot("alice", now=3.0)
    assert snapshot.reserved_requests == 2
    assert snapshot.reserved_units == 4.0
    assert snapshot.remaining_requests == 0
    assert snapshot.active_inflight == 2

    committed = store.commit(
        owner_id="alice",
        reservation_id=first.reservation_id,
        fencing_token=first.fencing_token,
        actual_units=1.0,
        now=4.0,
    )
    released = store.release(
        owner_id="alice",
        reservation_id=second.reservation_id,
        fencing_token=second.fencing_token,
        now=5.0,
    )
    assert committed.state == "committed" and committed.committed_units == 1.0
    assert released.state == "released"
    snapshot = store.snapshot("alice", now=6.0)
    assert snapshot.committed_requests == 1
    assert snapshot.committed_units == 1.0
    assert snapshot.reserved_requests == 0
    assert snapshot.remaining_requests == 1
    assert snapshot.remaining_units == 4.0

    store.close()
    reopened = TenantQuotaStore(path)
    snapshot = reopened.snapshot("alice", now=6.0)
    assert snapshot.committed_requests == 1
    assert snapshot.committed_units == 1.0


def test_request_unit_and_inflight_limits_fail_closed(tmp_path):
    store = TenantQuotaStore(tmp_path / "quota.sqlite3")
    configure(store, requests=2, units=10.0, inflight=2)
    store.reserve(owner_id="alice", reservation_id="r1", units=1.0, now=1.0)
    store.reserve(owner_id="alice", reservation_id="r2", units=1.0, now=1.0)
    with pytest.raises(QuotaExceededError, match="request"):
        store.reserve(owner_id="alice", reservation_id="r3", units=1.0, now=1.0)

    store = TenantQuotaStore(tmp_path / "quota-units.sqlite3")
    configure(store, requests=3, units=4.0, inflight=3)
    store.reserve(owner_id="alice", reservation_id="u1", units=2.0, now=1.0)
    store.reserve(owner_id="alice", reservation_id="u2", units=2.0, now=1.0)
    with pytest.raises(QuotaExceededError, match="unit"):
        store.reserve(owner_id="alice", reservation_id="u3", units=0.5, now=1.0)

    store = TenantQuotaStore(tmp_path / "quota-inflight.sqlite3")
    configure(store, requests=10, units=100.0, inflight=1)
    store.reserve(owner_id="alice", reservation_id="i1", units=1.0, now=1.0)
    with pytest.raises(QuotaExceededError, match="inflight"):
        store.reserve(owner_id="alice", reservation_id="i2", units=1.0, now=1.0)


def test_renewal_increments_fence_and_stale_token_cannot_commit(tmp_path):
    store = TenantQuotaStore(tmp_path / "quota.sqlite3")
    configure(store)
    reservation = store.reserve(
        owner_id="alice",
        reservation_id="request-a",
        units=2.0,
        now=1.0,
    )
    renewed = store.renew(
        owner_id="alice",
        reservation_id=reservation.reservation_id,
        fencing_token=reservation.fencing_token,
        now=5.0,
    )
    assert renewed.fencing_token == reservation.fencing_token + 1
    assert renewed.lease_expires_at == 15.0
    with pytest.raises(RuntimeError, match="stale"):
        store.commit(
            owner_id="alice",
            reservation_id=reservation.reservation_id,
            fencing_token=reservation.fencing_token,
            now=6.0,
        )
    committed = store.commit(
        owner_id="alice",
        reservation_id=renewed.reservation_id,
        fencing_token=renewed.fencing_token,
        now=6.0,
    )
    assert committed.state == "committed"


def test_expired_lease_frees_capacity_and_cannot_be_resurrected(tmp_path):
    store = TenantQuotaStore(tmp_path / "quota.sqlite3")
    configure(store, requests=10, units=10.0, inflight=1, lease=5)
    expired = store.reserve(
        owner_id="alice",
        reservation_id="old-request",
        units=2.0,
        now=1.0,
    )
    replacement = store.reserve(
        owner_id="alice",
        reservation_id="new-request",
        units=2.0,
        now=6.0,
    )
    assert replacement.state == "reserved"
    with pytest.raises(RuntimeError, match="stale"):
        store.commit(
            owner_id="alice",
            reservation_id=expired.reservation_id,
            fencing_token=expired.fencing_token,
            now=6.0,
        )


def test_owner_and_window_isolation_and_configuration_revision(tmp_path):
    store = TenantQuotaStore(tmp_path / "quota.sqlite3")
    assert configure(store, "alice", requests=1) == 1
    assert configure(store, "bob", requests=1) == 1
    assert configure(store, "alice", requests=2) == 2
    alice = store.reserve(owner_id="alice", reservation_id="alice-1", units=1.0, now=1.0)
    bob = store.reserve(owner_id="bob", reservation_id="bob-1", units=1.0, now=1.0)
    store.commit(
        owner_id="alice",
        reservation_id=alice.reservation_id,
        fencing_token=alice.fencing_token,
        now=2.0,
    )
    store.commit(
        owner_id="bob",
        reservation_id=bob.reservation_id,
        fencing_token=bob.fencing_token,
        now=2.0,
    )
    assert store.snapshot("alice", now=2.0).committed_requests == 1
    assert store.snapshot("bob", now=2.0).committed_requests == 1
    assert store.snapshot("alice", now=61.0).committed_requests == 0
    assert store.snapshot("bob", now=61.0).committed_requests == 0


def test_quota_contract_rejects_unsafe_values(tmp_path):
    with pytest.raises(ValueError, match="lease_seconds"):
        TenantQuotaConfig(
            request_limit=1,
            unit_limit=1.0,
            inflight_limit=1,
            window_seconds=10,
            lease_seconds=11,
        )
    store = TenantQuotaStore(tmp_path / "quota.sqlite3")
    configure(store)
    with pytest.raises(ValueError, match="reservation_id"):
        store.reserve(
            owner_id="alice",
            reservation_id="bad\tcontrol",
            units=1.0,
            now=1.0,
        )
    with pytest.raises(ValueError, match="positive"):
        store.reserve(
            owner_id="alice",
            reservation_id="negative-units",
            units=-1.0,
            now=1.0,
        )
