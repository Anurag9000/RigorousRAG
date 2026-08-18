from __future__ import annotations

import hashlib

import pytest

from orchestration.current_multi_region_authority import decide_current_region_authority
from orchestration.multi_region_authority import MultiRegionFailoverPolicy, RegionHealthObservation, SQLiteRegionAuthorityStore
from orchestration.multi_region_routing import ProviderRegionRoute, publish_authoritative_region_route


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def health(region: str, *, writes: bool = True) -> RegionHealthObservation:
    return RegionHealthObservation(region, 100.0, True, writes, 1.0, 99.0, sha(f"health:{region}:{writes}"))


def policy() -> MultiRegionFailoverPolicy:
    return MultiRegionFailoverPolicy("primary", ("secondary",), max_health_age_seconds=10.0, max_replication_lag_seconds=5.0, max_recovery_point_age_seconds=10.0)


def bootstrap(store: SQLiteRegionAuthorityStore):
    decision = decide_current_region_authority(owner_id="alice", service_id="write", current_region=None, observations=(health("primary"), health("secondary")), policy=policy(), now=100.0)
    return store.apply_decision(decision, expected_revision=None, now=100.0)


class Provider:
    def __init__(self, route=None, *, wrong_region=False, wrong_token=False, no_revision_advance=False):
        self.route = route
        self.wrong_region = wrong_region
        self.wrong_token = wrong_token
        self.no_revision_advance = no_revision_advance
        self.calls = []

    def read_route(self, *, owner_id, service_id):
        return self.route

    def compare_and_set_route(self, *, owner_id, service_id, expected_provider_revision, target_region_id, authority_fencing_token, idempotency_key):
        self.calls.append((owner_id, service_id, expected_provider_revision, target_region_id, authority_fencing_token, idempotency_key))
        current_revision = -1 if self.route is None else self.route.provider_revision
        if expected_provider_revision != (None if self.route is None else current_revision):
            raise RuntimeError("provider CAS mismatch")
        revision = current_revision if self.no_revision_advance else current_revision + 1
        if revision < 0:
            revision = 0
        self.route = ProviderRegionRoute(
            owner_id,
            service_id,
            "wrong" if self.wrong_region else target_region_id,
            revision,
            authority_fencing_token + 1 if self.wrong_token else authority_fencing_token,
            sha(f"provider:{target_region_id}:{authority_fencing_token}:{revision}"),
        )
        return self.route


def test_first_publication_projects_authority_and_fencing_token(tmp_path) -> None:
    store = SQLiteRegionAuthorityStore(tmp_path / "authority.sqlite3")
    authority = bootstrap(store)
    provider = Provider()
    receipt = publish_authoritative_region_route(owner_id="alice", service_id="write", authority_store=store, provider=provider)
    assert receipt.publication_performed
    assert receipt.authority_region == "primary"
    assert provider.route.region_id == "primary"
    assert provider.route.authority_fencing_token == authority.fencing_token
    assert len(receipt.receipt_sha256) == 64


def test_already_current_provider_route_is_idempotent_noop(tmp_path) -> None:
    store = SQLiteRegionAuthorityStore(tmp_path / "authority.sqlite3")
    authority = bootstrap(store)
    route = ProviderRegionRoute("alice", "write", "primary", 7, authority.fencing_token, sha("current-route"))
    provider = Provider(route)
    receipt = publish_authoritative_region_route(owner_id="alice", service_id="write", authority_store=store, provider=provider)
    assert not receipt.publication_performed
    assert provider.calls == []
    assert receipt.provider_revision == 7


def test_failover_publication_advances_provider_revision_and_authority_fence(tmp_path) -> None:
    store = SQLiteRegionAuthorityStore(tmp_path / "authority.sqlite3")
    primary = bootstrap(store)
    provider = Provider(ProviderRegionRoute("alice", "write", "primary", 3, primary.fencing_token, sha("old-route")))
    failover = decide_current_region_authority(owner_id="alice", service_id="write", current_region="primary", observations=(health("primary", writes=False), health("secondary")), policy=policy(), now=100.0)
    secondary = store.apply_decision(failover, expected_revision=primary.revision, now=101.0)
    receipt = publish_authoritative_region_route(owner_id="alice", service_id="write", authority_store=store, provider=provider)
    assert provider.route.region_id == "secondary"
    assert provider.route.authority_fencing_token == secondary.fencing_token
    assert provider.route.provider_revision == 4
    assert receipt.publication_performed


def test_provider_must_echo_authority_region_and_fencing_token(tmp_path) -> None:
    store = SQLiteRegionAuthorityStore(tmp_path / "authority.sqlite3")
    bootstrap(store)
    with pytest.raises(RuntimeError, match="authoritative region"):
        publish_authoritative_region_route(owner_id="alice", service_id="write", authority_store=store, provider=Provider(wrong_region=True))
    with pytest.raises(RuntimeError, match="fencing token"):
        publish_authoritative_region_route(owner_id="alice", service_id="write", authority_store=store, provider=Provider(wrong_token=True))


def test_provider_revision_must_advance_on_changed_route(tmp_path) -> None:
    store = SQLiteRegionAuthorityStore(tmp_path / "authority.sqlite3")
    authority = bootstrap(store)
    stale = ProviderRegionRoute("alice", "write", "secondary", 5, authority.fencing_token, sha("stale-route"))
    with pytest.raises(RuntimeError, match="revision did not advance"):
        publish_authoritative_region_route(owner_id="alice", service_id="write", authority_store=store, provider=Provider(stale, no_revision_advance=True))


def test_provider_route_scope_mismatch_is_rejected(tmp_path) -> None:
    store = SQLiteRegionAuthorityStore(tmp_path / "authority.sqlite3")
    authority = bootstrap(store)
    wrong_scope = ProviderRegionRoute("bob", "write", "primary", 1, authority.fencing_token, sha("wrong-scope"))
    with pytest.raises(RuntimeError, match="outside"):
        publish_authoritative_region_route(owner_id="alice", service_id="write", authority_store=store, provider=Provider(wrong_scope))
