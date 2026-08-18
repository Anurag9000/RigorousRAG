from __future__ import annotations

import hashlib

from evaluation.multi_region_observability import observations_from_region_authority, observations_from_region_health, observations_from_region_route_publication
from orchestration.multi_region_authority import RegionAuthorityRecord, RegionHealthObservation
from orchestration.multi_region_routing import ProviderRegionRoute, RegionRoutePublicationReceipt


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_region_health_observations_expose_recovery_freshness_without_content() -> None:
    rows = observations_from_region_health(
        (RegionHealthObservation("secondary", 90.0, True, False, 3.0, 88.0, sha("health")),),
        now=100.0,
    )
    by_name = {row.name: row for row in rows}
    assert by_name["multi_region.replication_lag_seconds"].value == 3.0
    assert by_name["multi_region.health_age_seconds"].value == 10.0
    assert by_name["multi_region.recovery_point_age_seconds"].value == 12.0
    assert by_name["multi_region.write_ready"].value == 0.0
    assert dict(rows[0].tags)["region_id"] == "secondary"


def test_authority_observations_bind_service_and_region_only() -> None:
    record = RegionAuthorityRecord("alice", "write", "primary", 4, 7, sha("policy"), sha("decision"), 100.0)
    rows = observations_from_region_authority(record)
    assert {row.name for row in rows} == {"multi_region.authority_revision", "multi_region.authority_fencing_token"}
    assert all(dict(row.tags)["service_id"] == "write" for row in rows)
    assert all(dict(row.tags)["region_id"] == "primary" for row in rows)


def test_route_publication_observations_report_provider_revision_and_mutation() -> None:
    route = ProviderRegionRoute("alice", "write", "primary", 9, 4, sha("provider"))
    payload = {
        "schema": "rigorousrag-region-route-publication/v1",
        "owner_id": "alice",
        "service_id": "write",
        "authority_region": "primary",
        "authority_revision": 7,
        "authority_fencing_token": 4,
        "authority_decision_sha256": sha("decision"),
        "before_route_sha256": None,
        "after_route_sha256": route.route_sha256,
        "provider_revision": 9,
        "publication_performed": True,
    }
    import json
    receipt_sha = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    receipt = RegionRoutePublicationReceipt(**payload, receipt_sha256=receipt_sha)
    rows = observations_from_region_route_publication(receipt)
    values = {row.name: row.value for row in rows}
    assert values["multi_region.route_publication_performed"] == 1.0
    assert values["multi_region.provider_route_revision"] == 9.0
