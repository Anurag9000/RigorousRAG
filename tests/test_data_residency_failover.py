from __future__ import annotations

import hashlib

import pytest

from orchestration.multi_region_authority import MultiRegionFailoverPolicy, RegionHealthObservation
from orchestration.residency_aware_failover import decide_residency_aware_region_authority
from security.data_residency import (
    DataResidencyPolicy,
    RegionDescriptor,
    ResidencyRule,
    SQLiteResidencyPolicyStore,
    evaluate_data_residency,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _region(region_id: str, country: str, *tags: str) -> RegionDescriptor:
    return RegionDescriptor(region_id, "provider", country, tuple(tags))


def _health(region_id: str, *, ready: bool = True) -> RegionHealthObservation:
    return RegionHealthObservation(region_id, 100.0, ready, ready, 0.5, 99.0, _sha(region_id))


def _policy() -> DataResidencyPolicy:
    return DataResidencyPolicy(
        "eu-source",
        (
            ResidencyRule("source_content", allowed_country_codes=("DE", "FR"), required_jurisdiction_tags=("eu",)),
            ResidencyRule("derived_index", allowed_country_codes=("DE", "FR", "US")),
        ),
    )


def test_residency_is_data_class_specific() -> None:
    us = _region("us-east", "US", "us")
    source = evaluate_data_residency(
        owner_id="owner",
        service_id="retrieval",
        region=us,
        data_classes=("source_content",),
        policy=_policy(),
    )
    derived = evaluate_data_residency(
        owner_id="owner",
        service_id="retrieval",
        region=us,
        data_classes=("derived_index",),
        policy=_policy(),
    )
    assert source.eligible is False
    assert derived.eligible is True


def test_residency_policy_promotion_is_monotonic_and_cas_bound(tmp_path) -> None:
    store = SQLiteResidencyPolicyStore(tmp_path / "residency.sqlite")
    first = store.promote(owner_id="owner", service_id="retrieval", policy=_policy(), expected_revision=0, now=1.0)
    assert first.revision == 1
    same = store.promote(owner_id="owner", service_id="retrieval", policy=_policy(), expected_revision=1, now=2.0)
    assert same.revision == 1
    stricter = DataResidencyPolicy(
        "de-only",
        (ResidencyRule("source_content", allowed_country_codes=("DE",)),),
    )
    second = store.promote(owner_id="owner", service_id="retrieval", policy=stricter, expected_revision=1, now=3.0)
    assert second.revision == 2
    with pytest.raises(RuntimeError, match="promotion CAS failed"):
        store.promote(owner_id="owner", service_id="retrieval", policy=_policy(), expected_revision=1, now=4.0)


def test_failover_never_selects_healthy_but_residency_forbidden_region() -> None:
    failover = MultiRegionFailoverPolicy("eu-primary", ("us-secondary",))
    decision = decide_residency_aware_region_authority(
        owner_id="owner",
        service_id="retrieval",
        current_region="eu-primary",
        observations=(_health("eu-primary", ready=False), _health("us-secondary")),
        region_descriptors={
            "eu-primary": _region("eu-primary", "DE", "eu"),
            "us-secondary": _region("us-secondary", "US", "us"),
        },
        failover_policy=failover,
        residency_policy=_policy(),
        data_classes=("source_content",),
        now=100.0,
    )
    assert decision.authority.action == "hold"
    assert decision.authority.target_region is None
    assert decision.eligible_region_ids == ("eu-primary",)


def test_residency_violation_forces_safe_exit_from_current_secondary() -> None:
    failover = MultiRegionFailoverPolicy("eu-primary", ("us-secondary",), allow_automatic_failback=False)
    decision = decide_residency_aware_region_authority(
        owner_id="owner",
        service_id="retrieval",
        current_region="us-secondary",
        observations=(_health("eu-primary"), _health("us-secondary")),
        region_descriptors={
            "eu-primary": _region("eu-primary", "DE", "eu"),
            "us-secondary": _region("us-secondary", "US", "us"),
        },
        failover_policy=failover,
        residency_policy=_policy(),
        data_classes=("source_content",),
        now=100.0,
    )
    assert decision.authority.action == "failback"
    assert decision.authority.target_region == "eu-primary"


def test_missing_rule_is_denied_by_default() -> None:
    decision = evaluate_data_residency(
        owner_id="owner",
        service_id="retrieval",
        region=_region("eu-primary", "DE", "eu"),
        data_classes=("backup",),
        policy=_policy(),
    )
    assert decision.eligible is False
    assert "backup:no_rule" in decision.reason_codes
