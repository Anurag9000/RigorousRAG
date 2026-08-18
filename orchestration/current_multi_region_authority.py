"""Authoritative multi-region decision wrapper with explicit failback recovery semantics.

`orchestration.multi_region_authority` owns the region health/fencing/store primitives.
This wrapper is the control-plane decision entrypoint. It preserves the base policy for
normal bootstrap/failover/no-change decisions while making one recovery rule explicit:
when the currently authoritative *secondary* is unhealthy and the primary is healthy,
an explicitly requested failback may return authority to the primary even when automatic
failback is disabled.

The wrapper exists separately so the correction can be applied without risking a stale
contents-API overwrite of the live base module during concurrent direct-to-main work.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

from orchestration.multi_region_authority import (
    MultiRegionFailoverPolicy,
    RegionAuthorityDecision,
    RegionHealthObservation,
    decide_region_authority,
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _healthy(
    observation: RegionHealthObservation | None,
    *,
    now: float,
    policy: MultiRegionFailoverPolicy,
) -> bool:
    if observation is None:
        return False
    return (
        observation.ready_for_reads
        and observation.ready_for_writes
        and now >= observation.observed_at
        and now - observation.observed_at <= policy.max_health_age_seconds
        and observation.replication_lag_seconds <= policy.max_replication_lag_seconds
        and now >= observation.recovery_point_at
        and now - observation.recovery_point_at <= policy.max_recovery_point_age_seconds
    )


def decide_current_region_authority(
    *,
    owner_id: str,
    service_id: str,
    current_region: str | None,
    observations: Sequence[RegionHealthObservation],
    policy: MultiRegionFailoverPolicy,
    now: float,
    explicit_failback: bool = False,
) -> RegionAuthorityDecision:
    """Return the authoritative region decision for control-plane use."""

    rows = tuple(observations)
    by_region = {row.region_id: row for row in rows}
    if len(by_region) != len(rows):
        raise ValueError("region observations must be unique")

    # Recovery override: an operator-requested failback is allowed from an unhealthy
    # secondary to a healthy primary. Automatic failback remains governed by the base
    # policy. We bind the exact same health-observation identities and policy digest as
    # every base RegionAuthorityDecision.
    if (
        explicit_failback
        and current_region is not None
        and current_region != policy.primary_region
        and not _healthy(by_region.get(current_region), now=now, policy=policy)
        and _healthy(by_region.get(policy.primary_region), now=now, policy=policy)
    ):
        observation_ids = tuple(
            sorted((region, row.observation_sha256) for region, row in by_region.items())
        )
        payload = {
            "schema": "rigorousrag-region-authority-decision/v1",
            "owner_id": owner_id,
            "service_id": service_id,
            "current_region": current_region,
            "target_region": policy.primary_region,
            "action": "failback",
            "policy_sha256": policy.policy_sha256,
            "observation_sha256s": observation_ids,
            "reason_codes": (),
        }
        return RegionAuthorityDecision(**payload, decision_sha256=_digest(payload))

    return decide_region_authority(
        owner_id=owner_id,
        service_id=service_id,
        current_region=current_region,
        observations=rows,
        policy=policy,
        now=now,
        explicit_failback=explicit_failback,
    )


__all__ = ["decide_current_region_authority"]
