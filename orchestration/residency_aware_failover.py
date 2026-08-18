"""Compose promoted data-residency policy with fenced multi-region authority decisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from orchestration.multi_region_authority import (
    MultiRegionFailoverPolicy,
    RegionAuthorityDecision,
    RegionHealthObservation,
    decide_region_authority,
)
from security.data_residency import (
    DataResidencyPolicy,
    RegionDescriptor,
    ResidencyDecision,
    evaluate_data_residency,
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


@dataclass(frozen=True)
class ResidencyAwareAuthorityDecision:
    authority: RegionAuthorityDecision
    residency_decisions: tuple[ResidencyDecision, ...]
    eligible_region_ids: tuple[str, ...]
    decision_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.authority, RegionAuthorityDecision):
            raise ValueError("authority must be RegionAuthorityDecision")
        decisions = tuple(self.residency_decisions)
        if not decisions or any(not isinstance(value, ResidencyDecision) for value in decisions):
            raise ValueError("residency_decisions must be a non-empty ResidencyDecision sequence")
        by_region = {value.region_id: value for value in decisions}
        if len(by_region) != len(decisions):
            raise ValueError("residency_decisions must have unique region ids")
        decisions = tuple(sorted(decisions, key=lambda value: value.region_id))
        object.__setattr__(self, "residency_decisions", decisions)
        eligible = tuple(sorted({_text(value, "eligible region id") for value in self.eligible_region_ids}))
        expected_eligible = tuple(sorted(value.region_id for value in decisions if value.eligible))
        if eligible != expected_eligible:
            raise ValueError("eligible_region_ids do not match residency decisions")
        if self.authority.target_region is not None and self.authority.action != "hold" and self.authority.target_region not in eligible:
            raise ValueError("authority target is not residency eligible")
        object.__setattr__(self, "eligible_region_ids", eligible)
        expected = _digest(self._payload())
        if self.decision_sha256 != expected:
            raise ValueError("decision_sha256 does not match residency-aware authority decision")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-residency-aware-authority/v1",
            "authority_decision_sha256": self.authority.decision_sha256,
            "residency_decision_sha256s": tuple(
                (value.region_id, value.decision_sha256) for value in self.residency_decisions
            ),
            "eligible_region_ids": self.eligible_region_ids,
        }


def _policy_filtered_observations(
    observations: Sequence[RegionHealthObservation],
    *,
    eligible_regions: set[str],
) -> tuple[RegionHealthObservation, ...]:
    """Preserve health evidence identity while making forbidden regions unavailable."""

    rows = tuple(observations)
    if not rows or any(not isinstance(value, RegionHealthObservation) for value in rows):
        raise ValueError("observations must be a non-empty RegionHealthObservation sequence")
    seen: set[str] = set()
    filtered: list[RegionHealthObservation] = []
    for value in rows:
        if value.region_id in seen:
            raise ValueError("region observations must have unique region ids")
        seen.add(value.region_id)
        if value.region_id in eligible_regions:
            filtered.append(value)
        else:
            filtered.append(
                RegionHealthObservation(
                    region_id=value.region_id,
                    observed_at=value.observed_at,
                    ready_for_reads=False,
                    ready_for_writes=False,
                    replication_lag_seconds=value.replication_lag_seconds,
                    recovery_point_at=value.recovery_point_at,
                    evidence_sha256=value.evidence_sha256,
                )
            )
    return tuple(filtered)


def decide_residency_aware_region_authority(
    *,
    owner_id: str,
    service_id: str,
    current_region: str | None,
    observations: Sequence[RegionHealthObservation],
    region_descriptors: Mapping[str, RegionDescriptor] | Sequence[RegionDescriptor],
    failover_policy: MultiRegionFailoverPolicy,
    residency_policy: DataResidencyPolicy,
    data_classes: Sequence[str],
    now: float,
    explicit_failback: bool = False,
) -> ResidencyAwareAuthorityDecision:
    """Fail closed on residency before applying health/RPO failover selection.

    All regions present in the failover policy require an explicit descriptor. A health
    observation for a residency-ineligible region remains part of the evidence set but is
    converted to an unavailable policy view before failover selection. If the current
    authority itself becomes residency-ineligible, the wrapper authorizes a safe return
    to a healthy eligible primary without requiring automatic failback.
    """

    if not isinstance(failover_policy, MultiRegionFailoverPolicy):
        raise ValueError("failover_policy must be MultiRegionFailoverPolicy")
    if not isinstance(residency_policy, DataResidencyPolicy):
        raise ValueError("residency_policy must be DataResidencyPolicy")
    if not isinstance(explicit_failback, bool):
        raise ValueError("explicit_failback must be boolean")
    if isinstance(region_descriptors, Mapping):
        descriptors = tuple(region_descriptors.values())
    else:
        descriptors = tuple(region_descriptors)
    if any(not isinstance(value, RegionDescriptor) for value in descriptors):
        raise ValueError("region_descriptors contains invalid values")
    by_region = {value.region_id: value for value in descriptors}
    if len(by_region) != len(descriptors):
        raise ValueError("region descriptors must have unique region ids")
    known = {failover_policy.primary_region, *failover_policy.failover_regions}
    if set(by_region) != known:
        raise ValueError("region descriptors must exactly cover the failover policy regions")

    residency = tuple(
        evaluate_data_residency(
            owner_id=owner_id,
            service_id=service_id,
            region=by_region[region_id],
            data_classes=data_classes,
            policy=residency_policy,
        )
        for region_id in sorted(known)
    )
    eligible = {value.region_id for value in residency if value.eligible}
    filtered = _policy_filtered_observations(observations, eligible_regions=eligible)
    selected_current = None if current_region is None else _text(current_region, "current_region")
    residency_forced_exit = selected_current is not None and selected_current not in eligible
    authority = decide_region_authority(
        owner_id=owner_id,
        service_id=service_id,
        current_region=selected_current,
        observations=filtered,
        policy=failover_policy,
        now=now,
        explicit_failback=explicit_failback or residency_forced_exit,
    )
    if authority.target_region is not None and authority.action != "hold" and authority.target_region not in eligible:
        raise RuntimeError("base failover selected a residency-ineligible target")
    payload = {
        "schema": "rigorousrag-residency-aware-authority/v1",
        "authority_decision_sha256": authority.decision_sha256,
        "residency_decision_sha256s": tuple((value.region_id, value.decision_sha256) for value in residency),
        "eligible_region_ids": tuple(sorted(eligible)),
    }
    return ResidencyAwareAuthorityDecision(
        authority=authority,
        residency_decisions=residency,
        eligible_region_ids=tuple(sorted(eligible)),
        decision_sha256=_digest(payload),
    )


__all__ = ["ResidencyAwareAuthorityDecision", "decide_residency_aware_region_authority"]
