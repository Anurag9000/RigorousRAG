"""Deterministic eligibility and sticky traffic assignment for retrieval interleaving.

This module does not derive user identities. Deployments supply a privacy-safe SHA-256
randomization-unit identifier according to their approved experiment policy (for example,
a pseudonymous account or session unit). Assignment is deterministic, owner-scoped and
bound to the experiment/policy digest.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from evaluation.retrieval_interleaving import InterleavingSpec


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _fraction(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be in [0, 1]")
    selected = float(value)
    if not math.isfinite(selected) or not 0.0 <= selected <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return selected


@dataclass(frozen=True)
class InterleavingTrafficPolicy:
    assignment_salt_sha256: str
    exclusion_group_id: str
    interleaving_fraction: float = 0.10
    baseline_holdout_fraction: float = 0.10
    allowed_domain_ids: tuple[str, ...] = ()
    allowed_route_ids: tuple[str, ...] = ()
    require_explicit_experiment_permission: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignment_salt_sha256", _sha(self.assignment_salt_sha256, "assignment_salt_sha256"))
        object.__setattr__(self, "exclusion_group_id", _text(self.exclusion_group_id, "exclusion_group_id"))
        interleaving = _fraction(self.interleaving_fraction, "interleaving_fraction")
        baseline = _fraction(self.baseline_holdout_fraction, "baseline_holdout_fraction")
        if interleaving + baseline > 1.0 + 1e-12:
            raise ValueError("interleaving and baseline fractions may not sum above one")
        object.__setattr__(self, "interleaving_fraction", interleaving)
        object.__setattr__(self, "baseline_holdout_fraction", baseline)
        domains = tuple(sorted({_text(value, "allowed domain id") for value in self.allowed_domain_ids}))
        routes = tuple(sorted({_text(value, "allowed route id") for value in self.allowed_route_ids}))
        object.__setattr__(self, "allowed_domain_ids", domains)
        object.__setattr__(self, "allowed_route_ids", routes)
        if not isinstance(self.require_explicit_experiment_permission, bool):
            raise ValueError("require_explicit_experiment_permission must be boolean")

    @property
    def policy_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-interleaving-traffic-policy/v1", **asdict(self)})


@dataclass(frozen=True)
class TrafficEligibilityContext:
    owner_id: str
    randomization_unit_sha256: str
    query_sha256: str
    domain_id: str
    route_id: str
    experiment_permitted: bool
    safety_blocked: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", _text(self.owner_id, "owner_id"))
        object.__setattr__(self, "randomization_unit_sha256", _sha(self.randomization_unit_sha256, "randomization_unit_sha256"))
        object.__setattr__(self, "query_sha256", _sha(self.query_sha256, "query_sha256"))
        object.__setattr__(self, "domain_id", _text(self.domain_id, "domain_id"))
        object.__setattr__(self, "route_id", _text(self.route_id, "route_id"))
        if not isinstance(self.experiment_permitted, bool) or not isinstance(self.safety_blocked, bool):
            raise ValueError("experiment_permitted and safety_blocked must be boolean")


@dataclass(frozen=True)
class TrafficAssignment:
    owner_id: str
    spec_sha256: str
    traffic_policy_sha256: str
    exclusion_group_id: str
    randomization_unit_sha256: str
    query_sha256: str
    domain_id: str
    route_id: str
    bucket: float
    arm: str
    reason_codes: tuple[str, ...]
    assignment_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", _text(self.owner_id, "owner_id"))
        for name in ("spec_sha256", "traffic_policy_sha256", "randomization_unit_sha256", "query_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        object.__setattr__(self, "exclusion_group_id", _text(self.exclusion_group_id, "exclusion_group_id"))
        object.__setattr__(self, "domain_id", _text(self.domain_id, "domain_id"))
        object.__setattr__(self, "route_id", _text(self.route_id, "route_id"))
        object.__setattr__(self, "bucket", _fraction(self.bucket, "bucket"))
        if self.arm not in {"interleaving", "baseline_only", "not_enrolled", "ineligible"}:
            raise ValueError("traffic assignment arm is invalid")
        reasons = tuple(sorted(set(self.reason_codes)))
        if self.arm in {"interleaving", "baseline_only"} and reasons:
            raise ValueError("eligible assigned arm may not contain failure reasons")
        if self.arm == "ineligible" and not reasons:
            raise ValueError("ineligible assignment requires reason codes")
        object.__setattr__(self, "reason_codes", reasons)
        expected = _digest(self._payload())
        provided = _sha(self.assignment_sha256, "assignment_sha256")
        if expected != provided:
            raise ValueError("assignment_sha256 does not match traffic assignment content")
        object.__setattr__(self, "assignment_sha256", provided)

    @property
    def experiment_exposed(self) -> bool:
        return self.arm == "interleaving"

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-interleaving-traffic-assignment/v1",
            "owner_id": self.owner_id,
            "spec_sha256": self.spec_sha256,
            "traffic_policy_sha256": self.traffic_policy_sha256,
            "exclusion_group_id": self.exclusion_group_id,
            "randomization_unit_sha256": self.randomization_unit_sha256,
            "query_sha256": self.query_sha256,
            "domain_id": self.domain_id,
            "route_id": self.route_id,
            "bucket": self.bucket,
            "arm": self.arm,
            "reason_codes": self.reason_codes,
        }


def _bucket(spec: InterleavingSpec, policy: InterleavingTrafficPolicy, unit_sha256: str) -> float:
    raw = hashlib.sha256(f"{policy.assignment_salt_sha256}:{spec.spec_sha256}:{unit_sha256}".encode("utf-8")).digest()
    integer = int.from_bytes(raw[:8], "big")
    return integer / float(1 << 64)


def assign_interleaving_traffic(
    spec: InterleavingSpec,
    policy: InterleavingTrafficPolicy,
    context: TrafficEligibilityContext,
) -> TrafficAssignment:
    if not isinstance(spec, InterleavingSpec) or not isinstance(policy, InterleavingTrafficPolicy) or not isinstance(context, TrafficEligibilityContext):
        raise ValueError("traffic assignment inputs have invalid types")
    reasons: list[str] = []
    if context.safety_blocked:
        reasons.append("safety_blocked")
    if policy.require_explicit_experiment_permission and not context.experiment_permitted:
        reasons.append("experiment_permission_missing")
    if policy.allowed_domain_ids and context.domain_id not in policy.allowed_domain_ids:
        reasons.append("domain_not_eligible")
    if policy.allowed_route_ids and context.route_id not in policy.allowed_route_ids:
        reasons.append("route_not_eligible")
    bucket = _bucket(spec, policy, context.randomization_unit_sha256)
    if reasons:
        arm = "ineligible"
    elif bucket < policy.interleaving_fraction:
        arm = "interleaving"
    elif bucket < policy.interleaving_fraction + policy.baseline_holdout_fraction:
        arm = "baseline_only"
    else:
        arm = "not_enrolled"
    payload = {
        "schema": "rigorousrag-interleaving-traffic-assignment/v1",
        "owner_id": context.owner_id,
        "spec_sha256": spec.spec_sha256,
        "traffic_policy_sha256": policy.policy_sha256,
        "exclusion_group_id": policy.exclusion_group_id,
        "randomization_unit_sha256": context.randomization_unit_sha256,
        "query_sha256": context.query_sha256,
        "domain_id": context.domain_id,
        "route_id": context.route_id,
        "bucket": bucket,
        "arm": arm,
        "reason_codes": tuple(sorted(set(reasons))),
    }
    return TrafficAssignment(**payload, assignment_sha256=_digest(payload))


__all__ = [
    "InterleavingTrafficPolicy",
    "TrafficAssignment",
    "TrafficEligibilityContext",
    "assign_interleaving_traffic",
]
