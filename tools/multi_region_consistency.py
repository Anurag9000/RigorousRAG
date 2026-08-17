"""Multi-region replication consistency metadata and failover eligibility policy.

The module records externally observed replication/generation state and evaluates whether a
candidate region is eligible for promotion. It does not perform replication, DNS changes,
or lease acquisition. Promotion still requires the deployment's own strongly consistent
fencing/lease service.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

_MAX_GENERATIONS = 10_000


def _text(value: Any, label: str, maximum: int = 500, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if (not selected and not allow_empty) or len(selected) > maximum or "\x00" in selected:
        raise ValueError(f"{label} is invalid")
    return selected


def _finite(value: Any, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(parsed) or parsed < minimum:
        raise ValueError(f"{label} is invalid")
    return parsed


def _sha(value: str, label: str, *, allow_empty: bool = False) -> str:
    selected = _text(value, label, 64, allow_empty=allow_empty).lower()
    if selected and (len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected)):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


@dataclass(frozen=True)
class RegionalGenerationState:
    region_id: str
    observed_at: float
    commit_sequence: int
    generations: Mapping[str, str]
    state_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_id", _text(self.region_id, "region_id", 256).lower())
        object.__setattr__(self, "observed_at", _finite(self.observed_at, "observed_at"))
        if isinstance(self.commit_sequence, bool) or not isinstance(self.commit_sequence, int) or self.commit_sequence < 0:
            raise ValueError("commit_sequence is invalid")
        if not isinstance(self.generations, Mapping) or len(self.generations) > _MAX_GENERATIONS:
            raise ValueError("generations are invalid")
        selected: dict[str, str] = {}
        for kind, fingerprint in self.generations.items():
            selected[_text(str(kind), "generation kind", 256)] = _sha(str(fingerprint), "generation fingerprint")
        object.__setattr__(self, "generations", dict(sorted(selected.items())))
        computed = hashlib.sha256(
            _canonical(
                {
                    "region_id": self.region_id,
                    "observed_at": self.observed_at,
                    "commit_sequence": self.commit_sequence,
                    "generations": self.generations,
                }
            )
        ).hexdigest()
        supplied = self.state_fingerprint.strip().lower()
        if supplied and _sha(supplied, "state_fingerprint") != computed:
            raise ValueError("state_fingerprint does not match regional state")
        object.__setattr__(self, "state_fingerprint", computed)


@dataclass(frozen=True)
class RegionalPrimaryLease:
    region_id: str
    lease_id: str
    fencing_epoch: int
    issued_at: float
    expires_at: float
    lease_authority_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_id", _text(self.region_id, "region_id", 256).lower())
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id", 500))
        if isinstance(self.fencing_epoch, bool) or not isinstance(self.fencing_epoch, int) or self.fencing_epoch < 1:
            raise ValueError("fencing_epoch must be positive")
        issued = _finite(self.issued_at, "issued_at")
        expires = _finite(self.expires_at, "expires_at")
        if expires <= issued:
            raise ValueError("expires_at must be greater than issued_at")
        object.__setattr__(self, "issued_at", issued)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(self, "lease_authority_fingerprint", _sha(self.lease_authority_fingerprint, "lease_authority_fingerprint"))

    def active_at(self, timestamp: float) -> bool:
        selected = _finite(timestamp, "timestamp")
        return self.issued_at <= selected < self.expires_at


@dataclass(frozen=True)
class FailoverPolicy:
    policy_id: str
    max_commit_lag: int
    max_observation_age_seconds: float
    require_generation_parity: tuple[str, ...]
    require_no_active_foreign_primary: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id", 256))
        if isinstance(self.max_commit_lag, bool) or not isinstance(self.max_commit_lag, int) or self.max_commit_lag < 0:
            raise ValueError("max_commit_lag is invalid")
        object.__setattr__(self, "max_observation_age_seconds", _finite(self.max_observation_age_seconds, "max_observation_age_seconds"))
        object.__setattr__(
            self,
            "require_generation_parity",
            tuple(sorted(set(_text(item, "generation kind", 256) for item in self.require_generation_parity))),
        )
        if not isinstance(self.require_no_active_foreign_primary, bool):
            raise ValueError("require_no_active_foreign_primary must be boolean")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True)
class FailoverEvaluation:
    candidate_region: str
    eligible: bool
    reasons: tuple[str, ...]
    missing_generation_parity: tuple[str, ...]
    commit_lag: int
    observation_age_seconds: float
    required_next_fencing_epoch: int
    policy_fingerprint: str
    source_state_fingerprint: str
    candidate_state_fingerprint: str
    fingerprint: str


def evaluate_failover(
    *,
    source: RegionalGenerationState,
    candidate: RegionalGenerationState,
    policy: FailoverPolicy,
    now: float,
    observed_primary_leases: Sequence[RegionalPrimaryLease] = (),
) -> FailoverEvaluation:
    if not isinstance(source, RegionalGenerationState) or not isinstance(candidate, RegionalGenerationState):
        raise TypeError("source and candidate must be RegionalGenerationState")
    if not isinstance(policy, FailoverPolicy):
        raise TypeError("policy must be FailoverPolicy")
    if source.region_id == candidate.region_id:
        raise ValueError("source and candidate regions must differ")
    selected_now = _finite(now, "now")
    if any(not isinstance(item, RegionalPrimaryLease) for item in observed_primary_leases):
        raise ValueError("observed_primary_leases contain an invalid value")

    reasons: list[str] = []
    commit_lag = max(0, source.commit_sequence - candidate.commit_sequence)
    if candidate.commit_sequence > source.commit_sequence:
        reasons.append("candidate_commit_sequence_is_ahead_of_source; operator review required")
    if commit_lag > policy.max_commit_lag:
        reasons.append(f"commit_lag_exceeds_policy:{commit_lag}>{policy.max_commit_lag}")
    observation_age = max(0.0, selected_now - candidate.observed_at)
    if observation_age > policy.max_observation_age_seconds:
        reasons.append("candidate_replication_observation_is_stale")

    missing: list[str] = []
    for kind in policy.require_generation_parity:
        source_generation = source.generations.get(kind)
        candidate_generation = candidate.generations.get(kind)
        if not source_generation or source_generation != candidate_generation:
            missing.append(kind)
    if missing:
        reasons.append("required_generation_parity_not_satisfied")

    active_leases = [item for item in observed_primary_leases if item.active_at(selected_now)]
    epochs = [item.fencing_epoch for item in observed_primary_leases]
    if policy.require_no_active_foreign_primary:
        foreign = [item for item in active_leases if item.region_id != candidate.region_id]
        if foreign:
            reasons.append("active_foreign_primary_lease_observed")
    candidate_active = [item for item in active_leases if item.region_id == candidate.region_id]
    if len(candidate_active) > 1:
        reasons.append("multiple_active_candidate_primary_leases_observed")
    required_epoch = (max(epochs) + 1) if epochs else 1
    # Even an eligible candidate must acquire a *new* externally fenced lease at this epoch
    # or greater before becoming writable; this evaluator never grants that lease itself.
    payload = {
        "candidate_region": candidate.region_id,
        "reasons": reasons,
        "missing_generation_parity": missing,
        "commit_lag": commit_lag,
        "observation_age_seconds": observation_age,
        "required_next_fencing_epoch": required_epoch,
        "policy": policy.fingerprint,
        "source": source.state_fingerprint,
        "candidate": candidate.state_fingerprint,
    }
    return FailoverEvaluation(
        candidate_region=candidate.region_id,
        eligible=not reasons,
        reasons=tuple(reasons),
        missing_generation_parity=tuple(missing),
        commit_lag=commit_lag,
        observation_age_seconds=observation_age,
        required_next_fencing_epoch=required_epoch,
        policy_fingerprint=policy.fingerprint,
        source_state_fingerprint=source.state_fingerprint,
        candidate_state_fingerprint=candidate.state_fingerprint,
        fingerprint=hashlib.sha256(_canonical(payload)).hexdigest(),
    )


def verify_writable_region(
    state: RegionalGenerationState,
    lease: RegionalPrimaryLease,
    *,
    expected_minimum_epoch: int,
    now: float,
) -> tuple[bool, tuple[str, ...]]:
    if not isinstance(state, RegionalGenerationState) or not isinstance(lease, RegionalPrimaryLease):
        raise TypeError("state/lease types are invalid")
    if isinstance(expected_minimum_epoch, bool) or not isinstance(expected_minimum_epoch, int) or expected_minimum_epoch < 1:
        raise ValueError("expected_minimum_epoch is invalid")
    reasons: list[str] = []
    if state.region_id != lease.region_id:
        reasons.append("lease_region_does_not_match_state")
    if lease.fencing_epoch < expected_minimum_epoch:
        reasons.append("lease_fencing_epoch_is_stale")
    if not lease.active_at(now):
        reasons.append("primary_lease_is_not_active")
    return not reasons, tuple(reasons)


__all__ = [
    "FailoverEvaluation",
    "FailoverPolicy",
    "RegionalGenerationState",
    "RegionalPrimaryLease",
    "evaluate_failover",
    "verify_writable_region",
]
