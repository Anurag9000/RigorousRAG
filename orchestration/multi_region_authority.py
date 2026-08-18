"""Fenced multi-region authority, failover and failback control-plane primitives.

This module owns no cloud provider API calls.  It decides whether a region is eligible to
be authoritative and persists one monotonic write-authority record per owner/service.
Data-plane adapters can require the returned fencing token before accepting mutations.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


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


def _number(value: Any, label: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    selected = float(value)
    if not math.isfinite(selected) or (nonnegative and selected < 0.0):
        raise ValueError(f"{label} is invalid")
    return selected


@dataclass(frozen=True)
class RegionHealthObservation:
    region_id: str
    observed_at: float
    ready_for_reads: bool
    ready_for_writes: bool
    replication_lag_seconds: float
    recovery_point_at: float
    evidence_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_id", _text(self.region_id, "region_id"))
        object.__setattr__(self, "observed_at", _number(self.observed_at, "observed_at"))
        if not isinstance(self.ready_for_reads, bool) or not isinstance(self.ready_for_writes, bool):
            raise ValueError("region readiness values must be boolean")
        object.__setattr__(self, "replication_lag_seconds", _number(self.replication_lag_seconds, "replication_lag_seconds"))
        object.__setattr__(self, "recovery_point_at", _number(self.recovery_point_at, "recovery_point_at"))
        if self.recovery_point_at > self.observed_at:
            raise ValueError("recovery_point_at cannot be later than observed_at")
        object.__setattr__(self, "evidence_sha256", _sha(self.evidence_sha256, "evidence_sha256"))

    @property
    def observation_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-region-health/v1", **asdict(self)})


@dataclass(frozen=True)
class MultiRegionFailoverPolicy:
    primary_region: str
    failover_regions: tuple[str, ...]
    max_health_age_seconds: float = 30.0
    max_replication_lag_seconds: float = 5.0
    max_recovery_point_age_seconds: float = 30.0
    require_primary_unhealthy_for_failover: bool = True
    allow_automatic_failback: bool = False

    def __post_init__(self) -> None:
        primary = _text(self.primary_region, "primary_region")
        object.__setattr__(self, "primary_region", primary)
        failovers = tuple(_text(value, "failover region") for value in self.failover_regions)
        if not failovers or len(set(failovers)) != len(failovers) or primary in failovers:
            raise ValueError("failover_regions must be unique, non-empty, and exclude primary")
        object.__setattr__(self, "failover_regions", failovers)
        for name in ("max_health_age_seconds", "max_replication_lag_seconds", "max_recovery_point_age_seconds"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if not isinstance(self.require_primary_unhealthy_for_failover, bool) or not isinstance(self.allow_automatic_failback, bool):
            raise ValueError("multi-region policy booleans are invalid")

    @property
    def policy_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-multi-region-failover-policy/v1", **asdict(self)})


@dataclass(frozen=True)
class RegionAuthorityDecision:
    owner_id: str
    service_id: str
    current_region: str | None
    target_region: str | None
    action: str
    policy_sha256: str
    observation_sha256s: tuple[tuple[str, str], ...]
    reason_codes: tuple[str, ...]
    decision_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", _text(self.owner_id, "owner_id"))
        object.__setattr__(self, "service_id", _text(self.service_id, "service_id"))
        if self.current_region is not None:
            object.__setattr__(self, "current_region", _text(self.current_region, "current_region"))
        if self.target_region is not None:
            object.__setattr__(self, "target_region", _text(self.target_region, "target_region"))
        if self.action not in {"bootstrap", "hold", "failover", "failback", "no_change"}:
            raise ValueError("region authority action is invalid")
        object.__setattr__(self, "policy_sha256", _sha(self.policy_sha256, "policy_sha256"))
        rows = tuple(sorted((_text(region, "observation region"), _sha(digest, "observation sha256")) for region, digest in self.observation_sha256s))
        if not rows or len({region for region, _ in rows}) != len(rows):
            raise ValueError("observation identities must be unique and non-empty")
        object.__setattr__(self, "observation_sha256s", rows)
        reasons = tuple(sorted(set(self.reason_codes)))
        if self.action in {"bootstrap", "failover", "failback", "no_change"} and reasons:
            raise ValueError("successful authority decision may not contain failure reasons")
        if self.action == "hold" and not reasons:
            raise ValueError("hold decision requires reason codes")
        object.__setattr__(self, "reason_codes", reasons)
        expected = _digest(self._payload())
        provided = _sha(self.decision_sha256, "decision_sha256")
        if expected != provided:
            raise ValueError("decision_sha256 does not match authority decision")
        object.__setattr__(self, "decision_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-region-authority-decision/v1",
            "owner_id": self.owner_id,
            "service_id": self.service_id,
            "current_region": self.current_region,
            "target_region": self.target_region,
            "action": self.action,
            "policy_sha256": self.policy_sha256,
            "observation_sha256s": self.observation_sha256s,
            "reason_codes": self.reason_codes,
        }


def _healthy(observation: RegionHealthObservation | None, *, now: float, policy: MultiRegionFailoverPolicy) -> bool:
    if observation is None:
        return False
    return (
        observation.ready_for_reads
        and observation.ready_for_writes
        and now - observation.observed_at <= policy.max_health_age_seconds
        and observation.replication_lag_seconds <= policy.max_replication_lag_seconds
        and now - observation.recovery_point_at <= policy.max_recovery_point_age_seconds
    )


def decide_region_authority(
    *,
    owner_id: str,
    service_id: str,
    current_region: str | None,
    observations: Sequence[RegionHealthObservation],
    policy: MultiRegionFailoverPolicy,
    now: float,
    explicit_failback: bool = False,
) -> RegionAuthorityDecision:
    owner = _text(owner_id, "owner_id")
    service = _text(service_id, "service_id")
    timestamp = _number(now, "now")
    if not isinstance(policy, MultiRegionFailoverPolicy):
        raise ValueError("policy must be MultiRegionFailoverPolicy")
    rows = tuple(observations)
    if any(not isinstance(row, RegionHealthObservation) for row in rows):
        raise ValueError("observations contains invalid values")
    by_region = {row.region_id: row for row in rows}
    if len(by_region) != len(rows):
        raise ValueError("region observations must be unique")
    known_regions = {policy.primary_region, *policy.failover_regions}
    if not set(by_region).issubset(known_regions):
        raise ValueError("observation contains a region outside the failover policy")
    current = None if current_region is None else _text(current_region, "current_region")
    if current is not None and current not in known_regions:
        raise ValueError("current_region is outside the failover policy")
    observation_ids = tuple(sorted((region, row.observation_sha256) for region, row in by_region.items()))
    healthy = {region: _healthy(by_region.get(region), now=timestamp, policy=policy) for region in known_regions}
    action: str
    target: str | None
    reasons: list[str] = []
    if current is None:
        if healthy[policy.primary_region]:
            action, target = "bootstrap", policy.primary_region
        else:
            target = next((region for region in policy.failover_regions if healthy[region]), None)
            if target is None:
                action = "hold"
                reasons.append("no_healthy_region_available")
            else:
                action = "bootstrap"
    elif healthy[current]:
        if current != policy.primary_region and healthy[policy.primary_region] and (policy.allow_automatic_failback or explicit_failback):
            action, target = "failback", policy.primary_region
        else:
            action, target = "no_change", current
    else:
        if current == policy.primary_region or not policy.require_primary_unhealthy_for_failover or not healthy[policy.primary_region]:
            target = next((region for region in policy.failover_regions if region != current and healthy[region]), None)
            if target is None and current != policy.primary_region and healthy[policy.primary_region]:
                target = policy.primary_region
            if target is None:
                action = "hold"
                reasons.append("no_safe_failover_target")
            else:
                action = "failback" if target == policy.primary_region else "failover"
        else:
            action, target = "hold", None
            reasons.append("primary_is_healthy")
    payload = {
        "schema": "rigorousrag-region-authority-decision/v1",
        "owner_id": owner,
        "service_id": service,
        "current_region": current,
        "target_region": target,
        "action": action,
        "policy_sha256": policy.policy_sha256,
        "observation_sha256s": observation_ids,
        "reason_codes": tuple(sorted(set(reasons))),
    }
    return RegionAuthorityDecision(**payload, decision_sha256=_digest(payload))


@dataclass(frozen=True)
class RegionAuthorityRecord:
    owner_id: str
    service_id: str
    authority_region: str
    fencing_token: int
    revision: int
    policy_sha256: str
    decision_sha256: str
    updated_at: float


class SQLiteRegionAuthorityStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS region_authority (
                    owner_id TEXT NOT NULL,
                    service_id TEXT NOT NULL,
                    authority_region TEXT NOT NULL,
                    fencing_token INTEGER NOT NULL,
                    revision INTEGER NOT NULL,
                    policy_sha256 TEXT NOT NULL,
                    decision_sha256 TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(owner_id,service_id)
                )"""
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> RegionAuthorityRecord:
        return RegionAuthorityRecord(
            row["owner_id"], row["service_id"], row["authority_region"], int(row["fencing_token"]), int(row["revision"]), row["policy_sha256"], row["decision_sha256"], float(row["updated_at"])
        )

    def get(self, *, owner_id: str, service_id: str) -> RegionAuthorityRecord | None:
        owner, service = _text(owner_id, "owner_id"), _text(service_id, "service_id")
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM region_authority WHERE owner_id=? AND service_id=?", (owner, service)).fetchone()
        return None if row is None else self._record(row)

    def apply_decision(
        self,
        decision: RegionAuthorityDecision,
        *,
        expected_revision: int | None,
        now: float,
    ) -> RegionAuthorityRecord:
        if decision.action == "hold" or decision.target_region is None:
            raise ValueError("hold/no-target decision cannot change region authority")
        timestamp = _number(now, "now")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM region_authority WHERE owner_id=? AND service_id=?", (decision.owner_id, decision.service_id)).fetchone()
            if row is None:
                if expected_revision not in (None, 0) or decision.current_region is not None or decision.action != "bootstrap":
                    raise RuntimeError("region authority bootstrap CAS failed")
                record = RegionAuthorityRecord(decision.owner_id, decision.service_id, decision.target_region, 1, 1, decision.policy_sha256, decision.decision_sha256, timestamp)
                connection.execute(
                    "INSERT INTO region_authority(owner_id,service_id,authority_region,fencing_token,revision,policy_sha256,decision_sha256,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                    (record.owner_id, record.service_id, record.authority_region, record.fencing_token, record.revision, record.policy_sha256, record.decision_sha256, record.updated_at),
                )
                return record
            current = self._record(row)
            if expected_revision is None or current.revision != expected_revision or decision.current_region != current.authority_region:
                raise RuntimeError("region authority transition CAS failed")
            if decision.action == "no_change":
                if decision.target_region != current.authority_region:
                    raise RuntimeError("no_change decision targets a different region")
                return current
            record = RegionAuthorityRecord(decision.owner_id, decision.service_id, decision.target_region, current.fencing_token + 1, current.revision + 1, decision.policy_sha256, decision.decision_sha256, timestamp)
            changed = connection.execute(
                """UPDATE region_authority SET authority_region=?,fencing_token=?,revision=?,policy_sha256=?,decision_sha256=?,updated_at=?
                   WHERE owner_id=? AND service_id=? AND revision=? AND fencing_token=?""",
                (record.authority_region, record.fencing_token, record.revision, record.policy_sha256, record.decision_sha256, record.updated_at, record.owner_id, record.service_id, current.revision, current.fencing_token),
            ).rowcount
            if changed != 1:
                raise RuntimeError("region authority transition lost CAS race")
            return record

    def assert_write_authority(
        self,
        *,
        owner_id: str,
        service_id: str,
        region_id: str,
        fencing_token: int,
    ) -> RegionAuthorityRecord:
        record = self.get(owner_id=owner_id, service_id=service_id)
        if record is None:
            raise RuntimeError("no region write authority is established")
        if isinstance(fencing_token, bool) or not isinstance(fencing_token, int) or fencing_token < 1:
            raise ValueError("fencing_token must be positive")
        if record.authority_region != _text(region_id, "region_id") or record.fencing_token != fencing_token:
            raise RuntimeError("stale or non-authoritative region write token")
        return record


__all__ = [
    "MultiRegionFailoverPolicy",
    "RegionAuthorityDecision",
    "RegionAuthorityRecord",
    "RegionHealthObservation",
    "SQLiteRegionAuthorityStore",
    "decide_region_authority",
]
