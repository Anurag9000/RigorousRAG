"""Content-addressed data-residency policy and durable promotion authority.

Residency is evaluated per data class rather than as one global region allow-list.  A
service may therefore permit derived indexes in a broader geography while constraining
source content, backups, model inputs, or key material more tightly.  The module stores
only bounded identifiers and policy metadata; it never stores document/query content.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_SUPPORTED_DATA_CLASSES = frozenset(
    {
        "source_content",
        "derived_index",
        "metadata",
        "audit",
        "backup",
        "key_material",
        "model_input",
    }
)
_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _sha(value: Any, label: str) -> str:
    selected = _text(value, label, 64).lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _time(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative") from exc
    if not math.isfinite(selected) or selected < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return selected


def _revision(value: Any, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _country(value: Any) -> str:
    selected = _text(value, "country_code", 2).upper()
    if len(selected) != 2 or not selected.isalpha():
        raise ValueError("country_code must be a two-letter code")
    return selected


def _data_class(value: Any) -> str:
    selected = _text(value, "data_class", 100).lower()
    if selected not in _SUPPORTED_DATA_CLASSES:
        raise ValueError(f"unsupported data_class {selected!r}")
    return selected


@dataclass(frozen=True)
class RegionDescriptor:
    region_id: str
    provider_id: str
    country_code: str
    jurisdiction_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_id", _text(self.region_id, "region_id"))
        object.__setattr__(self, "provider_id", _text(self.provider_id, "provider_id"))
        object.__setattr__(self, "country_code", _country(self.country_code))
        tags = tuple(sorted({_text(value, "jurisdiction tag", 200).lower() for value in self.jurisdiction_tags}))
        object.__setattr__(self, "jurisdiction_tags", tags)

    @property
    def descriptor_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-region-descriptor/v1", **asdict(self)})


@dataclass(frozen=True)
class ResidencyRule:
    data_class: str
    allowed_region_ids: tuple[str, ...] = ()
    allowed_country_codes: tuple[str, ...] = ()
    required_jurisdiction_tags: tuple[str, ...] = ()
    forbidden_jurisdiction_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_class", _data_class(self.data_class))
        regions = tuple(sorted({_text(value, "allowed region") for value in self.allowed_region_ids}))
        countries = tuple(sorted({_country(value) for value in self.allowed_country_codes}))
        required = tuple(sorted({_text(value, "required jurisdiction tag", 200).lower() for value in self.required_jurisdiction_tags}))
        forbidden = tuple(sorted({_text(value, "forbidden jurisdiction tag", 200).lower() for value in self.forbidden_jurisdiction_tags}))
        if not regions and not countries and not required:
            raise ValueError("residency rule must define at least one positive placement constraint")
        if set(required) & set(forbidden):
            raise ValueError("jurisdiction tags may not be both required and forbidden")
        object.__setattr__(self, "allowed_region_ids", regions)
        object.__setattr__(self, "allowed_country_codes", countries)
        object.__setattr__(self, "required_jurisdiction_tags", required)
        object.__setattr__(self, "forbidden_jurisdiction_tags", forbidden)

    def evaluate(self, region: RegionDescriptor) -> tuple[str, ...]:
        reasons: list[str] = []
        tags = set(region.jurisdiction_tags)
        if self.allowed_region_ids and region.region_id not in self.allowed_region_ids:
            reasons.append("region_not_allowed")
        if self.allowed_country_codes and region.country_code not in self.allowed_country_codes:
            reasons.append("country_not_allowed")
        missing = set(self.required_jurisdiction_tags) - tags
        if missing:
            reasons.append("required_jurisdiction_tag_missing")
        if set(self.forbidden_jurisdiction_tags) & tags:
            reasons.append("forbidden_jurisdiction_tag_present")
        return tuple(sorted(set(reasons)))


@dataclass(frozen=True)
class DataResidencyPolicy:
    policy_id: str
    rules: tuple[ResidencyRule, ...]
    default_deny: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id", 300))
        rules = tuple(self.rules)
        if not rules or any(not isinstance(rule, ResidencyRule) for rule in rules):
            raise ValueError("rules must be a non-empty ResidencyRule sequence")
        by_class = {rule.data_class: rule for rule in rules}
        if len(by_class) != len(rules):
            raise ValueError("data classes may appear at most once in a residency policy")
        object.__setattr__(self, "rules", tuple(sorted(rules, key=lambda rule: rule.data_class)))
        if not isinstance(self.default_deny, bool):
            raise ValueError("default_deny must be boolean")

    @property
    def policy_sha256(self) -> str:
        return _digest(
            {
                "schema": "rigorousrag-data-residency-policy/v1",
                "policy_id": self.policy_id,
                "rules": [asdict(rule) for rule in self.rules],
                "default_deny": self.default_deny,
            }
        )

    def rule_for(self, data_class: str) -> ResidencyRule | None:
        selected = _data_class(data_class)
        return next((rule for rule in self.rules if rule.data_class == selected), None)


@dataclass(frozen=True)
class ResidencyDecision:
    owner_id: str
    service_id: str
    region_id: str
    data_classes: tuple[str, ...]
    policy_sha256: str
    region_descriptor_sha256: str
    eligible: bool
    reason_codes: tuple[str, ...]
    decision_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", _text(self.owner_id, "owner_id"))
        object.__setattr__(self, "service_id", _text(self.service_id, "service_id"))
        object.__setattr__(self, "region_id", _text(self.region_id, "region_id"))
        classes = tuple(sorted({_data_class(value) for value in self.data_classes}))
        if not classes:
            raise ValueError("data_classes must be non-empty")
        object.__setattr__(self, "data_classes", classes)
        object.__setattr__(self, "policy_sha256", _sha(self.policy_sha256, "policy_sha256"))
        object.__setattr__(self, "region_descriptor_sha256", _sha(self.region_descriptor_sha256, "region_descriptor_sha256"))
        if not isinstance(self.eligible, bool):
            raise ValueError("eligible must be boolean")
        reasons = tuple(sorted({_text(reason, "reason code", 200) for reason in self.reason_codes}))
        if self.eligible and reasons:
            raise ValueError("eligible residency decision may not contain failure reasons")
        if not self.eligible and not reasons:
            raise ValueError("ineligible residency decision requires reason codes")
        object.__setattr__(self, "reason_codes", reasons)
        expected = _digest(self._payload())
        provided = _sha(self.decision_sha256, "decision_sha256")
        if expected != provided:
            raise ValueError("decision_sha256 does not match residency decision")
        object.__setattr__(self, "decision_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-residency-decision/v1",
            "owner_id": self.owner_id,
            "service_id": self.service_id,
            "region_id": self.region_id,
            "data_classes": self.data_classes,
            "policy_sha256": self.policy_sha256,
            "region_descriptor_sha256": self.region_descriptor_sha256,
            "eligible": self.eligible,
            "reason_codes": self.reason_codes,
        }


def evaluate_data_residency(
    *,
    owner_id: str,
    service_id: str,
    region: RegionDescriptor,
    data_classes: Sequence[str],
    policy: DataResidencyPolicy,
) -> ResidencyDecision:
    if not isinstance(region, RegionDescriptor):
        raise ValueError("region must be RegionDescriptor")
    if not isinstance(policy, DataResidencyPolicy):
        raise ValueError("policy must be DataResidencyPolicy")
    classes = tuple(sorted({_data_class(value) for value in data_classes}))
    if not classes:
        raise ValueError("data_classes must be non-empty")
    reasons: list[str] = []
    for selected in classes:
        rule = policy.rule_for(selected)
        if rule is None:
            if policy.default_deny:
                reasons.append(f"{selected}:no_rule")
            continue
        reasons.extend(f"{selected}:{reason}" for reason in rule.evaluate(region))
    payload = {
        "schema": "rigorousrag-residency-decision/v1",
        "owner_id": _text(owner_id, "owner_id"),
        "service_id": _text(service_id, "service_id"),
        "region_id": region.region_id,
        "data_classes": classes,
        "policy_sha256": policy.policy_sha256,
        "region_descriptor_sha256": region.descriptor_sha256,
        "eligible": not reasons,
        "reason_codes": tuple(sorted(set(reasons))),
    }
    return ResidencyDecision(**payload, decision_sha256=_digest(payload))


@dataclass(frozen=True)
class ResidencyPolicyRecord:
    owner_id: str
    service_id: str
    revision: int
    policy: DataResidencyPolicy
    promoted_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", _text(self.owner_id, "owner_id"))
        object.__setattr__(self, "service_id", _text(self.service_id, "service_id"))
        object.__setattr__(self, "revision", _revision(self.revision, "revision"))
        if not isinstance(self.policy, DataResidencyPolicy):
            raise ValueError("policy must be DataResidencyPolicy")
        object.__setattr__(self, "promoted_at", _time(self.promoted_at, "promoted_at"))


class SQLiteResidencyPolicyStore:
    """Monotonic CAS promotion store for owner/service residency policy revisions."""

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
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS residency_policy_history (
                    owner_id TEXT NOT NULL,
                    service_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    policy_sha256 TEXT NOT NULL,
                    policy_json TEXT NOT NULL,
                    promoted_at REAL NOT NULL,
                    PRIMARY KEY(owner_id,service_id,revision)
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS residency_policy_current (
                    owner_id TEXT NOT NULL,
                    service_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    policy_sha256 TEXT NOT NULL,
                    PRIMARY KEY(owner_id,service_id)
                )"""
            )

    @staticmethod
    def _decode_policy(raw: str, expected_sha256: str) -> DataResidencyPolicy:
        value = json.loads(raw)
        if not isinstance(value, Mapping):
            raise RuntimeError("persisted residency policy is invalid")
        rules = tuple(ResidencyRule(**row) for row in value.get("rules", ()))
        policy = DataResidencyPolicy(
            policy_id=value["policy_id"],
            rules=rules,
            default_deny=value.get("default_deny", True),
        )
        if policy.policy_sha256 != _sha(expected_sha256, "policy_sha256"):
            raise RuntimeError("persisted residency policy digest is corrupt")
        return policy

    def current(self, *, owner_id: str, service_id: str) -> ResidencyPolicyRecord | None:
        owner, service = _text(owner_id, "owner_id"), _text(service_id, "service_id")
        with self._connect() as connection:
            row = connection.execute(
                """SELECT h.owner_id,h.service_id,h.revision,h.policy_sha256,h.policy_json,h.promoted_at
                   FROM residency_policy_current c
                   JOIN residency_policy_history h
                     ON h.owner_id=c.owner_id AND h.service_id=c.service_id AND h.revision=c.revision
                   WHERE c.owner_id=? AND c.service_id=?""",
                (owner, service),
            ).fetchone()
        if row is None:
            return None
        policy = self._decode_policy(row["policy_json"], row["policy_sha256"])
        return ResidencyPolicyRecord(owner, service, int(row["revision"]), policy, float(row["promoted_at"]))

    def promote(
        self,
        *,
        owner_id: str,
        service_id: str,
        policy: DataResidencyPolicy,
        expected_revision: int | None,
        now: float,
    ) -> ResidencyPolicyRecord:
        owner, service = _text(owner_id, "owner_id"), _text(service_id, "service_id")
        if not isinstance(policy, DataResidencyPolicy):
            raise ValueError("policy must be DataResidencyPolicy")
        if expected_revision is not None:
            _revision(expected_revision, "expected_revision", allow_zero=True)
        promoted_at = _time(now, "now")
        policy_json = _canonical(
            {
                "policy_id": policy.policy_id,
                "rules": [asdict(rule) for rule in policy.rules],
                "default_deny": policy.default_deny,
            }
        ).decode("utf-8")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT revision,policy_sha256 FROM residency_policy_current WHERE owner_id=? AND service_id=?",
                (owner, service),
            ).fetchone()
            if current is None:
                if expected_revision not in (None, 0):
                    raise RuntimeError("residency policy bootstrap CAS failed")
                revision = 1
            else:
                current_revision = int(current["revision"])
                if expected_revision is None or expected_revision != current_revision:
                    raise RuntimeError("residency policy promotion CAS failed")
                if current["policy_sha256"] == policy.policy_sha256:
                    row = connection.execute(
                        """SELECT policy_json,promoted_at FROM residency_policy_history
                           WHERE owner_id=? AND service_id=? AND revision=?""",
                        (owner, service, current_revision),
                    ).fetchone()
                    if row is None:
                        raise RuntimeError("current residency policy history is missing")
                    current_policy = self._decode_policy(row["policy_json"], policy.policy_sha256)
                    return ResidencyPolicyRecord(owner, service, current_revision, current_policy, float(row["promoted_at"]))
                revision = current_revision + 1
            connection.execute(
                "INSERT INTO residency_policy_history(owner_id,service_id,revision,policy_sha256,policy_json,promoted_at) VALUES(?,?,?,?,?,?)",
                (owner, service, revision, policy.policy_sha256, policy_json, promoted_at),
            )
            if current is None:
                connection.execute(
                    "INSERT INTO residency_policy_current(owner_id,service_id,revision,policy_sha256) VALUES(?,?,?,?)",
                    (owner, service, revision, policy.policy_sha256),
                )
            else:
                changed = connection.execute(
                    """UPDATE residency_policy_current SET revision=?,policy_sha256=?
                       WHERE owner_id=? AND service_id=? AND revision=?""",
                    (revision, policy.policy_sha256, owner, service, expected_revision),
                ).rowcount
                if changed != 1:
                    raise RuntimeError("residency policy promotion lost CAS race")
        return ResidencyPolicyRecord(owner, service, revision, policy, promoted_at)

    def assert_current(
        self,
        *,
        owner_id: str,
        service_id: str,
        policy_sha256: str,
    ) -> ResidencyPolicyRecord:
        record = self.current(owner_id=owner_id, service_id=service_id)
        if record is None:
            raise RuntimeError("no residency policy is promoted for owner/service")
        if record.policy.policy_sha256 != _sha(policy_sha256, "policy_sha256"):
            raise RuntimeError("residency policy is stale or not authoritative")
        return record


__all__ = [
    "DataResidencyPolicy",
    "RegionDescriptor",
    "ResidencyDecision",
    "ResidencyPolicyRecord",
    "ResidencyRule",
    "SQLiteResidencyPolicyStore",
    "evaluate_data_residency",
]
