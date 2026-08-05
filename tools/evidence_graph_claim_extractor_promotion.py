"""Policy-gated benchmark promotion and exact-version activation for claim extractors."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import stat
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from tools.evidence_graph_claim_contracts import (
    _digest,
    _identifier,
    _integer,
    _sha256,
    _timestamp,
)
from tools.evidence_graph_claim_extractor_benchmark import (
    ScientificClaimExtractorBenchmarkSuite,
)
from tools.evidence_graph_claim_extractor_registry import (
    ScientificClaimExtractorRecord,
    ScientificClaimExtractorRegistry,
)
from tools.evidence_graph_relation_actor import (
    ReviewActorBinding,
    require_relation_review_actor,
)
from tools.security import normalize_owner_id

_PROMOTION_ACTIONS = frozenset({"promote", "rollback"})
_MAX_PATH = 4096
_MAX_LIMIT = 10_000
_MAX_ADMINS = 1_000
_MAX_SCOPE = 1_000
_MAX_POLICY_BYTES = 1_000_000
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _metric(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and between 0 and 1.")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and between 0 and 1.") from exc
    if not math.isfinite(selected) or not 0.0 <= selected <= 1.0:
        raise ValueError(f"{label} must be finite and between 0 and 1.")
    return selected


def _optional_digest(value: Any, label: str) -> str | None:
    return None if value is None else _digest(value, label)


def _scope_values(
    value: Any,
    label: str,
    *,
    owner_scope: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a bounded array.")
    if not 1 <= len(value) <= _MAX_SCOPE:
        raise ValueError(f"{label} must contain 1-{_MAX_SCOPE} entries.")
    selected: set[str] = set()
    for item in value:
        rendered = _identifier(item, label, 500)
        if rendered == "*":
            selected.add(rendered)
        elif owner_scope:
            selected.add(normalize_owner_id(rendered))
        else:
            selected.add(rendered)
    if "*" in selected and len(selected) != 1:
        raise ValueError(f"{label} wildcard may not be combined with explicit entries.")
    return tuple(sorted(selected))


def _allows(scope: tuple[str, ...], value: str) -> bool:
    return scope == ("*",) or value in scope


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("promotion policy contains a duplicate JSON key.")
        result[key] = value
    return result


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str], label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"{label} must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError(f"{label} is invalid.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for part in (absolute, *absolute.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(f"{label} could not be validated.") from exc
        if _redirecting(info):
            raise ValueError(f"{label} may not contain redirects.")
    return absolute


@dataclass(frozen=True)
class ClaimExtractorPromotionThresholds:
    minimum_case_count: int
    minimum_gold_count: int
    minimum_precision: float
    minimum_recall: float
    minimum_f1: float
    minimum_exact_evidence_accuracy: float
    minimum_exact_locator_accuracy: float
    minimum_mean_span_iou: float
    minimum_mean_claim_token_f1: float
    minimum_claim_type_accuracy: float
    minimum_modality_accuracy: float
    maximum_confidence_brier_score: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_case_count",
            _integer(self.minimum_case_count, "minimum_case_count", 1, 100_000_000),
        )
        object.__setattr__(
            self,
            "minimum_gold_count",
            _integer(self.minimum_gold_count, "minimum_gold_count", 1, 100_000_000),
        )
        for name in (
            "minimum_precision",
            "minimum_recall",
            "minimum_f1",
            "minimum_exact_evidence_accuracy",
            "minimum_exact_locator_accuracy",
            "minimum_mean_span_iou",
            "minimum_mean_claim_token_f1",
            "minimum_claim_type_accuracy",
            "minimum_modality_accuracy",
            "maximum_confidence_brier_score",
        ):
            object.__setattr__(self, name, _metric(getattr(self, name), name))
        if self.schema_version != 1:
            raise ValueError("promotion threshold schema is unsupported.")

    @property
    def thresholds_digest(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class ClaimExtractorPromotionAdministratorGrant:
    administrator_id: str
    owners: tuple[str, ...]
    extractor_names: tuple[str, ...]
    actions: tuple[str, ...]
    expires_at: float | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "administrator_id",
            _identifier(self.administrator_id, "administrator_id", 200),
        )
        object.__setattr__(
            self,
            "owners",
            _scope_values(self.owners, "owners", owner_scope=True),
        )
        object.__setattr__(
            self,
            "extractor_names",
            _scope_values(self.extractor_names, "extractor_names"),
        )
        actions = _scope_values(self.actions, "actions")
        if actions == ("*",) or any(value not in _PROMOTION_ACTIONS for value in actions):
            raise ValueError("promotion actions contains an unsupported value.")
        object.__setattr__(self, "actions", actions)
        if self.expires_at is not None:
            object.__setattr__(self, "expires_at", _timestamp(self.expires_at, "expires_at"))
        if self.schema_version != 1:
            raise ValueError("promotion administrator grant schema is unsupported.")

    @property
    def grant_digest(self) -> str:
        return _sha256(asdict(self))

    def permits(self, *, owner_id: str, extractor_name: str, action: str, now: float) -> bool:
        return bool(
            _allows(self.owners, owner_id)
            and _allows(self.extractor_names, extractor_name)
            and action in self.actions
            and (self.expires_at is None or now <= self.expires_at)
        )


@dataclass(frozen=True)
class ClaimExtractorPromotionPolicy:
    thresholds: ClaimExtractorPromotionThresholds
    administrators: tuple[ClaimExtractorPromotionAdministratorGrant, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.thresholds, ClaimExtractorPromotionThresholds):
            raise ValueError("thresholds must be ClaimExtractorPromotionThresholds.")
        if (
            not isinstance(self.administrators, tuple)
            or not 1 <= len(self.administrators) <= _MAX_ADMINS
            or any(
                not isinstance(value, ClaimExtractorPromotionAdministratorGrant)
                for value in self.administrators
            )
        ):
            raise ValueError("administrators must be a bounded non-empty tuple.")
        ordered = tuple(
            sorted(self.administrators, key=lambda value: value.administrator_id)
        )
        if len({value.administrator_id for value in ordered}) != len(ordered):
            raise ValueError("promotion administrator IDs must be unique.")
        object.__setattr__(self, "administrators", ordered)
        if self.schema_version != 1:
            raise ValueError("promotion policy schema is unsupported.")

    @property
    def policy_digest(self) -> str:
        return _sha256(asdict(self))

    def grant_for(self, administrator_id: str) -> ClaimExtractorPromotionAdministratorGrant:
        selected = _identifier(administrator_id, "administrator_id", 200)
        for grant in self.administrators:
            if grant.administrator_id == selected:
                return grant
        raise PermissionError("promotion administrator is not authorized.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ClaimExtractorPromotionPolicy":
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version",
            "thresholds",
            "administrators",
        }:
            raise ValueError("promotion policy schema is invalid.")
        if value["schema_version"] != 1:
            raise ValueError("promotion policy schema is unsupported.")
        thresholds = value["thresholds"]
        if not isinstance(thresholds, Mapping) or set(thresholds) != {
            "minimum_case_count",
            "minimum_gold_count",
            "minimum_precision",
            "minimum_recall",
            "minimum_f1",
            "minimum_exact_evidence_accuracy",
            "minimum_exact_locator_accuracy",
            "minimum_mean_span_iou",
            "minimum_mean_claim_token_f1",
            "minimum_claim_type_accuracy",
            "minimum_modality_accuracy",
            "maximum_confidence_brier_score",
        }:
            raise ValueError("promotion threshold schema is invalid.")
        raw_admins = value["administrators"]
        if (
            isinstance(raw_admins, (str, bytes, bytearray))
            or not isinstance(raw_admins, Sequence)
            or not 1 <= len(raw_admins) <= _MAX_ADMINS
        ):
            raise ValueError("administrators must be a bounded non-empty array.")
        allowed = {
            "administrator_id",
            "owners",
            "extractor_names",
            "actions",
            "expires_at",
        }
        required = allowed - {"expires_at"}
        administrators = []
        for raw in raw_admins:
            if not isinstance(raw, Mapping) or not required <= set(raw) <= allowed:
                raise ValueError("promotion administrator grant schema is invalid.")
            administrators.append(
                ClaimExtractorPromotionAdministratorGrant(
                    administrator_id=raw["administrator_id"],
                    owners=_scope_values(raw["owners"], "owners", owner_scope=True),
                    extractor_names=_scope_values(
                        raw["extractor_names"], "extractor_names"
                    ),
                    actions=_scope_values(raw["actions"], "actions"),
                    expires_at=raw.get("expires_at"),
                )
            )
        return cls(
            thresholds=ClaimExtractorPromotionThresholds(**dict(thresholds)),
            administrators=tuple(administrators),
        )


def load_claim_extractor_promotion_policy(
    *,
    path: str | os.PathLike[str] | None = None,
    policy_json: str | None = None,
) -> ClaimExtractorPromotionPolicy:
    if path is None and policy_json is None:
        configured_path = os.getenv("EVIDENCE_GRAPH_CLAIM_EXTRACTOR_PROMOTION_POLICY_PATH")
        configured_json = os.getenv("EVIDENCE_GRAPH_CLAIM_EXTRACTOR_PROMOTION_POLICY_JSON")
        path = configured_path if configured_path else None
        policy_json = configured_json if configured_json else None
    if (path is None) == (policy_json is None):
        raise RuntimeError("configure exactly one claim extractor promotion policy source.")
    if path is not None:
        selected = _path(path, "promotion policy path")
        descriptor = os.open(
            selected,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= _MAX_POLICY_BYTES:
                raise ValueError("promotion policy file is invalid or too large.")
            payload = os.read(descriptor, _MAX_POLICY_BYTES + 1)
            if len(payload) > _MAX_POLICY_BYTES:
                raise ValueError("promotion policy file is too large.")
        finally:
            os.close(descriptor)
    else:
        if not isinstance(policy_json, str):
            raise ValueError("promotion policy JSON must be text.")
        payload = policy_json.encode("utf-8")
        if not 1 <= len(payload) <= _MAX_POLICY_BYTES:
            raise ValueError("promotion policy size is invalid.")
    try:
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError("promotion policy JSON is invalid.") from exc
    return ClaimExtractorPromotionPolicy.from_mapping(raw)


@dataclass(frozen=True)
class ScientificClaimExtractorPromotionReport:
    owner_id: str
    extractor_name: str
    extractor_version: str
    extractor_record_digest: str
    benchmark_id: str
    benchmark_suite_digest: str
    policy_digest: str
    thresholds_digest: str
    eligible: bool
    reasons: tuple[str, ...]
    assessed_at: float
    report_digest: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(
            self,
            "extractor_name",
            _identifier(self.extractor_name, "extractor_name", 200),
        )
        object.__setattr__(
            self,
            "extractor_version",
            _identifier(self.extractor_version, "extractor_version", 200),
        )
        for name in (
            "extractor_record_digest",
            "benchmark_suite_digest",
            "policy_digest",
            "thresholds_digest",
            "report_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "benchmark_id",
            _identifier(self.benchmark_id, "benchmark_id", 500),
        )
        if not isinstance(self.eligible, bool):
            raise ValueError("eligible must be boolean.")
        if not isinstance(self.reasons, tuple) or any(
            not isinstance(value, str) for value in self.reasons
        ):
            raise ValueError("reasons must be a tuple.")
        normalized_reasons = tuple(
            sorted(_identifier(value, "reason", 200) for value in self.reasons)
        )
        if len(set(normalized_reasons)) != len(normalized_reasons):
            raise ValueError("promotion reasons contains duplicates.")
        if self.eligible != (len(normalized_reasons) == 0):
            raise ValueError("promotion eligibility differs from reasons.")
        object.__setattr__(self, "reasons", normalized_reasons)
        object.__setattr__(self, "assessed_at", _timestamp(self.assessed_at, "assessed_at"))
        if self.schema_version != 1:
            raise ValueError("promotion report schema is unsupported.")
        stable = {
            "scope": "rigorousrag-scientific-claim-extractor-promotion-report-v1",
            **{
                key: value
                for key, value in asdict(self).items()
                if key not in {"assessed_at", "report_digest"}
            },
        }
        if self.report_digest != _sha256(stable):
            raise ValueError("report_digest differs from promotion report.")


@dataclass(frozen=True)
class ScientificClaimExtractorActivation:
    activation_id: str
    owner_id: str
    extractor_name: str
    extractor_version: str
    extractor_record_digest: str
    promotion_report_digest: str
    action: str
    previous_activation_id: str | None
    actor_id: str
    actor_binding_method: str
    actor_binding_digest: str
    activated_at: float
    activation_digest: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "activation_id", _digest(self.activation_id, "activation_id"))
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(
            self,
            "extractor_name",
            _identifier(self.extractor_name, "extractor_name", 200),
        )
        object.__setattr__(
            self,
            "extractor_version",
            _identifier(self.extractor_version, "extractor_version", 200),
        )
        for name in (
            "extractor_record_digest",
            "promotion_report_digest",
            "actor_binding_digest",
            "activation_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        action = _identifier(self.action, "action", 30)
        if action not in _PROMOTION_ACTIONS:
            raise ValueError("activation action is unsupported.")
        object.__setattr__(self, "action", action)
        object.__setattr__(
            self,
            "previous_activation_id",
            _optional_digest(self.previous_activation_id, "previous_activation_id"),
        )
        object.__setattr__(self, "actor_id", _identifier(self.actor_id, "actor_id", 200))
        object.__setattr__(
            self,
            "actor_binding_method",
            _identifier(self.actor_binding_method, "actor_binding_method", 50),
        )
        object.__setattr__(self, "activated_at", _timestamp(self.activated_at, "activated_at"))
        if self.schema_version != 1:
            raise ValueError("activation schema is unsupported.")
        stable_identity = {
            "scope": "rigorousrag-scientific-claim-extractor-activation-id-v1",
            "owner_id": self.owner_id,
            "extractor_name": self.extractor_name,
            "extractor_version": self.extractor_version,
            "extractor_record_digest": self.extractor_record_digest,
            "promotion_report_digest": self.promotion_report_digest,
            "action": self.action,
            "previous_activation_id": self.previous_activation_id,
            "actor_id": self.actor_id,
            "actor_binding_method": self.actor_binding_method,
            "actor_binding_digest": self.actor_binding_digest,
        }
        if self.activation_id != _sha256(stable_identity):
            raise ValueError("activation_id differs from deterministic activation identity.")
        stable_record = {
            **stable_identity,
            "activation_id": self.activation_id,
            "activated_at": self.activated_at,
            "schema_version": self.schema_version,
        }
        if self.activation_digest != _sha256(stable_record):
            raise ValueError("activation_digest differs from activation record.")

    @classmethod
    def create(
        cls,
        *,
        owner_id: str,
        extractor_name: str,
        extractor_version: str,
        extractor_record_digest: str,
        promotion_report_digest: str,
        action: str,
        previous_activation_id: str | None,
        actor: ReviewActorBinding,
        activated_at: float,
    ) -> "ScientificClaimExtractorActivation":
        if not isinstance(actor, ReviewActorBinding):
            raise ValueError("actor must be ReviewActorBinding.")
        values = {
            "owner_id": normalize_owner_id(owner_id),
            "extractor_name": _identifier(extractor_name, "extractor_name", 200),
            "extractor_version": _identifier(extractor_version, "extractor_version", 200),
            "extractor_record_digest": _digest(
                extractor_record_digest, "extractor_record_digest"
            ),
            "promotion_report_digest": _digest(
                promotion_report_digest, "promotion_report_digest"
            ),
            "action": _identifier(action, "action", 30),
            "previous_activation_id": _optional_digest(
                previous_activation_id, "previous_activation_id"
            ),
            "actor_id": actor.actor_id,
            "actor_binding_method": actor.binding_method,
            "actor_binding_digest": actor.binding_digest,
        }
        activation_id = _sha256(
            {
                "scope": "rigorousrag-scientific-claim-extractor-activation-id-v1",
                **values,
            }
        )
        timestamp = _timestamp(activated_at, "activated_at")
        return cls(
            activation_id=activation_id,
            **values,
            activated_at=timestamp,
            activation_digest=_sha256(
                {
                    "scope": "rigorousrag-scientific-claim-extractor-activation-id-v1",
                    **values,
                    "activation_id": activation_id,
                    "activated_at": timestamp,
                    "schema_version": 1,
                }
            ),
        )


def assess_scientific_claim_extractor_promotion(
    *,
    extractor_record: ScientificClaimExtractorRecord,
    benchmark_suite: ScientificClaimExtractorBenchmarkSuite,
    policy: ClaimExtractorPromotionPolicy,
    now: float | None = None,
) -> ScientificClaimExtractorPromotionReport:
    if not isinstance(extractor_record, ScientificClaimExtractorRecord):
        raise ValueError("extractor_record must be ScientificClaimExtractorRecord.")
    if not isinstance(benchmark_suite, ScientificClaimExtractorBenchmarkSuite):
        raise ValueError("benchmark_suite must be ScientificClaimExtractorBenchmarkSuite.")
    if not isinstance(policy, ClaimExtractorPromotionPolicy):
        raise ValueError("policy must be ClaimExtractorPromotionPolicy.")
    if extractor_record.state != "active":
        raise PermissionError("only active extractor versions may be assessed for promotion.")
    if (
        benchmark_suite.owner_id != extractor_record.owner_id
        or benchmark_suite.extractor_name != extractor_record.extractor_name
        or benchmark_suite.extractor_version != extractor_record.extractor_version
        or benchmark_suite.extractor_record_digest != extractor_record.record_digest
    ):
        raise PermissionError("benchmark suite differs from exact extractor record scope.")
    thresholds = policy.thresholds
    reasons: list[str] = []
    checks = (
        (benchmark_suite.case_count >= thresholds.minimum_case_count, "case_count_below_floor"),
        (benchmark_suite.gold_count >= thresholds.minimum_gold_count, "gold_count_below_floor"),
        (benchmark_suite.precision >= thresholds.minimum_precision, "precision_below_floor"),
        (benchmark_suite.recall >= thresholds.minimum_recall, "recall_below_floor"),
        (benchmark_suite.f1 >= thresholds.minimum_f1, "f1_below_floor"),
        (
            benchmark_suite.exact_evidence_accuracy
            >= thresholds.minimum_exact_evidence_accuracy,
            "exact_evidence_accuracy_below_floor",
        ),
        (
            benchmark_suite.exact_locator_accuracy
            >= thresholds.minimum_exact_locator_accuracy,
            "exact_locator_accuracy_below_floor",
        ),
        (
            benchmark_suite.mean_span_iou >= thresholds.minimum_mean_span_iou,
            "mean_span_iou_below_floor",
        ),
        (
            benchmark_suite.mean_claim_token_f1
            >= thresholds.minimum_mean_claim_token_f1,
            "mean_claim_token_f1_below_floor",
        ),
        (
            benchmark_suite.claim_type_accuracy
            >= thresholds.minimum_claim_type_accuracy,
            "claim_type_accuracy_below_floor",
        ),
        (
            benchmark_suite.modality_accuracy
            >= thresholds.minimum_modality_accuracy,
            "modality_accuracy_below_floor",
        ),
        (
            benchmark_suite.confidence_brier_score
            <= thresholds.maximum_confidence_brier_score,
            "confidence_brier_score_above_ceiling",
        ),
    )
    reasons.extend(reason for passed, reason in checks if not passed)
    values = {
        "owner_id": extractor_record.owner_id,
        "extractor_name": extractor_record.extractor_name,
        "extractor_version": extractor_record.extractor_version,
        "extractor_record_digest": extractor_record.record_digest,
        "benchmark_id": benchmark_suite.benchmark_id,
        "benchmark_suite_digest": benchmark_suite.suite_digest,
        "policy_digest": policy.policy_digest,
        "thresholds_digest": thresholds.thresholds_digest,
        "eligible": not reasons,
        "reasons": tuple(sorted(reasons)),
        "schema_version": 1,
    }
    stable = {
        "scope": "rigorousrag-scientific-claim-extractor-promotion-report-v1",
        **values,
    }
    return ScientificClaimExtractorPromotionReport(
        **values,
        assessed_at=time.time() if now is None else now,
        report_digest=_sha256(stable),
    )


class ScientificClaimExtractorPromotionStore:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path, "promotion database path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("promotion database parent must be a directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("promotion database is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
            or self._file_identity() != self._database_identity
        ):
            raise RuntimeError("promotion database identity changed.")

    def _connect(self) -> sqlite3.Connection:
        self._verify()
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with sqlite3.connect(self.path, timeout=30.0, isolation_level=None) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS scientific_claim_extractor_promotion_reports (
                    report_digest TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    extractor_name TEXT NOT NULL,
                    extractor_version TEXT NOT NULL,
                    extractor_record_digest TEXT NOT NULL,
                    eligible INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    assessed_at REAL NOT NULL,
                    schema_version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS scientific_claim_extractor_activations (
                    activation_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    extractor_name TEXT NOT NULL,
                    extractor_version TEXT NOT NULL,
                    extractor_record_digest TEXT NOT NULL,
                    promotion_report_digest TEXT NOT NULL,
                    action TEXT NOT NULL,
                    previous_activation_id TEXT,
                    payload_json TEXT NOT NULL,
                    activated_at REAL NOT NULL,
                    schema_version INTEGER NOT NULL,
                    FOREIGN KEY(promotion_report_digest)
                        REFERENCES scientific_claim_extractor_promotion_reports(report_digest)
                        ON DELETE RESTRICT
                );
                CREATE TABLE IF NOT EXISTS scientific_claim_extractor_current (
                    owner_id TEXT NOT NULL,
                    extractor_name TEXT NOT NULL,
                    activation_id TEXT NOT NULL,
                    PRIMARY KEY(owner_id, extractor_name),
                    FOREIGN KEY(activation_id)
                        REFERENCES scientific_claim_extractor_activations(activation_id)
                        ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS scientific_claim_extractor_promotion_scope
                    ON scientific_claim_extractor_promotion_reports(
                        owner_id, extractor_name, assessed_at, report_digest
                    );
                CREATE INDEX IF NOT EXISTS scientific_claim_extractor_activation_scope
                    ON scientific_claim_extractor_activations(
                        owner_id, extractor_name, activated_at, activation_id
                    );
                """
            )

    @staticmethod
    def _report(row: sqlite3.Row) -> ScientificClaimExtractorPromotionReport:
        try:
            value = ScientificClaimExtractorPromotionReport(**json.loads(row["payload_json"]))
        except Exception as exc:
            raise RuntimeError("stored promotion report is corrupt.") from exc
        if (
            value.report_digest != row["report_digest"]
            or value.owner_id != row["owner_id"]
            or value.extractor_name != row["extractor_name"]
            or value.extractor_version != row["extractor_version"]
            or value.extractor_record_digest != row["extractor_record_digest"]
            or int(value.eligible) != int(row["eligible"])
            or value.assessed_at != float(row["assessed_at"])
        ):
            raise RuntimeError("stored promotion report columns are corrupt.")
        return value

    @staticmethod
    def _activation(row: sqlite3.Row) -> ScientificClaimExtractorActivation:
        try:
            value = ScientificClaimExtractorActivation(**json.loads(row["payload_json"]))
        except Exception as exc:
            raise RuntimeError("stored activation is corrupt.") from exc
        if (
            value.activation_id != row["activation_id"]
            or value.owner_id != row["owner_id"]
            or value.extractor_name != row["extractor_name"]
            or value.extractor_version != row["extractor_version"]
            or value.extractor_record_digest != row["extractor_record_digest"]
            or value.promotion_report_digest != row["promotion_report_digest"]
            or value.action != row["action"]
            or value.previous_activation_id != row["previous_activation_id"]
            or value.activated_at != float(row["activated_at"])
        ):
            raise RuntimeError("stored activation columns are corrupt.")
        return value

    def store_report(
        self,
        report: ScientificClaimExtractorPromotionReport,
    ) -> ScientificClaimExtractorPromotionReport:
        if not isinstance(report, ScientificClaimExtractorPromotionReport):
            raise ValueError("report must be ScientificClaimExtractorPromotionReport.")
        payload = json.dumps(asdict(report), sort_keys=True, separators=(",", ":"))
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM scientific_claim_extractor_promotion_reports "
                    "WHERE report_digest=?",
                    (report.report_digest,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO scientific_claim_extractor_promotion_reports "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)",
                        (
                            report.report_digest,
                            report.owner_id,
                            report.extractor_name,
                            report.extractor_version,
                            report.extractor_record_digest,
                            int(report.eligible),
                            payload,
                            report.assessed_at,
                        ),
                    )
                elif self._report(row) != report:
                    raise RuntimeError("promotion report digest collision detected.")
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get_report(report.report_digest)

    def get_report(self, report_digest: str) -> ScientificClaimExtractorPromotionReport:
        selected = _digest(report_digest, "report_digest")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM scientific_claim_extractor_promotion_reports "
                "WHERE report_digest=?",
                (selected,),
            ).fetchone()
        if row is None:
            raise KeyError(selected)
        return self._report(row)

    def current(
        self,
        *,
        owner_id: str,
        extractor_name: str,
    ) -> ScientificClaimExtractorActivation | None:
        owner = normalize_owner_id(owner_id)
        name = _identifier(extractor_name, "extractor_name", 200)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT a.* FROM scientific_claim_extractor_current c "
                "JOIN scientific_claim_extractor_activations a "
                "ON a.activation_id=c.activation_id "
                "WHERE c.owner_id=? AND c.extractor_name=?",
                (owner, name),
            ).fetchone()
        return None if row is None else self._activation(row)

    def activate(
        self,
        *,
        report: ScientificClaimExtractorPromotionReport,
        action: str,
        expected_current_activation_id: str | None,
        actor: ReviewActorBinding,
        now: float,
    ) -> ScientificClaimExtractorActivation:
        if not isinstance(report, ScientificClaimExtractorPromotionReport) or not report.eligible:
            raise PermissionError("only eligible promotion reports may be activated.")
        selected_action = _identifier(action, "action", 30)
        if selected_action not in _PROMOTION_ACTIONS:
            raise ValueError("activation action is unsupported.")
        expected = _optional_digest(
            expected_current_activation_id,
            "expected_current_activation_id",
        )
        timestamp = _timestamp(now, "now")
        stored_report = self.store_report(report)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                pointer = connection.execute(
                    "SELECT activation_id FROM scientific_claim_extractor_current "
                    "WHERE owner_id=? AND extractor_name=?",
                    (stored_report.owner_id, stored_report.extractor_name),
                ).fetchone()
                current_id = None if pointer is None else pointer["activation_id"]
                if current_id != expected:
                    raise RuntimeError("extractor current activation changed.")
                activation = ScientificClaimExtractorActivation.create(
                    owner_id=stored_report.owner_id,
                    extractor_name=stored_report.extractor_name,
                    extractor_version=stored_report.extractor_version,
                    extractor_record_digest=stored_report.extractor_record_digest,
                    promotion_report_digest=stored_report.report_digest,
                    action=selected_action,
                    previous_activation_id=current_id,
                    actor=actor,
                    activated_at=timestamp,
                )
                row = connection.execute(
                    "SELECT * FROM scientific_claim_extractor_activations "
                    "WHERE activation_id=?",
                    (activation.activation_id,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO scientific_claim_extractor_activations "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                        (
                            activation.activation_id,
                            activation.owner_id,
                            activation.extractor_name,
                            activation.extractor_version,
                            activation.extractor_record_digest,
                            activation.promotion_report_digest,
                            activation.action,
                            activation.previous_activation_id,
                            json.dumps(
                                asdict(activation),
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            activation.activated_at,
                        ),
                    )
                elif self._activation(row).activation_digest != activation.activation_digest:
                    raise RuntimeError("activation identity collision detected.")
                connection.execute(
                    "INSERT INTO scientific_claim_extractor_current "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT(owner_id, extractor_name) DO UPDATE "
                    "SET activation_id=excluded.activation_id",
                    (
                        activation.owner_id,
                        activation.extractor_name,
                        activation.activation_id,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        result = self.current(
            owner_id=stored_report.owner_id,
            extractor_name=stored_report.extractor_name,
        )
        if result is None:
            raise RuntimeError("activated extractor pointer disappeared.")
        return result

    def history(
        self,
        *,
        owner_id: str,
        extractor_name: str,
        limit: int = 100,
    ) -> tuple[ScientificClaimExtractorActivation, ...]:
        owner = normalize_owner_id(owner_id)
        name = _identifier(extractor_name, "extractor_name", 200)
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM scientific_claim_extractor_activations "
                "WHERE owner_id=? AND extractor_name=? "
                "ORDER BY activated_at DESC, activation_id DESC LIMIT ?",
                (owner, name, count),
            ).fetchall()
        return tuple(self._activation(row) for row in rows)


class GovernedScientificClaimExtractorPromotionService:
    def __init__(
        self,
        *,
        extractor_registry: ScientificClaimExtractorRegistry,
        promotion_store: ScientificClaimExtractorPromotionStore,
        policy: ClaimExtractorPromotionPolicy,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(extractor_registry, ScientificClaimExtractorRegistry):
            raise ValueError("extractor_registry must be ScientificClaimExtractorRegistry.")
        if not isinstance(promotion_store, ScientificClaimExtractorPromotionStore):
            raise ValueError("promotion_store must be ScientificClaimExtractorPromotionStore.")
        if not isinstance(policy, ClaimExtractorPromotionPolicy):
            raise ValueError("policy must be ClaimExtractorPromotionPolicy.")
        if not callable(clock):
            raise ValueError("clock must be callable.")
        self.extractor_registry = extractor_registry
        self.promotion_store = promotion_store
        self.policy = policy
        self.clock = clock

    def _authorize(
        self,
        *,
        actor: ReviewActorBinding,
        owner_id: str,
        extractor_name: str,
        action: str,
    ) -> float:
        if not isinstance(actor, ReviewActorBinding):
            raise ValueError("actor must be ReviewActorBinding.")
        require_relation_review_actor(actor.actor_id, binding=actor)
        owner = normalize_owner_id(owner_id)
        name = _identifier(extractor_name, "extractor_name", 200)
        selected_action = _identifier(action, "action", 30)
        if selected_action not in _PROMOTION_ACTIONS:
            raise ValueError("promotion action is unsupported.")
        now = _timestamp(self.clock(), "promotion time")
        grant = self.policy.grant_for(actor.actor_id)
        if not grant.permits(
            owner_id=owner,
            extractor_name=name,
            action=selected_action,
            now=now,
        ):
            raise PermissionError("promotion administrator grant does not permit this scope.")
        return now

    def promote(
        self,
        *,
        benchmark_suite: ScientificClaimExtractorBenchmarkSuite,
        expected_current_activation_id: str | None,
        actor: ReviewActorBinding,
    ) -> tuple[
        ScientificClaimExtractorPromotionReport,
        ScientificClaimExtractorActivation | None,
    ]:
        now = self._authorize(
            actor=actor,
            owner_id=benchmark_suite.owner_id,
            extractor_name=benchmark_suite.extractor_name,
            action="promote",
        )
        record = self.extractor_registry.require_active(
            owner_id=benchmark_suite.owner_id,
            extractor_name=benchmark_suite.extractor_name,
            extractor_version=benchmark_suite.extractor_version,
        )
        report = assess_scientific_claim_extractor_promotion(
            extractor_record=record,
            benchmark_suite=benchmark_suite,
            policy=self.policy,
            now=now,
        )
        stored_report = self.promotion_store.store_report(report)
        if not stored_report.eligible:
            return stored_report, None
        activation = self.promotion_store.activate(
            report=stored_report,
            action="promote",
            expected_current_activation_id=expected_current_activation_id,
            actor=actor,
            now=now,
        )
        return stored_report, activation

    def rollback(
        self,
        *,
        target_promotion_report_digest: str,
        expected_current_activation_id: str,
        actor: ReviewActorBinding,
    ) -> ScientificClaimExtractorActivation:
        report = self.promotion_store.get_report(target_promotion_report_digest)
        now = self._authorize(
            actor=actor,
            owner_id=report.owner_id,
            extractor_name=report.extractor_name,
            action="rollback",
        )
        record = self.extractor_registry.require_active(
            owner_id=report.owner_id,
            extractor_name=report.extractor_name,
            extractor_version=report.extractor_version,
        )
        if record.record_digest != report.extractor_record_digest or not report.eligible:
            raise PermissionError("rollback target is not an eligible active exact version.")
        return self.promotion_store.activate(
            report=report,
            action="rollback",
            expected_current_activation_id=expected_current_activation_id,
            actor=actor,
            now=now,
        )

    def resolve_current(
        self,
        *,
        owner_id: str,
        extractor_name: str,
    ) -> ScientificClaimExtractorRecord:
        activation = self.promotion_store.current(
            owner_id=owner_id,
            extractor_name=extractor_name,
        )
        if activation is None:
            raise KeyError((owner_id, extractor_name))
        record = self.extractor_registry.require_active(
            owner_id=activation.owner_id,
            extractor_name=activation.extractor_name,
            extractor_version=activation.extractor_version,
        )
        if record.record_digest != activation.extractor_record_digest:
            raise RuntimeError("current activation differs from exact extractor record.")
        report = self.promotion_store.get_report(activation.promotion_report_digest)
        if not report.eligible or report.extractor_record_digest != record.record_digest:
            raise RuntimeError("current activation lacks an eligible exact promotion report.")
        return record


__all__ = [
    "ClaimExtractorPromotionAdministratorGrant",
    "ClaimExtractorPromotionPolicy",
    "ClaimExtractorPromotionThresholds",
    "GovernedScientificClaimExtractorPromotionService",
    "ScientificClaimExtractorActivation",
    "ScientificClaimExtractorPromotionReport",
    "ScientificClaimExtractorPromotionStore",
    "assess_scientific_claim_extractor_promotion",
    "load_claim_extractor_promotion_policy",
]
