"""Governed registry and execution boundary for scientific claim extractors."""

from __future__ import annotations

import hashlib
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
    CLAIM_MODALITIES,
    CLAIM_TYPES,
    _digest,
    _identifier,
    _integer,
    _sha256,
    _timestamp,
)
from tools.evidence_graph_claim_extraction import (
    ScientificClaimExtractionBatch,
    extract_scientific_claim_proposals,
)
from tools.evidence_graph_relation_actor import (
    ReviewActorBinding,
    require_relation_review_actor,
)
from tools.security import normalize_owner_id

EXTRACTOR_KINDS = frozenset({"model", "rule"})
EXTRACTOR_ACTIONS = frozenset({"register", "retire"})
SCIENTIFIC_CLAIM_OUTPUT_SCHEMA_SHA256 = _sha256(
    {
        "scope": "rigorousrag-scientific-claim-output-schema-v1",
        "top_level": ["schema_version", "claims"],
        "claim_required": [
            "claim_key",
            "claim_text",
            "claim_type",
            "modality",
            "section_index",
            "char_start",
            "char_end",
            "confidence",
        ],
        "claim_optional": [
            "page_number",
            "supersedes_proposal_id",
            "metadata",
        ],
        "claim_types": sorted(CLAIM_TYPES),
        "modalities": sorted(CLAIM_MODALITIES),
    }
)

_MAX_PATH = 4096
_MAX_LIMIT = 10_000
_MAX_SCOPE_VALUES = 1_000
_MAX_POLICY_BYTES = 1_000_000
_MAX_ADMINS = 1_000
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_STATES = frozenset({"active", "retired"})


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


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


def _scope_values(
    value: Any,
    label: str,
    *,
    owner_scope: bool = False,
    allowed: frozenset[str] | None = None,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a bounded array.")
    if not 1 <= len(value) <= _MAX_SCOPE_VALUES:
        raise ValueError(f"{label} must contain 1-{_MAX_SCOPE_VALUES} entries.")
    selected: set[str] = set()
    for item in value:
        rendered = _identifier(item, label, 500)
        if rendered == "*":
            selected.add(rendered)
        elif owner_scope:
            selected.add(normalize_owner_id(rendered))
        else:
            selected.add(rendered.casefold() if label == "languages" else rendered)
    if "*" in selected and len(selected) != 1:
        raise ValueError(f"{label} wildcard may not be combined with explicit entries.")
    if allowed is not None and ("*" in selected or any(item not in allowed for item in selected)):
        raise ValueError(f"{label} contains an unsupported value.")
    return tuple(sorted(selected))


def _allows(scope: tuple[str, ...], value: str) -> bool:
    return scope == ("*",) or value in scope


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("extractor governance policy contains a duplicate JSON key.")
        result[key] = value
    return result


@dataclass(frozen=True)
class ScientificClaimExtractorRecord:
    owner_id: str
    extractor_name: str
    extractor_version: str
    extractor_kind: str
    implementation_sha256: str
    configuration_sha256: str
    output_schema_sha256: str
    supported_claim_types: tuple[str, ...]
    supported_modalities: tuple[str, ...]
    supported_languages: tuple[str, ...]
    state: str
    registered_actor_id: str
    registered_binding_method: str
    registered_binding_digest: str
    registered_at: float
    retired_actor_id: str | None
    retired_binding_method: str | None
    retired_binding_digest: str | None
    retired_at: float | None
    record_digest: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        name = _identifier(self.extractor_name, "extractor_name", 200)
        version = _identifier(self.extractor_version, "extractor_version", 200)
        kind = _identifier(self.extractor_kind, "extractor_kind", 20)
        if kind not in EXTRACTOR_KINDS:
            raise ValueError("extractor_kind is unsupported.")
        implementation = _digest(self.implementation_sha256, "implementation_sha256")
        configuration = _digest(self.configuration_sha256, "configuration_sha256")
        output_schema = _digest(self.output_schema_sha256, "output_schema_sha256")
        if output_schema != SCIENTIFIC_CLAIM_OUTPUT_SCHEMA_SHA256:
            raise ValueError("extractor output schema differs from the supported claim schema.")
        claim_types = _scope_values(
            self.supported_claim_types,
            "supported_claim_types",
            allowed=CLAIM_TYPES,
        )
        modalities = _scope_values(
            self.supported_modalities,
            "supported_modalities",
            allowed=CLAIM_MODALITIES,
        )
        languages = _scope_values(self.supported_languages, "languages")
        state = _identifier(self.state, "state", 30)
        if state not in _STATES:
            raise ValueError("extractor state is unsupported.")
        registered_actor = _identifier(
            self.registered_actor_id, "registered_actor_id", 200
        )
        registered_method = _identifier(
            self.registered_binding_method, "registered_binding_method", 50
        )
        registered_binding = _digest(
            self.registered_binding_digest, "registered_binding_digest"
        )
        registered_at = _timestamp(self.registered_at, "registered_at")
        retired_values = (
            self.retired_actor_id,
            self.retired_binding_method,
            self.retired_binding_digest,
            self.retired_at,
        )
        if state == "active":
            if any(value is not None for value in retired_values):
                raise ValueError("active extractor may not contain retirement fields.")
            retired_actor = retired_method = retired_binding = retired_at = None
        else:
            if any(value is None for value in retired_values):
                raise ValueError("retired extractor requires complete retirement fields.")
            retired_actor = _identifier(self.retired_actor_id, "retired_actor_id", 200)
            retired_method = _identifier(
                self.retired_binding_method, "retired_binding_method", 50
            )
            retired_binding = _digest(
                self.retired_binding_digest, "retired_binding_digest"
            )
            retired_at = _timestamp(self.retired_at, "retired_at")
            if retired_at < registered_at:
                raise ValueError("extractor retirement predates registration.")
        stable = {
            "scope": "rigorousrag-scientific-claim-extractor-record-v1",
            "owner_id": owner,
            "extractor_name": name,
            "extractor_version": version,
            "extractor_kind": kind,
            "implementation_sha256": implementation,
            "configuration_sha256": configuration,
            "output_schema_sha256": output_schema,
            "supported_claim_types": claim_types,
            "supported_modalities": modalities,
            "supported_languages": languages,
            "state": state,
            "registered_actor_id": registered_actor,
            "registered_binding_method": registered_method,
            "registered_binding_digest": registered_binding,
            "registered_at": registered_at,
            "retired_actor_id": retired_actor,
            "retired_binding_method": retired_method,
            "retired_binding_digest": retired_binding,
            "retired_at": retired_at,
            "schema_version": self.schema_version,
        }
        if self.schema_version != 1:
            raise ValueError("extractor registry schema is unsupported.")
        record_digest = _digest(self.record_digest, "record_digest")
        if record_digest != _canonical_digest(stable):
            raise ValueError("record_digest differs from extractor record.")
        for key, value in {
            "owner_id": owner,
            "extractor_name": name,
            "extractor_version": version,
            "extractor_kind": kind,
            "implementation_sha256": implementation,
            "configuration_sha256": configuration,
            "output_schema_sha256": output_schema,
            "supported_claim_types": claim_types,
            "supported_modalities": modalities,
            "supported_languages": languages,
            "state": state,
            "registered_actor_id": registered_actor,
            "registered_binding_method": registered_method,
            "registered_binding_digest": registered_binding,
            "registered_at": registered_at,
            "retired_actor_id": retired_actor,
            "retired_binding_method": retired_method,
            "retired_binding_digest": retired_binding,
            "retired_at": retired_at,
            "record_digest": record_digest,
        }.items():
            object.__setattr__(self, key, value)

    @classmethod
    def active(
        cls,
        *,
        owner_id: str,
        extractor_name: str,
        extractor_version: str,
        extractor_kind: str,
        implementation_sha256: str,
        configuration_sha256: str,
        supported_claim_types: Sequence[str],
        supported_modalities: Sequence[str],
        supported_languages: Sequence[str],
        actor: ReviewActorBinding,
        now: float,
    ) -> "ScientificClaimExtractorRecord":
        if not isinstance(actor, ReviewActorBinding):
            raise ValueError("actor must be ReviewActorBinding.")
        values = {
            "owner_id": normalize_owner_id(owner_id),
            "extractor_name": _identifier(extractor_name, "extractor_name", 200),
            "extractor_version": _identifier(extractor_version, "extractor_version", 200),
            "extractor_kind": _identifier(extractor_kind, "extractor_kind", 20),
            "implementation_sha256": _digest(
                implementation_sha256, "implementation_sha256"
            ),
            "configuration_sha256": _digest(
                configuration_sha256, "configuration_sha256"
            ),
            "output_schema_sha256": SCIENTIFIC_CLAIM_OUTPUT_SCHEMA_SHA256,
            "supported_claim_types": tuple(supported_claim_types),
            "supported_modalities": tuple(supported_modalities),
            "supported_languages": tuple(supported_languages),
            "state": "active",
            "registered_actor_id": actor.actor_id,
            "registered_binding_method": actor.binding_method,
            "registered_binding_digest": actor.binding_digest,
            "registered_at": _timestamp(now, "now"),
            "retired_actor_id": None,
            "retired_binding_method": None,
            "retired_binding_digest": None,
            "retired_at": None,
            "schema_version": 1,
        }
        normalized = cls(
            **values,
            record_digest="0" * 64,
        ) if False else None
        stable = {
            "scope": "rigorousrag-scientific-claim-extractor-record-v1",
            **{
                **values,
                "supported_claim_types": _scope_values(
                    values["supported_claim_types"],
                    "supported_claim_types",
                    allowed=CLAIM_TYPES,
                ),
                "supported_modalities": _scope_values(
                    values["supported_modalities"],
                    "supported_modalities",
                    allowed=CLAIM_MODALITIES,
                ),
                "supported_languages": _scope_values(
                    values["supported_languages"], "languages"
                ),
            },
        }
        return cls(**values, record_digest=_canonical_digest(stable))

    def retire(self, *, actor: ReviewActorBinding, now: float) -> "ScientificClaimExtractorRecord":
        if not isinstance(actor, ReviewActorBinding):
            raise ValueError("actor must be ReviewActorBinding.")
        if self.state == "retired":
            if (
                self.retired_actor_id != actor.actor_id
                or self.retired_binding_method != actor.binding_method
                or self.retired_binding_digest != actor.binding_digest
            ):
                raise RuntimeError("extractor is already retired by another actor binding.")
            return self
        values = {
            **asdict(self),
            "state": "retired",
            "retired_actor_id": actor.actor_id,
            "retired_binding_method": actor.binding_method,
            "retired_binding_digest": actor.binding_digest,
            "retired_at": max(_timestamp(now, "now"), self.registered_at),
        }
        values.pop("record_digest", None)
        return ScientificClaimExtractorRecord(
            **values,
            record_digest=_canonical_digest(
                {
                    "scope": "rigorousrag-scientific-claim-extractor-record-v1",
                    **values,
                }
            ),
        )


@dataclass(frozen=True)
class ClaimExtractorAdministratorGrant:
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
        if actions == ("*",) or any(value not in EXTRACTOR_ACTIONS for value in actions):
            raise ValueError("actions contains an unsupported value.")
        object.__setattr__(self, "actions", actions)
        if self.expires_at is not None:
            object.__setattr__(
                self,
                "expires_at",
                _timestamp(self.expires_at, "expires_at"),
            )
        if self.schema_version != 1:
            raise ValueError("extractor administrator grant schema is unsupported.")

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
class ClaimExtractorGovernancePolicy:
    administrators: tuple[ClaimExtractorAdministratorGrant, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.administrators, tuple)
            or not 1 <= len(self.administrators) <= _MAX_ADMINS
            or any(
                not isinstance(value, ClaimExtractorAdministratorGrant)
                for value in self.administrators
            )
        ):
            raise ValueError("administrators must be a bounded non-empty tuple.")
        ordered = tuple(
            sorted(self.administrators, key=lambda value: value.administrator_id)
        )
        if len({value.administrator_id for value in ordered}) != len(ordered):
            raise ValueError("extractor administrator IDs must be unique.")
        object.__setattr__(self, "administrators", ordered)
        if self.schema_version != 1:
            raise ValueError("extractor governance policy schema is unsupported.")

    @property
    def policy_digest(self) -> str:
        return _sha256(asdict(self))

    def grant_for(self, administrator_id: str) -> ClaimExtractorAdministratorGrant:
        selected = _identifier(administrator_id, "administrator_id", 200)
        for grant in self.administrators:
            if grant.administrator_id == selected:
                return grant
        raise PermissionError("extractor administrator is not authorized.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ClaimExtractorGovernancePolicy":
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version",
            "administrators",
        }:
            raise ValueError("extractor governance policy schema is invalid.")
        if value["schema_version"] != 1:
            raise ValueError("extractor governance policy schema is unsupported.")
        raw_values = value["administrators"]
        if (
            isinstance(raw_values, (str, bytes, bytearray))
            or not isinstance(raw_values, Sequence)
            or not 1 <= len(raw_values) <= _MAX_ADMINS
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
        grants = []
        for raw in raw_values:
            if not isinstance(raw, Mapping) or not required <= set(raw) <= allowed:
                raise ValueError("extractor administrator grant schema is invalid.")
            grants.append(
                ClaimExtractorAdministratorGrant(
                    administrator_id=raw["administrator_id"],
                    owners=_scope_values(raw["owners"], "owners", owner_scope=True),
                    extractor_names=_scope_values(
                        raw["extractor_names"], "extractor_names"
                    ),
                    actions=_scope_values(raw["actions"], "actions"),
                    expires_at=raw.get("expires_at"),
                )
            )
        return cls(administrators=tuple(grants))


def load_claim_extractor_governance_policy(
    *,
    path: str | os.PathLike[str] | None = None,
    policy_json: str | None = None,
) -> ClaimExtractorGovernancePolicy:
    if path is None and policy_json is None:
        configured_path = os.getenv("EVIDENCE_GRAPH_CLAIM_EXTRACTOR_POLICY_PATH")
        configured_json = os.getenv("EVIDENCE_GRAPH_CLAIM_EXTRACTOR_POLICY_JSON")
        path = configured_path if configured_path else None
        policy_json = configured_json if configured_json else None
    if (path is None) == (policy_json is None):
        raise RuntimeError("configure exactly one claim extractor governance policy source.")
    if path is not None:
        selected = _path(path, "extractor governance policy path")
        descriptor = os.open(
            selected,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= _MAX_POLICY_BYTES:
                raise ValueError("extractor governance policy file is invalid or too large.")
            payload = os.read(descriptor, _MAX_POLICY_BYTES + 1)
            if len(payload) > _MAX_POLICY_BYTES:
                raise ValueError("extractor governance policy file is too large.")
        finally:
            os.close(descriptor)
    else:
        payload = policy_json.encode("utf-8")
        if not 1 <= len(payload) <= _MAX_POLICY_BYTES:
            raise ValueError("extractor governance policy size is invalid.")
    try:
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError("extractor governance policy JSON is invalid.") from exc
    return ClaimExtractorGovernancePolicy.from_mapping(raw)


class ScientificClaimExtractorRegistry:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path, "extractor registry path")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("extractor registry parent must be a directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("extractor registry is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
            or self._file_identity() != self._database_identity
        ):
            raise RuntimeError("extractor registry identity changed.")

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
                CREATE TABLE IF NOT EXISTS scientific_claim_extractors (
                    owner_id TEXT NOT NULL,
                    extractor_name TEXT NOT NULL,
                    extractor_version TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    record_digest TEXT NOT NULL,
                    state TEXT NOT NULL,
                    registered_at REAL NOT NULL,
                    retired_at REAL,
                    schema_version INTEGER NOT NULL,
                    PRIMARY KEY(owner_id, extractor_name, extractor_version)
                );
                CREATE INDEX IF NOT EXISTS scientific_claim_extractor_scope
                    ON scientific_claim_extractors(
                        owner_id, extractor_name, state, registered_at, extractor_version
                    );
                """
            )

    @staticmethod
    def _value(row: sqlite3.Row) -> ScientificClaimExtractorRecord:
        try:
            raw = json.loads(
                row["payload_json"],
                object_pairs_hook=_pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
            )
            value = ScientificClaimExtractorRecord(**raw)
        except Exception as exc:
            raise RuntimeError("stored extractor record is corrupt.") from exc
        if (
            value.owner_id != row["owner_id"]
            or value.extractor_name != row["extractor_name"]
            or value.extractor_version != row["extractor_version"]
            or value.record_digest != row["record_digest"]
            or value.state != row["state"]
            or value.registered_at != float(row["registered_at"])
            or value.retired_at != row["retired_at"]
        ):
            raise RuntimeError("stored extractor record columns are corrupt.")
        return value

    @staticmethod
    def _registration_scope(value: ScientificClaimExtractorRecord) -> tuple[Any, ...]:
        return (
            value.owner_id,
            value.extractor_name,
            value.extractor_version,
            value.extractor_kind,
            value.implementation_sha256,
            value.configuration_sha256,
            value.output_schema_sha256,
            value.supported_claim_types,
            value.supported_modalities,
            value.supported_languages,
            value.registered_actor_id,
            value.registered_binding_method,
            value.registered_binding_digest,
        )

    def register(self, record: ScientificClaimExtractorRecord) -> ScientificClaimExtractorRecord:
        if not isinstance(record, ScientificClaimExtractorRecord) or record.state != "active":
            raise ValueError("record must be an active ScientificClaimExtractorRecord.")
        payload = json.dumps(
            asdict(record), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM scientific_claim_extractors "
                    "WHERE owner_id=? AND extractor_name=? AND extractor_version=?",
                    (record.owner_id, record.extractor_name, record.extractor_version),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO scientific_claim_extractors VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)",
                        (
                            record.owner_id,
                            record.extractor_name,
                            record.extractor_version,
                            payload,
                            record.record_digest,
                            record.state,
                            record.registered_at,
                            record.retired_at,
                        ),
                    )
                else:
                    existing = self._value(row)
                    if self._registration_scope(existing) != self._registration_scope(record):
                        raise RuntimeError("extractor version is already registered differently.")
                    if existing.state == "retired":
                        raise RuntimeError("retired extractor versions may not be reactivated.")
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(
            owner_id=record.owner_id,
            extractor_name=record.extractor_name,
            extractor_version=record.extractor_version,
        )

    def get(
        self,
        *,
        owner_id: str,
        extractor_name: str,
        extractor_version: str,
    ) -> ScientificClaimExtractorRecord:
        owner = normalize_owner_id(owner_id)
        name = _identifier(extractor_name, "extractor_name", 200)
        version = _identifier(extractor_version, "extractor_version", 200)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM scientific_claim_extractors "
                "WHERE owner_id=? AND extractor_name=? AND extractor_version=?",
                (owner, name, version),
            ).fetchone()
        if row is None:
            raise KeyError((owner, name, version))
        return self._value(row)

    def require_active(self, **kwargs: Any) -> ScientificClaimExtractorRecord:
        value = self.get(**kwargs)
        if value.state != "active":
            raise PermissionError("scientific claim extractor version is retired.")
        return value

    def retire(
        self,
        *,
        owner_id: str,
        extractor_name: str,
        extractor_version: str,
        actor: ReviewActorBinding,
        now: float,
    ) -> ScientificClaimExtractorRecord:
        current = self.get(
            owner_id=owner_id,
            extractor_name=extractor_name,
            extractor_version=extractor_version,
        )
        retired = current.retire(actor=actor, now=now)
        payload = json.dumps(
            asdict(retired), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM scientific_claim_extractors "
                    "WHERE owner_id=? AND extractor_name=? AND extractor_version=?",
                    (current.owner_id, current.extractor_name, current.extractor_version),
                ).fetchone()
                if row is None:
                    raise KeyError((current.owner_id, current.extractor_name, current.extractor_version))
                observed = self._value(row)
                if observed.record_digest != current.record_digest:
                    if observed.record_digest == retired.record_digest:
                        connection.execute("COMMIT")
                        return observed
                    raise RuntimeError("extractor record changed during retirement.")
                connection.execute(
                    "UPDATE scientific_claim_extractors SET payload_json=?, record_digest=?, "
                    "state='retired', retired_at=? WHERE owner_id=? AND extractor_name=? "
                    "AND extractor_version=? AND record_digest=?",
                    (
                        payload,
                        retired.record_digest,
                        retired.retired_at,
                        current.owner_id,
                        current.extractor_name,
                        current.extractor_version,
                        current.record_digest,
                    ),
                )
                if connection.total_changes != 1:
                    raise RuntimeError("extractor retirement compare-and-swap failed.")
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(
            owner_id=current.owner_id,
            extractor_name=current.extractor_name,
            extractor_version=current.extractor_version,
        )

    def list(
        self,
        *,
        owner_id: str,
        extractor_name: str | None = None,
        state: str | None = None,
        limit: int = 100,
    ) -> tuple[ScientificClaimExtractorRecord, ...]:
        owner = normalize_owner_id(owner_id)
        name = None if extractor_name is None else _identifier(
            extractor_name, "extractor_name", 200
        )
        selected_state = None if state is None else _identifier(state, "state", 30)
        if selected_state is not None and selected_state not in _STATES:
            raise ValueError("extractor state is unsupported.")
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        query = "SELECT * FROM scientific_claim_extractors WHERE owner_id=?"
        params: list[Any] = [owner]
        if name is not None:
            query += " AND extractor_name=?"
            params.append(name)
        if selected_state is not None:
            query += " AND state=?"
            params.append(selected_state)
        query += " ORDER BY extractor_name, registered_at, extractor_version LIMIT ?"
        params.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(self._value(row) for row in rows)


class GovernedScientificClaimExtractorService:
    def __init__(
        self,
        *,
        registry: ScientificClaimExtractorRegistry,
        policy: ClaimExtractorGovernancePolicy,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(registry, ScientificClaimExtractorRegistry):
            raise ValueError("registry must be ScientificClaimExtractorRegistry.")
        if not isinstance(policy, ClaimExtractorGovernancePolicy):
            raise ValueError("policy must be ClaimExtractorGovernancePolicy.")
        if not callable(clock):
            raise ValueError("clock must be callable.")
        self.registry = registry
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
        if selected_action not in EXTRACTOR_ACTIONS:
            raise ValueError("extractor action is unsupported.")
        now = _timestamp(self.clock(), "governance time")
        grant = self.policy.grant_for(actor.actor_id)
        if not grant.permits(
            owner_id=owner,
            extractor_name=name,
            action=selected_action,
            now=now,
        ):
            raise PermissionError("extractor administrator grant does not permit this scope.")
        return now

    def register(self, *, actor: ReviewActorBinding, **kwargs: Any) -> ScientificClaimExtractorRecord:
        now = self._authorize(
            actor=actor,
            owner_id=kwargs["owner_id"],
            extractor_name=kwargs["extractor_name"],
            action="register",
        )
        return self.registry.register(
            ScientificClaimExtractorRecord.active(actor=actor, now=now, **kwargs)
        )

    def retire(
        self,
        *,
        actor: ReviewActorBinding,
        owner_id: str,
        extractor_name: str,
        extractor_version: str,
    ) -> ScientificClaimExtractorRecord:
        now = self._authorize(
            actor=actor,
            owner_id=owner_id,
            extractor_name=extractor_name,
            action="retire",
        )
        return self.registry.retire(
            owner_id=owner_id,
            extractor_name=extractor_name,
            extractor_version=extractor_version,
            actor=actor,
            now=now,
        )


def extract_registered_scientific_claim_proposals(
    document: Any,
    extractor_output: Any,
    *,
    owner_id: str,
    generation: int,
    profile_fingerprint: str,
    proposer_id: str,
    extractor_name: str,
    extractor_version: str,
    language: str,
    registry: ScientificClaimExtractorRegistry,
    now: float | None = None,
) -> ScientificClaimExtractionBatch:
    """Run the closed adapter only for an exact active governed extractor profile."""

    if not isinstance(registry, ScientificClaimExtractorRegistry):
        raise ValueError("registry must be ScientificClaimExtractorRegistry.")
    record = registry.require_active(
        owner_id=owner_id,
        extractor_name=extractor_name,
        extractor_version=extractor_version,
    )
    selected_language = _identifier(language, "language", 100).casefold()
    if not _allows(record.supported_languages, selected_language):
        raise PermissionError("extractor is not registered for this language.")
    batch = extract_scientific_claim_proposals(
        document,
        extractor_output,
        owner_id=owner_id,
        generation=generation,
        profile_fingerprint=profile_fingerprint,
        proposer_id=proposer_id,
        extractor_name=extractor_name,
        extractor_version=extractor_version,
        now=now,
    )
    for proposal in batch.proposals:
        if proposal.proposer_kind != record.extractor_kind:
            raise PermissionError("proposal kind differs from registered extractor kind.")
        if not _allows(record.supported_claim_types, proposal.claim_type):
            raise PermissionError("extractor emitted an unregistered claim type.")
        if not _allows(record.supported_modalities, proposal.modality):
            raise PermissionError("extractor emitted an unregistered modality.")
    return batch


__all__ = [
    "EXTRACTOR_ACTIONS",
    "EXTRACTOR_KINDS",
    "SCIENTIFIC_CLAIM_OUTPUT_SCHEMA_SHA256",
    "ClaimExtractorAdministratorGrant",
    "ClaimExtractorGovernancePolicy",
    "GovernedScientificClaimExtractorService",
    "ScientificClaimExtractorRecord",
    "ScientificClaimExtractorRegistry",
    "extract_registered_scientific_claim_proposals",
    "load_claim_extractor_governance_policy",
]
