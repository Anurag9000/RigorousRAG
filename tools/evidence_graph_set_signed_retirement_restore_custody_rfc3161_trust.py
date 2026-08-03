"""Governed external TSA trust profiles for RFC 3161 custody verification."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from cryptography import x509

from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _integer,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_rfc3161_contracts import (
    MAX_INPUT_BYTES,
    Rfc3161TimestampVerificationReceipt,
    canonical_digest,
    optional_oid,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_rfc3161_io import (
    read_regular,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_rfc3161_verify import (
    verify_rfc3161_timestamp_response,
)
from tools.security import normalize_owner_id

_SCHEMA_VERSION = 1
_STATES = frozenset({"active", "retired"})
_MAX_LIMIT = 10_000
_MAX_SIGNERS = 100
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_TABLE = "evidence_graph_restore_custody_rfc3161_trust_profiles"


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _registry_path(value: str | os.PathLike[str]) -> Path:
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("RFC 3161 trust registry path is invalid.")
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
            raise ValueError(
                "RFC 3161 trust registry path could not be validated."
            ) from exc
        if _redirecting(info):
            raise ValueError("RFC 3161 trust registry path may not contain redirects.")
    return absolute


def _actor(value: ReviewActorBinding) -> ReviewActorBinding:
    if not isinstance(value, ReviewActorBinding):
        raise ValueError("actor must be ReviewActorBinding.")
    _identifier(value.actor_id, "actor_id", 200)
    _identifier(value.binding_method, "binding_method", 50)
    _digest(value.binding_digest, "binding_digest")
    _timestamp(value.loaded_at, "loaded_at")
    return value


def _signers(values: Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("allowed signer fingerprints must be an iterable.")
    rendered = sorted(
        {
            _digest(value, "allowed_signer_certificate_sha256")
            for value in values
        }
    )
    if len(rendered) > _MAX_SIGNERS:
        raise ValueError("allowed signer fingerprints exceed the limit.")
    return tuple(rendered)


def _bundle_digest(path: str | os.PathLike[str] | None, label: str) -> str | None:
    if path is None:
        return None
    payload = read_regular(path, label=label, maximum=MAX_INPUT_BYTES)
    return hashlib.sha256(payload).hexdigest()


def _validate_trust_anchor_bundle(path: str | os.PathLike[str]) -> str:
    payload = read_regular(
        path,
        label="trust_anchor_bundle_path",
        maximum=MAX_INPUT_BYTES,
    )
    try:
        certificates = x509.load_pem_x509_certificates(payload)
    except ValueError as exc:
        raise ValueError("trust anchor bundle is not valid PEM certificates.") from exc
    if not certificates:
        raise ValueError("trust anchor bundle contains no certificates.")
    for certificate in certificates:
        try:
            constraints = certificate.extensions.get_extension_for_class(
                x509.BasicConstraints
            )
        except x509.ExtensionNotFound as exc:
            raise ValueError(
                "trust anchor certificate lacks basic constraints."
            ) from exc
        if not constraints.value.ca:
            raise ValueError("trust anchor bundle contains a non-CA certificate.")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Rfc3161TrustProfile:
    owner_id: str
    profile_id: str
    policy_oid: str
    trust_anchor_bundle_sha256: str
    untrusted_bundle_sha256: str | None
    crl_bundle_sha256: str | None
    allowed_signer_certificate_sha256: tuple[str, ...]
    valid_from: float
    valid_until: float | None
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
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        profile = _identifier(self.profile_id, "profile_id", 200)
        policy = optional_oid(self.policy_oid, "policy_oid")
        if policy is None:
            raise ValueError("trust profile requires a policy OID.")
        trust = _digest(
            self.trust_anchor_bundle_sha256,
            "trust_anchor_bundle_sha256",
        )
        untrusted = (
            None
            if self.untrusted_bundle_sha256 is None
            else _digest(
                self.untrusted_bundle_sha256,
                "untrusted_bundle_sha256",
            )
        )
        crl = (
            None
            if self.crl_bundle_sha256 is None
            else _digest(self.crl_bundle_sha256, "crl_bundle_sha256")
        )
        signers = _signers(self.allowed_signer_certificate_sha256)
        valid_from = _timestamp(self.valid_from, "valid_from")
        valid_until = (
            None
            if self.valid_until is None
            else _timestamp(self.valid_until, "valid_until")
        )
        if valid_until is not None and valid_until < valid_from:
            raise ValueError("trust profile validity window is reversed.")
        state = _identifier(self.state, "state", 30)
        if state not in _STATES:
            raise ValueError("trust profile state is unsupported.")
        registered_actor = _identifier(
            self.registered_actor_id,
            "registered_actor_id",
            200,
        )
        registered_method = _identifier(
            self.registered_binding_method,
            "registered_binding_method",
            50,
        )
        registered_binding = _digest(
            self.registered_binding_digest,
            "registered_binding_digest",
        )
        registered_at = _timestamp(self.registered_at, "registered_at")
        if state == "active":
            if any(
                value is not None
                for value in (
                    self.retired_actor_id,
                    self.retired_binding_method,
                    self.retired_binding_digest,
                    self.retired_at,
                )
            ):
                raise ValueError(
                    "active trust profile may not contain retirement fields."
                )
            retired_actor = retired_method = retired_binding = retired_at = None
        else:
            if any(
                value is None
                for value in (
                    self.retired_actor_id,
                    self.retired_binding_method,
                    self.retired_binding_digest,
                    self.retired_at,
                )
            ):
                raise ValueError(
                    "retired trust profile requires retirement fields."
                )
            retired_actor = _identifier(
                self.retired_actor_id,
                "retired_actor_id",
                200,
            )
            retired_method = _identifier(
                self.retired_binding_method,
                "retired_binding_method",
                50,
            )
            retired_binding = _digest(
                self.retired_binding_digest,
                "retired_binding_digest",
            )
            retired_at = _timestamp(self.retired_at, "retired_at")
            if retired_at < registered_at:
                raise ValueError("trust profile retirement predates registration.")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("trust profile schema is unsupported.")
        stable = {
            "scope": "rigorousrag-rfc3161-trust-profile-v1",
            "owner_id": owner,
            "profile_id": profile,
            "policy_oid": policy,
            "trust_anchor_bundle_sha256": trust,
            "untrusted_bundle_sha256": untrusted,
            "crl_bundle_sha256": crl,
            "allowed_signer_certificate_sha256": list(signers),
            "valid_from": valid_from,
            "valid_until": valid_until,
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
        record_digest = _digest(self.record_digest, "record_digest")
        if record_digest != canonical_digest(stable):
            raise ValueError("record_digest differs from RFC 3161 trust profile.")
        for name, value in stable.items():
            if name not in {"scope", "allowed_signer_certificate_sha256"}:
                object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "allowed_signer_certificate_sha256",
            signers,
        )
        object.__setattr__(self, "record_digest", record_digest)

    @classmethod
    def active(
        cls,
        *,
        owner_id: str,
        profile_id: str,
        policy_oid: str,
        trust_anchor_bundle_sha256: str,
        untrusted_bundle_sha256: str | None,
        crl_bundle_sha256: str | None,
        allowed_signer_certificate_sha256: Iterable[str] | None,
        valid_from: float,
        valid_until: float | None,
        actor: ReviewActorBinding,
        now: float,
    ) -> "Rfc3161TrustProfile":
        selected_actor = _actor(actor)
        values = {
            "owner_id": normalize_owner_id(owner_id),
            "profile_id": _identifier(profile_id, "profile_id", 200),
            "policy_oid": optional_oid(policy_oid, "policy_oid"),
            "trust_anchor_bundle_sha256": _digest(
                trust_anchor_bundle_sha256,
                "trust_anchor_bundle_sha256",
            ),
            "untrusted_bundle_sha256": untrusted_bundle_sha256,
            "crl_bundle_sha256": crl_bundle_sha256,
            "allowed_signer_certificate_sha256": _signers(
                allowed_signer_certificate_sha256
            ),
            "valid_from": _timestamp(valid_from, "valid_from"),
            "valid_until": (
                None
                if valid_until is None
                else _timestamp(valid_until, "valid_until")
            ),
            "state": "active",
            "registered_actor_id": selected_actor.actor_id,
            "registered_binding_method": selected_actor.binding_method,
            "registered_binding_digest": selected_actor.binding_digest,
            "registered_at": _timestamp(now, "now"),
            "retired_actor_id": None,
            "retired_binding_method": None,
            "retired_binding_digest": None,
            "retired_at": None,
            "schema_version": _SCHEMA_VERSION,
        }
        stable = {
            "scope": "rigorousrag-rfc3161-trust-profile-v1",
            **values,
        }
        stable["allowed_signer_certificate_sha256"] = list(
            values["allowed_signer_certificate_sha256"]
        )
        return cls(**values, record_digest=canonical_digest(stable))

    def retire(
        self,
        *,
        actor: ReviewActorBinding,
        now: float,
    ) -> "Rfc3161TrustProfile":
        selected_actor = _actor(actor)
        if self.state == "retired":
            if (
                self.retired_actor_id != selected_actor.actor_id
                or self.retired_binding_method != selected_actor.binding_method
                or self.retired_binding_digest != selected_actor.binding_digest
            ):
                raise RuntimeError(
                    "trust profile already retired by another actor."
                )
            return self
        values = asdict(self)
        values.pop("record_digest")
        values.update(
            state="retired",
            retired_actor_id=selected_actor.actor_id,
            retired_binding_method=selected_actor.binding_method,
            retired_binding_digest=selected_actor.binding_digest,
            retired_at=max(_timestamp(now, "now"), self.registered_at),
        )
        stable = {
            "scope": "rigorousrag-rfc3161-trust-profile-v1",
            **values,
        }
        stable["allowed_signer_certificate_sha256"] = list(
            values["allowed_signer_certificate_sha256"]
        )
        return Rfc3161TrustProfile(
            **values,
            record_digest=canonical_digest(stable),
        )

    def permits(self, receipt: Rfc3161TimestampVerificationReceipt) -> bool:
        if not isinstance(receipt, Rfc3161TimestampVerificationReceipt):
            raise ValueError(
                "receipt must be an RFC 3161 verification receipt."
            )
        if receipt.owner_id != self.owner_id or receipt.policy_oid != self.policy_oid:
            return False
        if receipt.trust_anchor_bundle_sha256 != self.trust_anchor_bundle_sha256:
            return False
        if receipt.untrusted_bundle_sha256 != self.untrusted_bundle_sha256:
            return False
        if receipt.crl_bundle_sha256 != self.crl_bundle_sha256:
            return False
        if (
            self.allowed_signer_certificate_sha256
            and receipt.signer_certificate_sha256
            not in self.allowed_signer_certificate_sha256
        ):
            return False
        upper = self.valid_until
        if self.retired_at is not None:
            upper = self.retired_at if upper is None else min(upper, self.retired_at)
        return receipt.generated_at_unix >= self.valid_from and (
            upper is None or receipt.generated_at_unix <= upper
        )


class Rfc3161TrustRegistry:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _registry_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("RFC 3161 trust registry parent must be a directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("RFC 3161 trust registry is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino))
            != self._parent_identity
        ):
            raise RuntimeError("RFC 3161 trust registry parent identity changed.")
        if self._file_identity() != self._database_identity:
            raise RuntimeError("RFC 3161 trust registry identity changed.")

    def _connect(self) -> sqlite3.Connection:
        self._verify()
        connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
        ) as connection:
            connection.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS {_TABLE} (
                    owner_id TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    registered_at REAL NOT NULL,
                    retired_at REAL,
                    PRIMARY KEY(owner_id, profile_id)
                );
                CREATE INDEX IF NOT EXISTS rfc3161_trust_profile_state
                    ON {_TABLE}(owner_id, state, registered_at, profile_id);
                """
            )

    @staticmethod
    def _decode(value: str) -> Rfc3161TrustProfile:
        try:
            def pairs(values):
                rendered = {}
                for key, item in values:
                    if key in rendered:
                        raise ValueError("duplicate key")
                    rendered[key] = item
                return rendered

            raw = json.loads(
                value,
                object_pairs_hook=pairs,
                parse_constant=lambda token: (
                    _ for _ in ()
                ).throw(ValueError(token)),
            )
        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
            RecursionError,
        ) as exc:
            raise RuntimeError(
                "stored RFC 3161 trust profile is corrupt."
            ) from exc
        if not isinstance(raw, dict) or set(raw) != set(
            Rfc3161TrustProfile.__dataclass_fields__
        ):
            raise RuntimeError(
                "stored RFC 3161 trust profile schema is corrupt."
            )
        try:
            raw["allowed_signer_certificate_sha256"] = tuple(
                raw["allowed_signer_certificate_sha256"]
            )
            return Rfc3161TrustProfile(**raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "stored RFC 3161 trust profile is corrupt."
            ) from exc

    @staticmethod
    def _encode(value: Rfc3161TrustProfile) -> str:
        return json.dumps(
            asdict(value),
            sort_keys=True,
            separators=(",", ":"),
        )

    def register(
        self,
        value: Rfc3161TrustProfile,
    ) -> Rfc3161TrustProfile:
        if not isinstance(value, Rfc3161TrustProfile) or value.state != "active":
            raise ValueError("active RFC 3161 trust profile is required.")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    f"SELECT record_json FROM {_TABLE} "
                    "WHERE owner_id=? AND profile_id=?",
                    (value.owner_id, value.profile_id),
                ).fetchone()
                if row is not None:
                    stored = self._decode(row["record_json"])
                    if stored.record_digest != value.record_digest:
                        raise RuntimeError(
                            "RFC 3161 trust profile identity collision."
                        )
                    connection.execute("COMMIT")
                    return stored
                connection.execute(
                    f"INSERT INTO {_TABLE} VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        value.owner_id,
                        value.profile_id,
                        self._encode(value),
                        value.state,
                        value.registered_at,
                        value.retired_at,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return value

    def get(
        self,
        *,
        owner_id: str,
        profile_id: str,
    ) -> Rfc3161TrustProfile:
        owner = normalize_owner_id(owner_id)
        profile = _identifier(profile_id, "profile_id", 200)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT record_json FROM {_TABLE} "
                "WHERE owner_id=? AND profile_id=?",
                (owner, profile),
            ).fetchone()
        if row is None:
            raise KeyError(profile)
        return self._decode(row["record_json"])

    def list(
        self,
        *,
        owner_id: str,
        state: str | None = None,
        limit: int = 100,
    ) -> tuple[Rfc3161TrustProfile, ...]:
        owner = normalize_owner_id(owner_id)
        selected_state = (
            None if state is None else _identifier(state, "state", 30)
        )
        if selected_state is not None and selected_state not in _STATES:
            raise ValueError("trust profile state is unsupported.")
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        query = f"SELECT record_json FROM {_TABLE} WHERE owner_id=?"
        params: list[Any] = [owner]
        if selected_state is not None:
            query += " AND state=?"
            params.append(selected_state)
        query += " ORDER BY registered_at DESC, profile_id DESC LIMIT ?"
        params.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(self._decode(row["record_json"]) for row in rows)

    def retire(
        self,
        *,
        owner_id: str,
        profile_id: str,
        actor: ReviewActorBinding,
        now: float | None = None,
    ) -> Rfc3161TrustProfile:
        owner = normalize_owner_id(owner_id)
        profile = _identifier(profile_id, "profile_id", 200)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    f"SELECT record_json FROM {_TABLE} "
                    "WHERE owner_id=? AND profile_id=?",
                    (owner, profile),
                ).fetchone()
                if row is None:
                    raise KeyError(profile)
                current = self._decode(row["record_json"])
                retired = current.retire(actor=actor, now=timestamp)
                connection.execute(
                    f"UPDATE {_TABLE} "
                    "SET record_json=?, state=?, retired_at=? "
                    "WHERE owner_id=? AND profile_id=?",
                    (
                        self._encode(retired),
                        retired.state,
                        retired.retired_at,
                        owner,
                        profile,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return retired


def register_rfc3161_trust_profile(
    *,
    registry: Rfc3161TrustRegistry,
    owner_id: str,
    profile_id: str,
    policy_oid: str,
    trust_anchor_bundle_path: str | os.PathLike[str],
    actor: ReviewActorBinding,
    valid_from: float,
    valid_until: float | None = None,
    untrusted_bundle_path: str | os.PathLike[str] | None = None,
    crl_bundle_path: str | os.PathLike[str] | None = None,
    allowed_signer_certificate_sha256: Iterable[str] | None = None,
    now: float | None = None,
) -> Rfc3161TrustProfile:
    if not isinstance(registry, Rfc3161TrustRegistry):
        raise ValueError("registry must be Rfc3161TrustRegistry.")
    timestamp = _timestamp(time.time() if now is None else now, "now")
    value = Rfc3161TrustProfile.active(
        owner_id=owner_id,
        profile_id=profile_id,
        policy_oid=policy_oid,
        trust_anchor_bundle_sha256=_validate_trust_anchor_bundle(
            trust_anchor_bundle_path
        ),
        untrusted_bundle_sha256=_bundle_digest(
            untrusted_bundle_path,
            "untrusted_bundle_path",
        ),
        crl_bundle_sha256=_bundle_digest(
            crl_bundle_path,
            "crl_bundle_path",
        ),
        allowed_signer_certificate_sha256=(
            allowed_signer_certificate_sha256
        ),
        valid_from=valid_from,
        valid_until=valid_until,
        actor=actor,
        now=timestamp,
    )
    return registry.register(value)


def verify_rfc3161_timestamp_response_with_profile(
    *,
    registry: Rfc3161TrustRegistry,
    owner_id: str,
    profile_id: str,
    request_bundle_path: str | os.PathLike[str],
    response_path: str | os.PathLike[str],
    trust_anchor_bundle_path: str | os.PathLike[str],
    output_receipt_path: str | os.PathLike[str] | None = None,
    untrusted_bundle_path: str | os.PathLike[str] | None = None,
    crl_bundle_path: str | os.PathLike[str] | None = None,
    openssl_binary: str = "openssl",
    timeout_seconds: int = 30,
    now: float | None = None,
    maximum_future_seconds: float = 300.0,
) -> tuple[Rfc3161TimestampVerificationReceipt, Rfc3161TrustProfile]:
    if not isinstance(registry, Rfc3161TrustRegistry):
        raise ValueError("registry must be Rfc3161TrustRegistry.")
    profile = registry.get(owner_id=owner_id, profile_id=profile_id)
    if (
        _validate_trust_anchor_bundle(trust_anchor_bundle_path)
        != profile.trust_anchor_bundle_sha256
    ):
        raise PermissionError(
            "trust anchor bundle differs from governed profile."
        )
    if (
        _bundle_digest(untrusted_bundle_path, "untrusted_bundle_path")
        != profile.untrusted_bundle_sha256
    ):
        raise PermissionError("untrusted bundle differs from governed profile.")
    if (
        _bundle_digest(crl_bundle_path, "crl_bundle_path")
        != profile.crl_bundle_sha256
    ):
        raise PermissionError("CRL bundle differs from governed profile.")
    receipt = verify_rfc3161_timestamp_response(
        request_bundle_path=request_bundle_path,
        response_path=response_path,
        trust_anchor_bundle_path=trust_anchor_bundle_path,
        output_receipt_path=None,
        untrusted_bundle_path=untrusted_bundle_path,
        crl_bundle_path=crl_bundle_path,
        expected_policy_oid=profile.policy_oid,
        openssl_binary=openssl_binary,
        timeout_seconds=timeout_seconds,
        now=now,
        maximum_future_seconds=maximum_future_seconds,
    )
    if not profile.permits(receipt):
        raise PermissionError(
            "RFC 3161 receipt is outside governed trust profile scope."
        )
    if output_receipt_path is not None:
        from tools.evidence_graph_set_signed_retirement_snapshot import (
            _atomic_create,
            _canonical_bytes,
            _path,
        )

        _atomic_create(
            _path(output_receipt_path, label="output_receipt_path"),
            _canonical_bytes(receipt.public_payload()) + b"\n",
        )
    return receipt, profile


__all__ = [
    "Rfc3161TrustProfile",
    "Rfc3161TrustRegistry",
    "register_rfc3161_trust_profile",
    "verify_rfc3161_timestamp_response_with_profile",
]
