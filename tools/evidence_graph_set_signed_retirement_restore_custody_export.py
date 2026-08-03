"""Deterministic external chain-of-custody manifests for signed restores."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _integer,
    _timestamp,
    deterministic_signed_retirement_restore_id,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_boundary import (
    verify_post_restore_comparison_receipt,
    verify_pre_restore_backup_receipt,
)
from tools.evidence_graph_set_signed_retirement_restore_mutation import (
    inspect_restored_target,
    target_path_digest,
    validate_terminal_snapshot,
)
from tools.evidence_graph_set_signed_retirement_snapshot import (
    _atomic_create,
    _canonical_bytes,
    _pairs,
    _path,
)
from tools.evidence_graph_set_signed_retirement_snapshot_boundary import (
    verify_signed_retirement_snapshot,
)
from tools.security import normalize_owner_id

_SCHEMA_VERSION = 1
_ENVELOPE_SCHEMA_VERSION = 1
_MAX_MANIFEST_BYTES = 32 * 1024 * 1024
_MAX_KEY_BYTES = 1024 * 1024
_MAX_RECORDS = 10_000
_HOLD_STATUSES = frozenset({"active", "inactive", "not_checked"})


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _actor_digest(value: str) -> str:
    actor = _identifier(value, "actor_id", 200)
    return hashlib.sha256(actor.encode("utf-8")).hexdigest()


def _optional_digest(value: Any, label: str) -> str | None:
    return None if value is None else _digest(value, label)


def _read_regular(
    path: str | os.PathLike[str],
    *,
    label: str,
    maximum: int,
) -> bytes:
    selected = _path(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(selected, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file.")
        if before.st_size <= 0 or before.st_size > maximum:
            raise ValueError(f"{label} size is invalid.")
        remaining = int(before.st_size)
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError(f"{label} changed while being read.")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError(f"{label} grew while being read.")
        after = os.fstat(descriptor)
        if (
            int(after.st_dev) != int(before.st_dev)
            or int(after.st_ino) != int(before.st_ino)
            or int(after.st_size) != int(before.st_size)
        ):
            raise RuntimeError(f"{label} identity changed while being read.")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _decode_json(path: str | os.PathLike[str], *, label: str) -> dict[str, Any]:
    payload = _read_regular(path, label=label, maximum=_MAX_MANIFEST_BYTES)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{label} JSON is invalid.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object.")
    return value


@dataclass(frozen=True)
class CustodyArtifactEvidence:
    artifact_id: str
    backup_path_digest: str
    receipt_path_digest: str
    backup_sha256: str
    backup_size_bytes: int
    receipt_digest: str
    actor_id_digest: str
    binding_method: str
    binding_digest: str
    completed_at: float

    def __post_init__(self) -> None:
        for field in (
            "artifact_id",
            "backup_path_digest",
            "receipt_path_digest",
            "backup_sha256",
            "receipt_digest",
            "actor_id_digest",
            "binding_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        object.__setattr__(
            self,
            "backup_size_bytes",
            _integer(
                self.backup_size_bytes,
                "backup_size_bytes",
                1,
                1024 * 1024 * 1024 * 1024,
            ),
        )
        object.__setattr__(
            self,
            "binding_method",
            _identifier(self.binding_method, "binding_method", 50),
        )
        object.__setattr__(
            self,
            "completed_at",
            _timestamp(self.completed_at, "completed_at"),
        )


@dataclass(frozen=True)
class RestoreChainOfCustodyManifest:
    owner_id: str
    restore_id: str
    snapshot_digest: str
    target_path_digest: str
    snapshot_record_count: int
    restore_target_verification_digest: str
    restore_completed_at: float
    custody_id: str
    custody_manifest_digest: str
    pre_receipt_digest: str
    backup_sha256: str
    backup_size_bytes: int
    pre_actor_id_digest: str
    pre_binding_method: str
    pre_binding_digest: str
    pre_bound_at: float
    post_receipt_digest: str
    post_target_verification_digest: str
    post_actor_id_digest: str
    post_binding_method: str
    post_binding_digest: str
    post_bound_at: float
    legal_hold_status: str
    artifacts: tuple[CustodyArtifactEvidence, ...]
    generated_at: float
    chain_digest: str
    schema_version: int = _SCHEMA_VERSION
    contains_source_text: bool = False
    contains_assertion_secrets: bool = False
    contains_raw_paths: bool = False
    mutation_performed: bool = False

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        restore = _digest(self.restore_id, "restore_id")
        snapshot = _digest(self.snapshot_digest, "snapshot_digest")
        target = _digest(self.target_path_digest, "target_path_digest")
        if restore != deterministic_signed_retirement_restore_id(
            owner_id=owner,
            snapshot_digest=snapshot,
            target_path_digest=target,
        ):
            raise ValueError("restore_id differs from chain scope.")
        records = _integer(
            self.snapshot_record_count,
            "snapshot_record_count",
            1,
            _MAX_RECORDS,
        )
        for field in (
            "restore_target_verification_digest",
            "custody_id",
            "custody_manifest_digest",
            "pre_receipt_digest",
            "backup_sha256",
            "pre_actor_id_digest",
            "pre_binding_digest",
            "post_receipt_digest",
            "post_target_verification_digest",
            "post_actor_id_digest",
            "post_binding_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if self.restore_target_verification_digest != self.post_target_verification_digest:
            raise ValueError("restore and post-receipt verification digests differ.")
        object.__setattr__(
            self,
            "backup_size_bytes",
            _integer(
                self.backup_size_bytes,
                "backup_size_bytes",
                1,
                1024 * 1024 * 1024 * 1024,
            ),
        )
        for field in ("pre_binding_method", "post_binding_method"):
            object.__setattr__(
                self,
                field,
                _identifier(getattr(self, field), field, 50),
            )
        restore_completed = _timestamp(
            self.restore_completed_at,
            "restore_completed_at",
        )
        pre_bound = _timestamp(self.pre_bound_at, "pre_bound_at")
        post_bound = _timestamp(self.post_bound_at, "post_bound_at")
        generated = _timestamp(self.generated_at, "generated_at")
        if post_bound < pre_bound or generated < post_bound or generated < restore_completed:
            raise ValueError("chain timestamps are not monotonic.")
        status = _identifier(self.legal_hold_status, "legal_hold_status", 30)
        if status not in _HOLD_STATUSES:
            raise ValueError("legal hold status is unsupported.")
        artifacts = tuple(self.artifacts)
        if not artifacts or len(artifacts) > _MAX_RECORDS:
            raise ValueError("chain requires bounded completed artifact evidence.")
        seen: set[str] = set()
        for value in artifacts:
            if not isinstance(value, CustodyArtifactEvidence):
                raise ValueError("artifact evidence is invalid.")
            if value.artifact_id in seen:
                raise ValueError("chain contains duplicate artifact IDs.")
            seen.add(value.artifact_id)
            if (
                value.backup_sha256 != self.backup_sha256
                or value.backup_size_bytes != self.backup_size_bytes
                or value.receipt_digest != self.pre_receipt_digest
                or value.binding_method != self.pre_binding_method
                or value.binding_digest != self.pre_binding_digest
                or value.actor_id_digest != self.pre_actor_id_digest
            ):
                raise ValueError("artifact evidence differs from pre-restore custody.")
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("chain schema is unsupported.")
        for field in (
            "contains_source_text",
            "contains_assertion_secrets",
            "contains_raw_paths",
            "mutation_performed",
        ):
            if getattr(self, field) is not False:
                raise ValueError(f"{field} must be false.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "restore_id", restore)
        object.__setattr__(self, "snapshot_digest", snapshot)
        object.__setattr__(self, "target_path_digest", target)
        object.__setattr__(self, "snapshot_record_count", records)
        object.__setattr__(self, "restore_completed_at", restore_completed)
        object.__setattr__(self, "pre_bound_at", pre_bound)
        object.__setattr__(self, "post_bound_at", post_bound)
        object.__setattr__(self, "legal_hold_status", status)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "generated_at", generated)
        stable = self._stable_payload()
        digest = _digest(self.chain_digest, "chain_digest")
        if digest != _canonical_digest(stable):
            raise ValueError("chain_digest differs from manifest.")
        object.__setattr__(self, "chain_digest", digest)

    def _stable_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("chain_digest", None)
        for key in (
            "contains_source_text",
            "contains_assertion_secrets",
            "contains_raw_paths",
            "mutation_performed",
        ):
            payload.pop(key, None)
        return {
            "scope": "rigorousrag-external-restore-chain-of-custody-v1",
            **payload,
        }

    def public_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuthenticatedCustodyEnvelope:
    algorithm: str
    key_id: str
    manifest: RestoreChainOfCustodyManifest
    authentication_tag: str
    schema_version: int = _ENVELOPE_SCHEMA_VERSION
    contains_key_material: bool = False
    mutation_performed: bool = False

    def __post_init__(self) -> None:
        algorithm = _identifier(self.algorithm, "algorithm", 30)
        if algorithm != "hmac-sha256":
            raise ValueError("authentication algorithm is unsupported.")
        key_id = _identifier(self.key_id, "key_id", 200)
        if not isinstance(self.manifest, RestoreChainOfCustodyManifest):
            raise ValueError("authenticated envelope manifest is invalid.")
        tag = _digest(self.authentication_tag, "authentication_tag")
        if self.schema_version != _ENVELOPE_SCHEMA_VERSION:
            raise ValueError("authenticated envelope schema is unsupported.")
        if self.contains_key_material is not False or self.mutation_performed is not False:
            raise ValueError("authenticated envelope safety flags must be false.")
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "key_id", key_id)
        object.__setattr__(self, "authentication_tag", tag)

    def authenticated_payload(self) -> dict[str, Any]:
        return {
            "scope": "rigorousrag-restore-custody-hmac-envelope-v1",
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "manifest": self.manifest.public_payload(),
            "schema_version": self.schema_version,
        }

    def public_payload(self) -> dict[str, Any]:
        return asdict(self)


def _manifest_from_dict(raw: dict[str, Any]) -> RestoreChainOfCustodyManifest:
    expected = {
        "owner_id",
        "restore_id",
        "snapshot_digest",
        "target_path_digest",
        "snapshot_record_count",
        "restore_target_verification_digest",
        "restore_completed_at",
        "custody_id",
        "custody_manifest_digest",
        "pre_receipt_digest",
        "backup_sha256",
        "backup_size_bytes",
        "pre_actor_id_digest",
        "pre_binding_method",
        "pre_binding_digest",
        "pre_bound_at",
        "post_receipt_digest",
        "post_target_verification_digest",
        "post_actor_id_digest",
        "post_binding_method",
        "post_binding_digest",
        "post_bound_at",
        "legal_hold_status",
        "artifacts",
        "generated_at",
        "chain_digest",
        "schema_version",
        "contains_source_text",
        "contains_assertion_secrets",
        "contains_raw_paths",
        "mutation_performed",
    }
    if set(raw) != expected or not isinstance(raw["artifacts"], list):
        raise ValueError("chain manifest schema is invalid.")
    artifacts = tuple(CustodyArtifactEvidence(**value) for value in raw["artifacts"])
    return RestoreChainOfCustodyManifest(
        **{**raw, "artifacts": artifacts}
    )


def build_restore_chain_of_custody(
    *,
    restore_id: str,
    snapshot_path: str | os.PathLike[str],
    target_db_path: str | os.PathLike[str],
    backup_path: str | os.PathLike[str],
    pre_receipt_path: str | os.PathLike[str],
    post_receipt_path: str | os.PathLike[str],
    restore_journal: Any,
    custody_store: Any,
    artifact_journal: Any,
    hold_store: Any | None = None,
    now: float | None = None,
    limit: int = _MAX_RECORDS,
) -> RestoreChainOfCustodyManifest:
    selected_restore = _digest(restore_id, "restore_id")
    count = _integer(limit, "limit", 1, _MAX_RECORDS)
    snapshot = verify_signed_retirement_snapshot(snapshot_path)
    validate_terminal_snapshot(snapshot)
    target_digest = target_path_digest(target_db_path)
    expected_restore = deterministic_signed_retirement_restore_id(
        owner_id=snapshot.owner_id,
        snapshot_digest=snapshot.snapshot_digest,
        target_path_digest=target_digest,
    )
    if selected_restore != expected_restore:
        raise RuntimeError("restore ID differs from live snapshot/target scope.")
    if not callable(getattr(restore_journal, "get", None)):
        raise ValueError("restore journal lacks the required read boundary.")
    restore = restore_journal.get(selected_restore)
    if (
        restore.state != "completed"
        or restore.phase != "verified"
        or restore.owner_id != snapshot.owner_id
        or restore.snapshot_digest != snapshot.snapshot_digest
        or restore.target_path_digest != target_digest
        or restore.snapshot_record_count != snapshot.record_count
        or restore.target_verification_digest is None
        or restore.completed_at is None
    ):
        raise RuntimeError("restore intent is not one completed exact chain scope.")
    disposition, current_verification = inspect_restored_target(
        snapshot=snapshot,
        target_db_path=target_db_path,
    )
    if disposition != "exact" or current_verification != restore.target_verification_digest:
        raise RuntimeError("live restored target differs from completed restore intent.")
    pre = verify_pre_restore_backup_receipt(
        receipt_path=pre_receipt_path,
        backup_path=backup_path,
    )
    post = verify_post_restore_comparison_receipt(post_receipt_path)
    if (
        pre.owner_id != snapshot.owner_id
        or pre.snapshot_digest != snapshot.snapshot_digest
        or pre.target_path_digest != target_digest
        or post.owner_id != snapshot.owner_id
        or post.restore_id != selected_restore
        or post.snapshot_digest != snapshot.snapshot_digest
        or post.target_path_digest != target_digest
        or post.pre_restore_receipt_digest != pre.receipt_digest
        or post.backup_sha256 != pre.backup_sha256
        or post.target_verification_digest != current_verification
        or post.target_record_count != snapshot.record_count
    ):
        raise RuntimeError("live custody receipts differ from completed restore scope.")
    if not callable(getattr(custody_store, "list", None)):
        raise ValueError("custody store lacks the required read boundary.")
    custody_values = tuple(
        custody_store.list(
            owner_id=snapshot.owner_id,
            state="post_bound",
            limit=count,
        )
    )
    if len(custody_values) >= count:
        raise RuntimeError("custody manifest lookup reached the bounded result limit.")
    custody_matches = [
        value for value in custody_values if value.restore_id == selected_restore
    ]
    if len(custody_matches) != 1:
        raise RuntimeError("chain requires exactly one post-bound custody manifest.")
    custody = custody_matches[0]
    if (
        custody.snapshot_digest != snapshot.snapshot_digest
        or custody.target_path_digest != target_digest
        or custody.pre_receipt_digest != pre.receipt_digest
        or custody.backup_sha256 != pre.backup_sha256
        or custody.backup_size_bytes != pre.backup_size_bytes
        or custody.pre_bound_actor_id != pre.actor_id
        or custody.pre_bound_method != pre.binding_method
        or custody.pre_bound_binding_digest != pre.binding_digest
        or custody.post_receipt_digest != post.receipt_digest
        or custody.target_verification_digest != post.target_verification_digest
        or custody.post_bound_actor_id != post.actor_id
        or custody.post_bound_method != post.binding_method
        or custody.post_bound_binding_digest != post.binding_digest
        or custody.post_bound_at != post.compared_at
    ):
        raise RuntimeError("custody manifest differs from live receipt evidence.")
    if not callable(getattr(artifact_journal, "list", None)):
        raise ValueError("artifact journal lacks the required read boundary.")
    artifact_values = tuple(
        artifact_journal.list(
            owner_id=snapshot.owner_id,
            state="completed",
            limit=count,
        )
    )
    if len(artifact_values) >= count:
        raise RuntimeError("artifact lookup reached the bounded result limit.")
    artifacts: list[CustodyArtifactEvidence] = []
    for value in artifact_values:
        derived_restore = deterministic_signed_retirement_restore_id(
            owner_id=value.owner_id,
            snapshot_digest=value.snapshot_digest,
            target_path_digest=value.target_path_digest,
        )
        if derived_restore != selected_restore:
            continue
        if (
            value.phase != "verified"
            or value.disposition != "paired"
            or value.backup_sha256 != pre.backup_sha256
            or value.backup_size_bytes != pre.backup_size_bytes
            or value.receipt_digest != pre.receipt_digest
            or value.receipt_actor_id != pre.actor_id
            or value.receipt_binding_method != pre.binding_method
            or value.receipt_binding_digest != pre.binding_digest
            or value.completed_at is None
        ):
            continue
        artifacts.append(
            CustodyArtifactEvidence(
                artifact_id=value.artifact_id,
                backup_path_digest=value.backup_path_digest,
                receipt_path_digest=value.receipt_path_digest,
                backup_sha256=value.backup_sha256,
                backup_size_bytes=value.backup_size_bytes,
                receipt_digest=value.receipt_digest,
                actor_id_digest=_actor_digest(value.receipt_actor_id),
                binding_method=value.receipt_binding_method,
                binding_digest=value.receipt_binding_digest,
                completed_at=value.completed_at,
            )
        )
    if not artifacts:
        raise RuntimeError("chain requires a completed artifact pair for this receipt.")
    artifacts.sort(key=lambda value: value.artifact_id)
    hold_status = "not_checked"
    if hold_store is not None:
        if not callable(getattr(hold_store, "active_restore_ids", None)):
            raise ValueError("hold store lacks the required read boundary.")
        active = hold_store.active_restore_ids(
            owner_id=snapshot.owner_id,
            limit=count,
        )
        hold_status = "active" if selected_restore in active else "inactive"
    generated = _timestamp(time.time() if now is None else now, "now")
    stable = {
        "scope": "rigorousrag-external-restore-chain-of-custody-v1",
        "owner_id": snapshot.owner_id,
        "restore_id": selected_restore,
        "snapshot_digest": snapshot.snapshot_digest,
        "target_path_digest": target_digest,
        "snapshot_record_count": snapshot.record_count,
        "restore_target_verification_digest": current_verification,
        "restore_completed_at": restore.completed_at,
        "custody_id": custody.custody_id,
        "custody_manifest_digest": custody.manifest_digest,
        "pre_receipt_digest": pre.receipt_digest,
        "backup_sha256": pre.backup_sha256,
        "backup_size_bytes": pre.backup_size_bytes,
        "pre_actor_id_digest": _actor_digest(pre.actor_id),
        "pre_binding_method": pre.binding_method,
        "pre_binding_digest": pre.binding_digest,
        "pre_bound_at": custody.pre_bound_at,
        "post_receipt_digest": post.receipt_digest,
        "post_target_verification_digest": post.target_verification_digest,
        "post_actor_id_digest": _actor_digest(post.actor_id),
        "post_binding_method": post.binding_method,
        "post_binding_digest": post.binding_digest,
        "post_bound_at": post.compared_at,
        "legal_hold_status": hold_status,
        "artifacts": [asdict(value) for value in artifacts],
        "generated_at": generated,
        "schema_version": _SCHEMA_VERSION,
    }
    return RestoreChainOfCustodyManifest(
        owner_id=snapshot.owner_id,
        restore_id=selected_restore,
        snapshot_digest=snapshot.snapshot_digest,
        target_path_digest=target_digest,
        snapshot_record_count=snapshot.record_count,
        restore_target_verification_digest=current_verification,
        restore_completed_at=restore.completed_at,
        custody_id=custody.custody_id,
        custody_manifest_digest=custody.manifest_digest,
        pre_receipt_digest=pre.receipt_digest,
        backup_sha256=pre.backup_sha256,
        backup_size_bytes=pre.backup_size_bytes,
        pre_actor_id_digest=_actor_digest(pre.actor_id),
        pre_binding_method=pre.binding_method,
        pre_binding_digest=pre.binding_digest,
        pre_bound_at=custody.pre_bound_at,
        post_receipt_digest=post.receipt_digest,
        post_target_verification_digest=post.target_verification_digest,
        post_actor_id_digest=_actor_digest(post.actor_id),
        post_binding_method=post.binding_method,
        post_binding_digest=post.binding_digest,
        post_bound_at=post.compared_at,
        legal_hold_status=hold_status,
        artifacts=tuple(artifacts),
        generated_at=generated,
        chain_digest=_canonical_digest(stable),
    )


def export_restore_chain_of_custody(
    *,
    output_path: str | os.PathLike[str],
    **kwargs: Any,
) -> RestoreChainOfCustodyManifest:
    manifest = build_restore_chain_of_custody(**kwargs)
    output = _path(output_path, label="output_path")
    _atomic_create(output, _canonical_bytes(manifest.public_payload()) + b"\n")
    return manifest


def verify_restore_chain_of_custody(
    path: str | os.PathLike[str],
) -> RestoreChainOfCustodyManifest:
    return _manifest_from_dict(_decode_json(path, label="chain_manifest"))


def _read_hmac_key(path: str | os.PathLike[str]) -> bytes:
    key = _read_regular(path, label="hmac_key_path", maximum=_MAX_KEY_BYTES)
    if len(key) < 32:
        raise ValueError("HMAC key must contain at least 32 bytes.")
    return key


def authenticate_restore_chain_of_custody(
    *,
    manifest_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    key_id: str,
    key_path: str | os.PathLike[str],
) -> AuthenticatedCustodyEnvelope:
    manifest = verify_restore_chain_of_custody(manifest_path)
    selected_key_id = _identifier(key_id, "key_id", 200)
    key = _read_hmac_key(key_path)
    authenticated = {
        "scope": "rigorousrag-restore-custody-hmac-envelope-v1",
        "algorithm": "hmac-sha256",
        "key_id": selected_key_id,
        "manifest": manifest.public_payload(),
        "schema_version": _ENVELOPE_SCHEMA_VERSION,
    }
    tag = hmac.new(key, _canonical_bytes(authenticated), hashlib.sha256).hexdigest()
    envelope = AuthenticatedCustodyEnvelope(
        algorithm="hmac-sha256",
        key_id=selected_key_id,
        manifest=manifest,
        authentication_tag=tag,
    )
    output = _path(output_path, label="output_path")
    _atomic_create(output, _canonical_bytes(envelope.public_payload()) + b"\n")
    return envelope


def verify_authenticated_restore_chain_of_custody(
    *,
    envelope_path: str | os.PathLike[str],
    key_path: str | os.PathLike[str],
    expected_key_id: str | None = None,
) -> AuthenticatedCustodyEnvelope:
    raw = _decode_json(envelope_path, label="authenticated_envelope")
    expected = {
        "algorithm",
        "key_id",
        "manifest",
        "authentication_tag",
        "schema_version",
        "contains_key_material",
        "mutation_performed",
    }
    if set(raw) != expected or not isinstance(raw["manifest"], dict):
        raise ValueError("authenticated envelope schema is invalid.")
    manifest = _manifest_from_dict(raw["manifest"])
    envelope = AuthenticatedCustodyEnvelope(
        **{**raw, "manifest": manifest}
    )
    if expected_key_id is not None and envelope.key_id != _identifier(
        expected_key_id,
        "expected_key_id",
        200,
    ):
        raise PermissionError("authenticated envelope key ID differs.")
    key = _read_hmac_key(key_path)
    expected_tag = hmac.new(
        key,
        _canonical_bytes(envelope.authenticated_payload()),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected_tag, envelope.authentication_tag):
        raise PermissionError("authenticated envelope verification failed.")
    return envelope


__all__ = [
    "AuthenticatedCustodyEnvelope",
    "CustodyArtifactEvidence",
    "RestoreChainOfCustodyManifest",
    "authenticate_restore_chain_of_custody",
    "build_restore_chain_of_custody",
    "export_restore_chain_of_custody",
    "verify_authenticated_restore_chain_of_custody",
    "verify_restore_chain_of_custody",
]
