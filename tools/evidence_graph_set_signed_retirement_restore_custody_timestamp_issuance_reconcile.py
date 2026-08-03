"""Crash-recoverable one-serial custody timestamp issuance."""

from __future__ import annotations

import base64
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tools import evidence_graph_set_signed_retirement_restore_custody_export as _export
from tools.evidence_graph_set_signed_retirement_restore_custody_signature import (
    _load_private,
    _public_fingerprint,
    verify_signed_restore_chain_of_custody,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp import (
    CustodyTimestampAttestation,
    _canonical_digest,
    _envelope_digest,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_contracts import (
    CustodyTimestampIssuanceAttempt,
    timestamp_output_path_digest,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_journal import (
    CustodyTimestampIssuanceJournal,
)
from tools.evidence_graph_set_signed_retirement_snapshot import (
    _atomic_create,
    _canonical_bytes,
    _path,
)


class CustodyTimestampIssuanceRecoveryError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        issuance_id: str,
        state: str,
        phase: str,
    ) -> None:
        self.issuance_id = issuance_id
        self.state = state
        self.phase = phase
        super().__init__(message)


@dataclass(frozen=True)
class CustodyTimestampIssuanceExecution:
    issuance_id: str
    serial: str
    state: str
    phase: str
    attestation_digest: str
    output_path_digest: str
    verification_digest: str | None
    attempt_count: int
    output_created: bool
    existing_exact_output_reused: bool
    journal_mutation_performed: bool = True
    private_key_material_returned: bool = False
    raw_output_path_returned: bool = False


def _attestation_bytes(value: CustodyTimestampAttestation) -> bytes:
    return _canonical_bytes(value.public_payload())


def _attestation_digest(value: CustodyTimestampAttestation) -> str:
    return hashlib.sha256(_attestation_bytes(value)).hexdigest()


def _read_output(path: Path) -> bytes:
    reader = getattr(_export, "_read_regular", None)
    if callable(reader):
        return reader(
            path,
            label="timestamp_output",
            maximum=32 * 1024 * 1024,
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if info.st_size <= 0 or info.st_size > 32 * 1024 * 1024:
            raise ValueError("timestamp output size is invalid.")
        payload = os.read(descriptor, int(info.st_size) + 1)
        if len(payload) != info.st_size:
            raise RuntimeError("timestamp output changed while being read.")
        return payload
    finally:
        os.close(descriptor)


def _prepare_attestation(
    *,
    registry: Any,
    owner_id: str,
    authority_id: str,
    key_id: str,
    authority_private_key_path: str | os.PathLike[str],
    signed_envelope_path: str | os.PathLike[str],
    custody_signer_public_key_path: str | os.PathLike[str],
    asserted_at: float,
    nonce: bytes,
) -> CustodyTimestampAttestation:
    record = registry.get(
        owner_id=owner_id,
        authority_id=authority_id,
        key_id=key_id,
    )
    if record.state != "active":
        raise PermissionError("timestamp authority key is not active.")
    private_key = _load_private(authority_private_key_path)
    fingerprint = _public_fingerprint(private_key.public_key())
    if fingerprint != record.public_key_sha256:
        raise PermissionError("timestamp authority private key differs from registry.")
    envelope = verify_signed_restore_chain_of_custody(
        envelope_path=signed_envelope_path,
        public_key_path=custody_signer_public_key_path,
    )
    if envelope.manifest.owner_id != record.owner_id:
        raise PermissionError("custody envelope owner differs from authority registry.")
    if asserted_at < envelope.manifest.generated_at:
        raise ValueError("timestamp assertion predates custody manifest generation.")
    if asserted_at < record.registered_at:
        raise PermissionError("timestamp assertion predates authority registration.")
    if len(nonce) < 16 or len(nonce) > 1024:
        raise ValueError("timestamp nonce length is invalid.")
    nonce_digest = hashlib.sha256(nonce).hexdigest()
    stable = {
        "scope": "rigorousrag-restore-custody-timestamp-attestation-v1",
        "owner_id": record.owner_id,
        "authority_id": record.authority_id,
        "key_id": record.key_id,
        "algorithm": "ed25519",
        "public_key_sha256": fingerprint,
        "custody_envelope_sha256": _envelope_digest(envelope),
        "custody_manifest_digest": envelope.manifest.custody_manifest_digest,
        "custody_chain_digest": envelope.manifest.chain_digest,
        "asserted_at": asserted_at,
        "nonce_sha256": nonce_digest,
        "schema_version": 1,
    }
    serial = _canonical_digest(stable)
    signature = base64.b64encode(
        private_key.sign(_canonical_bytes(stable))
    ).decode("ascii")
    return CustodyTimestampAttestation(
        **{key: value for key, value in stable.items() if key != "scope"},
        serial=serial,
        signature=signature,
    )


def seed_custody_timestamp_issuance(
    *,
    journal: CustodyTimestampIssuanceJournal,
    registry: Any,
    owner_id: str,
    authority_id: str,
    key_id: str,
    authority_private_key_path: str | os.PathLike[str],
    signed_envelope_path: str | os.PathLike[str],
    custody_signer_public_key_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    confirm_output_path_digest: str,
    max_attempts: int = 3,
    now: float | None = None,
    nonce: bytes | None = None,
) -> tuple[CustodyTimestampIssuanceAttempt, CustodyTimestampAttestation]:
    timestamp = float(time.time() if now is None else now)
    output = _path(output_path, label="output_path")
    output_digest = timestamp_output_path_digest(output)
    if output_digest != confirm_output_path_digest:
        raise ValueError("timestamp output path confirmation differs.")
    if output.exists():
        raise FileExistsError("timestamp output already exists before issuance intent.")
    selected_nonce = os.urandom(32) if nonce is None else bytes(nonce)
    attestation = _prepare_attestation(
        registry=registry,
        owner_id=owner_id,
        authority_id=authority_id,
        key_id=key_id,
        authority_private_key_path=authority_private_key_path,
        signed_envelope_path=signed_envelope_path,
        custody_signer_public_key_path=custody_signer_public_key_path,
        asserted_at=timestamp,
        nonce=selected_nonce,
    )
    attempt = CustodyTimestampIssuanceAttempt.create(
        owner_id=attestation.owner_id,
        authority_id=attestation.authority_id,
        key_id=attestation.key_id,
        serial=attestation.serial,
        attestation_digest=_attestation_digest(attestation),
        output_path_digest=output_digest,
        max_attempts=max_attempts,
        now=timestamp,
    )
    return journal.seed(attempt, attestation=attestation), attestation


def _failure_name(exc: BaseException) -> str:
    name = type(exc).__name__
    return name if len(name) <= 200 else "TimestampIssuanceFailure"


def execute_custody_timestamp_issuance(
    issuance_id: str,
    *,
    worker_id: str,
    lease_seconds: int,
    output_path: str | os.PathLike[str],
    journal: CustodyTimestampIssuanceJournal,
    registry: Any,
    now: float | None = None,
    _phase_hook: Callable[[str], None] | None = None,
) -> CustodyTimestampIssuanceExecution:
    timestamp = float(time.time() if now is None else now)
    current = journal.get(issuance_id)
    if current.state == "completed":
        return CustodyTimestampIssuanceExecution(
            issuance_id=current.issuance_id,
            serial=current.serial,
            state=current.state,
            phase=current.phase,
            attestation_digest=current.attestation_digest,
            output_path_digest=current.output_path_digest,
            verification_digest=current.verification_digest,
            attempt_count=current.attempt_count,
            output_created=False,
            existing_exact_output_reused=True,
        )
    claimed = journal.claim(
        issuance_id,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        now=timestamp,
    )
    output_created = False
    reused = False
    try:
        output = _path(output_path, label="output_path")
        if timestamp_output_path_digest(output) != claimed.output_path_digest:
            raise RuntimeError("timestamp output path differs from issuance intent.")
        attestation = journal.get_attestation(claimed.issuance_id)
        record = registry.get(
            owner_id=claimed.owner_id,
            authority_id=claimed.authority_id,
            key_id=claimed.key_id,
        )
        if record.public_key_sha256 != attestation.public_key_sha256:
            raise PermissionError("timestamp authority registry fingerprint changed.")
        if attestation.asserted_at < record.registered_at:
            raise PermissionError("timestamp assertion predates authority registration.")
        if record.retired_at is not None and attestation.asserted_at > record.retired_at:
            raise PermissionError("timestamp assertion postdates authority retirement.")
        expected = _attestation_bytes(attestation) + b"\n"
        if claimed.phase == "output_published":
            if not output.exists():
                raise RuntimeError("published timestamp output is missing.")
            actual = _read_output(output)
            if actual != expected:
                raise RuntimeError(
                    "published timestamp output differs from issuance intent."
                )
            reused = True
        elif output.exists():
            actual = _read_output(output)
            if actual != expected:
                raise RuntimeError("timestamp output collision detected.")
            reused = True
        else:
            _atomic_create(output, expected)
            output_created = True
        if _phase_hook is not None:
            _phase_hook("after_output_publish")
        current = journal.record_output_published(
            claimed.issuance_id,
            worker_id=worker_id,
            now=timestamp,
        )
        if _phase_hook is not None:
            _phase_hook("after_output_phase")
        actual = _read_output(output)
        if actual != expected:
            raise RuntimeError("timestamp output changed before completion.")
        verification = hashlib.sha256(
            _canonical_bytes(
                {
                    "scope": "rigorousrag-custody-timestamp-issuance-verification-v1",
                    "issuance_id": current.issuance_id,
                    "serial": current.serial,
                    "attestation_digest": current.attestation_digest,
                    "output_path_digest": current.output_path_digest,
                    "output_sha256": hashlib.sha256(actual).hexdigest(),
                }
            )
        ).hexdigest()
        completed = journal.complete(
            current.issuance_id,
            worker_id=worker_id,
            verification_digest=verification,
            now=timestamp,
        )
        return CustodyTimestampIssuanceExecution(
            issuance_id=completed.issuance_id,
            serial=completed.serial,
            state=completed.state,
            phase=completed.phase,
            attestation_digest=completed.attestation_digest,
            output_path_digest=completed.output_path_digest,
            verification_digest=completed.verification_digest,
            attempt_count=completed.attempt_count,
            output_created=output_created,
            existing_exact_output_reused=reused,
        )
    except Exception as exc:
        failure = _failure_name(exc)
        current = journal.get(claimed.issuance_id)
        if current.state == "running":
            try:
                current = journal.fail(
                    current.issuance_id,
                    worker_id=worker_id,
                    failure_type=failure,
                    now=timestamp,
                )
            except (KeyError, RuntimeError):
                current = journal.get(claimed.issuance_id)
        raise CustodyTimestampIssuanceRecoveryError(
            f"custody timestamp issuance failed ({failure}).",
            issuance_id=current.issuance_id,
            state=current.state,
            phase=current.phase,
        ) from exc


def execute_next_custody_timestamp_issuance(
    *,
    owner_id: str,
    worker_id: str,
    lease_seconds: int,
    output_path_resolver: Callable[
        [CustodyTimestampIssuanceAttempt],
        str | os.PathLike[str],
    ],
    journal: CustodyTimestampIssuanceJournal,
    registry: Any,
    now: float | None = None,
) -> CustodyTimestampIssuanceExecution | None:
    timestamp = float(time.time() if now is None else now)
    issuance_id = journal.next_claimable_id(owner_id=owner_id, now=timestamp)
    if issuance_id is None:
        return None
    attempt = journal.get(issuance_id)
    return execute_custody_timestamp_issuance(
        issuance_id,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        output_path=output_path_resolver(attempt),
        journal=journal,
        registry=registry,
        now=timestamp,
    )


__all__ = [
    "CustodyTimestampIssuanceExecution",
    "CustodyTimestampIssuanceRecoveryError",
    "execute_custody_timestamp_issuance",
    "execute_next_custody_timestamp_issuance",
    "seed_custody_timestamp_issuance",
]
